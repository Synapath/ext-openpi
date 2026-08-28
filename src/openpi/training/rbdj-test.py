from collections import Counter
import hashlib
import math

from flax import nnx, traverse_util
import jax
import numpy as np
import pytest

from openpi.training import config, rbdj
from openpi import transforms


def test_scientific_config_and_strict_count():
    c = config.get_config("pi05_g2_rbdj_three_task_s0_strict")
    assert (c.batch_size, c.num_train_steps, c.seed, c.fsdp_devices) == (64, 10000, 0, 1)
    assert c.ema_decay is None and c.model.action_horizon == 50 and c.model.pi05
    assert (c.lr_schedule.warmup_steps, c.lr_schedule.decay_steps) == (500, 10000)
    assert (c.lr_schedule.peak_lr, c.lr_schedule.decay_lr) == (2.5e-5, 2.5e-6)
    assert not c.data.adapt_to_pi and c.data.use_delta_joint_actions
    state = jax.eval_shape(lambda key: nnx.state(c.model.create(key)), jax.random.key(0))
    trainable = traverse_util.flatten_dict(state.filter(c.trainable_filter).to_pure_dict(), sep="/")
    assert sum(math.prod(v.shape) for v in trainable.values()) == 52153376
    assert not any("img" in k for k in trainable)
    assert all(
        "lora" in k or k.split("/")[0] in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}
        for k in trainable
    )


def test_missing_artifacts_cannot_fall_back_to_random():
    c = config.get_config("pi05_g2_rbdj_three_task_s0_strict")
    with pytest.raises(ValueError, match="no random fallback"):
        rbdj.ManifestDataLoader(c, c.data.base_config)


class FakeDraws:
    def __getitem__(self, i):
        return i // 64, i % 64, 9, 900, i % 20


def cursor_loader():
    loader = object.__new__(rbdj.ManifestDataLoader)
    loader.draws = FakeDraws()
    loader.committed_updates = 0
    loader.stop_update = 10
    loader.pending = None
    loader.prefix = hashlib.sha256()
    loader.task_counts, loader.episode_counts = Counter(), Counter()
    loader.identity = {"draw_manifest_sha256": "draw", "split_sha256": "split", "view_manifest_sha256": "view"}
    loader.readback_path = None
    return loader


def test_prefetch_is_not_a_committed_cursor_and_restore_recomputes_prefix():
    loader = cursor_loader()
    loader.pending = np.array([loader.draws[i] for i in range(64)], dtype=np.uint32)
    with pytest.raises(ValueError):
        loader.cursor_receipt(0)
    with pytest.raises(ValueError):
        loader.commit_batch(2)
    loader.commit_batch(1)
    receipt = loader.cursor_receipt(1)
    restored = cursor_loader()
    restored.restore_cursor(receipt, 1)
    assert restored.cursor_receipt(1) == receipt
    assert restored.draws[restored.committed_updates * 64] == loader.draws[64]
    with pytest.raises(ValueError):
        restored.commit_batch(2)


@pytest.mark.parametrize("key", ["completed_updates", "prefix_sha256", "draw_manifest_sha256", "episode_counts"])
def test_cursor_rejects_corruption(key):
    loader = cursor_loader()
    receipt = loader.cursor_receipt(0)
    receipt[key] = 2 if key == "completed_updates" else "bad"
    with pytest.raises(ValueError):
        loader.restore_cursor(receipt, 0)


def test_native_arm_delta_and_gripper_absolute_roundtrip():
    rng = np.random.default_rng(0)
    state = rng.normal(size=14).astype(np.float32)
    actions = rng.normal(size=(50, 14)).astype(np.float32)
    mask = transforms.make_bool_mask(6, -1, 6, -1)
    delta = transforms.DeltaActions(mask)({"state": state.copy(), "actions": actions.copy()})
    np.testing.assert_array_equal(delta["actions"][:, [6, 13]], actions[:, [6, 13]])
    restored = transforms.AbsoluteActions(mask)(delta)["actions"]
    np.testing.assert_allclose(restored, actions, atol=5e-7, rtol=0)


def test_orbax_roundtrip_binds_committed_cursor(tmp_path):
    import json
    import jax.numpy as jnp
    import optax
    from openpi.shared import array_typing as at
    from openpi.training import checkpoints, utils

    tiny = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    params = nnx.state(tiny)
    tx = optax.adam(1e-3)
    with at.disable_typechecking():
        state = utils.TrainState(
            step=jnp.asarray(1),
            params=params,
            model_def=nnx.graphdef(tiny),
            opt_state=tx.init(params),
            tx=tx,
            ema_decay=None,
        )
    loader = cursor_loader()
    loader._data_config = config.DataConfig()
    loader.pending = np.array([loader.draws[i] for i in range(64)], dtype=np.uint32)
    loader.commit_batch(1)
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoints", keep_period=None, overwrite=False, resume=False, checkpoint_steps={0}
    )
    try:
        checkpoints.save_state(manager, state, loader, 0)
        manager.wait_until_finished()
        restored_loader = cursor_loader()
        restored = checkpoints.restore_state(manager, state, restored_loader, step=0)
        assert int(restored.step) == 1
        assert restored_loader.cursor_receipt(1) == loader.cursor_receipt(1)
        path = tmp_path / "checkpoints/0/assets/data-cursor.json"
        receipt = json.loads(path.read_text())
        receipt["prefix_sha256"] = "invalid"
        path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="prefix"):
            checkpoints.restore_state(manager, state, cursor_loader(), step=0)
    finally:
        manager.close()
