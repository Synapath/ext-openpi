import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.training import checkpoints
from openpi.training import diagnostics
from openpi.training import utils


def toy_state(*, partial=True):
    params = nnx.state(
        nnx.Dict(
            {
                "frozen": nnx.Param(jnp.array([1.0078125, -3.0], dtype=jnp.bfloat16)),
                "trainable": nnx.Param(jnp.array([2.0, 4.0])),
            }
        )
    )
    ema = nnx.state(nnx.Dict({"trainable": nnx.Param(jnp.array([1.0, 3.0]))})) if partial else params
    return utils.TrainState(
        step=jnp.array(2, jnp.int32),
        params=params,
        model_def=nnx.graphdef(nnx.Dict()),
        opt_state=(),
        tx=optax.sgd(0.01),
        ema_decay=0.99,
        ema_params=ema,
        ema_trainable_only=partial,
    )


@pytest.mark.parametrize("partial", [False, True])
def test_checkpoint_split_merge_and_frozen_identity(partial):
    state = toy_state(partial=partial)
    before = np.asarray(state.params["frozen"].value).tobytes()
    train_state, params = checkpoints._split_params(state)  # noqa: SLF001 - test serialization boundary
    restored = checkpoints._merge_params(train_state, {"params": params})  # noqa: SLF001
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.asarray(params["frozen"].value).tobytes() == before
    assert np.asarray(state.params["frozen"].value).tobytes() == before
    np.testing.assert_array_equal(state.params["trainable"].value, [2.0, 4.0])
    if partial:
        np.testing.assert_array_equal(params["trainable"].value, [1.0, 3.0])
        assert list(restored.ema_params) == ["trainable"]


def test_partial_ema_recurrence_never_touches_frozen():
    state = toy_state()
    expected = np.array([1.0, 3.0], np.float32)
    for _ in range(35):
        state = dataclasses.replace(
            state,
            ema_params=jax.tree.map(
                lambda a, b: 0.99 * a + 0.01 * b, state.ema_params, nnx.State({"trainable": state.params["trainable"]})
            ),
        )
        expected = 0.99 * expected + 0.01 * np.array([2.0, 4.0], np.float32)
    np.testing.assert_allclose(state.ema_params["trainable"].value, expected, rtol=1e-6, atol=1e-7)
    assert checkpoints.inference_params(state)["frozen"].value.dtype == jnp.bfloat16
    np.testing.assert_array_equal(checkpoints.inference_params(state)["frozen"].value, state.params["frozen"].value)


def test_flow_decompositions_are_element_weighted():
    x = jnp.asarray(np.random.default_rng(0).uniform(size=(3, 32, 32)))
    m = diagnostics.flow_metrics(x)
    np.testing.assert_allclose(m["flow_mse"], (14 * m["flow_mse_real14"] + 18 * m["flow_mse_pad18"]) / 32, rtol=1e-6)
    np.testing.assert_allclose(
        m["flow_mse_real14"],
        (
            6 * m["flow_mse_left_arm"]
            + 6 * m["flow_mse_right_arm"]
            + m["flow_mse_left_gripper"]
            + m["flow_mse_right_gripper"]
        )
        / 14,
        rtol=1e-6,
    )
    np.testing.assert_allclose(m["flow_mse_real14"], (m["flow_mse_exec_prefix"] + m["flow_mse_future"]) / 2, rtol=1e-6)
    assert m["element_count"] == 3 * 32 * 32
    np.testing.assert_allclose(m["flow_mse_prefix10"], np.asarray(x)[:, :10, :14].mean(), rtol=1e-6)


def test_clip_uses_individual_events():
    metrics = [diagnostics.optimizer_metrics([jnp.array([v])], 1.0) for v in (0.5, 2.0)]
    assert sum(float(x["optim/clip_fraction"]) for x in metrics) / 2 == 0.5
    assert float(metrics[1]["optim/clip_scale"]) == 0.5
    assert int(diagnostics.nonfinite_count(jnp.array([1.0, jnp.nan, jnp.inf]))) == 2


def test_temporal_mask_uses_global_elements_and_zero_padding_gradient():
    from openpi.models.model import weight_action_loss

    mask = jnp.arange(32)[None, :] < jnp.array([1, 32])[:, None]
    losses = jnp.ones((2, 32)).at[0].set(4.0)
    value = jnp.mean(weight_action_loss(losses, mask))
    np.testing.assert_allclose(value, 36 / 33, rtol=1e-6)
    gradient = jax.grad(lambda x: jnp.mean(weight_action_loss(x, mask)))(losses)
    np.testing.assert_array_equal(gradient[0, 1:], 0)
    np.testing.assert_allclose(gradient[mask], 1 / 33, rtol=1e-6)
    squared = jnp.broadcast_to(losses[..., None], (2, 32, 32))
    metrics = diagnostics.flow_metrics(squared, valid_mask=mask)
    assert metrics["element_count"] == 33 * 32
    np.testing.assert_allclose(metrics["flow_mse"], value, rtol=1e-6)
    only_one = diagnostics.flow_metrics(squared[:1], valid_mask=mask[:1])
    assert only_one["flow_mse_future_count"] == 0
    assert only_one["flow_mse_future"] == 0


def test_log_window_flow_uses_valid_counts_but_loss_averages_updates():
    result = diagnostics.reduce_step_metrics(
        {
            "train/flow_mse": jnp.array([4.0, 1.0]),
            "train/flow_mse_count": jnp.array([1, 32]),
            "loss": jnp.array([4.0, 1.0]),
        }
    )
    np.testing.assert_allclose(result["train/flow_mse"], 36 / 33, rtol=1e-6)
    assert result["train/flow_mse_count"] == 33
    assert result["loss"] == 2.5
