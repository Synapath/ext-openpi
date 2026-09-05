from openpi.training.probes import aggregate
from openpi.training.probes import select_anchors


def test_episode_weighting_does_not_favor_longer_probe_inventory():
    rows = [{"episode_id": 1}, {"episode_id": 1}, {"episode_id": 2}]
    result = aggregate({"flow_mse": [1.0, 3.0, 10.0]}, rows)
    assert result["flow_mse"] == 6.0
    assert result["flow_mse_element_weighted"] == 14 / 3
    assert result["anchor_noise_count"] == 3
    assert result["episode_count"] == 2


def test_probe_rng_identity_is_order_independent_and_covers_episodes():
    rows = [{"episode_id": i, "task_index": 23, "split": "train", "anchors": 168, "length": 200} for i in range(90)]
    split = {"episodes": rows, "revision": "test-only"}
    anchors = select_anchors(split, "train")
    assert len(anchors) == 128
    assert len({r["episode_id"] for r in anchors}) == 90
    assert len({(r["episode_id"], r["frame"]) for r in anchors}) == 128
    assert anchors == select_anchors({**split, "episodes": rows[::-1]}, "train")
    assert all(0 <= r["frame"] < 168 for r in anchors)


def test_masked_probe_aggregates_elements_inside_each_episode():
    rows = [{"episode_id": 1}, {"episode_id": 1}, {"episode_id": 2}]
    result = aggregate({"flow_mse": [4.0, 1.0, 10.0]}, rows, {"flow_mse": [1, 32, 2]})
    assert result["flow_mse"] == ((36 / 33) + 10) / 2
    assert result["flow_mse_element_weighted"] == 56 / 35
    assert result["flow_mse_count"] == 35
    empty = aggregate({"future": [0.0, 0.0, 0.0]}, rows, {"future": [0, 0, 0]})
    assert empty["future_count"] == 0
    assert "future" not in empty


def test_native_padding_also_excludes_the_sources_final_fake_action():
    from manip_datasets.robodojo_pi05 import MASKED_SCHEMA
    import numpy as np
    import pytest

    from openpi.training.rbdj import temporal_mask

    split = {"schema": MASKED_SCHEMA, "horizon": 32}
    row = {"length": 200}
    native = {"action_is_pad": np.arange(32) + 198 >= 200}
    mask = temporal_mask(native, split, row, 198)
    assert mask.sum() == 1
    assert not native["action_is_pad"][1]  # Native accepts the source's final fake action.
    assert not mask[1]
    with pytest.raises(ValueError, match="native temporal padding differs"):
        temporal_mask({"action_is_pad": np.zeros(32, dtype=bool)}, split, row, 198)
