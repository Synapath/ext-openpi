import copy

import numpy as np
import pytest

from openpi.policies import aloha_policy


def _example() -> dict:
    example = aloha_policy.make_aloha_example()
    example["actions"] = np.arange(14, dtype=np.float32)[None, :]
    return example


def test_head_only_preserves_non_camera_inputs_and_masks_wrists() -> None:
    example = _example()
    three_view = aloha_policy.AlohaInputs(
        adapt_to_pi=False,
        enabled_cameras=("cam_high", "cam_left_wrist", "cam_right_wrist"),
    )(copy.deepcopy(example))
    head_only = aloha_policy.AlohaInputs(
        adapt_to_pi=False,
        enabled_cameras=("cam_high",),
    )(copy.deepcopy(example))

    np.testing.assert_array_equal(head_only["image"]["base_0_rgb"], three_view["image"]["base_0_rgb"])
    np.testing.assert_array_equal(head_only["state"], three_view["state"])
    np.testing.assert_array_equal(head_only["actions"], three_view["actions"])
    assert head_only["prompt"] == three_view["prompt"]

    assert bool(three_view["image_mask"]["base_0_rgb"])
    assert bool(three_view["image_mask"]["left_wrist_0_rgb"])
    assert bool(three_view["image_mask"]["right_wrist_0_rgb"])
    assert bool(head_only["image_mask"]["base_0_rgb"])
    assert not bool(head_only["image_mask"]["left_wrist_0_rgb"])
    assert not bool(head_only["image_mask"]["right_wrist_0_rgb"])
    assert not np.any(head_only["image"]["left_wrist_0_rgb"])
    assert not np.any(head_only["image"]["right_wrist_0_rgb"])


def test_head_camera_cannot_be_disabled() -> None:
    with pytest.raises(ValueError, match="cam_high must remain enabled"):
        aloha_policy.AlohaInputs(enabled_cameras=("cam_left_wrist",))(_example())
