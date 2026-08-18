import math

from flax import nnx
from flax import traverse_util
import jax
import pytest

import openpi.training.config as config_module

FRAME_COUNTS = (9133, 8398, 5990, 4892, 7188)
REPO_ID = "RoboTwin-g51-easy5-clean-aloha_agilex-joint"


@pytest.mark.parametrize("camera_condition", ["head_only", "three_view"])
@pytest.mark.parametrize("seed", [0, 1])
def test_g51_camera_recipe_contract(camera_condition: str, seed: int) -> None:
    name = f"pi05_g51_easy5_{camera_condition}_s{seed}_builtin_dual_lora"
    config = config_module.get_config(name)
    data_config = config.data.create(config.assets_dirs, config.model)
    expected_cameras = (
        ("cam_high",)
        if camera_condition == "head_only"
        else ("cam_high", "cam_left_wrist", "cam_right_wrist")
    )

    assert config.batch_size == 32
    assert config.seed == seed
    assert config.ema_decay is None
    assert config.fsdp_devices == 1
    assert config.num_train_steps == 11_126
    assert config.lr_schedule.warmup_steps == 311
    assert config.lr_schedule.peak_lr == 2.5e-5
    assert config.lr_schedule.decay_steps == 11_126
    assert config.lr_schedule.decay_lr == 2.5e-6
    assert data_config.repo_id == REPO_ID
    assert tuple(data_config.task_frame_counts) == FRAME_COUNTS
    assert data_config.task_sampling_exponent == 0.0
    assert config.data.enabled_cameras == expected_cameras
    assert config.policy_metadata is not None
    assert config.policy_metadata["recipe"] == "builtin-dual-lora"
    assert config.policy_metadata["training_purpose"]["stage"] == "G5.1"
    assert config.policy_metadata["training_purpose"]["training_seed"] == seed
    assert config.policy_metadata["input_config"]["camera_condition"] == camera_condition


def test_g51_trainable_parameter_count_matches_frozen_recipe() -> None:
    config = config_module.get_config("pi05_g51_easy5_three_view_s0_builtin_dual_lora")
    state = jax.eval_shape(lambda rng: nnx.state(config.model.create(rng)), jax.random.key(config.seed))
    trainable = traverse_util.flatten_dict(state.filter(config.trainable_filter).to_pure_dict())
    observed = sum(math.prod(value.shape) for value in trainable.values())

    assert observed == 466_957_072
