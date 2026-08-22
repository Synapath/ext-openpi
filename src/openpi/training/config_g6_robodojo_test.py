import math

from flax import nnx
from flax import traverse_util
import jax

import openpi.training.config as config_module
import openpi.training.optimizer as optimizer_module

CONFIG_NAME = "pi05_g6_rbdj_joint4_s0_builtin_dual_lora"
REPO_ID = "RoboDojo-g6-joint4-arx_x5-joint"
FRAME_COUNTS = (18_558, 27_219, 16_144, 36_100)


def test_g6_robodojo_recipe_contract() -> None:
    config = config_module.get_config(CONFIG_NAME)
    data_config = config.data.create(config.assets_dirs, config.model)

    assert config.project_name == "manip-pi05-rbdj"
    assert config.model.pi05 is True
    assert config.model.action_horizon == 50
    assert config.batch_size == 32
    assert config.seed == 0
    assert config.ema_decay is None
    assert config.fsdp_devices == 1
    assert config.num_workers == 8
    assert config.num_train_steps == 30_632
    assert config.lr_schedule.warmup_steps == 857
    assert config.lr_schedule.peak_lr == 2.5e-5
    assert config.lr_schedule.decay_steps == 30_632
    assert config.lr_schedule.decay_lr == 2.5e-6

    assert isinstance(config.optimizer, optimizer_module.AdamW)
    assert config.optimizer.b1 == 0.9
    assert config.optimizer.b2 == 0.95
    assert config.optimizer.eps == 1e-8
    assert config.optimizer.weight_decay == 1e-10
    assert config.optimizer.clip_gradient_norm == 1.0

    assert data_config.repo_id == REPO_ID
    assert len(data_config.episode_indices) == 200
    assert tuple(data_config.episode_indices[:2]) == (3_000, 3_001)
    assert tuple(data_config.episode_indices[48:52]) == (3_048, 3_049, 300, 301)
    assert tuple(data_config.episode_indices[-2:]) == (848, 849)
    assert len(set(data_config.episode_indices)) == 200
    assert tuple(data_config.task_frame_counts) == FRAME_COUNTS
    assert sum(data_config.task_frame_counts) == 98_021
    assert data_config.task_sampling_exponent == 0.0
    assert config.data.enabled_cameras == ("cam_high", "cam_left_wrist", "cam_right_wrist")
    assert config.policy_metadata is not None
    assert config.policy_metadata["recipe"] == "builtin-dual-lora"
    assert config.policy_metadata["training_purpose"]["stage"] == "G6"
    assert config.policy_metadata["task_info"]["held_out_eval_task"] == "stack_blocks_by_language"


def test_g6_robodojo_trainable_parameter_count() -> None:
    config = config_module.get_config(CONFIG_NAME)
    state = jax.eval_shape(lambda rng: nnx.state(config.model.create(rng)), jax.random.key(config.seed))
    trainable = traverse_util.flatten_dict(state.filter(config.trainable_filter).to_pure_dict())
    observed = sum(math.prod(value.shape) for value in trainable.values())

    assert observed == 466_957_072
