import math

from flax import nnx
from flax import traverse_util
import jax

import openpi.training.config as config_module


def test_g34_joint_recipe_contract() -> None:
    config = config_module.get_config("pi05_g34_joint4_builtin_dual_lora")
    state = jax.eval_shape(lambda rng: nnx.state(config.model.create(rng)), jax.random.key(config.seed))
    trainable = traverse_util.flatten_dict(state.filter(config.trainable_filter).to_pure_dict())
    observed = sum(math.prod(value.shape) for value in trainable.values())
    data_config = config.data.create(config.assets_dirs, config.model)

    assert observed == 466_957_072
    assert config.batch_size == 32
    assert config.seed == 0
    assert config.ema_decay is None
    assert data_config.repo_id == "RoboTwin-g34-joint4-aloha_agilex-joint"
    assert tuple(data_config.task_frame_counts) == (7188, 5554, 6129, 14084)
    assert data_config.task_sampling_exponent == 0.43
    assert config.policy_metadata is not None
    assert config.policy_metadata["recipe"] == "builtin-dual-lora"
    assert config.policy_metadata["training_purpose"]["stage"] == "G3.4"
