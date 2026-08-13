import math

from flax import nnx
from flax import traverse_util
import jax
import pytest

import openpi.training.config as config_module

EXPECTED = {
    "clean": {
        "repo_id": "RoboTwin-g4-j10-clean-aloha_agilex-joint",
        "frame_counts": (4728, 7188, 5554, 12122, 4252, 6129, 14084, 5682, 7451, 7349),
    },
    "mixed": {
        "repo_id": "RoboTwin-g4-j10-mixed-aloha_agilex-joint",
        "frame_counts": (4820, 7412, 5709, 13462, 4258, 6320, 14571, 6038, 7712, 7683),
    },
}


@pytest.mark.parametrize("distribution", ["clean", "mixed"])
def test_g4_joint10_matched_recipe_contract(distribution: str) -> None:
    config = config_module.get_config(f"pi05_g4_j10_{distribution}_builtin_dual_lora")
    state = jax.eval_shape(lambda rng: nnx.state(config.model.create(rng)), jax.random.key(config.seed))
    trainable = traverse_util.flatten_dict(state.filter(config.trainable_filter).to_pure_dict())
    observed = sum(math.prod(value.shape) for value in trainable.values())
    data_config = config.data.create(config.assets_dirs, config.model)

    assert observed == 466_957_072
    assert config.batch_size == 32
    assert config.seed == 0
    assert config.ema_decay is None
    assert config.num_train_steps == 35_748
    assert data_config.repo_id == EXPECTED[distribution]["repo_id"]
    assert tuple(data_config.task_frame_counts) == EXPECTED[distribution]["frame_counts"]
    assert data_config.task_sampling_exponent == 0.43
    assert config.policy_metadata is not None
    assert config.policy_metadata["recipe"] == "builtin-dual-lora"
    assert config.policy_metadata["training_purpose"]["stage"] == "G4.2"
    assert config.policy_metadata["training_purpose"]["controlled_axis"] == "environment_distribution"
    assert config.policy_metadata["task_info"]["environment_distribution"] == distribution
