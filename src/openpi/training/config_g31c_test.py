import math

from flax import nnx
from flax import traverse_util
import jax
import pytest

import openpi.training.config as config_module


EXPECTED_TRAINABLE = {
    "pi05_g31c_strict_dual_lora_interface": 52_153_376,
    "pi05_g31c_action_expert_interface": 430_098_464,
    "pi05_g31c_builtin_dual_lora": 466_957_072,
    "pi05_g31c_vlm_lora_expert_full": 872_771_344,
}


@pytest.mark.parametrize(("name", "expected"), EXPECTED_TRAINABLE.items())
def test_g31c_recipe_trainable_parameter_count(name: str, expected: int) -> None:
    config = config_module.get_config(name)
    state = jax.eval_shape(lambda rng: nnx.state(config.model.create(rng)), jax.random.key(config.seed))
    trainable = traverse_util.flatten_dict(state.filter(config.trainable_filter).to_pure_dict())
    observed = sum(math.prod(value.shape) for value in trainable.values())

    assert observed == expected
    assert config.ema_decay is None
    assert config.fsdp_devices == 1
    assert config.policy_metadata is not None
    assert set(config.policy_metadata) == {
        "task_info",
        "robot_config",
        "input_config",
        "training_purpose",
        "recipe",
    }
