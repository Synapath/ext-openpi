import math

from flax import nnx
from flax import traverse_util
import jax
import pytest

import openpi.training.config as config_module


TASKS = ("lift_pot", "handover_block")
RECIPES = {
    "builtin_dual_lora": ("builtin-dual-lora", 466_957_072),
    "vlm_lora_expert_full": ("vlm-lora-expert-full", 872_771_344),
}


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize(("name_suffix", "recipe_expected"), RECIPES.items())
def test_g33_recipe_identity_and_trainable_parameter_count(
    task: str, name_suffix: str, recipe_expected: tuple[str, int]
) -> None:
    recipe, expected = recipe_expected
    config = config_module.get_config(f"pi05_g33_{task}_{name_suffix}")
    state = jax.eval_shape(lambda rng: nnx.state(config.model.create(rng)), jax.random.key(config.seed))
    trainable = traverse_util.flatten_dict(state.filter(config.trainable_filter).to_pure_dict())
    observed = sum(math.prod(value.shape) for value in trainable.values())

    assert observed == expected
    assert config.seed == 0
    assert config.ema_decay is None
    assert config.fsdp_devices == 1
    assert config.policy_metadata is not None
    assert config.policy_metadata["task_info"] == {
        "benchmark": "RoboTwin-2.0",
        "task": task,
        "task_config": "demo_clean",
        "scene": "Easy",
    }
    assert config.policy_metadata["recipe"] == recipe
    assert config.policy_metadata["training_purpose"] == {
        "stage": "G3.3",
        "type": "cross-task-start-factorial",
        "comparison_axes": ["recipe", "start"],
        "training_seed": 0,
    }
