import math

from flax import nnx
from flax import traverse_util
import jax
import pytest

from openpi.training import config

TASKS = {
    "pour": "pour_liquid_into_cup",
    "stack": "stack_bowls",
}
TRAINABLE_COUNTS = {
    "strict": 52_153_376,
    "builtin": 466_957_072,
}


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("recipe", TRAINABLE_COUNTS)
def test_g2debug_single_task_matrix(task, recipe):
    value = config.get_config(f"pi05_g2debug_{task}_{recipe}")
    assert value.policy_metadata["task_info"]["task"] == TASKS[task]
    assert value.policy_metadata["robot_config"]["prediction_horizon"] == 50
    assert value.policy_metadata["robot_config"]["execution_horizon"] == 16
    assert (value.batch_size, value.num_train_steps, value.seed, value.fsdp_devices) == (32, 6000, 0, 1)
    assert value.ema_decay is None
    assert value.model.action_horizon == 50
    assert value.model.pi05
    assert (value.lr_schedule.warmup_steps, value.lr_schedule.decay_steps) == (300, 6000)
    assert (value.lr_schedule.peak_lr, value.lr_schedule.decay_lr) == (2.5e-5, 2.5e-6)
    assert not value.data.adapt_to_pi
    assert value.data.use_delta_joint_actions
    assert tuple(value.data.enabled_cameras) == ("cam_high", "cam_left_wrist", "cam_right_wrist")

    state = jax.eval_shape(lambda key: nnx.state(value.model.create(key)), jax.random.key(0))
    trainable = traverse_util.flatten_dict(state.filter(value.trainable_filter).to_pure_dict(), sep="/")
    assert sum(math.prod(parameter.shape) for parameter in trainable.values()) == TRAINABLE_COUNTS[recipe]
