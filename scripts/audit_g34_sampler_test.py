import numpy as np

from scripts import audit_g34_sampler as sampler_audit


def test_replay_indices_is_deterministic_and_drops_partial_batch() -> None:
    first = sampler_audit.replay_indices(batch_size=32, updates=1100, seed=0)
    second = sampler_audit.replay_indices(batch_size=32, updates=1100, seed=0)

    assert len(first) == 35200
    np.testing.assert_array_equal(first, second)
    assert np.all((first >= 0) & (first < sum(sampler_audit.FRAME_COUNTS)))


def test_task_indices_follow_frozen_contiguous_ranges() -> None:
    boundaries = np.cumsum(sampler_audit.FRAME_COUNTS)
    observed = sampler_audit.task_indices(np.asarray([0, boundaries[0] - 1, boundaries[0], boundaries[-1] - 1]))

    np.testing.assert_array_equal(observed, [0, 0, 1, 3])
