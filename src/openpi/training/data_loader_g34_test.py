import numpy as np
import pytest

from openpi.training import data_loader

COUNTS = (7188, 5554, 6129, 14084)


def test_g34_task_sampling_probabilities() -> None:
    observed = data_loader.task_sampling_probabilities(COUNTS, 0.43)

    np.testing.assert_allclose(observed, [0.24014258, 0.21493554, 0.22423595, 0.32068592], atol=5e-9)
    assert observed.sum() == pytest.approx(1.0)


def test_weighted_task_sampler_is_deterministic() -> None:
    first = list(data_loader.create_weighted_task_sampler(COUNTS, 0.43, dataset_size=sum(COUNTS), seed=0))
    second = list(data_loader.create_weighted_task_sampler(COUNTS, 0.43, dataset_size=sum(COUNTS), seed=0))

    assert first == second
    assert len(first) == sum(COUNTS)


def test_weighted_task_sampler_rejects_bad_dataset_size() -> None:
    with pytest.raises(ValueError, match="dataset has"):
        data_loader.create_weighted_task_sampler(COUNTS, 0.43, dataset_size=sum(COUNTS) - 1, seed=0)
