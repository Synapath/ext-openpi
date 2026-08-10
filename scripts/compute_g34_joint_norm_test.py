import numpy as np

from scripts import compute_g34_joint_norm as joint_norm


def test_weighted_stats() -> None:
    values = np.asarray([[0.0], [2.0], [10.0]])
    weights = np.asarray([0.25, 0.25, 0.5])

    observed = joint_norm.weighted_stats(values, weights)

    np.testing.assert_allclose(observed.mean, [5.5])
    np.testing.assert_allclose(observed.std, [np.sqrt(20.75)])
    np.testing.assert_allclose(observed.q01, [0.0])
    np.testing.assert_allclose(observed.q99, [10.0])


def test_weighted_quantile_respects_large_weight() -> None:
    values = np.asarray([[0.0], [1.0], [2.0]])
    weights = np.asarray([0.1, 0.8, 0.1])

    np.testing.assert_allclose(joint_norm.weighted_quantile(values, weights, 0.5), [1.0])
