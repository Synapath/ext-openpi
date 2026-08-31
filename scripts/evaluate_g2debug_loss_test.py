import numpy as np

from scripts import evaluate_g2debug_loss as evaluator


def test_aggregate_reproduces_native_full_loss():
    rng = np.random.default_rng(0)
    components = rng.normal(size=(6, 50, 14)).astype(np.float32) ** 2
    native = components.mean(axis=-1)
    phases = ["start", "start", "mid", "mid", "late", "late"]

    result = evaluator.aggregate(native, components, phases)

    assert result["native_component_max_abs_error"] <= 5e-7
    assert result["native_full_component_aggregation_abs_error"] <= 2e-6
    assert result["native_segment_aggregation_abs_error"] <= 5e-7
    assert len(result["physical_action_components_14"]) == 14
    assert result["model_action_dim"] == 14


def test_aggregate_reports_physical_and_padded_dimensions_separately():
    components = np.ones((3, 50, 32), dtype=np.float32)
    native = components.mean(axis=-1)

    result = evaluator.aggregate(native, components, ["start", "mid", "late"])

    assert result["model_action_dim"] == 32
    assert len(result["physical_action_components_14"]) == 14
    assert len(result["padded_action_components"]) == 18
