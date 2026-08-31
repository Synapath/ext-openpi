import numpy as np

from scripts import audit_g2debug_checkpoint as auditor


def test_pure_parameter_arrays_keep_array_leaf_paths_without_value_suffix():
    tree = {
        "vision": {"kernel": np.zeros((2, 3), dtype=np.float32)},
        "expert": {"lora": {"a": np.ones((4,), dtype=np.float32)}},
    }

    arrays = auditor.pure_parameter_arrays(tree)

    assert set(arrays) == {"vision/kernel", "expert/lora/a"}
    assert arrays["vision/kernel"].shape == (2, 3)
    assert arrays["expert/lora/a"].dtype == np.float32
