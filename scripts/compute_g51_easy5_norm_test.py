from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import compute_g51_easy5_norm as joint_norm


def test_load_numeric_dataset_uses_episode_bounded_future_actions(tmp_path: Path) -> None:
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    states = np.arange(10 * 14, dtype=np.float32).reshape(10, 14)
    table = pa.table(
        {
            "observation.state": states.tolist(),
            "action": states.tolist(),
            "canonical_task_index": [[index // 2] for index in range(10)],
            "episode_index": [index // 2 for index in range(10)],
            "frame_index": [index % 2 for index in range(10)],
            "index": list(range(10)),
        }
    )
    pq.write_table(table, data / "part.parquet")

    observed_states, actions, task_indices = joint_norm.load_numeric_dataset(
        tmp_path,
        action_horizon=3,
        frame_counts=(2, 2, 2, 2, 2),
    )

    np.testing.assert_array_equal(observed_states, states)
    np.testing.assert_array_equal(task_indices, np.repeat(np.arange(5), 2))
    assert actions.shape == (10, 3, 14)
    np.testing.assert_array_equal(actions[1][:, [6, 13]], np.repeat(states[1:2, [6, 13]], 3, axis=0))
