import json

from manip_datasets import robodojo_pi05 as source
import pytest

from scripts import compile_g2debug_data as compiler


def synthetic_split():
    rows = []
    for task_number, (_, instruction) in enumerate(source.TASKS.values()):
        for episode in range(100):
            length = 60 + episode
            row = {
                "episode_index": task_number * 100 + episode,
                "tasks": [instruction],
                "length": length,
                "dataset_from_index": (task_number * 100 + episode) * 1000,
                "dataset_to_index": (task_number * 100 + episode) * 1000 + length,
                "data/chunk_index": 0,
                "data/file_index": task_number,
            }
            for camera in source.CAMERAS:
                key = "videos/observation.images." + camera
                row.update(
                    {
                        key + "/chunk_index": 0,
                        key + "/file_index": task_number,
                        key + "/from_timestamp": 0,
                        key + "/to_timestamp": length / 25,
                    }
                )
            rows.append(row)
    return source.compile_split(rows, {"synthetic": "test"})


def test_single_task_draws_are_deterministic_and_holdout_free(tmp_path):
    split = tmp_path / "split.json"
    source.write_json(split, synthetic_split())
    first = compiler.generate_single_task_draws(split, tmp_path / "first", task="stack_bowls", updates=7, batch_size=8)
    second = compiler.generate_single_task_draws(
        split, tmp_path / "second", task="stack_bowls", updates=7, batch_size=8
    )
    assert first["binary_sha256"] == second["binary_sha256"]
    assert first["task_counts"] == {"stack_bowls": 56}
    with source.DrawManifest(tmp_path / "first/draw-manifest.json", split, verify=False) as draws:
        holdout = {row["episode_id"] for row in json.loads(split.read_text())["episodes"] if row["split"] == "holdout"}
        assert all(row[2] == 30 and row[3] not in holdout for row in (draws[i] for i in range(len(draws))))


def test_single_task_draws_refuse_unknown_task_and_overwrite(tmp_path):
    split = tmp_path / "split.json"
    source.write_json(split, synthetic_split())
    compiler.generate_single_task_draws(split, tmp_path / "draws", task="pour_liquid_into_cup", updates=1, batch_size=2)
    with pytest.raises(FileExistsError):
        compiler.generate_single_task_draws(
            split, tmp_path / "draws", task="pour_liquid_into_cup", updates=1, batch_size=2
        )
    with pytest.raises(ValueError, match="known task"):
        compiler.generate_single_task_draws(split, tmp_path / "bad", task="unknown")
