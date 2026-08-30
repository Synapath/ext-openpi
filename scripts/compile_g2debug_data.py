"""Compile frozen single-task PI05 draws and normalization for G2.0-debug."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import random

from manip_datasets import robodojo_pi05 as source
import numpy as np

ASSET_IDS = {
    "pour_liquid_into_cup": "rbdj-g2debug-pour-v1",
    "stack_bowls": "rbdj-g2debug-stack-v1",
}


def generate_single_task_draws(
    split_path: Path,
    output: Path,
    *,
    task: str,
    updates: int = 6000,
    batch_size: int = 32,
) -> dict:
    split = json.loads(split_path.read_text())
    source.validate_split(split)
    if task not in source.TASKS or updates < 1 or batch_size < 1:
        raise ValueError("known task and positive updates/batch are required")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "draw-manifest.json"
    binary_path = output / "draws.bin"
    episodes = [row for row in split["episodes"] if row["task"] == task and row["split"] == "train"]
    if len(episodes) != 90:
        raise ValueError(f"{task}: expected the frozen 90 training episodes")
    ends, total_anchors = [], 0
    for row in episodes:
        total_anchors += row["anchors"]
        ends.append(total_anchors)

    draws = updates * batch_size
    rng = random.Random(0)
    order = [task] * draws
    # Preserve the historical generator's RNG advancement even though all labels
    # are identical; the bytes, not a future RNG recreation, remain authoritative.
    rng.shuffle(order)
    episode_counts = Counter()
    with binary_path.open("xb") as stream:
        for index in range(draws):
            anchor = rng.randrange(total_anchors)
            episode_offset = bisect_right(ends, anchor)
            frame = anchor - (ends[episode_offset - 1] if episode_offset else 0)
            episode = episodes[episode_offset]
            stream.write(
                source.RECORD.pack(
                    index // batch_size,
                    index % batch_size,
                    episode["task_index"],
                    episode["episode_id"],
                    frame,
                )
            )
            episode_counts[str(episode["episode_id"])] += 1
    result = {
        "schema": source.SCHEMA,
        "split_sha256": source.sha256(split_path),
        "split_identity": source.canonical_hash(split),
        "binary": "draws.bin",
        "binary_sha256": source.sha256(binary_path),
        "record_format": "<IIIII",
        "fields": ["update", "slot", "task_index", "episode_id", "frame_idx"],
        "updates": updates,
        "batch_size": batch_size,
        "draws": draws,
        "seed": 0,
        "algorithm": "cpython-mt19937-shuffle-randrange-v1",
        "task_counts": {task: draws},
        "episode_counts": dict(episode_counts),
        "eligible_anchors": total_anchors,
        "train_episode_ids": [row["episode_id"] for row in episodes],
    }
    source.write_json(manifest_path, result)
    return result


def compute_single_task_norm(view: Path, split_path: Path, output: Path, *, task: str) -> dict:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    split = json.loads(split_path.read_text())
    source.validate_split(split)
    episodes = [row for row in split["episodes"] if row["task"] == task and row["split"] == "train"]
    if len(episodes) != 90:
        raise ValueError(f"{task}: expected the frozen 90 training episodes")
    episode_ids = [row["episode_id"] for row in episodes]
    tables = [pq.read_table(path) for path in sorted((view / "data").glob("chunk-*/*.parquet"))]
    table = pa.concat_tables(tables)
    table = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(episode_ids, type=pa.int64())))
    table = table.sort_by([("index", "ascending")])
    epids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if (
        states.shape != actions.shape
        or states.shape[1:] != (14,)
        or not np.isfinite(states).all()
        or not np.isfinite(actions).all()
    ):
        raise ValueError("single-task numeric inventory is malformed or non-finite")

    norm_states, norm_actions = [], []
    arm_mask = np.array([True] * 6 + [False] + [True] * 6 + [False])
    for row in episodes:
        positions = np.flatnonzero(epids == row["episode_id"])
        if len(positions) != row["length"]:
            raise ValueError(f"episode length mismatch: {row['episode_id']}")
        state_values, action_values = states[positions], actions[positions]
        count = row["anchors"]
        future = np.arange(count)[:, None] + np.arange(50)[None, :]
        chunks = action_values[future].copy()
        chunks[..., arm_mask] -= state_values[:count, None, arm_mask]
        norm_states.append(state_values[:count])
        norm_actions.append(chunks.reshape(-1, 14))
    state_population = np.concatenate(norm_states)
    action_population = np.concatenate(norm_actions)
    stats = {
        "state": source.weighted_statistics(state_population, np.ones(len(state_population))),
        "actions": source.weighted_statistics(action_population, np.ones(len(action_population))),
    }
    output.mkdir(parents=True, exist_ok=False)
    norm_path = output / "norm_stats.json"
    source.write_json(norm_path, {"norm_stats": stats})
    return {
        "schema": "g2debug-single-task-norm-v1",
        "task": task,
        "split_sha256": source.sha256(split_path),
        "train_episode_ids": episode_ids,
        "train_episodes": len(episode_ids),
        "eligible_anchors": len(state_population),
        "action_targets": len(action_population),
        "norm_sha256": source.sha256(norm_path),
        "population": "train-only/uniform-eligible-anchor/uniform-H50-position",
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--view", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=sorted(source.TASKS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    draws = generate_single_task_draws(args.split, args.output / "draws", task=args.task)
    asset_id = ASSET_IDS[args.task]
    norm = compute_single_task_norm(args.view, args.split, args.output / "assets" / asset_id, task=args.task)
    source.write_json(args.output / "receipt.json", {"draws": draws, "norm": norm, "norm_asset_id": asset_id})
    print(json.dumps({"task": args.task, "draws_sha256": draws["binary_sha256"], "norm_sha256": norm["norm_sha256"]}))


if __name__ == "__main__":
    main()
