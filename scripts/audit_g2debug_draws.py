"""Audit frozen G2.0-debug single-task draws and task-mode distributions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from manip_datasets import robodojo_pi05 as source
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from scripts import g2debug_common
except ImportError:  # Direct execution adds scripts/ rather than the repository root.
    import g2debug_common


def episode_hand_classes(view: Path, episodes: list[dict]) -> dict[int, dict]:
    tables = [pq.read_table(path) for path in sorted((view / "data").glob("chunk-*/*.parquet"))]
    table = pa.concat_tables(tables)
    ids = [row["episode_id"] for row in episodes]
    table = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(ids, type=pa.int64())))
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
        raise ValueError("malformed task numeric table")
    result = {}
    for row in episodes:
        positions = np.flatnonzero(epids == row["episode_id"])
        if len(positions) != row["length"]:
            raise ValueError(f"episode length mismatch: {row['episode_id']}")
        delta = np.abs(actions[positions[: row["anchors"]]] - states[positions[: row["anchors"]]])
        left = float(delta[:, :6].sum())
        right = float(delta[:, 7:13].sum())
        left_events = np.flatnonzero(delta[:, :6].sum(axis=1) > 1e-3)
        right_events = np.flatnonzero(delta[:, 7:13].sum(axis=1) > 1e-3)
        left_first = int(left_events[0]) if len(left_events) else row["anchors"]
        right_first = int(right_events[0]) if len(right_events) else row["anchors"]
        result[row["episode_id"]] = {
            "hand": "left" if left_first <= right_first else "right",
            "left_first_active_frame": left_first,
            "right_first_active_frame": right_first,
            "left_arm_next_frame_l1": left,
            "right_arm_next_frame_l1": right,
        }
    return result


def audit(task: str, output: Path) -> dict:
    split_path = g2debug_common.DATA_ROOT / "split.json"
    split = json.loads(split_path.read_text())
    source.validate_split(split)
    episodes = [row for row in split["episodes"] if row["task"] == task and row["split"] == "train"]
    if len(episodes) != 90:
        raise ValueError("frozen single-task audit requires exactly 90 training episodes")
    episode_rows = {row["episode_id"]: row for row in episodes}
    hand_classes = episode_hand_classes(g2debug_common.VIEW_ROOT, episodes)
    draw_root = g2debug_common.DATA_ROOT / task / "draws"
    identities = Counter()
    phases, hands = Counter(), Counter()
    with source.DrawManifest(draw_root / "draw-manifest.json", split_path, verify=False) as draws:
        for index in range(len(draws)):
            _, _, _, episode_id, frame = draws[index]
            row = episode_rows[episode_id]
            identities[(episode_id, frame)] += 1
            phases[("start", "mid", "late")[min(2, 3 * frame // row["anchors"])]] += 1
            hands[hand_classes[episode_id]["hand"]] += 1
        total_draws = len(draws)
        draw_identity = {
            "manifest_sha256": source.sha256(draw_root / "draw-manifest.json"),
            "binary_sha256": source.sha256(draw_root / draws.metadata["binary"]),
        }
    eligible = sum(row["anchors"] for row in episodes)
    multiplicities = Counter(identities.values())
    result = {
        "schema": "g2debug-draw-distribution-audit-v1",
        "status": "PASS",
        "task": task,
        "split_sha256": source.sha256(split_path),
        "draw_identity": draw_identity,
        "draws": total_draws,
        "eligible_anchors": eligible,
        "nominal_passes": total_draws / eligible,
        "unique_anchors_drawn": len(identities),
        "anchor_coverage_fraction": len(identities) / eligible,
        "repeat_draw_fraction": 1 - len(identities) / total_draws,
        "draw_multiplicity_counts": {str(key): value for key, value in sorted(multiplicities.items())},
        "phase_draw_counts": dict(phases),
        "phase_draw_fractions": {key: value / total_draws for key, value in phases.items()},
        "active_hand_rule": "episode earliest frame with arm sum(abs(action_next-state)) > 1e-3 over six joints; ties left; grippers excluded",
        "active_hand_draw_counts": dict(hands),
        "active_hand_draw_fractions": {key: value / total_draws for key, value in hands.items()},
        "episode_hand_classes": {str(key): value for key, value in sorted(hand_classes.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"draw audit output already exists: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--task", required=True, choices=sorted({task for task, _ in g2debug_common.TASKS.values()}))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.task, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
