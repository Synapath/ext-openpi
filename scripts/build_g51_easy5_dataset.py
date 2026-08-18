#!/usr/bin/env python3
"""Build the canonical three-camera G5.1 Easy5 LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm

from scripts.build_g34_joint_dataset import CAMERAS
from scripts.build_g34_joint_dataset import features
from scripts.build_g34_joint_dataset import load_episode
from scripts.build_g34_joint_dataset import sha256_file
from scripts.build_g34_joint_dataset import stable_tree_sha256

TASKS = (
    "place_empty_cup",
    "place_container_plate",
    "press_stapler",
    "turn_switch",
    "adjust_bottle",
)


def source_root(args: argparse.Namespace, task: str) -> Path:
    if task == "adjust_bottle":
        return args.adjust_bottle_root
    return args.generated_root / "demo_clean" / task / "aloha_agilex"


def source_files(args: argparse.Namespace, task: str) -> list[Path]:
    paths = sorted((source_root(args, task) / "data").glob("episode_*.hdf5"))
    if len(paths) != 50:
        raise ValueError(f"Expected 50 source episodes for {task}, found {len(paths)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=Path("/data/manip/datasets/robotwin-g5/g51-easy5-clean"),
    )
    parser.add_argument(
        "--adjust-bottle-root",
        type=Path,
        default=Path("/data/manip/datasets/robotwin/adjust_bottle/aloha_agilex"),
    )
    parser.add_argument("--repo-id", default="RoboTwin-g51-easy5-clean-aloha_agilex-joint")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root or HF_LEROBOT_HOME / args.repo_id
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    if args.manifest.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {args.manifest}")

    all_sources = [(task, source_files(args, task)) for task in TASKS]
    first = load_episode(all_sources[0][1][0], 0)
    image_shape = tuple(first["images"]["cam_high"].shape[1:])
    if image_shape != (240, 320, 3):
        raise ValueError(f"Expected source images shaped (240, 320, 3), got {image_shape}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=50,
        robot_type="aloha_agilex",
        features=features(image_shape),
        use_videos=False,
    )
    episodes: list[dict[str, Any]] = []
    global_episode_index = 0
    global_frame_index = 0
    task_frame_counts: dict[str, int] = {}
    task_episode_counts: dict[str, int] = {}
    for canonical_task_index, (task, paths) in enumerate(all_sources):
        task_frames = 0
        for local_episode_index, path in enumerate(tqdm.tqdm(paths, desc=task, unit="episode")):
            loaded = (
                first
                if canonical_task_index == 0 and local_episode_index == 0
                else load_episode(path, local_episode_index)
            )
            start = global_frame_index
            for frame_index in range(len(loaded["state"])):
                frame = {
                    "observation.state": loaded["state"][frame_index],
                    "action": loaded["action"][frame_index],
                    "canonical_task_index": np.asarray([canonical_task_index], dtype=np.int64),
                    "task": loaded["prompt"],
                }
                for camera, images in loaded["images"].items():
                    frame[f"observation.images.{camera}"] = images[frame_index]
                dataset.add_frame(frame)
            dataset.save_episode()
            frame_count = len(loaded["state"])
            global_frame_index += frame_count
            task_frames += frame_count
            episodes.append(
                {
                    "canonical_task_id": task,
                    "canonical_task_index": canonical_task_index,
                    "global_episode_index": global_episode_index,
                    "local_episode_index": local_episode_index,
                    "global_frame_start": start,
                    "global_frame_end_exclusive": global_frame_index,
                    "frame_count": frame_count,
                    "instruction": loaded["prompt"],
                    "source_path": str(path.resolve()),
                    "source_sha256": sha256_file(path),
                }
            )
            global_episode_index += 1
        task_frame_counts[task] = task_frames
        task_episode_counts[task] = len(paths)

    dataset.finalize()
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if info["total_episodes"] != 250 or info["total_frames"] != global_frame_index:
        raise ValueError(f"LeRobot metadata totals are inconsistent: {info}")
    manifest = {
        "schema_version": "g51-easy5-dataset-manifest-v1",
        "status": "PASS",
        "repo_id": args.repo_id,
        "dataset_root": str(output_root.resolve()),
        "dataset_tree_sha256": stable_tree_sha256(output_root),
        "info_sha256": sha256_file(info_path),
        "fps": 50,
        "camera_order": list(CAMERAS.values()),
        "canonical_task_order": list(TASKS),
        "task_episode_counts": task_episode_counts,
        "task_frame_counts": task_frame_counts,
        "total_episodes": global_episode_index,
        "total_frames": global_frame_index,
        "instruction_rule": "source_instructions[local_episode_index % len(source_instructions)]",
        "action_rule": "next_state_with_final_state_repeated",
        "episodes": episodes,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS repo_id={args.repo_id} episodes={global_episode_index} frames={global_frame_index}")


if __name__ == "__main__":
    main()
