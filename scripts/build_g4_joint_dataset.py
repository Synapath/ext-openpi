#!/usr/bin/env python3
"""Build a deterministic G4 Joint10 LeRobot dataset from a frozen source manifest."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm

TASKS = (
    "grab_roller",
    "adjust_bottle",
    "lift_pot",
    "dump_bin_bigbin",
    "click_alarmclock",
    "pick_dual_bottles",
    "handover_block",
    "beat_block_hammer",
    "place_a2b_left",
    "place_a2b_right",
)
CAMERAS = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
MOTORS = (
    "left_0",
    "left_1",
    "left_2",
    "left_3",
    "left_4",
    "left_5",
    "left_ee_0",
    "right_0",
    "right_1",
    "right_2",
    "right_3",
    "right_4",
    "right_5",
    "right_ee_0",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tree_inventory(root: Path) -> list[tuple[str, int, int]]:
    return [
        (path.relative_to(root).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def stable_tree_sha256(root: Path) -> str:
    before = tree_inventory(root)
    digest = tree_sha256(root)
    after = tree_inventory(root)
    if before != after:
        changed = sorted({entry[0] for entry in set(before) ^ set(after)})
        raise RuntimeError(f"Dataset tree changed while hashing: {changed}")
    return digest


def decode_images(encoded: Any) -> np.ndarray:
    frames = []
    for index, value in enumerate(encoded):
        image = cv2.imdecode(np.frombuffer(bytes(value).rstrip(b"\0"), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image frame {index}.")
        frames.append(image)
    return np.stack(frames)


def next_state_actions(state: np.ndarray) -> np.ndarray:
    if state.ndim != 2 or state.shape[1] != 14 or len(state) == 0:
        raise ValueError(f"Expected non-empty state array shaped [T, 14], got {state.shape}.")
    action = np.empty_like(state, dtype=np.float32)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    return action


def load_episode(path: Path, expected_frames: int) -> dict[str, Any]:
    with h5py.File(path, "r") as episode:
        left_ee = np.asarray(episode["state/left_ee_joint_states"][:]).reshape(-1, 1)
        right_ee = np.asarray(episode["state/right_ee_joint_states"][:]).reshape(-1, 1)
        left = np.concatenate([episode["state/left_arm_joint_states"][:], left_ee], axis=1)
        right = np.concatenate([episode["state/right_arm_joint_states"][:], right_ee], axis=1)
        state = np.concatenate([left, right], axis=1).astype(np.float32)
        images = {output: decode_images(episode[f"vision/{source}/colors"][:]) for source, output in CAMERAS.items()}
    if len(state) != expected_frames:
        raise ValueError(f"Frame count mismatch in {path}: expected {expected_frames}, observed {len(state)}.")
    if any(len(value) != len(state) for value in images.values()):
        raise ValueError(f"Camera/state length mismatch in {path}.")
    if not np.isfinite(state).all():
        raise ValueError(f"Non-finite state in {path}.")
    return {"state": state, "action": next_state_actions(state), "images": images}


def features(image_shape: tuple[int, int, int]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": MOTORS},
        "action": {"dtype": "float32", "shape": (14,), "names": MOTORS},
        "canonical_task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    for camera in CAMERAS.values():
        result[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ("height", "width", "channels"),
        }
    return result


def read_source_manifest(path: Path) -> tuple[dict[str, Any], list[tuple[str, list[dict[str, Any]]]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "g4-j10-source-manifest-v1":
        raise ValueError(f"Unexpected source manifest schema: {raw.get('schema_version')!r}")
    if raw.get("task_count") != 10 or raw.get("episode_count") != 500:
        raise ValueError("G4 source manifest must describe exactly 10 tasks and 500 episodes.")
    if set(raw.get("tasks", {})) != set(TASKS):
        raise ValueError(f"Task set mismatch: {sorted(raw.get('tasks', {}))}")
    ordered = []
    for task in TASKS:
        episodes = raw["tasks"][task]
        if len(episodes) != 50:
            raise ValueError(f"Expected 50 episodes for {task}, found {len(episodes)}.")
        if any(item.get("task") != task for item in episodes):
            raise ValueError(f"Task identity mismatch inside {task} records.")
        ordered.append((task, episodes))
    return raw, ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root or HF_LEROBOT_HOME / args.repo_id
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    if args.output_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {args.output_manifest}")

    source, ordered = read_source_manifest(args.source_manifest)
    first_record = ordered[0][1][0]
    first_path = Path(first_record["source_hdf5"])
    if sha256_file(first_path) != first_record["sha256"]:
        raise ValueError(f"Source hash mismatch: {first_path}")
    first = load_episode(first_path, first_record["frames"])
    image_shape = tuple(first["images"]["cam_high"].shape[1:])
    if image_shape != (240, 320, 3):
        raise ValueError(f"Expected source images shaped (240, 320, 3), got {image_shape}.")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=50,
        robot_type="aloha_agilex",
        features=features(image_shape),
        use_videos=False,
    )
    output_episodes = []
    task_frame_counts: dict[str, int] = {}
    factor_episode_counts: dict[str, dict[str, int]] = {}
    global_frame_index = 0
    global_episode_index = 0
    for task_index, (task, records) in enumerate(ordered):
        task_frames = 0
        factor_counts: collections.Counter[str] = collections.Counter()
        for local_index, record in enumerate(tqdm.tqdm(records, desc=task, unit="episode")):
            path = Path(record["source_hdf5"])
            observed_sha256 = sha256_file(path)
            if observed_sha256 != record["sha256"]:
                raise ValueError(f"Source hash mismatch: {path}")
            loaded = first if task_index == 0 and local_index == 0 else load_episode(path, record["frames"])
            prompt = str(record["instruction"]).strip()
            if not prompt:
                raise ValueError(f"Empty instruction in source record for {path}.")
            start = global_frame_index
            for frame_index in range(len(loaded["state"])):
                frame = {
                    "observation.state": loaded["state"][frame_index],
                    "action": loaded["action"][frame_index],
                    "canonical_task_index": np.asarray([task_index], dtype=np.int64),
                    "task": prompt,
                }
                for camera, images in loaded["images"].items():
                    frame[f"observation.images.{camera}"] = images[frame_index]
                dataset.add_frame(frame)
            dataset.save_episode()
            frames = len(loaded["state"])
            global_frame_index += frames
            task_frames += frames
            factor_counts[record["factor"]] += 1
            output_episodes.append(
                {
                    "canonical_task_id": task,
                    "canonical_task_index": task_index,
                    "global_episode_index": global_episode_index,
                    "local_episode_index": local_index,
                    "global_frame_start": start,
                    "global_frame_end_exclusive": global_frame_index,
                    "frame_count": frames,
                    "factor": record["factor"],
                    "instruction": prompt,
                    "source_path": str(path.resolve()),
                    "source_sha256": observed_sha256,
                }
            )
            global_episode_index += 1
        task_frame_counts[task] = task_frames
        factor_episode_counts[task] = dict(sorted(factor_counts.items()))

    dataset.finalize()
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info["total_episodes"] != 500 or info["total_frames"] != source["frame_count"]:
        raise ValueError(f"LeRobot metadata totals are inconsistent: {info}")
    if sum(task_frame_counts.values()) != source["frame_count"]:
        raise ValueError("Per-task frame totals do not match the frozen source manifest.")

    output = {
        "schema_version": "g4-joint-dataset-manifest-v1",
        "status": "PASS",
        "policy_dataset": source["policy_dataset"],
        "repo_id": args.repo_id,
        "dataset_root": str(output_root.resolve()),
        "dataset_tree_sha256": stable_tree_sha256(output_root),
        "info_sha256": sha256_file(info_path),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "source_content_sha256": source["content_sha256"],
        "fps": 50,
        "camera_order": list(CAMERAS.values()),
        "canonical_task_order": list(TASKS),
        "task_frame_counts": task_frame_counts,
        "factor_episode_counts": factor_episode_counts,
        "total_episodes": global_episode_index,
        "total_frames": global_frame_index,
        "instruction_rule": "frozen source-manifest instruction, constant within each episode",
        "action_rule": "next_state_with_final_state_repeated",
        "episodes": output_episodes,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS repo_id={args.repo_id} episodes={global_episode_index} frames={global_frame_index}")


if __name__ == "__main__":
    main()
