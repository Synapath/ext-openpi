#!/usr/bin/env python3
"""Build the deterministic four-task LeRobot dataset frozen for manip G3.4."""

from __future__ import annotations

import argparse
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
    ("adjust_bottle", 7188),
    ("lift_pot", 5554),
    ("pick_dual_bottles", 6129),
    ("handover_block", 14084),
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


def tree_sha256(root: Path) -> str:
    """Hash relative paths and contents for every regular file below root."""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def tree_inventory(root: Path) -> list[tuple[str, int, int]]:
    """Relative path, size and mtime_ns for every regular file below root."""
    return [
        (path.relative_to(root).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def stable_tree_sha256(root: Path) -> str:
    """Hash the tree and prove nothing wrote to it while hashing.

    LeRobot only writes parquet footers and episode metadata when its writers are
    closed, so hashing a still-open dataset yields a digest of a partially written
    tree. Callers must close the dataset first; this guard turns any remaining
    write race into a hard failure instead of an unreproducible digest.
    """
    before = tree_inventory(root)
    digest = tree_sha256(root)
    after = tree_inventory(root)
    if before != after:
        changed = sorted({entry[0] for entry in set(before) ^ set(after)})
        raise RuntimeError(f"Dataset tree changed while hashing; digest is not reproducible: {changed}")
    return digest


def choose_instruction(raw: Any, episode_index: int) -> str:
    """Choose one original prompt deterministically for the whole source episode."""
    instructions = list(raw)
    if not instructions:
        raise ValueError("Source episode contains no instructions.")
    value = instructions[episode_index % len(instructions)]
    if isinstance(value, bytes | np.bytes_):
        value = bytes(value).rstrip(b"\0").decode("utf-8")
    prompt = str(value).strip()
    if not prompt:
        raise ValueError("Selected source instruction is empty.")
    return prompt


def read_instructions(dataset: h5py.Dataset) -> list[Any]:
    """Read both legacy instruction arrays and scalar JSON instruction lists."""
    raw = dataset[()]
    if isinstance(raw, bytes | np.bytes_):
        text = bytes(raw).rstrip(b"\0").decode("utf-8")
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(decoded, list):
            return decoded
        if isinstance(decoded, str):
            return [decoded]
        raise ValueError(f"Scalar instructions JSON must decode to a string or list, got {type(decoded).__name__}.")
    if isinstance(raw, np.ndarray):
        return raw.tolist()
    return [raw]


def decode_images(encoded: Any) -> np.ndarray:
    frames = []
    for index, value in enumerate(encoded):
        buffer = bytes(value).rstrip(b"\0")
        image = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
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


def load_episode(path: Path, local_episode_index: int) -> dict[str, Any]:
    with h5py.File(path, "r") as episode:
        left_ee = np.asarray(episode["state/left_ee_joint_states"][:]).reshape(-1, 1)
        right_ee = np.asarray(episode["state/right_ee_joint_states"][:]).reshape(-1, 1)
        left = np.concatenate(
            [episode["state/left_arm_joint_states"][:], left_ee],
            axis=1,
        )
        right = np.concatenate(
            [episode["state/right_arm_joint_states"][:], right_ee],
            axis=1,
        )
        state = np.concatenate([left, right], axis=1).astype(np.float32)
        images = {output: decode_images(episode[f"vision/{source}/colors"][:]) for source, output in CAMERAS.items()}
        instruction_key = "instructions" if "instructions" in episode else "instruction"
        prompt = choose_instruction(read_instructions(episode[instruction_key]), local_episode_index)
    if any(len(value) != len(state) for value in images.values()):
        raise ValueError(f"Camera/state length mismatch in {path}.")
    return {"state": state, "action": next_state_actions(state), "images": images, "prompt": prompt}


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


def source_files(source_root: Path, task_id: str) -> list[Path]:
    paths = sorted((source_root / task_id / "aloha_agilex" / "data").glob("episode_*.hdf5"))
    if len(paths) != 50:
        raise ValueError(f"Expected 50 source episodes for {task_id}, found {len(paths)}.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data/manip/datasets/robotwin"))
    parser.add_argument("--repo-id", default="RoboTwin-g34-joint4-aloha_agilex-joint")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root or HF_LEROBOT_HOME / args.repo_id
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    if args.manifest.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {args.manifest}")

    all_sources = [
        (task_id, expected_frames, source_files(args.source_root, task_id)) for task_id, expected_frames in TASKS
    ]
    first = load_episode(all_sources[0][2][0], 0)
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
    episodes = []
    global_episode_index = 0
    global_frame_index = 0
    task_totals: dict[str, int] = {}
    for canonical_task_index, (task_id, expected_frames, paths) in enumerate(all_sources):
        task_frames = 0
        for local_episode_index, path in enumerate(tqdm.tqdm(paths, desc=task_id, unit="episode")):
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
                    "canonical_task_id": task_id,
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
        if task_frames != expected_frames:
            raise ValueError(f"Frame count mismatch for {task_id}: expected {expected_frames}, observed {task_frames}.")
        task_totals[task_id] = task_frames

    # Close the parquet writers so footers and episode metadata land on disk before
    # anything reads or hashes the tree. Without this the dataset is only completed
    # incidentally at interpreter shutdown, after the manifest has been written.
    dataset.finalize()

    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if info["total_episodes"] != 200 or info["total_frames"] != sum(count for _, count in TASKS):
        raise ValueError(f"LeRobot metadata totals are inconsistent: {info}")
    manifest = {
        "schema_version": "g34-joint-dataset-manifest-v1",
        "status": "PASS",
        "repo_id": args.repo_id,
        "dataset_root": str(output_root.resolve()),
        "dataset_tree_sha256": stable_tree_sha256(output_root),
        "info_sha256": sha256_file(info_path),
        "fps": 50,
        "camera_order": list(CAMERAS.values()),
        "canonical_task_order": [task_id for task_id, _ in TASKS],
        "task_frame_counts": task_totals,
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
