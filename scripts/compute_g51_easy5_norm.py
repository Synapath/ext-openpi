#!/usr/bin/env python3
"""Compute the shared task-balanced joint norm and readiness manifest for G5.1."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import openpi.shared.normalize as normalize
import openpi.training.config as config_module
import openpi.training.data_loader as data_loader_module
from scripts.compute_g34_joint_norm import outlier_rates
from scripts.compute_g34_joint_norm import stats_json
from scripts.compute_g34_joint_norm import weighted_stats

TASKS = (
    "place_empty_cup",
    "place_container_plate",
    "press_stapler",
    "turn_switch",
    "adjust_bottle",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_numeric_dataset(
    root: Path,
    action_horizon: int,
    frame_counts: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parquet_paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No LeRobot parquet files under {root / 'data'}")
    table = pa.concat_tables(
        [
            pq.read_table(
                path,
                columns=[
                    "observation.state",
                    "action",
                    "canonical_task_index",
                    "episode_index",
                    "frame_index",
                    "index",
                ],
            )
            for path in parquet_paths
        ]
    )
    indices = np.asarray(table["index"].to_pylist(), dtype=np.int64).reshape(-1)
    order = np.argsort(indices)
    if not np.array_equal(indices[order], np.arange(len(indices))):
        raise ValueError("LeRobot global frame indices are not contiguous")
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[order]
    source_actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[order]
    task_indices = np.asarray(table["canonical_task_index"].to_pylist(), dtype=np.int64).reshape(-1)[order]
    episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64).reshape(-1)[order]
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64).reshape(-1)[order]
    observed_counts = tuple(int(np.sum(task_indices == index)) for index in range(len(TASKS)))
    if states.shape != (sum(frame_counts), 14) or observed_counts != frame_counts:
        raise ValueError(f"Frame contract mismatch: states={states.shape}, task_counts={observed_counts}")

    episode_ends = np.empty(len(states), dtype=np.int64)
    for episode_index in np.unique(episode_indices):
        positions = np.flatnonzero(episode_indices == episode_index)
        if not np.array_equal(frame_indices[positions], np.arange(len(positions))):
            raise ValueError(f"Episode {episode_index} frame indices are not contiguous")
        episode_ends[positions] = positions[-1] + 1
    offsets = np.arange(action_horizon, dtype=np.int64)
    rows = np.arange(len(states), dtype=np.int64)
    future = np.minimum(rows[:, None] + offsets[None, :], episode_ends[:, None] - 1)
    actions = source_actions[future]
    arm_mask = np.asarray([True] * 6 + [False] + [True] * 6 + [False])
    actions[..., arm_mask] -= states[:, None, arm_mask]
    return states, actions, task_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_g34_joint4_builtin_dual_lora")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--readiness-output", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.output_dir / "norm_stats.json", args.audit_output, args.readiness_output)
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Refusing to overwrite an existing output: {outputs}")

    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if tuple(dataset_manifest["canonical_task_order"]) != TASKS:
        raise ValueError("Dataset task order does not match G5.1")
    frame_counts = tuple(int(dataset_manifest["task_frame_counts"][task]) for task in TASKS)
    config = config_module.get_config(args.config_name)
    states, actions, task_indices = load_numeric_dataset(
        args.dataset_root,
        config.model.action_horizon,
        frame_counts,
    )
    probabilities = np.full(len(TASKS), 1.0 / len(TASKS), dtype=np.float64)
    count_array = np.asarray(frame_counts)
    state_weights = probabilities[task_indices] / count_array[task_indices]
    action_values = actions.reshape(-1, actions.shape[-1])
    action_task_indices = np.repeat(task_indices, config.model.action_horizon)
    action_weights = probabilities[action_task_indices] / (
        count_array[action_task_indices] * config.model.action_horizon
    )
    norm_stats = {
        "state": weighted_stats(states, state_weights),
        "actions": weighted_stats(action_values, action_weights),
    }

    assert isinstance(config.data, config_module.LeRobotAlohaDataConfig)
    data_factory = dataclasses.replace(
        config.data,
        repo_id=dataset_manifest["repo_id"],
        assets=config_module.AssetsConfig(asset_id=dataset_manifest["repo_id"]),
        base_config=config_module.DataConfig(
            prompt_from_task=True,
            task_frame_counts=frame_counts,
            task_sampling_exponent=0.0,
        ),
    )
    data_config = data_factory.create(config.assets_dirs, config.model)
    official_dataset = data_loader_module.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
    )
    official_dataset = data_loader_module.TransformedDataset(
        official_dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs],
    )
    reference_checks = []
    for index in (0, len(states) // 2, len(states) - 1):
        reference = official_dataset[index]
        check = {
            "index": index,
            "state_equal": bool(np.array_equal(np.asarray(reference["state"]), states[index])),
            "actions_equal": bool(np.array_equal(np.asarray(reference["actions"]), actions[index])),
        }
        reference_checks.append(check)
    if not all(check["state_equal"] and check["actions_equal"] for check in reference_checks):
        raise RuntimeError(f"Numeric transform does not match official pipeline: {reference_checks}")

    normalize.save(args.output_dir, norm_stats)
    norm_path = args.output_dir / "norm_stats.json"
    per_task: dict[str, Any] = {}
    for task_index, task in enumerate(TASKS):
        state_mask = task_indices == task_index
        action_mask = action_task_indices == task_index
        task_state_weights = np.ones(int(state_mask.sum()), dtype=np.float64)
        task_action_weights = np.ones(int(action_mask.sum()), dtype=np.float64)
        per_task[task] = {
            "frames": int(state_mask.sum()),
            "sampling_probability": float(probabilities[task_index]),
            "state": stats_json(weighted_stats(states[state_mask], task_state_weights)),
            "actions": stats_json(weighted_stats(action_values[action_mask], task_action_weights)),
            "outliers_against_global": {
                "state": outlier_rates(states[state_mask], task_state_weights, norm_stats["state"]),
                "actions": outlier_rates(action_values[action_mask], task_action_weights, norm_stats["actions"]),
            },
        }
    norm_sha = sha256_file(norm_path)
    audit = {
        "schema_version": "g51-task-balanced-joint-norm-audit-v1",
        "status": "PASS",
        "repo_id": dataset_manifest["repo_id"],
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "frame_counts": dict(zip(TASKS, frame_counts, strict=True)),
        "sampling_probabilities": dict(zip(TASKS, probabilities.tolist(), strict=True)),
        "action_horizon": config.model.action_horizon,
        "weighting_rule": "uniform task p=0.2; uniform frame starts within task",
        "official_transform_reference_checks": reference_checks,
        "global": {key: stats_json(value) for key, value in norm_stats.items()},
        "per_task": per_task,
        "norm_stats": str(norm_path.resolve()),
        "norm_stats_sha256": norm_sha,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readiness = {
        **dataset_manifest,
        "schema_version": "g51-dataset-norm-readiness-v1",
        "status": "PASS",
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "norm_stats": str(norm_path.resolve()),
        "norm_stats_sha256": norm_sha,
        "norm_audit": str(args.audit_output.resolve()),
        "norm_audit_sha256": sha256_file(args.audit_output),
        "task_sampling_probabilities": dict(zip(TASKS, probabilities.tolist(), strict=True)),
    }
    args.readiness_output.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS norm={norm_path} sha256={norm_sha}")


if __name__ == "__main__":
    main()
