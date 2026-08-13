#!/usr/bin/env python3
"""Compute exact task-sampler-weighted normalization for one G4 Joint10 dataset."""

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

EXPONENT = 0.43


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> np.ndarray:
    if values.ndim != 2 or weights.shape != (len(values),):
        raise ValueError(f"Bad weighted quantile shapes: values={values.shape}, weights={weights.shape}.")
    result = np.empty(values.shape[1], dtype=np.float64)
    target = quantile * weights.sum()
    for dimension in range(values.shape[1]):
        order = np.argsort(values[:, dimension], kind="stable")
        position = int(np.searchsorted(np.cumsum(weights[order]), target, side="left"))
        result[dimension] = values[order[min(position, len(order) - 1)], dimension]
    return result


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> normalize.NormStats:
    values = values.astype(np.float64, copy=False)
    total = weights.sum()
    mean = np.sum(values * weights[:, None], axis=0) / total
    second = np.sum(values**2 * weights[:, None], axis=0) / total
    return normalize.NormStats(
        mean=mean,
        std=np.sqrt(np.maximum(0.0, second - mean**2)),
        q01=weighted_quantile(values, weights, 0.01),
        q99=weighted_quantile(values, weights, 0.99),
    )


def stats_json(stats: normalize.NormStats) -> dict[str, list[float]]:
    return {
        "mean": np.asarray(stats.mean).tolist(),
        "std": np.asarray(stats.std).tolist(),
        "q01": np.asarray(stats.q01).tolist(),
        "q99": np.asarray(stats.q99).tolist(),
    }


def load_numeric_dataset(
    root: Path, task_frame_counts: tuple[int, ...], action_horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parquet_paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No LeRobot parquet files under {root / 'data'}.")
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
        raise ValueError("LeRobot global frame indices are not contiguous.")
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[order]
    source_actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[order]
    task_indices = np.asarray(table["canonical_task_index"].to_pylist(), dtype=np.int64).reshape(-1)[order]
    episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64).reshape(-1)[order]
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64).reshape(-1)[order]
    observed_counts = tuple(int(np.sum(task_indices == index)) for index in range(len(task_frame_counts)))
    if states.shape != (sum(task_frame_counts), 14) or observed_counts != task_frame_counts:
        raise ValueError(f"Frame contract mismatch: states={states.shape}, task_counts={observed_counts}.")

    episode_ends = np.empty(len(states), dtype=np.int64)
    for episode_index in np.unique(episode_indices):
        positions = np.flatnonzero(episode_indices == episode_index)
        if not np.array_equal(frame_indices[positions], np.arange(len(positions))):
            raise ValueError(f"Episode {episode_index} frame indices are not contiguous.")
        episode_ends[positions] = positions[-1] + 1
    offsets = np.arange(action_horizon, dtype=np.int64)
    rows = np.arange(len(states), dtype=np.int64)
    future = np.minimum(rows[:, None] + offsets[None, :], episode_ends[:, None] - 1)
    actions = source_actions[future]
    arm_mask = np.asarray([True] * 6 + [False] + [True] * 6 + [False])
    actions[..., arm_mask] -= states[:, None, arm_mask]
    return states, actions, task_indices


def outlier_rates(values: np.ndarray, weights: np.ndarray, stats: normalize.NormStats) -> dict[str, list[float]]:
    normalized = weights / weights.sum()
    return {
        "below_global_q01": np.sum((values < np.asarray(stats.q01)) * normalized[:, None], axis=0).tolist(),
        "above_global_q99": np.sum((values > np.asarray(stats.q99)) * normalized[:, None], axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / "norm_stats.json").exists() or args.audit_output.exists():
        raise FileExistsError("Refusing to overwrite an existing norm or audit artifact.")

    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("repo_id") != args.repo_id:
        raise ValueError("Dataset manifest is not PASS or does not match repo-id.")
    if Path(manifest["dataset_root"]).resolve() != args.dataset_root.resolve():
        raise ValueError("Dataset root does not match its frozen manifest.")
    tasks = tuple(manifest["canonical_task_order"])
    frame_counts = tuple(int(manifest["task_frame_counts"][task]) for task in tasks)

    config = config_module.get_config("pi05_g34_joint4_builtin_dual_lora")
    states, actions, task_indices = load_numeric_dataset(args.dataset_root, frame_counts, config.model.action_horizon)
    probabilities = data_loader_module.task_sampling_probabilities(frame_counts, EXPONENT)
    counts_array = np.asarray(frame_counts)
    state_weights = probabilities[task_indices] / counts_array[task_indices]
    action_values = actions.reshape(-1, actions.shape[-1])
    action_task_indices = np.repeat(task_indices, config.model.action_horizon)
    action_weights = probabilities[action_task_indices] / (
        counts_array[action_task_indices] * config.model.action_horizon
    )
    norm_stats = {
        "state": weighted_stats(states, state_weights),
        "actions": weighted_stats(action_values, action_weights),
    }

    data_factory = dataclasses.replace(config.data, repo_id=args.repo_id)
    data_config = data_factory.create(config.assets_dirs, config.model)
    official_dataset = data_loader_module.create_torch_dataset(data_config, config.model.action_horizon, config.model)
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
    for task_index, task in enumerate(tasks):
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

    raw = norm_path.read_bytes()
    audit = {
        "schema_version": "g4-weighted-joint-norm-audit-v1",
        "status": "PASS",
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_manifest_sha256": hashlib.sha256(args.dataset_manifest.read_bytes()).hexdigest(),
        "task_order": tasks,
        "frame_counts": dict(zip(tasks, frame_counts, strict=True)),
        "sampling_exponent": EXPONENT,
        "sampling_probabilities": dict(zip(tasks, probabilities.tolist(), strict=True)),
        "action_horizon": config.model.action_horizon,
        "weighting_rule": "task p_i proportional to n_i**0.43; uniform frame starts within task",
        "official_transform_reference_checks": reference_checks,
        "global": {key: stats_json(value) for key, value in norm_stats.items()},
        "per_task": per_task,
        "norm_stats": str(norm_path.resolve()),
        "norm_stats_sha256": hashlib.sha256(raw).hexdigest(),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS norm={norm_path} sha256={audit['norm_stats_sha256']}")


if __name__ == "__main__":
    main()
