#!/usr/bin/env python3
"""Compute the exact sampler-weighted joint normalization artifact for G3.4."""

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

TASKS = ("adjust_bottle", "lift_pot", "pick_dual_bottles", "handover_block")
FRAME_COUNTS = (7188, 5554, 6129, 14084)
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


def load_numeric_dataset(root: Path, action_horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    observed_counts = tuple(int(np.sum(task_indices == index)) for index in range(len(TASKS)))
    if states.shape != (sum(FRAME_COUNTS), 14) or observed_counts != FRAME_COUNTS:
        raise ValueError(f"Frozen frame contract mismatch: states={states.shape}, task_counts={observed_counts}.")

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
    parser.add_argument("--config-name", default="pi05_g34_joint4_builtin_dual_lora")
    parser.add_argument("--repo-id", default="RoboTwin-g34-joint4-aloha_agilex-joint")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / "norm_stats.json").exists() or args.audit_output.exists():
        raise FileExistsError("Refusing to overwrite an existing norm or audit artifact.")

    config = config_module.get_config(args.config_name)
    states, actions, task_indices = load_numeric_dataset(args.dataset_root, config.model.action_horizon)
    probabilities = data_loader_module.task_sampling_probabilities(FRAME_COUNTS, EXPONENT)
    state_weights = probabilities[task_indices] / np.asarray(FRAME_COUNTS)[task_indices]
    action_values = actions.reshape(-1, actions.shape[-1])
    action_task_indices = np.repeat(task_indices, config.model.action_horizon)
    action_weights = probabilities[action_task_indices] / (
        np.asarray(FRAME_COUNTS)[action_task_indices] * config.model.action_horizon
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
    for task_index, task_id in enumerate(TASKS):
        state_mask = task_indices == task_index
        action_mask = action_task_indices == task_index
        task_state_weights = np.ones(int(state_mask.sum()), dtype=np.float64)
        task_action_weights = np.ones(int(action_mask.sum()), dtype=np.float64)
        per_task[task_id] = {
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
        "schema_version": "g34-weighted-joint-norm-audit-v1",
        "status": "PASS",
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root.resolve()),
        "frame_counts": dict(zip(TASKS, FRAME_COUNTS, strict=True)),
        "sampling_exponent": EXPONENT,
        "sampling_probabilities": dict(zip(TASKS, probabilities.tolist(), strict=True)),
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
