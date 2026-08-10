#!/usr/bin/env python3
"""Replay and audit the deterministic G3.4 weighted sampler."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.training import data_loader

TASKS = ("adjust_bottle", "lift_pot", "pick_dual_bottles", "handover_block")
FRAME_COUNTS = (7188, 5554, 6129, 14084)
EXPONENT = 0.43


def replay_indices(batch_size: int, updates: int, seed: int) -> np.ndarray:
    dataset_size = sum(FRAME_COUNTS)
    sampler = data_loader.create_weighted_task_sampler(
        FRAME_COUNTS,
        EXPONENT,
        dataset_size=dataset_size,
        seed=seed,
    )
    required = batch_size * updates
    usable_per_epoch = dataset_size // batch_size * batch_size
    parts = []
    while required:
        epoch = np.fromiter(sampler, dtype=np.int64, count=dataset_size)[:usable_per_epoch]
        take = min(required, usable_per_epoch)
        parts.append(epoch[:take])
        required -= take
    return np.concatenate(parts)


def task_indices(frame_indices: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.cumsum(FRAME_COUNTS), frame_indices, side="right")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite sampler audit: {args.output}")
    if args.batch_size <= 0 or args.updates <= 0:
        raise ValueError("Batch size and updates must be positive.")

    indices = replay_indices(args.batch_size, args.updates, args.seed)
    observed_task_indices = task_indices(indices)
    observed_counts = np.bincount(observed_task_indices, minlength=len(TASKS))
    observed_probabilities = observed_counts / len(indices)
    expected_probabilities = data_loader.task_sampling_probabilities(FRAME_COUNTS, EXPONENT)
    absolute_errors = np.abs(observed_probabilities - expected_probabilities)
    checks = {
        "draw_count_exact": len(indices) == args.batch_size * args.updates,
        "all_indices_in_range": bool(np.all((indices >= 0) & (indices < sum(FRAME_COUNTS)))),
        "max_probability_error_at_most_0.005": float(absolute_errors.max()) <= 0.005,
    }
    result = {
        "schema_version": "g34-weighted-sampler-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "updates": args.updates,
        "draws": len(indices),
        "dataset_frames": sum(FRAME_COUNTS),
        "usable_draws_per_epoch_after_drop_last": sum(FRAME_COUNTS) // args.batch_size * args.batch_size,
        "sampling_exponent": EXPONENT,
        "expected_probabilities": dict(zip(TASKS, expected_probabilities.tolist(), strict=True)),
        "observed_counts": dict(zip(TASKS, observed_counts.tolist(), strict=True)),
        "observed_probabilities": dict(zip(TASKS, observed_probabilities.tolist(), strict=True)),
        "absolute_probability_errors": dict(zip(TASKS, absolute_errors.tolist(), strict=True)),
        "sampled_index_sequence_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{result['status']} draws={len(indices)} max_error={absolute_errors.max():.8f}")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
