"""Measure the real G2.0-debug video/augmentation loader without optimizer updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import g2debug_common
import numpy as np
import psutil

from openpi.training import data_loader


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values), q))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True, choices=sorted(g2debug_common.TASKS))
    parser.add_argument("--workers", required=True, type=int, choices=(0, 4, 8, 12, 16))
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.batches < 1 or args.output.exists():
        raise ValueError("positive batches and a fresh output path are required")
    config = g2debug_common.effective_config(args.config, workers=args.workers)
    io_before = psutil.disk_io_counters()
    cpu_before = psutil.cpu_times()
    available_before = psutil.virtual_memory().available
    create_started = time.perf_counter()
    loader = data_loader.create_data_loader(config, num_batches=args.batches)
    create_seconds = time.perf_counter() - create_started
    waits: list[float] = []
    started = time.perf_counter()
    iterator = iter(loader)
    try:
        for completed_updates in range(1, args.batches + 1):
            batch_started = time.perf_counter()
            observation, actions = next(iterator)
            waits.append(time.perf_counter() - batch_started)
            leaves = [*observation.images.values(), observation.state, actions]
            if not all(np.isfinite(np.asarray(value)).all() for value in leaves):
                raise FloatingPointError("non-finite real-loader batch")
            loader.commit_batch(completed_updates)
    finally:
        loader.close()
    elapsed = time.perf_counter() - started
    io_after = psutil.disk_io_counters()
    cpu_after = psutil.cpu_times()
    receipt = {
        "schema": "g2debug-loader-benchmark-v1",
        "config": args.config,
        "task": config.policy_metadata["task_info"]["task"],
        "workers": args.workers,
        "batches": args.batches,
        "batch_size": config.batch_size,
        "samples": args.batches * config.batch_size,
        "create_seconds": create_seconds,
        "elapsed_seconds": elapsed,
        "batches_per_second": args.batches / elapsed,
        "samples_per_second": args.batches * config.batch_size / elapsed,
        "batch_wait_seconds": {
            "mean": statistics.fmean(waits),
            "median": statistics.median(waits),
            "p95": percentile(waits, 0.95),
            "max": max(waits),
        },
        "cpu_seconds_delta": {
            "user": cpu_after.user - cpu_before.user,
            "system": cpu_after.system - cpu_before.system,
            "iowait": cpu_after.iowait - cpu_before.iowait,
        },
        "disk_bytes_delta": {
            "read": io_after.read_bytes - io_before.read_bytes,
            "write": io_after.write_bytes - io_before.write_bytes,
        },
        "available_memory_bytes": {
            "before": available_before,
            "after": psutil.virtual_memory().available,
        },
        "draw_cursor": loader.cursor_receipt(args.batches),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
