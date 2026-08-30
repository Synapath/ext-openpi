"""Freeze effective configs and trainable identities for the G2.0-debug matrix."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import subprocess

from flax import nnx
from flax import traverse_util
import g2debug_common
import jax

EXPECTED_TRAINABLES = {"strict": 52_153_376, "builtin": 466_957_072}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(workers: int, output: Path) -> dict:
    if workers <= 0 or output.exists():
        raise ValueError("a positive frozen worker count and fresh output directory are required")
    output.mkdir(parents=True)
    configs = {}
    scientific_rows = []
    for name in g2debug_common.TASKS:
        value = g2debug_common.effective_config(name, workers=workers)
        state = jax.eval_shape(
            lambda key, model_config=value.model: nnx.state(model_config.create(key)), jax.random.key(value.seed)
        )
        flat_all = traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
        flat_trainable = traverse_util.flatten_dict(state.filter(value.trainable_filter).to_pure_dict(), sep="/")
        trainable_count = sum(math.prod(parameter.shape) for parameter in flat_trainable.values())
        recipe = "builtin" if name.endswith("_builtin") else "strict"
        if trainable_count != EXPECTED_TRAINABLES[recipe]:
            raise ValueError(f"{name}: trainable count drift {trainable_count}")
        task, asset_id = g2debug_common.TASKS[name]
        task_root = g2debug_common.DATA_ROOT / task
        row = {
            "name": name,
            "run_name": value.exp_name,
            "task": task,
            "recipe": value.policy_metadata["recipe"],
            "trainable_count": trainable_count,
            "parameter_count": sum(math.prod(parameter.shape) for parameter in flat_all.values()),
            "parameter_leaves": len(flat_all),
            "trainable_leaves": len(flat_trainable),
            "trainable_parameters": [
                {"name": key, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
                for key, parameter in sorted(flat_trainable.items())
            ],
            "asset_id": asset_id,
            "norm_sha256": sha256(task_root / "assets" / asset_id / "norm_stats.json"),
            "draw_manifest_sha256": sha256(task_root / "draws/draw-manifest.json"),
            "draw_binary_sha256": sha256(task_root / "draws/draws.bin"),
        }
        configs[name] = row
        (output / f"{name}.txt").write_text(repr(value) + "\n")
        scientific_rows.append(
            {
                "batch_size": value.batch_size,
                "ema_decay": value.ema_decay,
                "fsdp_devices": value.fsdp_devices,
                "lr_schedule": dataclasses.asdict(value.lr_schedule),
                "model": {
                    "action_dim": value.model.action_dim,
                    "action_horizon": value.model.action_horizon,
                    "dtype": value.model.dtype,
                    "pi05": value.model.pi05,
                },
                "num_train_steps": value.num_train_steps,
                "num_workers": value.num_workers,
                "optimizer": dataclasses.asdict(value.optimizer),
                "seed": value.seed,
            }
        )
    if any(row != scientific_rows[0] for row in scientific_rows[1:]):
        raise ValueError("matrix common scientific fields drift")
    result = {
        "schema": "g2debug-effective-config-matrix-v1",
        "status": "PASS",
        "workers": workers,
        "common_scientific_config": scientific_rows[0],
        "configs": configs,
        "split_sha256": sha256(g2debug_common.DATA_ROOT / "split.json"),
        "base_params_tree": str(g2debug_common.BASE_PARAMS),
        "external_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "external_clean": not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip(),
    }
    (output / "matrix.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = freeze(args.workers, args.output)
    print(json.dumps({key: result[key] for key in ("status", "workers", "external_commit", "external_clean")}))


if __name__ == "__main__":
    main()
