"""Independently audit one G2.0-debug checkpoint on CPU without updating it."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from flax import nnx
import jax
import jax.numpy as jnp
from manip_datasets.robodojo_pi05 import RECORD
from manip_datasets.robodojo_pi05 import DrawManifest
import numpy as np
import orbax.checkpoint as ocp

from openpi.shared import nnx_utils

try:
    from scripts import g2debug_common
except ImportError:  # Direct execution adds scripts/ rather than the repository root.
    import g2debug_common  # type: ignore[no-redef]


EXPECTED_TRAINABLES = {"strict": 52_153_376, "builtin": 466_957_072}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten(tree, path=()):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from flatten(value, (*path, str(key)))
    elif isinstance(tree, tuple | list):
        for index, value in enumerate(tree):
            yield from flatten(value, (*path, str(index)))
    elif tree is not None:
        yield path, tree


def restore(path: Path):
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(path)
        args = jax.tree.map(
            lambda value: ocp.ArrayRestoreArgs(restore_type=np.ndarray)
            if hasattr(value, "shape")
            else ocp.RestoreArgs(),
            metadata,
        )
        return checkpointer.restore(path, restore_args=args)


def fresh_reference(config):
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    _, model_rng = jax.random.split(init_rng)
    model = config.model.create(model_rng)
    state = nnx.state(model)
    partial = config.weight_loader.load(state.to_pure_dict())
    state.replace_by_pure_dict(partial)
    state = nnx_utils.state_map(
        state,
        config.freeze_filter,
        lambda parameter: parameter.replace(parameter.value.astype(jnp.bfloat16)),
    )
    all_parameters = {
        "/".join(path[:-1]): np.asarray(value)
        for path, value in flatten(state.to_pure_dict())
        if path[-1] == "value"
    }
    trainable_names = {
        "/".join(path[:-1])
        for path, _ in flatten(state.filter(config.trainable_filter).to_pure_dict())
        if path[-1] == "value"
    }
    return all_parameters, trainable_names


def audit(config_name: str, checkpoint: Path, updates: int, workers: int) -> dict:
    started = time.monotonic()
    if not (checkpoint / "_CHECKPOINT_METADATA").is_file():
        raise FileNotFoundError(f"incomplete native checkpoint: {checkpoint}")
    config = g2debug_common.effective_config(config_name, workers=workers)
    recipe = "builtin" if config_name.endswith("_builtin") else "strict"
    expected, trainable_names = fresh_reference(config)
    trainable_count = sum(expected[name].size for name in trainable_names)
    if trainable_count != EXPECTED_TRAINABLES[recipe]:
        raise ValueError(f"trainable count drift: {trainable_count}")

    restored_params = restore(checkpoint / "params")["params"]
    actual = {
        "/".join(path[:-1]): np.asarray(value)
        for path, value in flatten(restored_params)
        if path[-1] == "value"
    }
    if set(actual) != set(expected):
        raise ValueError("checkpoint/fresh parameter name set mismatch")
    changed_trainables = 0
    frozen_parameters = 0
    tree_digest = hashlib.sha256()
    for name in sorted(actual):
        value = actual[name]
        reference = expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype or not np.isfinite(value).all():
            raise ValueError(f"shape/dtype/finite parameter gate: {name}")
        value_digest = hashlib.sha256(value.tobytes()).hexdigest()
        reference_digest = hashlib.sha256(reference.tobytes()).hexdigest()
        tree_digest.update(f"{name}\0{value_digest}\n".encode())
        if name in trainable_names:
            changed_trainables += value_digest != reference_digest
        else:
            frozen_parameters += 1
            if value_digest != reference_digest:
                raise ValueError(f"frozen parameter drift: {name}")
    if updates and not changed_trainables:
        raise ValueError("no trainable parameter changed from fresh initialization")

    del restored_params, actual, expected
    restored_state = restore(checkpoint / "train_state")
    if int(restored_state["step"]) != updates or str(restored_state["step"].dtype) != "int32":
        raise ValueError("optimizer step drift")
    if restored_state["params"] or restored_state["ema_params"] is not None:
        raise ValueError("EMA/parameter split drift")
    moment_names = {"mu": set(), "nu": set()}
    optimizer_leaves = 0
    optimizer_bytes = 0
    counters = []
    for path, value in flatten(restored_state["opt_state"]):
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite optimizer leaf: {'/'.join(path)}")
        optimizer_leaves += 1
        optimizer_bytes += array.nbytes
        if path[-1] == "count":
            if array.shape != () or str(array.dtype) != "int32" or int(array) != updates:
                raise ValueError(f"optimizer counter drift: {'/'.join(path)}")
            counters.append("/".join(path))
        for moment, names in moment_names.items():
            if moment in path:
                name = "/".join(path[path.index(moment) + 1 : -1])
                if path[-1] != "value" or name not in trainable_names:
                    raise ValueError(f"optimizer moment identity drift: {'/'.join(path)}")
                names.add(name)
    if len(counters) != 2 or any(names != trainable_names for names in moment_names.values()):
        raise ValueError("AdamW counter or moment coverage drift")

    cursor = json.loads((checkpoint / "assets/data-cursor.json").read_text())
    task, asset_id = g2debug_common.TASKS[config_name]
    task_root = g2debug_common.DATA_ROOT / task
    identity_paths = {
        "draw_manifest_sha256": task_root / "draws/draw-manifest.json",
        "split_sha256": g2debug_common.DATA_ROOT / "split.json",
        "view_manifest_sha256": g2debug_common.VIEW_ROOT / "view-manifest.json",
    }
    if cursor.get("schema") != "rbdj-cursor-v1" or cursor.get("completed_updates") != updates:
        raise ValueError("data cursor update drift")
    if any(cursor.get(key) != sha256(path) for key, path in identity_paths.items()):
        raise ValueError("data cursor source identity drift")
    prefix, task_counts, episode_counts = hashlib.sha256(), Counter(), Counter()
    with DrawManifest(
        task_root / "draws/draw-manifest.json",
        g2debug_common.DATA_ROOT / "split.json",
        verify=False,
    ) as draws:
        for index in range(updates * config.batch_size):
            row = draws[index]
            prefix.update(RECORD.pack(*row))
            task_counts[str(row[2])] += 1
            episode_counts[str(row[3])] += 1
    if (
        cursor.get("prefix_sha256") != prefix.hexdigest()
        or cursor.get("task_counts") != dict(task_counts)
        or cursor.get("episode_counts") != dict(episode_counts)
    ):
        raise ValueError("data cursor prefix/count drift")

    checkpoint_norm = checkpoint / "assets" / asset_id / "norm_stats.json"
    source_norm = task_root / "assets" / asset_id / "norm_stats.json"
    if json.loads(checkpoint_norm.read_text()) != json.loads(source_norm.read_text()):
        raise ValueError("normalization value drift")
    files = [path for path in sorted(checkpoint.rglob("*")) if path.is_file()]
    return {
        "schema": "g2debug-checkpoint-content-audit-v1",
        "status": "PASS_CPU_CHECKPOINT_CONTENT",
        "cpu_only": True,
        "config": config_name,
        "recipe": recipe,
        "checkpoint": str(checkpoint.resolve()),
        "updates": updates,
        "trainable_count": trainable_count,
        "trainable_parameters": len(trainable_names),
        "changed_trainable_parameters": changed_trainables,
        "frozen_parameters": frozen_parameters,
        "frozen_diff_zero": True,
        "parameter_tree_sha256": tree_digest.hexdigest(),
        "optimizer_leaves": optimizer_leaves,
        "optimizer_bytes": optimizer_bytes,
        "optimizer_counters": counters,
        "optimizer_moment_coverage": {key: len(value) for key, value in moment_names.items()},
        "cursor": cursor,
        "normalization_values_exact": True,
        "normalization_checkpoint_sha256": sha256(checkpoint_norm),
        "files": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "elapsed_seconds": time.monotonic() - started,
        "not_verified_here": ["finite action inference", "H50 holdout loss", "policy rollout"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True, choices=sorted(g2debug_common.TASKS))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--updates", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.config, args.checkpoint, args.updates, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cursor"}, sort_keys=True))


if __name__ == "__main__":
    main()
