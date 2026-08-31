"""Evaluate frozen G2.0-debug H50 validation loss without optimizer updates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from manip_datasets import robodojo_pi05 as source
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from openpi.training import data_loader

try:
    from scripts import g2debug_common
except ImportError:  # Direct execution adds scripts/ rather than the repository root.
    import g2debug_common

SEGMENTS = {"prefix_1_16": (0, 16), "middle_17_32": (16, 32), "tail_33_50": (32, 50)}


def compile_validation_manifest(config_name: str, output: Path, *, anchors_per_episode: int = 10) -> dict:
    if anchors_per_episode < 3:
        raise ValueError("anchors_per_episode must cover all three phases")
    task, _ = g2debug_common.TASKS[config_name]
    split_path = g2debug_common.DATA_ROOT / "split.json"
    split = json.loads(split_path.read_text())
    source.validate_split(split)
    episodes = [row for row in split["episodes"] if row["task"] == task and row["split"] == "holdout"]
    if len(episodes) != 10:
        raise ValueError(f"{task}: frozen split must contain exactly 10 validation episodes")
    anchors = []
    for row in episodes:
        frames = [
            min((2 * index + 1) * row["anchors"] // (2 * anchors_per_episode), row["anchors"] - 1)
            for index in range(anchors_per_episode)
        ]
        if len(set(frames)) != anchors_per_episode:
            raise ValueError(f"{row['episode_id']}: validation anchors are not unique")
        for frame in frames:
            phase_index = min(2, 3 * frame // row["anchors"])
            anchors.append(
                {
                    "episode_id": row["episode_id"],
                    "frame_idx": frame,
                    "phase": ("start", "mid", "late")[phase_index],
                    "task_index": row["task_index"],
                }
            )
    result = {
        "schema": "g2debug-h50-validation-v1",
        "task": task,
        "split_sha256": source.sha256(split_path),
        "episode_ids": [row["episode_id"] for row in episodes],
        "anchors_per_episode": anchors_per_episode,
        "anchors": anchors,
        "action_horizon": 50,
        "action_dim": 14,
        "flow_seed": 0,
        "flow_rng": "jax.random.fold_in(key(0), batch_index); fixed batch_size and order",
        "sampling": "equal-per-episode/equal-width-midpoint-v1",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise FileExistsError(f"existing validation identity differs: {output}")
    else:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


class ValidationDataset:
    def __init__(self, native, transform, manifest: dict, split: dict):
        self.native = native
        self.transform = transform
        self.anchors = manifest["anchors"]
        episode_rows = {row["episode_id"]: row for row in split["episodes"] if row["split"] == "holdout"}
        epids = np.asarray(native.hf_dataset["episode_index"], dtype=np.int64).reshape(-1)
        frames = np.asarray(native.hf_dataset["frame_index"], dtype=np.int64).reshape(-1)
        self.offsets = {}
        for episode_id in manifest["episode_ids"]:
            row = episode_rows[episode_id]
            positions = np.flatnonzero(epids == episode_id)
            if (
                len(positions) != row["length"]
                or not np.array_equal(frames[positions], np.arange(row["length"]))
                or not np.array_equal(positions, np.arange(positions[0], positions[0] + len(positions)))
            ):
                raise ValueError(f"validation native mapping mismatch: {episode_id}")
            self.offsets[episode_id] = int(positions[0])

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        identity = self.anchors[index]
        item = self.native[self.offsets[identity["episode_id"]] + identity["frame_idx"]]
        actual = (int(item["task_index"]), int(item["episode_index"]), int(item["frame_index"]))
        expected = (identity["task_index"], identity["episode_id"], identity["frame_idx"])
        if actual != expected or np.asarray(item["action_is_pad"]).any():
            raise ValueError(f"validation identity or H50 padding drift: {actual} != {expected}")
        value = self.transform(item)
        value["_validation_index"] = np.asarray(index, dtype=np.int64)
        return value


def create_validation_loader(config, manifest: dict, *, batch_size: int, workers: int):
    data_config = config.data.create(config.assets_dirs, config.model)
    split = json.loads((g2debug_common.DATA_ROOT / "split.json").read_text())
    native = data_loader.lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=Path(data_config.dataset_root),
        episodes=manifest["episode_ids"],
        download_videos=False,
        delta_timestamps={key: [step / 25 for step in range(50)] for key in data_config.action_sequence_keys},
        video_backend=data_config.video_backend,
    )
    if data_config.norm_stats is None:
        raise ValueError("validation must use frozen train-only normalization")
    transform = transforms.compose(
        [
            transforms.PromptFromLeRobotTask(native.meta.tasks),
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    dataset = ValidationDataset(native, transform, manifest, split)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "collate_fn": data_loader._collate_fn,  # noqa: SLF001
        "drop_last": False,
        "worker_init_fn": _worker_init,
    }
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    return torch.utils.data.DataLoader(**kwargs)


def _worker_init(_):
    torch.set_num_threads(1)


def aggregate(native: np.ndarray, components: np.ndarray, phases: list[str]) -> dict:
    if native.ndim != 2 or components.shape[:2] != native.shape or components.shape[-1] < 14 or native.shape[1] != 50:
        raise ValueError("expected native [N,50] and components [N,50,D>=14]")
    component_reduced = components.mean(axis=-1)
    max_component_error = float(np.max(np.abs(native - component_reduced)))
    full = float(native.mean())
    full_component_error = abs(full - float(components.mean()))
    segments = {name: float(native[:, start:end].mean()) for name, (start, end) in SEGMENTS.items()}
    weighted = float(sum(segments[name] * (end - start) for name, (start, end) in SEGMENTS.items()) / 50)
    aggregation_error = abs(full - weighted)
    if max_component_error > 5e-7 or aggregation_error > 5e-7 or full_component_error > 2e-6:
        raise ValueError(
            "native loss reproduction failed: "
            f"position_component={max_component_error} full_component={full_component_error} "
            f"segment={aggregation_error}"
        )
    phase_metrics = {}
    phase_array = np.asarray(phases)
    for phase in ("start", "mid", "late"):
        phase_native = native[phase_array == phase]
        phase_metrics[phase] = {
            "native_full_h50": float(phase_native.mean()),
            **{name: float(phase_native[:, start:end].mean()) for name, (start, end) in SEGMENTS.items()},
        }
    return {
        "native_full_h50": full,
        **segments,
        "physical_action_components_14": components[:, :, :14].mean(axis=(0, 1)).tolist(),
        "padded_action_components": components[:, :, 14:].mean(axis=(0, 1)).tolist(),
        "model_action_dim": components.shape[-1],
        "phase_metrics": phase_metrics,
        "native_component_max_abs_error": max_component_error,
        "native_full_component_aggregation_abs_error": full_component_error,
        "native_segment_aggregation_abs_error": aggregation_error,
    }


def evaluate(args) -> dict:
    manifest = compile_validation_manifest(args.config, args.manifest, anchors_per_episode=args.anchors_per_episode)
    if len(manifest["anchors"]) % args.batch_size:
        raise ValueError("fixed validation count must be divisible by batch size")
    config = g2debug_common.effective_config(args.config, workers=args.workers)
    if args.fresh_base:
        initialized = config.model.create(jax.random.key(config.seed))
        initial_params = nnx.state(initialized).to_pure_dict()
        # CheckpointWeightLoader supplies the dense official base and retains
        # deterministic task-recipe parameters that do not exist in that base.
        merged_params = config.weight_loader.load(initial_params)
        model = config.model.load(jax.tree.map(jnp.asarray, merged_params))
    else:
        model = config.model.load(model_lib.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16))
    model.eval()

    native_loss = nnx_utils.module_jit(model.compute_loss)
    component_loss = nnx_utils.module_jit(model.compute_loss_components)
    action_sampler = nnx_utils.module_jit(model.sample_actions)
    loader = create_validation_loader(config, manifest, batch_size=args.batch_size, workers=args.workers)
    natives, components, indices = [], [], []
    native_probe_max_abs_error = None
    action_inference = None
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        validation_indices = np.asarray(batch.pop("_validation_index"), dtype=np.int64)
        observation = model_lib.Observation.from_dict(batch)
        actions = batch["actions"]
        rng = jax.random.fold_in(jax.random.key(manifest["flow_seed"]), batch_index)
        component = np.asarray(jax.device_get(component_loss(rng, observation, actions)))
        reduced_component = component.mean(axis=-1)
        if batch_index == 0:
            native_probe = np.asarray(jax.device_get(native_loss(rng, observation, actions)))
            native_probe_max_abs_error = float(np.max(np.abs(native_probe - reduced_component)))
            if native_probe_max_abs_error > 5e-7:
                raise ValueError(f"independent native loss probe mismatch: {native_probe_max_abs_error}")
            sampled_actions = np.asarray(
                jax.device_get(
                    action_sampler(
                        jax.random.fold_in(jax.random.key(manifest["flow_seed"] + 1), batch_index),
                        observation,
                        num_steps=10,
                    )
                )
            )
            if sampled_actions.shape != (args.batch_size, 50, 32) or not np.isfinite(sampled_actions).all():
                raise ValueError(f"finite H50 action inference gate failed: {sampled_actions.shape}")
            action_inference = {
                "status": "PASS_FINITE_ACTION_INFERENCE",
                "samples": args.batch_size,
                "shape": list(sampled_actions.shape),
                "num_denoise_steps": 10,
                "seed": manifest["flow_seed"] + 1,
                "physical_action_dim": 14,
                "model_action_dim": sampled_actions.shape[-1],
                "max_abs_physical_action": float(np.max(np.abs(sampled_actions[:, :, :14]))),
            }
        natives.append(reduced_component)
        components.append(np.asarray(component))
        indices.extend(validation_indices.tolist())
    if indices != list(range(len(manifest["anchors"]))):
        raise ValueError("validation loader order/coverage drift")
    result = {
        "schema": "g2debug-h50-loss-result-v1",
        "status": "PASS",
        "config": args.config,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_params_identity": _tree_identity(args.checkpoint / "params"),
        "fresh_base_merge": args.fresh_base,
        "validation_manifest": str(args.manifest.resolve()),
        "validation_manifest_sha256": source.sha256(args.manifest),
        "samples": len(indices),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "elapsed_seconds": time.perf_counter() - started,
        "independent_native_probe_samples": args.batch_size,
        "independent_native_probe_max_abs_error": native_probe_max_abs_error,
        "action_inference": action_inference,
        "metrics": aggregate(
            np.concatenate(natives),
            np.concatenate(components),
            [row["phase"] for row in manifest["anchors"]],
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _tree_identity(root: Path) -> dict:
    files = []
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode() + b"\0" + str(size).encode() + b"\n")
        files.append({"path": relative, "size": size})
    if not files:
        raise FileNotFoundError(f"checkpoint params tree is empty: {root}")
    return {"files": len(files), "bytes": sum(row["size"] for row in files), "structure_sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True, choices=sorted(g2debug_common.TASKS))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--anchors-per-episode", type=int, default=10)
    parser.add_argument("--fresh-base", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), sort_keys=True))


if __name__ == "__main__":
    main()
