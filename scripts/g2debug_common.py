"""Frozen paths and effective-config construction for the G2.0-debug 2x2 run."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from openpi.training import config as training_config
from openpi.training import weight_loaders

RUN_ROOT = Path("/data/manip/outputs/ego-sim-eval/g2-rbdj/g2.0-debug-pi05-single-task-2x2-v1")
DATA_ROOT = RUN_ROOT / "execution/data-compiler"
VIEW_ROOT = Path("/data/manip/data/datasets/rbdj-g2-three-task-pi05-view-v1")
BASE_PARAMS = Path("/data/manip/data/openpi-assets/checkpoints/pi05_base/params")
TASKS = {
    "pi05_g2debug_pour_strict": ("pour_liquid_into_cup", "rbdj-g2debug-pour-v1"),
    "pi05_g2debug_pour_builtin": ("pour_liquid_into_cup", "rbdj-g2debug-pour-v1"),
    "pi05_g2debug_stack_strict": ("stack_bowls", "rbdj-g2debug-stack-v1"),
    "pi05_g2debug_stack_builtin": ("stack_bowls", "rbdj-g2debug-stack-v1"),
}
RUN_NAMES = {
    "pi05_g2debug_pour_strict": "g2.0-debug-pi05-pour-strict-s0-b32-u6k-v1",
    "pi05_g2debug_pour_builtin": "g2.0-debug-pi05-pour-builtin-s0-b32-u6k-v1",
    "pi05_g2debug_stack_strict": "g2.0-debug-pi05-stack-strict-s0-b32-u6k-v1",
    "pi05_g2debug_stack_builtin": "g2.0-debug-pi05-stack-builtin-s0-b32-u6k-v1",
}


def effective_config(
    name: str,
    *,
    workers: int,
    checkpoint_dir: Path | None = None,
    exp_name: str | None = None,
):
    if name not in TASKS or workers < 0:
        raise ValueError("known G2.0-debug config and non-negative workers are required")
    task, asset_id = TASKS[name]
    split_path = DATA_ROOT / "split.json"
    split = json.loads(split_path.read_text())
    episode_ids = tuple(
        row["episode_id"] for row in split["episodes"] if row["task"] == task and row["split"] == "train"
    )
    if len(episode_ids) != 90:
        raise ValueError("frozen single-task split must contain exactly 90 training episodes")
    value = training_config.get_config(name)
    data = dataclasses.replace(
        value.data,
        assets=training_config.AssetsConfig(
            assets_dir=str(DATA_ROOT / task / "assets"),
            asset_id=asset_id,
        ),
        base_config=dataclasses.replace(
            value.data.base_config,
            episode_indices=episode_ids,
            dataset_root=str(VIEW_ROOT),
            split_manifest_path=str(split_path),
            draw_manifest_path=str(DATA_ROOT / task / "draws/draw-manifest.json"),
        ),
    )
    replacements = {
        "data": data,
        "weight_loader": weight_loaders.CheckpointWeightLoader(str(BASE_PARAMS)),
        "num_workers": workers,
        "exp_name": exp_name or RUN_NAMES[name],
    }
    if checkpoint_dir is not None:
        replacements["checkpoint_dir_override"] = str(checkpoint_dir)
    return dataclasses.replace(value, **replacements)
