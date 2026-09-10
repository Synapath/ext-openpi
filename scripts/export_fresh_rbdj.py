"""Export the trainer's zero-update initialization with task-specific norm assets.

Run on CPU with a training binding; this creates no optimizer updates and never
reads a fine-tuned checkpoint. The output directory must be new.
"""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import time

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import traverse_util

from openpi.training import checkpoints, config, weight_loaders
import train


def export(binding, output):
    if output.exists():
        raise FileExistsError(output)
    if jax.default_backend() != "cpu":
        raise ValueError("fresh reference export is assigned to CPU")
    started = time.time()
    recipe = dataclasses.replace(
        config.get_config(binding["recipe"]),
        weight_loader=weight_loaders.CheckpointWeightLoader(binding["base_params"]),
    )
    _, init_rng = jax.random.split(jax.random.key(recipe.seed))
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, 1), ("batch", "fsdp"))
    with mesh:
        state, _ = train.init_train_state(recipe, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    if int(state.step) != 0:
        raise ValueError("reference must have zero updates")
    params = jax.device_get(checkpoints.inference_params(state))
    leaves = traverse_util.flatten_dict(params.to_pure_dict(), sep="/")
    for name, value in leaves.items():
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite initial parameter: {name}")
    lora_b = [v for k, v in leaves.items() if "lora_b" in k]
    if not lora_b:
        raise ValueError("LoRA recipe must contain adapter parameters")
    output.mkdir(parents=True)
    with ocp.PyTreeCheckpointer() as saver:
        saver.save(output / "params", {"params": params})
    asset = recipe.data.assets.asset_id
    shutil.copytree(Path(binding["assets_dir"]) / asset, output / "assets" / asset)
    receipt = dict(
        recipe=recipe.name,
        seed=recipe.seed,
        completed_updates=0,
        base_params=binding["base_params"],
        reference="trainer zero-update initialization; native OpenPI LoRA initialization retained",
        lora_b_leaves=len(lora_b),
        zero_lora_b_leaves=sum(np.count_nonzero(v) == 0 for v in lora_b),
        norm_sha256=hashlib.sha256((output / "assets" / asset / "norm_stats.json").read_bytes()).hexdigest(),
        elapsed_s=time.time() - started,
        parameter_hashes={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in leaves.items()},
    )
    (output / "export.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("binding", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(json.loads(args.binding.read_text()), args.output)
