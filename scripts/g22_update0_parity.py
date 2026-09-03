"""Verify official dense -> builtin dual-LoRA update-0 parity for G2.2."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from flax import nnx
from flax import traverse_util
import jax
import numpy as np

from openpi.models import model as model_lib
from openpi.training import config as config_lib
from openpi.training import data_loader as data_loader_lib

CONFIG_NAME = "pi05_g22_classify_official_s0_b128_builtin_dual_lora"
CHECKPOINT = Path("/data/xiaoliu/manip/data/robodojo-cfb06d1/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def main() -> None:
    config = config_lib.get_config(CONFIG_NAME)
    loader = data_loader_lib.create_data_loader(config, num_batches=1)
    try:
        observation, _ = next(iter(loader))
        observation = jax.tree.map(lambda value: value[:1], observation)
    finally:
        loader.close()

    dense_params = model_lib.restore_params(CHECKPOINT / "params")
    dense_config = config_lib.get_config("pi05_base_aloha_full_sim_arx-x5_seed_0")
    dense_model = dense_config.model.load(dense_params)

    reference_model = config.model.create(jax.random.key(config.seed))
    graphdef, reference_state = nnx.split(reference_model)
    lora_params = config.weight_loader.load(reference_state.to_pure_dict())
    reference_state.replace_by_pure_dict(lora_params)
    lora_model = nnx.merge(graphdef, reference_state)

    noise = jax.random.normal(
        jax.random.key(20260903),
        (1, config.model.action_horizon, config.model.action_dim),
    )
    dense_action = np.asarray(
        jax.jit(lambda obs, value: dense_model.sample_actions(jax.random.key(0), obs, num_steps=1, noise=value))(
            observation, noise
        )
    )
    lora_action = np.asarray(
        jax.jit(lambda obs, value: lora_model.sample_actions(jax.random.key(0), obs, num_steps=1, noise=value))(
            observation, noise
        )
    )
    max_abs_diff = float(np.max(np.abs(dense_action - lora_action)))
    if not np.isfinite(dense_action).all() or not np.isfinite(lora_action).all():
        raise FloatingPointError("update-0 action is non-finite")
    if max_abs_diff > 1e-4:
        raise ValueError(f"update-0 dense/LoRA parity drift: {max_abs_diff}")

    shape = jax.eval_shape(lambda key: nnx.state(config.model.create(key)), jax.random.key(config.seed))
    flattened = traverse_util.flatten_dict(shape.to_pure_dict(), sep="/")
    trainable = traverse_util.flatten_dict(shape.filter(config.trainable_filter).to_pure_dict(), sep="/")
    trainable_count = sum(math.prod(value.shape) for value in trainable.values())
    total_count = sum(math.prod(value.shape) for value in flattened.values())
    if trainable_count != 466_957_072:
        raise ValueError(f"trainable count drift: {trainable_count}")

    receipt = {
        "schema": "g22-update0-parity-v1",
        "status": "PASS",
        "config": CONFIG_NAME,
        "base_checkpoint": str(CHECKPOINT),
        "official_norm_sha256": sha256(CHECKPOINT / "assets/arx_x5_sim/norm_stats.json"),
        "sample": "first frozen exact-draw training observation",
        "action_shape": list(dense_action.shape),
        "sampler_steps": 1,
        "dense_action_sha256": array_hash(dense_action),
        "lora_action_sha256": array_hash(lora_action),
        "max_abs_diff": max_abs_diff,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_fraction": trainable_count / total_count,
        "finite": True,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
