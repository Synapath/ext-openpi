"""Verify official dense -> builtin dual-LoRA update-0 parity for G2.2."""

from __future__ import annotations

import gc
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

    dense_config = config_lib.get_config("pi05_base_aloha_full_sim_arx-x5_seed_0")
    noise = jax.random.normal(
        jax.random.key(20260903),
        (1, config.model.action_horizon, config.model.action_dim),
    )

    dense_model = dense_config.model.load(model_lib.restore_params(CHECKPOINT / "params"))
    dense_graphdef, dense_state = nnx.split(dense_model)
    dense_action = np.asarray(
        jax.jit(
            lambda state, obs, value: nnx.merge(dense_graphdef, state).sample_actions(
                jax.random.key(0), obs, num_steps=1, noise=value
            )
        )(dense_state, observation, noise)
    )
    del dense_model, dense_state
    jax.clear_caches()
    gc.collect()

    reference_model = config.model.create(jax.random.key(config.seed))
    graphdef, reference_state = nnx.split(reference_model)
    lora_params = config.weight_loader.load(reference_state.to_pure_dict())
    reference_state.replace_by_pure_dict(lora_params)
    lora_action = np.asarray(
        jax.jit(
            lambda state, obs, value: nnx.merge(graphdef, state).sample_actions(
                jax.random.key(0), obs, num_steps=1, noise=value
            )
        )(reference_state, observation, noise)
    )
    max_abs_diff = float(np.max(np.abs(dense_action - lora_action)))
    mean_abs_diff = float(np.mean(np.abs(dense_action - lora_action)))
    dense_abs_max = float(np.max(np.abs(dense_action)))
    relative_max_diff = max_abs_diff / max(dense_abs_max, 1e-12)
    if not np.isfinite(dense_action).all() or not np.isfinite(lora_action).all():
        raise FloatingPointError("update-0 action is non-finite")

    adapter_pairs: dict[str, dict[str, bool]] = {}
    for name, value in traverse_util.flatten_dict(reference_state.to_pure_dict(), sep="/").items():
        if name.endswith(("lora_a", "lora_b")):
            factor = name[-1]
            stem = name[:-1]
            adapter_pairs.setdefault(stem, {})[factor] = not np.asarray(value).any()
    if not adapter_pairs or any(
        set(factors) != {"a", "b"} or not any(factors.values()) for factors in adapter_pairs.values()
    ):
        raise ValueError("LoRA update-0 factors do not contain one zero factor per pair")

    # Dense and LoRA variants use different BF16 einsum paths even when every
    # adapter product is exactly zero. Bound per-element, aggregate, and
    # scale-relative drift at roughly one BF16 ULP at action scale.
    parity_tolerance = {"max_abs": 0.02, "mean_abs": 0.002, "relative_max": 0.01}
    if (
        max_abs_diff > parity_tolerance["max_abs"]
        or mean_abs_diff > parity_tolerance["mean_abs"]
        or relative_max_diff > parity_tolerance["relative_max"]
    ):
        raise ValueError(
            f"update-0 dense/LoRA parity drift: max={max_abs_diff} mean={mean_abs_diff} relative={relative_max_diff}"
        )

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
        "mean_abs_diff": mean_abs_diff,
        "relative_max_diff": relative_max_diff,
        "parity_tolerance": parity_tolerance,
        "lora_pairs": len(adapter_pairs),
        "lora_zero_factor_per_pair": True,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_fraction": trainable_count / total_count,
        "finite": True,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
