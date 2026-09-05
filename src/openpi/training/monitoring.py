"""Low-frequency health receipts. Missing hardware telemetry is explicit."""

import hashlib
import json
from pathlib import Path
import subprocess

from flax import traverse_util
import jax
import numpy as np
import optax

from openpi.training import checkpoints


def frozen_hashes(params, freeze_filter):
    return {
        "/".join(str(k) for k in key): hashlib.sha256(np.asarray(jax.device_get(value)).tobytes()).hexdigest()
        for key, value in traverse_util.flatten_dict(params.filter(freeze_filter).to_pure_dict()).items()
    }


def verify_frozen(state, config, path):
    path = Path(path)
    raw = frozen_hashes(state.params, config.freeze_filter)
    if path.exists():
        expected = json.loads(path.read_text())
    else:
        if int(state.step) != 0:
            raise ValueError("cannot establish fresh frozen identity from a resumed checkpoint")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, sort_keys=True) + "\n")
        expected = raw
    ema = frozen_hashes(checkpoints.inference_params(state), config.freeze_filter)
    for current in (raw, ema):
        if current != expected:
            raise ValueError("frozen raw/EMA identity drift")
    return {
        "health/frozen_raw_changed_leaves": 0,
        "health/frozen_ema_changed_leaves": 0,
        "health/frozen_leaf_count": len(expected),
    }


def trainable_health(state, config):
    raw = state.params.filter(config.trainable_filter)
    ema = checkpoints.inference_params(state).filter(config.trainable_filter)
    diff = jax.tree.map(lambda a, b: a - b, raw, ema)
    return {"health/raw_ema_trainable_rel_l2": float(optax.global_norm(diff) / (optax.global_norm(raw) + 1e-12))}


def system_metrics():
    result = {}
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        for line in text.splitlines():
            index, memory, util, temp = [int(x.strip()) for x in line.split(",")]
            for key, val in (("gpu_memory_reserved_mib", memory), ("gpu_util", util), ("temperature_c", temp)):
                result[f"system/gpu{index}/{key}"] = val
        result["system/nvml_available"] = 1
    except (OSError, ValueError, subprocess.SubprocessError):
        result["system/nvml_available"] = 0
    for device in jax.local_devices():
        stats = device.memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            result[f"system/gpu{device.id}/jax_peak_active_gib"] = stats["peak_bytes_in_use"] / 2**30
    return result
