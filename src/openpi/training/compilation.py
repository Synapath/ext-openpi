"""Explicit compilation identity for exact-draw training and resume."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import stat

import jax
from jax.experimental.compilation_cache import compilation_cache
import jaxlib
import numpy as np

from openpi.training import sharding


def require_canonical_step(state) -> None:
    step = state.step
    if tuple(step.shape) != () or step.dtype != np.dtype("int32") or getattr(step, "weak_type", False):
        raise ValueError("Exact-draw training requires a strong scalar int32 step")


def _signature(leaves):
    return [
        {"shape": list(x.shape), "dtype": str(x.dtype), "weak_type": bool(getattr(x, "weak_type", False))}
        for x in leaves
    ]


def _record(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Compilation identity drift: {path.name}")
    else:
        with path.open("xb") as stream:
            stream.write(payload)


@dataclasses.dataclass
class CompiledStep:
    executable: object
    state_tree: object

    def __call__(self, rng, state, batch):
        require_canonical_step(state)
        leaves, info = self.executable(rng, jax.tree.leaves(state), jax.tree.leaves(batch))
        return self.state_tree.unflatten(leaves), info


def compile_step(step, rng, state, batch, *, mesh, state_sharding, data_sharding, replicated_sharding):
    """Compile native math once, and require identical IR/cache inputs on resume.

    Flat outer arguments keep AOT ArgInfo metadata out of type-checked native
    dataclass constructors. The inner step and all optimizer math are unchanged.
    """
    require_canonical_step(state)
    cache = Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).expanduser()
    cache.mkdir(parents=True, mode=0o700, exist_ok=True)
    if cache.stat().st_uid != os.getuid() or stat.S_IMODE(cache.stat().st_mode) & 0o077:
        raise ValueError("Compilation cache must be private and owned by the current user")
    receipts = Path(os.environ["OPENPI_COMPILATION_RECEIPT_DIR"])
    receipts.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    # Earlier initialization kernels may already have opened JAX's process cache.
    # Rebind via the public API before compiling the guarded native update.
    compilation_cache.reset_cache()
    flat_state, state_tree = jax.tree.flatten(state)
    flat_batch, batch_tree = jax.tree.flatten(batch)
    shards = jax.tree.leaves(state_sharding)

    def call(key, state_leaves, batch_leaves):
        new_state, info = step(key, state_tree.unflatten(state_leaves), batch_tree.unflatten(batch_leaves))
        if jax.tree.structure(new_state) != state_tree:
            raise ValueError("Native train step changed the state structure")
        return jax.tree.leaves(new_state), info

    compiled_call = jax.jit(
        call,
        in_shardings=(replicated_sharding, shards, data_sharding),
        out_shardings=(shards, replicated_sharding),
        donate_argnums=(1,),
    )
    with sharding.set_mesh(mesh):
        lowered = compiled_call.lower(rng, flat_state, flat_batch)
        unoptimized = lowered.as_text().encode()
        executable = lowered.compile()
    optimized = executable.as_text().encode()
    identity = {
        "state": _signature(flat_state),
        "batch": _signature(flat_batch),
        "rng": _signature([rng]),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "devices": [{"id": d.id, "kind": d.device_kind, "platform": d.platform} for d in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "unoptimized_sha256": hashlib.sha256(unoptimized).hexdigest(),
        "optimized_sha256": hashlib.sha256(optimized).hexdigest(),
    }
    _record(receipts / "unoptimized.mlir", unoptimized)
    _record(receipts / "optimized.hlo", optimized)
    _record(receipts / "identity.json", (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode())
    return CompiledStep(executable, state_tree)
