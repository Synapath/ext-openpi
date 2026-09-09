"""Token-only frozen-feature optimizer and independent token checkpoints."""

from dataclasses import asdict
import hashlib
import os
from pathlib import Path

from flax import serialization
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models.rl_token import ARToken
from openpi.models.rl_token import ARTokenConfig
from openpi.models.rl_token import validate_features


@nnx.jit
def _step(token, optimizer, prefix, mask):
    loss, grad = nnx.value_and_grad(lambda m: m.loss(prefix, mask))(token)
    optimizer.update(grad)
    return loss, optax.global_norm(grad.to_pure_dict())


@nnx.jit
def _gradient(token, prefix, mask):
    return nnx.value_and_grad(lambda m: m.loss(prefix, mask))(token)


@nnx.jit
def _apply_gradient(optimizer, grad):
    optimizer.update(grad)


class TokenTrainer:
    def __init__(self, config=ARTokenConfig(), *, seed=0, learning_rate=1e-4, base_id):
        if not base_id or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("base identity and positive LR required")
        self.token = ARToken(config, nnx.Rngs(seed))
        self.lr = learning_rate
        self.base_id = base_id
        self.optimizer = nnx.Optimizer(self.token, optax.adam(learning_rate))
        self.updates = 0

    def update(self, prefix, mask):
        validate_features(prefix, mask)
        if np.shape(prefix)[-1] != self.token.config.dim:
            raise ValueError("feature width")
        # NNX mutates graph state; retain old state to roll back a nonfinite update.
        before = nnx.state((self.token, self.optimizer))
        loss, grad = _step(self.token, self.optimizer, jnp.asarray(prefix), jnp.asarray(mask))
        if not all(
            np.isfinite(np.asarray(x)).all()
            for x in jax.tree.leaves((nnx.state((self.token, self.optimizer)), loss, grad))
        ):
            nnx.update((self.token, self.optimizer), before)
            raise FloatingPointError("nonfinite token update rolled back")
        self.updates += 1
        return {"token/reconstruction_l2": float(loss), "token/grad_norm": float(grad)}

    def update_accumulated(self, microbatches):
        """One Adam step for an example-weighted batch, without retaining activations."""
        before = nnx.state((self.token, self.optimizer))
        total, count, grad_sum = 0.0, 0, None
        try:
            for prefix, mask in microbatches:
                validate_features(prefix, mask)
                if prefix.shape[-1] != self.token.config.dim:
                    raise ValueError("feature width")
                loss, grad = _gradient(self.token, jnp.asarray(prefix), jnp.asarray(mask))
                n = prefix.shape[0]
                total += float(loss) * n
                count += n
                weighted = jax.tree.map(lambda x, n=n: x.astype(jnp.float32) * n, grad)
                grad_sum = weighted if grad_sum is None else jax.tree.map(jnp.add, grad_sum, weighted)
            if not count:
                raise ValueError("empty accumulated batch")
            grad = jax.tree.map(lambda x: x / count, grad_sum)
            norm = optax.global_norm(grad.to_pure_dict())
            if not np.isfinite([total / count, float(norm)]).all():
                raise FloatingPointError("nonfinite accumulated gradient")
            _apply_gradient(self.optimizer, grad)
            if not all(
                np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(nnx.state((self.token, self.optimizer)))
            ):
                raise FloatingPointError("nonfinite token state")
        except BaseException:
            nnx.update((self.token, self.optimizer), before)
            raise
        self.updates += 1
        return {"token/reconstruction_l2": total / count, "token/grad_norm": float(norm)}

    def save(self, path):
        payload = {
            "schema": "pi05-ar-token-v1",
            "config": asdict(self.token.config),
            "learning_rate": self.lr,
            "base_id": self.base_id,
            "updates": self.updates,
            "state": serialization.to_state_dict(
                jax.device_get(nnx.state((self.token, self.optimizer)).to_pure_dict())
            ),
        }
        raw = serialization.msgpack_serialize(payload)
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        temp = path.with_name(path.name + ".partial")
        with temp.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, path)
        temp.unlink()
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(cls, path, *, expected_sha256, base_id):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("token hash mismatch")
        payload = serialization.msgpack_restore(raw)
        if payload["schema"] != "pi05-ar-token-v1" or payload["base_id"] != base_id:
            raise ValueError("token/base identity mismatch")
        obj = cls(ARTokenConfig(**payload["config"]), learning_rate=payload["learning_rate"], base_id=base_id)
        state = nnx.state((obj.token, obj.optimizer))
        restored = serialization.from_state_dict(state.to_pure_dict(), payload["state"])
        state.replace_by_pure_dict(restored)
        nnx.update((obj.token, obj.optimizer), jax.tree.map(jnp.asarray, state))
        obj.updates = payload["updates"]
        return obj
