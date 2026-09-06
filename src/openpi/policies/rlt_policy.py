"""Opt-in AR-token provider; uses the trained Policy's exact transforms and base."""

import copy
import threading

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_api
from openpi.shared import nnx_utils


class TokenPolicyModel(nnx.Module):
    def __init__(self, base, token):
        self.base = base
        self.token = token

    def sample(self, rng, observation, **kwargs):
        actions, prefix, mask = self.base.sample_actions_with_prefix(rng, observation, **kwargs)
        return actions, self.token.encode(prefix, mask)


class RLTPolicy:
    def __init__(self, policy, token, *, base_id, token_id):
        if policy._is_pytorch_model:
            raise ValueError("RLT provider requires the frozen JAX Pi0 backend")
        if not base_id or not token_id:
            raise ValueError("immutable base and token identities required")
        self.policy, self.base_id, self.token_id = policy, base_id, token_id
        self._model = TokenPolicyModel(policy._model, token)
        self._sample = nnx_utils.module_jit(self._model.sample)
        self._lock = threading.Lock()

    def infer(self, obs, *, noise=None):
        # Transforms may mutate arrays. Never share raw observation storage.
        with self._lock:
            inputs = self.policy._input_transform(copy.deepcopy(obs))
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
            self.policy._rng, rng = jax.random.split(self.policy._rng)
            kwargs = dict(self.policy._sample_kwargs)
            if noise is not None:
                noise = jnp.asarray(noise)
                kwargs["noise"] = noise[None] if noise.ndim == 2 else noise
            actions, token = self._sample(rng, model_api.Observation.from_dict(inputs), **kwargs)
            outputs = self.policy._output_transform(
                {
                    "state": np.array(inputs["state"][0]),
                    "actions": np.array(actions[0]),
                }
            )
            outputs.update(rl_token=np.array(token[0]), base_id=self.base_id, token_id=self.token_id)
            return outputs

    @property
    def metadata(self):
        return {**self.policy.metadata, "base_id": self.base_id, "token_id": self.token_id}
