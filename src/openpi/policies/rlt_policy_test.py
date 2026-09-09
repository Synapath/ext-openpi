from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.rl_token import ARToken
from openpi.models.rl_token import ARTokenConfig
from openpi.policies.rlt_policy import RLTPolicy


class Base(nnx.Module):
    def sample_actions_with_prefix(self, rng, obs, *, noise=None):
        prefix = jnp.ones((1, 3, 8)) * obs.state[0, 0]
        return jnp.ones((1, 32, 14)), prefix, jnp.ones((1, 3), dtype=bool)


def test_transform_copy_and_feature_not_action_denormalized():
    def transform(obs):
        obs["state"][0] += 1
        return {"state": obs["state"], "image": {}, "image_mask": {}}

    def output(obs):
        assert "rl_token" not in obs
        obs["actions"] *= 7
        return obs

    policy = SimpleNamespace(
        _is_pytorch_model=False,
        _model=Base(),
        _input_transform=transform,
        _output_transform=output,
        _rng=jax.random.key(1),
        _sample_kwargs={},
        metadata={},
    )
    token = ARToken(ARTokenConfig(dim=8, num_heads=2, mlp_dim=16, num_layers=1), nnx.Rngs(1))
    provider = RLTPolicy(policy, token, base_id="b1", token_id="t1")
    obs = {"state": np.zeros(14, dtype=np.float32)}
    result = provider.infer(obs)
    np.testing.assert_array_equal(obs["state"], 0)
    np.testing.assert_array_equal(result["actions"], 7)
    np.testing.assert_allclose(
        result["rl_token"], np.asarray(token.encode(jnp.ones((1, 3, 8)), jnp.ones((1, 3), dtype=bool)))[0], atol=1e-5
    )
