import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.rl_token import ARToken
from openpi.models.rl_token import ARTokenConfig
from openpi.models.rl_token import validate_features


def fixture():
    token = ARToken(ARTokenConfig(dim=8, num_heads=2, mlp_dim=16, num_layers=2), nnx.Rngs(12))
    x = jax.random.normal(jax.random.key(0), (3, 5, 8))
    mask = jnp.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 1, 0]], dtype=bool)
    return token, x, mask


def test_padding_and_gradient_isolation():
    token, x, mask = fixture()
    dirty = jnp.where(mask[..., None], x, jnp.nan)
    np.testing.assert_allclose(token.encode(x, mask), token.encode(dirty, mask), atol=1e-6)
    assert np.isfinite(token.loss(dirty, mask))
    grad = jax.grad(lambda a: token.loss(a, mask))(x)
    np.testing.assert_array_equal(grad, 0)


def test_decoder_causal_and_token_dependence():
    token, x, mask = fixture()
    z = token.encode(x, mask)
    y = token.decoder(z, x, mask)
    changed = x.at[:, 2:].add(100)
    np.testing.assert_allclose(y[:, :3], token.decoder(z, changed, mask)[:, :3], atol=1e-6)
    assert not np.allclose(y[:, 0], token.decoder(jnp.zeros_like(z), x, mask)[:, 0])


def test_mask_validation_and_config():
    _, x, mask = fixture()
    validate_features(x, mask)
    with pytest.raises(ValueError):
        validate_features(x, mask.at[0].set(False))
    with pytest.raises(ValueError):
        validate_features(x, mask.astype(int))
    with pytest.raises(ValueError):
        ARTokenConfig(dim=9, num_heads=2)


def test_finite_parameter_update_and_restore():
    import optax

    token, x, mask = fixture()
    opt = nnx.Optimizer(token, optax.adam(1e-3))
    before = float(token.loss(x, mask))
    loss, grads = nnx.value_and_grad(lambda m: m.loss(x, mask))(token)
    opt.update(grads)
    assert np.isfinite(loss) and float(token.loss(x, mask)) < before
    clone, _, _ = fixture()
    nnx.update(clone, nnx.state(token))
    np.testing.assert_array_equal(token.encode(x, mask), clone.encode(x, mask))


def test_token_optimizer_checkpoint_continuity(tmp_path):
    from openpi.training.rl_token import TokenTrainer

    token, x, mask = fixture()
    a = TokenTrainer(token.config, base_id="b1")
    a.update(x, mask)
    sha = a.save(tmp_path / "token.msgpack")
    b = TokenTrainer.load(tmp_path / "token.msgpack", expected_sha256=sha, base_id="b1")
    assert a.update(x, mask) == b.update(x, mask)
    for x, y in zip(
        jax.tree.leaves(nnx.state((a.token, a.optimizer))),
        jax.tree.leaves(nnx.state((b.token, b.optimizer))),
        strict=True,
    ):
        np.testing.assert_array_equal(x, y)


def test_pinned_reference_encoder_decoder_and_gradients():
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[4] / "ref/pravsels-openpi/src/openpi/models/pi0_rl.py"
    tree = ast.parse(source.read_text())
    names = {"RLTokenTransformerBlock", "RLTokenEncoder", "RLTokenDecoder"}
    env = {"nnx": nnx, "jax": jax, "jnp": jnp}
    exec(
        compile(
            ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names], type_ignores=[]),
            "pinned-reference",
            "exec",
        ),
        env,
    )
    token, x, mask = fixture()
    mask = jnp.ones_like(mask)
    reference = env["RLTokenEncoder"](8, 2, 16, 2, nnx.Rngs(12))
    nnx.update(reference, nnx.state(token.encoder))
    np.testing.assert_allclose(token.encode(x, mask), reference(x, mask), rtol=1e-6, atol=1e-6)
    ref_decoder = env["RLTokenDecoder"](8, 2, 16, 2, nnx.Rngs(1))
    nnx.update(ref_decoder, nnx.state(token.decoder))
    z = token.encode(x, mask)
    np.testing.assert_allclose(token.decoder(z, x, mask), ref_decoder(z, x, mask), rtol=1e-6, atol=1e-6)
    g1 = nnx.grad(lambda m: m(z, x, mask).sum())(token.decoder)
    g2 = nnx.grad(lambda m: m(z, x, mask).sum())(ref_decoder)
    for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g2), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
