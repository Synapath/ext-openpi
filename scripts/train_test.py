import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import compilation
from openpi.training import config as _config
from openpi.training import sharding

from . import train


def test_progress_metrics(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENPI_NOMINAL_DATASET_SIZE", "100")
    metrics = train.progress_metrics(
        step=4,
        start_step=0,
        num_train_steps=20,
        batch_size=8,
        elapsed_seconds=2.0,
        interval_updates=4,
        learning_rate=2.5e-5,
    )
    assert metrics == {
        "learning_rate": 2.5e-5,
        "step_time_seconds": 0.5,
        "steps_per_second": 2.0,
        "samples_per_second": 16.0,
        "sample_draws": 40,
        "progress_percent": 25.0,
        "eta_hours": 15 * 0.5 / 3600,
        "is_compile_interval": 0,
        "nominal_dataset_passes": 0.4,
    }


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 1), (1, 0), (99, 0), (100, 1), (30630, 0), (30631, 1)],
)
def test_should_log_step(step: int, expected: int):
    assert train.should_log_step(step=step, num_train_steps=30_632, log_interval=100) is bool(expected)


def test_should_log_step_rejects_nonpositive_interval():
    with pytest.raises(ValueError, match="log_interval"):
        train.should_log_step(step=0, num_train_steps=2, log_interval=0)


def test_tracking_metadata(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENPI_TRACKING_METADATA_JSON", '{"recipe":"strict"}')
    monkeypatch.setenv("OPENPI_NOMINAL_DATASET_SIZE", "7188")
    assert train.wandb_config_overrides() == {
        "nominal_dataset_size": 7188,
        "run_metadata": {"recipe": "strict"},
    }


def test_requested_checkpoint_steps(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENPI_CHECKPOINT_STEPS", raising=False)
    assert train._requested_checkpoint_steps(10) is None  # noqa: SLF001
    monkeypatch.setenv("OPENPI_CHECKPOINT_STEPS", "4,9")
    assert train._requested_checkpoint_steps(10) == frozenset({4, 9})  # noqa: SLF001


@pytest.mark.parametrize("value", ["", "4,4,9", "9,4", "4,8", "x,9", "-1,9"])
def test_requested_checkpoint_steps_rejects_invalid(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("OPENPI_CHECKPOINT_STEPS", value)
    with pytest.raises(ValueError, match="OPENPI_CHECKPOINT_STEPS"):
        train._requested_checkpoint_steps(10)  # noqa: SLF001


def test_append_jsonl(tmp_path: pathlib.Path):
    output = tmp_path / "metrics.jsonl"
    train._append_jsonl(output, {"step": 0, "loss": 0.5})  # noqa: SLF001
    train._append_jsonl(output, {"step": 1, "loss": 0.25})  # noqa: SLF001
    assert output.read_text().splitlines() == [
        '{"loss": 0.5, "step": 0}',
        '{"loss": 0.25, "step": 1}',
    ]


def test_capacity_parameter_norm_has_single_positional_sharding():
    source = pathlib.Path(__file__).with_name("train_capacity.py").read_text()
    assert "in_shardings=(train_state_sharding,)" in source


def test_formal_parameter_norm_has_single_positional_sharding():
    source = pathlib.Path(__file__).with_name("train.py").read_text()
    assert "in_shardings=(train_state_sharding,)" in source


def test_fresh_step_shape_is_strong_int32():
    import jax
    import numpy as np

    config = _config.get_config("pi05_g2_rbdj_three_task_s0_strict")
    shape, _ = train.init_train_state(config, jax.random.key(0), sharding.make_mesh(1), resume=True)
    assert shape.step.dtype == np.dtype("int32")
    assert not shape.step.weak_type
    compilation.require_canonical_step(shape)


def test_compilation_identity_roundtrip_and_drift(tmp_path, monkeypatch):
    from typing import NamedTuple

    import jax
    import jax.numpy as jnp

    class State(NamedTuple):
        step: object
        value: object

    def call(rng, state, batch):
        del rng
        return State(state.step + 1, state.value + batch), {"sum": state.value.sum()}

    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENPI_COMPILATION_RECEIPT_DIR", str(tmp_path / "identity"))
    mesh = sharding.make_mesh(1)
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = State(jnp.asarray(0, dtype=jnp.int32), jnp.ones(2))
    shards = State(rep, rep)
    for _ in range(2):
        compiled = compilation.compile_step(
            call,
            jax.random.key(0),
            state,
            jnp.ones(2),
            mesh=mesh,
            state_sharding=shards,
            data_sharding=rep,
            replicated_sharding=rep,
        )
        assert callable(compiled)
    assert list((tmp_path / "cache").glob("*-cache"))
    (tmp_path / "identity/identity.json").write_text("drift")
    with pytest.raises(ValueError, match="identity drift"):
        compilation.compile_step(
            call,
            jax.random.key(0),
            state,
            jnp.ones(2),
            mesh=mesh,
            state_sharding=shards,
            data_sharding=rep,
            replicated_sharding=rep,
        )
    with pytest.raises(ValueError, match="strong scalar int32"):
        compilation.require_canonical_step(State(jnp.asarray(0), jnp.ones(2)))


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
