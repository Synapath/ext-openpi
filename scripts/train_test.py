import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

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


def test_tracking_metadata(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENPI_TRACKING_METADATA_JSON", '{"recipe":"strict"}')
    monkeypatch.setenv("OPENPI_NOMINAL_DATASET_SIZE", "7188")
    assert train.wandb_config_overrides() == {
        "nominal_dataset_size": 7188,
        "run_metadata": {"recipe": "strict"},
    }


def test_append_jsonl(tmp_path: pathlib.Path):
    output = tmp_path / "metrics.jsonl"
    train._append_jsonl(output, {"step": 0, "loss": 0.5})  # noqa: SLF001
    train._append_jsonl(output, {"step": 1, "loss": 0.25})  # noqa: SLF001
    assert output.read_text().splitlines() == [
        '{"loss": 0.5, "step": 0}',
        '{"loss": 0.25, "step": 1}',
    ]


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
