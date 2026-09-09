import dataclasses
import json
import os
import pathlib

import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str, monkeypatch):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("OPENPI_METRICS_JSONL", str(metrics_path))
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        num_workers=0,  # Keep this CPU trainer regression independent of multiprocessing.
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
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [row["train/updates"] for row in records] == [1, 2, 3, 4]
    assert all(np.isfinite(row[key]) for row in records for key in ("loss", "grad_norm", "param_norm"))


def test_rlt_rejects_unverified_backend_before_creating_run(tmp_path, monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    config = dataclasses.replace(
        _config.get_config("pi05_rlt_charger_r1_0a"),
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="must-not-start",
    )
    with pytest.raises(ValueError, match="requires XLA_FLAGS"):
        train.main(config)
    assert not config.checkpoint_dir.exists()


def test_numeric_wandb_excludes_paths_images_code_and_dependency_inventory(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config = dataclasses.replace(
        _config.get_config("pi05_rlt_charger_r1_0a"),
        checkpoint_base_dir=str(tmp_path / "private-checkpoints"),
        exp_name="numeric-only",
    )
    config.checkpoint_dir.mkdir(parents=True)
    monkeypatch.setenv("OPENPI_TRACKING_METADATA_JSON", json.dumps({
        "host_path": "/private/host", "source_manifest_sha256": "a" * 64,
    }))
    captured = []
    fake_run = SimpleNamespace(id="test-id", log_code=lambda *_: pytest.fail("code uploaded"))
    monkeypatch.setattr(train.wandb, "init", lambda **kw: captured.append(kw))
    monkeypatch.setattr(train.wandb, "run", fake_run)
    monkeypatch.setattr(train.wandb, "define_metric", lambda *_a, **_k: None)
    monkeypatch.setattr(train.wandb, "log", lambda *_a, **_k: pytest.fail("image history uploaded"))
    monkeypatch.setattr(train.wandb, "Image", lambda *_a, **_k: pytest.fail("image created"))
    train.init_wandb(config, resuming=False, log_code=True)
    train.log_first_batch_images(config, None, 0)
    first = captured[0]
    assert first["project"] == "rlt_pi05_rbdj"
    assert first["resume"] == "never"
    assert "/private" not in json.dumps(first["config"])
    assert first["config"]["source_manifest_sha256"] == "a" * 64
    assert first["settings"].x_save_requirements is False
    assert first["settings"].x_disable_machine_info is False
    assert first["settings"].x_disable_meta is True
    assert first["settings"].x_disable_stats is False
    assert first["settings"].save_code is False
    assert first["settings"].console == "off"
    train.init_wandb(config, resuming=True, log_code=True)
    assert captured[1]["id"] == "test-id"
    assert captured[1]["resume"] == "must"
    assert captured[1]["settings"].x_save_requirements is False
