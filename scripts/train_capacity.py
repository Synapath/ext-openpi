"""Run the real JAX train step without creating a checkpoint.

This runner is intentionally small: it reuses ``train.init_train_state`` and
``train.train_step`` so capacity measurements exercise the same model, optimizer,
data path, sharding, and update math as production training. Only checkpoint
manager creation and image logging are omitted.
"""

import dataclasses
import functools
import json
import os
from pathlib import Path
import platform
import time

import etils.epath as epath
import flax.training.common_utils as common_utils
import jax
import jax.numpy as jnp
import train as _train
import wandb

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


def _append_jsonl(path: Path | None, value: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def main(config: _config.TrainConfig) -> None:
    _train.init_logging()
    if config.resume or config.overwrite:
        raise ValueError("capacity runner does not support resume or overwrite")
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by device count {jax.device_count()}."
        )

    metrics_path_value = os.environ.get("OPENPI_METRICS_JSONL")
    metrics_path = Path(metrics_path_value) if metrics_path_value else None
    if metrics_path is not None and metrics_path.exists():
        raise FileExistsError(f"metrics identity already exists: {metrics_path}")

    _train.logging.info("Capacity runner on %s; checkpoints disabled", platform.node())
    jax_cache_dir = epath.Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")).expanduser()
    jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    if config.wandb_enabled:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        if overrides := _train.wandb_config_overrides():
            wandb.config.update(overrides, allow_val_change=True)
        if run_id_path := os.environ.get("OPENPI_WANDB_ID_PATH"):
            Path(run_id_path).write_text(f"{wandb.run.id}\n", encoding="utf-8")
    else:
        wandb.init(mode="disabled")

    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    _train.logging.info("Initialized data loader:\n%s", training_utils.array_tree_to_info(batch))

    train_state, train_state_sharding = _train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    _train.logging.info("Initialized train state:\n%s", training_utils.array_tree_to_info(train_state.params))
    ptrain_step = jax.jit(
        functools.partial(_train.train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    pparameter_norm = jax.jit(
        _train.parameter_norm,
        in_shardings=train_state_sharding,
        out_shardings=replicated_sharding,
    )
    lr_schedule = config.lr_schedule.create()

    for step in range(config.num_train_steps):
        update_started = time.perf_counter()
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        reduced_info = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest([info])))
        reduced_info["param_norm"] = jax.device_get(pparameter_norm(train_state))
        reduced_info.update(
            _train.progress_metrics(
                step=step,
                start_step=0,
                num_train_steps=config.num_train_steps,
                batch_size=config.batch_size,
                elapsed_seconds=time.perf_counter() - update_started,
                interval_updates=1,
                learning_rate=float(jax.device_get(lr_schedule(step))),
            )
        )
        serializable = {key: float(value) for key, value in reduced_info.items()}
        serializable["step"] = step
        _append_jsonl(metrics_path, serializable)
        _train.logging.info("CAPACITY_METRIC %s", json.dumps(serializable, sort_keys=True))
        wandb.log(reduced_info, step=step)
        batch = next(data_iter)

    wandb.finish()
    _train.logging.info("Capacity run completed without checkpoint creation")


if __name__ == "__main__":
    main(_config.cli())
