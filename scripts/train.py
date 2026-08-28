import concurrent.futures
import dataclasses
import functools
import json
import logging
import os
from pathlib import Path
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.compilation as _compilation
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def _append_jsonl(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _nominal_dataset_size() -> int | None:
    value = os.environ.get("OPENPI_NOMINAL_DATASET_SIZE")
    if value is None:
        return None
    size = int(value)
    if size <= 0:
        raise ValueError("OPENPI_NOMINAL_DATASET_SIZE must be positive")
    return size


def _requested_checkpoint_steps(num_train_steps: int) -> frozenset[int] | None:
    value = os.environ.get("OPENPI_CHECKPOINT_STEPS")
    if value is None:
        return None
    try:
        steps = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("OPENPI_CHECKPOINT_STEPS must be comma-separated integers") from exc
    if not steps or steps != sorted(set(steps)):
        raise ValueError("OPENPI_CHECKPOINT_STEPS must be non-empty, sorted, and unique")
    if steps[0] < 0 or steps[-1] != num_train_steps - 1:
        raise ValueError("OPENPI_CHECKPOINT_STEPS must be non-negative and end at the final step")
    return frozenset(steps)


def _tracking_metadata() -> dict[str, Any] | None:
    value = os.environ.get("OPENPI_TRACKING_METADATA_JSON")
    if value is None:
        return None
    metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise ValueError("OPENPI_TRACKING_METADATA_JSON must encode an object")
    return metadata


def wandb_config_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if dataset_size := _nominal_dataset_size():
        overrides["nominal_dataset_size"] = dataset_size
    if metadata := _tracking_metadata():
        overrides["run_metadata"] = metadata
    return overrides


def progress_metrics(
    *,
    step: int,
    start_step: int,
    num_train_steps: int,
    batch_size: int,
    elapsed_seconds: float,
    interval_updates: int,
    learning_rate: float,
) -> dict[str, float | int]:
    if interval_updates <= 0:
        raise ValueError("interval_updates must be positive")
    elapsed_seconds = max(elapsed_seconds, 1e-12)
    completed_updates = step + 1
    step_time = elapsed_seconds / interval_updates
    metrics: dict[str, float | int] = {
        "learning_rate": learning_rate,
        "step_time_seconds": step_time,
        "steps_per_second": interval_updates / elapsed_seconds,
        "samples_per_second": interval_updates * batch_size / elapsed_seconds,
        "sample_draws": completed_updates * batch_size,
        "progress_percent": 100.0 * completed_updates / num_train_steps,
        "eta_hours": max(num_train_steps - completed_updates, 0) * step_time / 3600.0,
        "is_compile_interval": int(step == start_step),
    }
    if dataset_size := _nominal_dataset_size():
        metrics["nominal_dataset_passes"] = completed_updates * batch_size / dataset_size
    return metrics


def should_log_step(*, step: int, num_train_steps: int, log_interval: int) -> bool:
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    return step % log_interval == 0 or step == num_train_steps - 1


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if overrides := wandb_config_overrides():
        wandb.config.update(overrides, allow_val_change=True)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=jnp.asarray(0, dtype=jnp.int32),
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    }
    return new_state, info


def parameter_norm(state: training_utils.TrainState) -> at.Array:
    """Compute the full kernel parameter norm for periodic monitoring."""
    model = nnx.merge(state.model_def, state.params)
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return optax.global_norm(kernel_params)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax_cache_dir = epath.Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")).expanduser()
    jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    requested_checkpoint_steps = _requested_checkpoint_steps(config.num_train_steps)
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        checkpoint_steps=requested_checkpoint_steps,
    )
    metrics_path_value = os.environ.get("OPENPI_METRICS_JSONL")
    metrics_path = Path(metrics_path_value) if metrics_path_value else None
    if metrics_path is not None and metrics_path.exists() and not resuming:
        raise FileExistsError(f"metrics identity already exists: {metrics_path}")
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    exact_draws = hasattr(data_loader, "commit_batch")
    if exact_draws:
        # Restore the committed cursor BEFORE constructing/prefetching an iterator.
        train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
        if resuming:
            train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        data_iter = iter(data_loader)
        batch = next(data_iter)
    else:
        data_iter = iter(data_loader)
        overlap_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="first-batch") as executor:
            batch_future = executor.submit(next, data_iter)
            state_started = time.perf_counter()
            train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
            jax.block_until_ready(train_state)
            state_ready_seconds = time.perf_counter() - state_started
            batch = batch_future.result()
            jax.block_until_ready(batch)
        logging.info(
            "Initialized first batch and train state with overlap: total_seconds=%.3f state_seconds=%.3f",
            time.perf_counter() - overlap_started,
            state_ready_seconds,
        )
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming and not exact_draws:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    if exact_draws:
        ptrain_step = _compilation.compile_step(
            functools.partial(train_step, config),
            train_rng,
            train_state,
            batch,
            mesh=mesh,
            state_sharding=train_state_sharding,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    pparameter_norm = jax.jit(
        parameter_norm,
        in_shardings=(train_state_sharding,),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    lr_schedule = config.lr_schedule.create()
    interval_started = time.perf_counter()
    last_requested_checkpoint: int | None = None
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        if exact_draws:
            checked_info = jax.device_get(info)
            if not all(np.isfinite(value).all() for value in jax.tree.leaves(checked_info)):
                raise FloatingPointError("non-finite G2 optimizer update; exposure not acknowledged")
            data_loader.commit_batch(step + 1)
        infos.append(info)
        if should_log_step(step=step, num_train_steps=config.num_train_steps, log_interval=config.log_interval):
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            reduced_info["param_norm"] = jax.device_get(pparameter_norm(train_state))
            now = time.perf_counter()
            reduced_info.update(
                progress_metrics(
                    step=step,
                    start_step=start_step,
                    num_train_steps=config.num_train_steps,
                    batch_size=config.batch_size,
                    elapsed_seconds=now - interval_started,
                    interval_updates=len(infos),
                    learning_rate=float(jax.device_get(lr_schedule(step))),
                )
            )
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            serializable = {key: float(value) for key, value in reduced_info.items()}
            serializable["step"] = step
            _append_jsonl(metrics_path, serializable)
            infos = []
            interval_started = now
        if not exact_draws:
            batch = next(data_iter)

        default_save = (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1
        should_save = step in requested_checkpoint_steps if requested_checkpoint_steps is not None else default_save
        if should_save:
            save_started = time.perf_counter()
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            last_requested_checkpoint = step
            checkpoint_info = {
                "checkpoint_save_requested_step": step,
                "checkpoint_save_enqueue_seconds": time.perf_counter() - save_started,
            }
            wandb.log(checkpoint_info, step=step)
            logging.info(
                "Checkpoint save requested: step=%d enqueue_seconds=%.3f",
                step,
                checkpoint_info["checkpoint_save_enqueue_seconds"],
            )
        if exact_draws and step + 1 < config.num_train_steps:
            batch = next(data_iter)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_wait_started = time.perf_counter()
    checkpoint_manager.wait_until_finished()
    if last_requested_checkpoint is not None:
        wandb.log(
            {
                "latest_complete_checkpoint_step": last_requested_checkpoint,
                "checkpoint_final_wait_seconds": time.perf_counter() - checkpoint_wait_started,
            },
            step=last_requested_checkpoint,
        )
    wandb.finish()
    if exact_draws:
        data_loader.close()


if __name__ == "__main__":
    main(_config.cli())
