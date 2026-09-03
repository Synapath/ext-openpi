import jax
import orbax.checkpoint as ocp

from openpi.training import checkpoints


def test_requested_checkpoint_steps_override_periodic_retention(tmp_path):
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoints",
        keep_period=5,
        overwrite=False,
        resume=False,
        checkpoint_steps={4, 9},
    )
    try:
        assert not resuming
        assert manager._options.max_to_keep == 2  # noqa: SLF001
        assert manager._options.keep_period is None  # noqa: SLF001
    finally:
        manager.close()


def test_checkpoint_manager_disables_replica_parallel_array_transfers(tmp_path):
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoints",
        keep_period=None,
        overwrite=False,
        resume=False,
    )
    try:
        handler = ocp.type_handlers.get_type_handler(jax.Array)
        assert not handler._use_replica_parallel  # noqa: SLF001
    finally:
        manager.close()
