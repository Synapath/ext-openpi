"""Bind a versioned local RoboDojo view to the existing trainer, with bounded monitoring.

The JSON binding owns paths/run identity, not model math. The recipe remains in
training.config; smoke/resume retain the full draw stream and learning schedule.
"""

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def worker(binding):
    from openpi.training import config, weight_loaders
    import train

    split = json.loads(Path(binding["split"]).read_text())
    c = config.get_config(binding["recipe"])
    base = dataclasses.replace(
        c.data.base_config,
        dataset_root=binding["view"],
        split_manifest_path=binding["split"],
        draw_manifest_path=binding["draws"],
        episode_indices=tuple(r["episode_id"] for r in split["episodes"] if r["split"] == "train"),
    )
    c = dataclasses.replace(
        c,
        exp_name=binding["name"],
        checkpoint_base_dir=binding["checkpoint_root"],
        num_workers=binding.get("workers", 8),
        fsdp_devices=binding.get("fsdp_devices", c.fsdp_devices),
        log_interval=binding.get("log_interval", 100),
        resume=binding.get("resume", False),
        max_updates=binding.get("max_updates"),
        weight_loader=weight_loaders.CheckpointWeightLoader(binding["base_params"]),
        data=dataclasses.replace(
            c.data,
            base_config=base,
            assets=config.AssetsConfig(assets_dir=binding["assets_dir"], asset_id=c.data.assets.asset_id),
        ),
    )
    out = Path(binding["execution"])
    save(out / "effective-config.json", dataclasses.asdict(c))
    logging.basicConfig()
    train.main(c)
    save(
        out / "completion.json",
        dict(
            status="TRAINER_RETURNED",
            completed_updates=c.max_updates or c.num_train_steps,
            checkpoint_dir=str(c.checkpoint_dir),
            epoch=time.time(),
        ),
    )


def supervise(path, binding):
    out = Path(binding["execution"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "started.json").exists():
        raise FileExistsError("new execution attempt required")
    env = os.environ.copy()
    env.update(binding["environment"])
    env.update(
        OPENPI_METRICS_JSONL=str(out / "metrics.jsonl"),
        OPENPI_DIAGNOSTICS_DIR=binding["diagnostics_dir"],
        OPENPI_CHECKPOINT_STEPS=",".join(map(str, binding["checkpoint_steps"])),
        WANDB_ENTITY="synapath_ai",
        WANDB_PROJECT="rlt_pi05_rbdj",
        WANDB_RUN_GROUP=binding["group"],
        WANDB_JOB_TYPE=binding.get("job_type", "train"),
        WANDB_MODE="online",
        WANDB_DIR=str(out),
        WANDB_CONSOLE="off",
        OPENPI_COMPILATION_RECEIPT_DIR=str(out / "compilation"),
    )
    # This run is fresh unless the checkpoint's stored W&B identity explicitly resumes it.
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_NAME"):
        env.pop(key, None)
    cache = Path(env["JAX_COMPILATION_CACHE_DIR"])
    cache.mkdir(parents=True, mode=0o700, exist_ok=True)
    if cache.stat().st_uid != os.getuid() or cache.stat().st_mode & 0o077:
        raise ValueError("JAX compilation cache must be owned and private before worker initialization")
    gpus = binding["environment"]["CUDA_VISIBLE_DEVICES"]
    rows = subprocess.check_output(
        ["nvidia-smi", "-i", gpus, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    if any(int(row.strip()) > 512 for row in rows):
        raise RuntimeError("training GPUs occupied")
    if shutil.disk_usage("/data").free < binding["min_disk_free_bytes"]:
        raise RuntimeError("local disk reserve")
    log = (out / "trainer.log").open("x")
    start = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), str(path), "--worker"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    save(out / "started.json", dict(epoch=start, pid=proc.pid, gpus=gpus, binding=str(path)))
    stopping = None

    def interrupted(signum, _frame):
        nonlocal stopping
        stopping = f"signal {signum}"

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while proc.poll() is None:
            elapsed = time.time() - start
            rows = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    gpus,
                    "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
            health = [list(map(int, row.split(","))) for row in rows]
            if elapsed > binding["max_seconds"]:
                stopping = "wall budget"
            if shutil.disk_usage("/data").free < binding["min_disk_free_bytes"]:
                stopping = "local disk reserve"
            if shutil.disk_usage(binding["checkpoint_root"]).free < binding["min_disk_free_bytes"]:
                stopping = "checkpoint disk reserve"
            if any(row[-1] >= 85 for row in health):
                stopping = "GPU temperature >=85C"
            progress = {}
            metrics = out / "metrics.jsonl"
            if metrics.exists():
                for line in metrics.read_text().splitlines()[-12:]:
                    try:
                        progress.update(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            update = progress.get("train/updates", 0)
            seconds_per_update = progress.get("step_time_seconds")
            save(
                out / "monitor.json",
                dict(
                    epoch=time.time(),
                    elapsed_s=elapsed,
                    gpu_hours=elapsed * len(health) / 3600,
                    completed_updates=update,
                    health=health,
                    last_metrics=progress,
                    eta_update_seconds=(binding.get("max_updates", 20000) - update) * seconds_per_update
                    if seconds_per_update
                    else None,
                    stop_reason=stopping,
                ),
            )
            if stopping:
                break
            time.sleep(10)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        log.close()
        save(
            out / "exit.json",
            dict(epoch=time.time(), elapsed_s=time.time() - start, returncode=proc.returncode, stop_reason=stopping),
        )
    if proc.returncode or stopping:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("binding", type=Path)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    b = json.loads(args.binding.read_text())
    worker(b) if args.worker else supervise(args.binding.resolve(), b)
