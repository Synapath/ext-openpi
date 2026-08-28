"""Native LeRobot/transform adapter for the shared G2 exposure artifact."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import jax
import numpy as np
import torch

from manip_datasets.robodojo import DrawManifest, RECORD, sha256
from openpi import transforms
from openpi.models import model
from openpi.training import data_loader


class ManifestDataset:
    def __init__(self, native, transform, manifest_path, split_path, episodes):
        self.native, self.transform = native, transform
        self.manifest_path, self.split_path = manifest_path, split_path
        self.draws = None
        epids = np.asarray(native.hf_dataset["episode_index"], dtype=np.int64).reshape(-1)
        frames = np.asarray(native.hf_dataset["frame_index"], dtype=np.int64).reshape(-1)
        self.offsets = {}
        for r in episodes:
            positions = np.flatnonzero(epids == r["episode_id"])
            if (
                len(positions) != r["length"]
                or not np.array_equal(frames[positions], np.arange(r["length"]))
                or not np.array_equal(positions, np.arange(positions[0], positions[0] + len(positions)))
            ):
                raise ValueError("native dataset episode mapping mismatch")
            self.offsets[r["episode_id"]] = int(positions[0])
        if set(epids.tolist()) != set(self.offsets):
            raise ValueError("native dataset includes an unexpected episode")
        with DrawManifest(manifest_path, split_path) as draws:
            self.length = len(draws)

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["draws"] = None
        return state

    def __getitem__(self, draw_index):
        if self.draws is None:
            self.draws = DrawManifest(self.manifest_path, self.split_path, verify=False)
        identity = self.draws[int(draw_index)]
        _, _, task, episode, frame = identity
        item = self.native[self.offsets[episode] + frame]
        actual = (int(item["task_index"]), int(item["episode_index"]), int(item["frame_index"]))
        if actual != (task, episode, frame):
            raise ValueError(f"native loader identity drift: {actual} != {identity}")
        if np.asarray(item["action_is_pad"]).any():
            raise ValueError("unexpected temporal padding at common anchor")
        item = self.transform(item)
        item["_draw_identity"] = np.array(identity, dtype=np.uint32)
        return item


def worker_init(_):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    torch.set_num_threads(1)


class ManifestDataLoader:
    def __init__(self, config, data_config, *, sharding=None, num_batches=None):
        if not all((data_config.dataset_root, data_config.draw_manifest_path, data_config.split_manifest_path)):
            raise ValueError("G2 requires explicit local root, split and draw manifest; no random fallback")
        if jax.process_count() != 1 or config.batch_size != 64 or config.model.action_horizon != 50:
            raise ValueError("G2 requires single-process JAX B64/H50")
        self.draws = DrawManifest(data_config.draw_manifest_path, data_config.split_manifest_path)
        if self.draws.metadata["batch_size"] != 64 or self.draws.metadata["updates"] != 10000:
            raise ValueError("G2 shared exposure must be B64/U10000")
        self._data_config = data_config
        episodes = [r for r in self.draws.split["episodes"] if r["split"] == "train"]
        ids = [r["episode_id"] for r in episodes]
        if list(data_config.episode_indices) != ids:
            raise ValueError("G2 config must explicitly list the 270 sorted train episode IDs")
        root = Path(data_config.dataset_root)
        view_path = root / "view-manifest.json"
        view = json.loads(view_path.read_text())
        if view["split_sha256"] != self.draws.metadata["split_sha256"] or view["train_episodes"] != 270:
            raise ValueError("derived view does not match this split")
        native = data_loader.lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            root=root,
            episodes=ids,
            download_videos=False,
            delta_timestamps={key: [t / 25 for t in range(50)] for key in data_config.action_sequence_keys},
            video_backend=data_config.video_backend,
        )
        if data_config.norm_stats is None:
            raise ValueError("G2 train-only normalization is required")
        transform = transforms.compose(
            [
                transforms.PromptFromLeRobotTask(native.meta.tasks),
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self.dataset = ManifestDataset(
            native, transform, data_config.draw_manifest_path, data_config.split_manifest_path, episodes
        )
        self.sharding = sharding
        self.workers = config.num_workers
        self.stop_update = min(
            config.num_train_steps, num_batches if num_batches is not None else config.num_train_steps
        )
        if not 0 < self.stop_update <= 10000:
            raise ValueError("G2 update limit outside draw manifest")
        self.committed_updates = 0
        self.pending = None
        self.prefix = hashlib.sha256()
        self.task_counts, self.episode_counts = Counter(), Counter()
        self.identity = dict(
            draw_manifest_sha256=sha256(data_config.draw_manifest_path),
            split_sha256=sha256(data_config.split_manifest_path),
            view_manifest_sha256=sha256(view_path),
        )
        self.readback_path = os.environ.get("OPENPI_DRAW_READBACK")
        self._torch_loader = None

    def data_config(self):
        return self._data_config

    def __iter__(self):
        if self.pending is not None:
            raise ValueError("previous batch has not been acknowledged")
        start = self.committed_updates
        kwargs = dict(
            dataset=self.dataset,
            batch_size=64,
            sampler=range(start * 64, self.stop_update * 64),
            num_workers=self.workers,
            collate_fn=data_loader._collate_fn,
            drop_last=False,
            worker_init_fn=worker_init,
            generator=torch.Generator().manual_seed(0),
        )
        if self.workers:
            kwargs.update(
                multiprocessing_context=multiprocessing.get_context("spawn"), persistent_workers=True, prefetch_factor=2
            )
        self._torch_loader = torch.utils.data.DataLoader(**kwargs)
        for update, batch in enumerate(self._torch_loader, start=start):
            if self.pending is not None or update != self.committed_updates:
                raise ValueError("prefetch cannot advance committed exposure")
            actual = batch.pop("_draw_identity")
            if not np.array_equal(actual, np.asarray(self.draws.batch(update), dtype=np.uint32)):
                raise ValueError("collated actual-consumption readback mismatch")
            self.pending = actual
            if self.sharding is not None:
                batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(self.sharding, x), batch)
            yield model.Observation.from_dict(batch), batch["actions"]

    def commit_batch(self, completed_updates):
        if self.pending is None or completed_updates != self.committed_updates + 1:
            raise ValueError("invalid optimizer/exposure acknowledgement")
        actual = self.pending
        for row in actual:
            self.prefix.update(RECORD.pack(*(int(x) for x in row)))
            self.task_counts[str(int(row[2]))] += 1
            self.episode_counts[str(int(row[3]))] += 1
        if self.readback_path:
            record = dict(
                completed_updates=completed_updates, actual_draws=actual.tolist(), prefix_sha256=self.prefix.hexdigest()
            )
            with Path(self.readback_path).open("a") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.committed_updates = completed_updates
        self.pending = None

    def cursor_receipt(self, completed_updates):
        if self.pending is not None or completed_updates != self.committed_updates:
            raise ValueError("checkpoint cursor does not match completed updates")
        return dict(
            schema="rbdj-cursor-v1",
            **self.identity,
            completed_updates=completed_updates,
            prefix_sha256=self.prefix.hexdigest(),
            task_counts=dict(self.task_counts),
            episode_counts=dict(self.episode_counts),
        )

    def restore_cursor(self, receipt, completed_updates):
        if (
            receipt.get("schema") != "rbdj-cursor-v1"
            or completed_updates != receipt["completed_updates"]
            or any(receipt[k] != v for k, v in self.identity.items())
            or not 0 <= completed_updates <= self.stop_update
            or self.pending is not None
        ):
            raise ValueError("checkpoint/exposure identity mismatch")
        prefix, tasks, episodes = hashlib.sha256(), Counter(), Counter()
        for i in range(completed_updates * 64):
            row = self.draws[i]
            prefix.update(RECORD.pack(*row))
            tasks[str(row[2])] += 1
            episodes[str(row[3])] += 1
        if (
            prefix.hexdigest() != receipt["prefix_sha256"]
            or dict(tasks) != receipt["task_counts"]
            or dict(episodes) != receipt["episode_counts"]
        ):
            raise ValueError("checkpoint draw-prefix verification failed")
        self.prefix, self.task_counts, self.episode_counts = prefix, tasks, episodes
        self.committed_updates = completed_updates

    def close(self):
        if self._torch_loader is not None:
            iterator = getattr(self._torch_loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
        self.draws.close()
