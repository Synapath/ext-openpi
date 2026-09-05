"""Fixed episode-stratified probes with per-anchor RNG independent of training."""

from collections import defaultdict
import dataclasses
import hashlib
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model
from openpi.training import checkpoints
from openpi.training import data_loader
from openpi.training import diagnostics
from openpi.training.rbdj import temporal_mask


def select_anchors(split, partition, count=128):
    rows = [r for r in split["episodes"] if r["split"] == partition]
    rows.sort(key=lambda r: hashlib.sha256(f"pi-rlt-probe-v1|{r['episode_id']}".encode()).digest())
    if count < len(rows):
        raise ValueError("probe must cover every episode")
    result = []
    for i, row in enumerate(rows):
        n = count // len(rows) + (i < count % len(rows))
        frames = [(2 * k + 1) * row["anchors"] // (2 * n) for k in range(n)]
        if len(set(frames)) != n:
            raise ValueError("duplicate probe anchors")
        for frame in frames:
            identity = f"pi-rlt-probe-v1|{split['revision']}|{row['episode_id']}|{frame}"
            result.append(
                {
                    "episode_id": row["episode_id"],
                    "frame": frame,
                    "task_index": row["task_index"],
                    "progress_bin": min(2, 3 * frame // row["length"]),
                    "rng_seed": int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little"),
                }
            )
    return sorted(result, key=lambda r: (r["episode_id"], r["frame"]))


def aggregate(values, records, counts=None):
    """Element-weight within each episode, then episode-equal primary values."""
    episodes = sorted({r["episode_id"] for r in records})
    ids = np.asarray([r["episode_id"] for r in records])
    result = {}
    for key, value in values.items():
        raw = np.asarray(value, dtype=np.float64)
        weights = np.ones(len(records)) if counts is None else np.asarray(counts[key], dtype=np.float64)
        if (
            raw.shape != (len(records),)
            or weights.shape != raw.shape
            or not np.isfinite(raw).all()
            or not np.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError(f"invalid probe aggregate: {key}")
        result[key + "_count"] = int(weights.sum())
        episode_means = [
            float(np.sum(raw[ids == e] * weights[ids == e]) / weights[ids == e].sum())
            for e in episodes
            if weights[ids == e].sum() > 0
        ]
        result[key + "_episode_count"] = len(episode_means)
        if episode_means:
            result[key] = float(np.mean(episode_means))
            result[key + "_element_weighted"] = float(np.sum(raw * weights) / weights.sum())
    result.update(episode_count=len(episodes), anchor_noise_count=len(records))
    return result


class FixedProbes:
    def __init__(self, config, output, *, microbatch=8):
        self.config, self.microbatch = config, microbatch
        dc = config.data.create(config.assets_dirs, config.model)
        split_path = Path(dc.split_manifest_path)
        split = json.loads(split_path.read_text())
        episodes = {r["episode_id"]: r for r in split["episodes"]}
        self.records = {
            key: select_anchors(split, partition) for key, partition in (("train", "train"), ("val", "holdout"))
        }
        identity = {
            "schema": "pi-rlt-probes-v2-valid-elements",
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "horizon": config.model.action_horizon,
            "noise_draws_per_anchor": 2,
            "microbatch": microbatch,
            "records": self.records,
        }
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and json.loads(output.read_text()) != identity:
            raise ValueError("probe identity drift")
        if not output.exists():
            output.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        native = data_loader.lerobot_dataset.LeRobotDataset(
            dc.repo_id,
            root=Path(dc.dataset_root),
            episodes=[r["episode_id"] for r in split["episodes"]],
            download_videos=False,
            video_backend=dc.video_backend,
            delta_timestamps={key: [t / 25 for t in range(split["horizon"])] for key in dc.action_sequence_keys},
        )
        if dc.norm_stats is None:
            raise ValueError("probes require frozen train-only normalization")
        self.norm_stats = dc.norm_stats
        transform = transforms.compose(
            [
                transforms.PromptFromLeRobotTask(native.meta.tasks),
                *dc.repack_transforms.inputs,
                *dc.data_transforms.inputs,
                transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
                *dc.model_transforms.inputs,
            ]
        )
        epids = np.asarray(native.hf_dataset["episode_index"]).reshape(-1)
        frames = np.asarray(native.hf_dataset["frame_index"]).reshape(-1)
        self.batches, self.raw = {}, {}
        for partition, records in self.records.items():
            items, raw_items = [], []
            for r in records:
                positions = np.flatnonzero((epids == r["episode_id"]) & (frames == r["frame"]))
                if len(positions) != 1:
                    raise ValueError("probe native episode/frame identity mismatch")
                item = native[int(positions[0])]
                if int(item["task_index"]) != r["task_index"]:
                    raise ValueError("probe task/padding mismatch")
                mask = temporal_mask(item, split, episodes[r["episode_id"]], r["frame"])
                raw_items.append(
                    {
                        "state": np.asarray(item["observation.state"]),
                        "actions": np.asarray(item["action"]),
                        "valid_mask": np.ones(split["horizon"], dtype=bool) if mask is None else mask,
                    }
                )
                transformed = transform(item)
                if mask is not None:
                    transformed["action_valid_mask"] = mask
                    transformed["actions"] = np.where(mask[:, None], transformed["actions"], 0)
                items.append(transformed)
            self.batches[partition] = data_loader._collate_fn(items)  # noqa: SLF001 - share native loader collation
            self.raw[partition] = raw_items
        self._compiled = None

    def _compile(self, graphdef):
        def call(params, observation, actions, keys):
            m = nnx.merge(graphdef, params)
            m.eval()

            def one(obs, target, key):
                obs = jax.tree.map(lambda x: x[None], obs)
                squared = m.compute_loss_components(key, obs, target[None], train=False)
                return diagnostics.flow_metrics(squared, valid_mask=obs.action_valid_mask)

            return jax.vmap(one)(observation, actions, keys)

        self._compiled = jax.jit(call)

    def evaluate(self, state):
        if self._compiled is None:
            self._compile(state.model_def)
        result = {}
        for weight, params in (("raw", state.params), ("ema", checkpoints.inference_params(state))):
            for partition, batch in self.batches.items():
                rows, values, counts = [], defaultdict(list), defaultdict(list)
                records = self.records[partition]
                for draw in range(2):
                    for start in range(0, len(records), self.microbatch):
                        selected = records[start : start + self.microbatch]
                        sliced = jax.tree.map(lambda x, start=start: x[start : start + self.microbatch], batch)
                        keys = jnp.stack([jax.random.fold_in(jax.random.key(r["rng_seed"]), draw) for r in selected])
                        out = jax.device_get(
                            self._compiled(params, model.Observation.from_dict(sliced), sliced["actions"], keys)
                        )
                        for key, val in out.items():
                            if "count" not in key:
                                values[key].extend(val.tolist())
                                counts[key].extend(out[key + "_count"].tolist())
                        for r, key in zip(selected, keys, strict=True):
                            time_key = jax.random.split(key, 3)[2]
                            time = float(jax.random.beta(time_key, 1.5, 1, (1,))[0] * 0.999 + 0.001)
                            rows.append({**r, "time_bin": min(3, int(time * 4))})
                prefix = f"probe/{partition}/{weight}/"
                result[prefix + "element_count"] = sum(counts["flow_mse"])
                result[prefix + "real_element_count"] = sum(counts["flow_mse_real14"])
                result.update({prefix + k: v for k, v in aggregate(values, rows, counts).items()})
                for field, count in (("progress_bin", 3), ("time_bin", 4)):
                    for i in range(count):
                        mask = np.asarray([r[field] == i for r in rows])
                        result[prefix + f"{field}_{i}_count"] = int(mask.sum())
                        if mask.any():
                            subset = aggregate(
                                {"flow_mse_real14": np.asarray(values["flow_mse_real14"])[mask]},
                                [r for r, keep in zip(rows, mask, strict=True) if keep],
                                {"flow_mse_real14": np.asarray(counts["flow_mse_real14"])[mask]},
                            )
                            result[prefix + f"{field}_{i}"] = subset["flow_mse_real14"]
        for weight in ("raw", "ema"):
            result[f"probe/gap_{weight}"] = (
                result[f"probe/val/{weight}/flow_mse"] - result[f"probe/train/{weight}/flow_mse"]
            )
        result["probe/ema_minus_raw"] = result["probe/val/ema/flow_mse"] - result["probe/val/raw/flow_mse"]
        return result

    def evaluate_actions(self, state):
        def call(params, observation, keys):
            m = nnx.merge(state.model_def, params)
            m.eval()

            def one(obs, key):
                obs = jax.tree.map(lambda x: x[None], obs)
                obs = dataclasses.replace(obs, action_valid_mask=None)
                noise = jax.random.normal(key, (1, self.config.model.action_horizon, 32))
                return m.sample_actions(key, obs, num_steps=10, noise=noise)[0]

            return jax.vmap(one)(observation, keys)

        sample = jax.jit(call)
        records = self.records["val"]
        indices = []
        for episode in sorted({r["episode_id"] for r in records}):
            eligible = [i for i, r in enumerate(records) if r["episode_id"] == episode]
            indices.extend(eligible[i] for i in np.linspace(0, len(eligible) - 1, min(8, len(eligible)), dtype=int))
        result = {}
        arms = np.array([*range(6), *range(7, 13)])
        for weight, params in (("raw", state.params), ("ema", checkpoints.inference_params(state))):
            predictions = []
            for start in range(0, len(indices), self.microbatch):
                ids = indices[start : start + self.microbatch]
                batch = jax.tree.map(lambda x, ids=ids: x[ids], self.batches["val"])
                keys = jnp.stack([jax.random.fold_in(jax.random.key(records[i]["rng_seed"]), 100) for i in ids])
                predictions.extend(np.asarray(sample(params, model.Observation.from_dict(batch), keys)))
            pred = np.asarray(predictions)[..., :14]
            pred = transforms.Unnormalize({"actions": self.norm_stats["actions"]}, use_quantiles=True)(
                {"actions": pred}
            )["actions"]
            anchors = np.stack([self.raw["val"][i]["state"] for i in indices])
            pred[..., arms] += anchors[:, None, arms]
            target = np.stack([self.raw["val"][i]["actions"] for i in indices])
            if not np.isfinite(pred).all():
                raise ValueError("non-finite full-sampler action")
            valid = np.stack([self.raw["val"][i]["valid_mask"] for i in indices])
            selected_records = [records[i] for i in indices]
            for horizon in (10, 16, 20, self.config.model.action_horizon):
                delta = pred[:, :horizon] - target[:, :horizon]
                mask = valid[:, :horizon]
                adjacent = mask[:, 1:] & mask[:, :-1]
                prefix = f"action/{weight}/prefix{horizon}/"

                def record(name, error, selected, *, root_mean=False, records=selected_records, output_prefix=prefix):
                    if error.ndim == 2:
                        error = error[..., None]
                    counts = selected.sum(axis=1) * error.shape[-1]
                    means = np.where(selected[..., None], error, 0).sum(axis=(1, 2)) / np.maximum(counts, 1)
                    stats = aggregate({name: means}, records, {name: counts})
                    if root_mean:
                        for key in (name, name + "_element_weighted"):
                            if key in stats:
                                stats[key] = float(np.sqrt(stats[key]))
                    result.update({output_prefix + k: v for k, v in stats.items() if k.startswith(name)})

                record("arm_mae_rad", np.abs(delta[..., arms]), mask)
                record("arm_rmse_rad", np.square(delta[..., arms]), mask, root_mean=True)
                record("gripper_mae", np.abs(delta[..., [6, 13]]), mask)
                jumps = np.diff(pred[:, :horizon], axis=1)
                record("arm_jump_rad_per_step", np.abs(jumps[..., arms]), adjacent)
                record("arm_delta_action_error_rad_per_step", np.abs(np.diff(delta, axis=1)[..., arms]), adjacent)
                record("gripper_delta_action_error_per_step", np.abs(np.diff(delta, axis=1)[..., [6, 13]]), adjacent)
                for joint in arms:
                    record(f"joint_{joint:02d}_mae_rad", np.abs(delta[..., joint]), mask)
            result[f"action/{weight}/anchor_count"] = len(indices)
        return result
