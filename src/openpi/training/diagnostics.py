"""Small native-flow reductions and optimizer diagnostics; no sampling RNG."""

from flax import traverse_util
import jax
import jax.numpy as jnp
import optax


def flow_metrics(squared, *, execution_horizon=16, valid_mask=None):
    if squared.ndim != 3 or squared.shape[-1] != 32 or not 0 < execution_horizon < squared.shape[1]:
        raise ValueError("expected [B,H,32] with 0 < E < H")
    mask = jnp.ones(squared.shape[:2], dtype=bool) if valid_mask is None else valid_mask
    if mask.shape != squared.shape[:2]:
        raise ValueError("temporal mask must match [B,H]")
    result = {}

    def reduce(name, values, selected):
        if values.ndim == 2:
            values = values[..., None]
        count = selected.sum() * values.shape[-1]
        result[name] = jnp.where(selected[..., None], values, 0).sum() / jnp.maximum(count, 1)
        result[name + "_count"] = count

    real = squared[..., :14]
    for name, values in (
        ("flow_mse", squared),
        ("flow_mse_real14", real),
        ("flow_mse_pad18", squared[..., 14:]),
        ("flow_mse_left_arm", squared[..., :6]),
        ("flow_mse_right_arm", squared[..., 7:13]),
        ("flow_mse_left_gripper", squared[..., 6]),
        ("flow_mse_right_gripper", squared[..., 13]),
    ):
        reduce(name, values, mask)
    reduce("flow_mse_exec_prefix", real[:, :execution_horizon], mask[:, :execution_horizon])
    reduce("flow_mse_future", real[:, execution_horizon:], mask[:, execution_horizon:])
    for i in range(32):
        reduce(f"flow_mse_dim_{i:02d}", squared[..., i], mask)
    for n in (10, 16, 20):
        if n <= squared.shape[1]:
            reduce(f"flow_mse_prefix{n}", real[:, :n], mask[:, :n])
    result.update(
        element_count=result["flow_mse_count"],
        real_element_count=result["flow_mse_real14_count"],
        exec_element_count=result["flow_mse_exec_prefix_count"],
        valid_time_count=mask.sum(),
        nominal_element_count=squared.size,
    )
    return result


def module_name(path):
    name = "/".join(str(x) for x in path)
    if name.startswith("PaliGemma/img/"):
        return "vision"
    if "lora" in name:
        return "expert_lora" if "_1/" in name else "vlm_lora"
    if path[0] in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        return "interface"
    raise ValueError(f"unclassified trainable parameter: {name}")


def module_metrics(params, grads, updates):
    p, g, u = [traverse_util.flatten_dict(x.to_pure_dict()) for x in (params, grads, updates)]
    groups = {name: [] for name in ("vision", "vlm_lora", "expert_lora", "interface")}
    for path in p:
        groups[module_name(path)].append(path)
    result = {}
    for name, paths in groups.items():
        weight = optax.global_norm([p[k] for k in paths])
        update = optax.global_norm([u[k] for k in paths])
        result[f"optim/{name}/grad_norm"] = optax.global_norm([g[k] for k in paths])
        result[f"optim/{name}/update_norm"] = update
        result[f"optim/{name}/weight_norm"] = weight
        result[f"optim/{name}/update_to_weight"] = update / (weight + 1e-12)
    return result


def nonfinite_count(*trees):
    return sum(jnp.sum(~jnp.isfinite(x)) for x in jax.tree.leaves(trees))


def optimizer_metrics(grads, clip):
    norm = optax.global_norm(grads)
    return {
        "optim/grad_norm_preclip": norm,
        "optim/clip_fraction": (norm > clip).astype(jnp.float32),
        "optim/clip_scale": jnp.where(norm < clip, 1.0, clip / jnp.maximum(norm, 1e-12)),
    }


def reduce_step_metrics(stacked):
    """Report flow groups over actual elements; retain mean per-update optimizer loss."""
    result = jax.tree.map(jnp.mean, stacked)
    for key, values in stacked.items():
        if key.startswith("train/") and key.endswith("count"):
            result[key] = jnp.sum(values)
        elif key.startswith("train/flow_mse") and key + "_count" in stacked:
            counts = stacked[key + "_count"]
            result[key] = jnp.sum(values * counts) / jnp.maximum(jnp.sum(counts), 1)
    if "train/nominal_element_count" in result:
        result["train/temporal_padding_fraction"] = (
            1 - result["train/element_count"] / result["train/nominal_element_count"]
        )
    return result
