from openpi.training import config as _config


def test_g22_classify_official_recipe_is_frozen():
    config = _config.get_config("pi05_g22_classify_official_s0_b128_builtin_dual_lora")
    data = config.data.create(config.assets_dirs, config.model)

    assert config.batch_size == 128
    assert config.seed == 0
    assert config.ema_decay == 0.99
    assert config.fsdp_devices == 2
    assert config.num_workers == 8
    assert config.num_train_steps == 60_000
    assert config.model.lora_zero_init_b
    assert config.lr_schedule.warmup_steps == 253
    assert config.lr_schedule.decay_steps == 40_000
    assert tuple(data.episode_indices) == tuple(range(200, 300))
    assert data.prompt_from_task
    assert data.asset_id == "arx_x5_sim"
    assert data.dataset_root.endswith("/g22-classify-view")
    assert data.draw_manifest_path.endswith("/execution/data/draw-manifest.json")
    assert config.policy_metadata["recipe"] == "builtin-dual-lora"


def test_g22_classify_train90_val10_recipe_is_frozen():
    config = _config.get_config("pi05_g22_classify_official_s0_b128_builtin_dual_lora_train90_val10")
    data = config.data.create(config.assets_dirs, config.model)

    assert config.batch_size == 128
    assert config.seed == 0
    assert config.ema_decay == 0.99
    assert config.fsdp_devices == 2
    assert config.num_train_steps == 60_000
    assert config.validation_interval == 1_000
    assert config.lr_schedule.warmup_steps == 253
    assert config.lr_schedule.decay_steps == 40_000
    assert tuple(data.validation_episode_indices) == (
        201,
        207,
        208,
        211,
        213,
        223,
        248,
        254,
        291,
        294,
    )
    assert len(data.episode_indices) == 90
    assert set(data.episode_indices).isdisjoint(data.validation_episode_indices)
    assert set(data.episode_indices) | set(data.validation_episode_indices) == set(range(200, 300))
    assert data.dataset_root.endswith("/g22-classify-train90-val10-view")
    assert data.draw_manifest_path.endswith("/execution/data/draw-manifest.json")
    assert config.policy_metadata["recipe"] == "builtin-dual-lora"
    assert config.policy_metadata["robot_config"]["execution_horizon"] == 20
