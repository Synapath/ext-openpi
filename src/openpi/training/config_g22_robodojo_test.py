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
    assert config.lr_schedule.warmup_steps == 253
    assert config.lr_schedule.decay_steps == 40_000
    assert tuple(data.episode_indices) == tuple(range(200, 300))
    assert data.prompt_from_task
    assert data.asset_id == "arx_x5_sim"
    assert data.dataset_root.endswith("/g22-classify-view")
    assert data.draw_manifest_path.endswith("/execution/data/draw-manifest.json")
    assert config.policy_metadata["recipe"] == "builtin-dual-lora"
