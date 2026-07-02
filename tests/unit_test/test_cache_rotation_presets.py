import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_rotation.cache_rotation_lib import compute_training_plan, resolve_profile


def test_resolve_profile_selects_balanced_4gpu_2tb_defaults():
    profile = resolve_profile(gpus=4, nvme_size="2tb", preset="balanced")

    assert profile.gpus == 4
    assert profile.nvme_size == "2tb"
    assert profile.cache_slot_gb == 750
    assert profile.shard_epochs == 2
    assert profile.per_gpu_batch == 96
    assert profile.num_workers == 10
    assert profile.build_workers == 4


def test_resolve_profile_selects_throughput_8gpu_4tb_defaults():
    profile = resolve_profile(gpus=8, nvme_size="4tb", preset="throughput")

    assert profile.gpus == 8
    assert profile.nvme_size == "4tb"
    assert profile.cache_slot_gb == 1600
    assert profile.shard_epochs == 3
    assert profile.per_gpu_batch == 96
    assert profile.num_workers == 8
    assert profile.build_workers == 6


def test_compute_training_plan_matches_bridgedp_preload_repeat_factor():
    plan = compute_training_plan(
        total_episodes=1000,
        shard_episodes=200,
        total_epochs=100,
        shard_epochs=2,
        gpus=4,
        per_gpu_batch=96,
        grad_accum=1,
        current_global_step=0,
    )

    assert plan.global_batch == 384
    assert plan.total_max_steps == math.ceil(50 * 1000 * 100 / 384)
    assert plan.stage_steps == math.ceil(50 * 200 * 2 / 384)
    assert plan.stage_end_step == plan.stage_steps
