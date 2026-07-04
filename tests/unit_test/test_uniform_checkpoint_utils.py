def test_uniform_checkpoint_targets_cover_full_training_evenly():
    from scripts.train.base_train.uniform_checkpoint_utils import compute_uniform_checkpoint_targets

    total_steps = 2_559_063
    targets = compute_uniform_checkpoint_targets(total_steps=total_steps, checkpoint_count=20)

    assert len(targets) == 20
    assert targets[0] == 127_954
    assert targets[-1] == total_steps
    assert targets == sorted(set(targets))

    gaps = [right - left for left, right in zip([0] + targets[:-1], targets)]
    assert max(gaps) - min(gaps) <= 1


def test_full_100epoch_uniform_archives_include_10epoch_equivalent_checkpoint():
    from scripts.cache_rotation.cache_rotation_lib import compute_training_plan
    from scripts.train.base_train.uniform_checkpoint_utils import compute_uniform_checkpoint_targets

    total_episodes = 196_536
    full_100 = compute_training_plan(
        total_episodes=total_episodes,
        shard_episodes=total_episodes,
        total_epochs=100,
        shard_epochs=100,
        gpus=4,
        per_gpu_batch=96,
        grad_accum=1,
        current_global_step=0,
    )
    full_10 = compute_training_plan(
        total_episodes=total_episodes,
        shard_episodes=total_episodes,
        total_epochs=10,
        shard_epochs=10,
        gpus=4,
        per_gpu_batch=96,
        grad_accum=1,
        current_global_step=0,
    )

    targets = compute_uniform_checkpoint_targets(total_steps=full_100.total_max_steps, checkpoint_count=20)

    assert full_100.total_max_steps == 2_559_063
    assert full_10.total_max_steps == 255_907
    assert targets[1] == full_10.total_max_steps


def test_uniform_checkpoint_targets_can_be_disabled():
    from scripts.train.base_train.uniform_checkpoint_utils import compute_uniform_checkpoint_targets

    assert compute_uniform_checkpoint_targets(total_steps=1000, checkpoint_count=0) == []
