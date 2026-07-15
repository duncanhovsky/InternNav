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


def test_full_16epoch_uniform_archives_include_2epoch_equivalent_checkpoint():
    from scripts.cache_rotation.cache_rotation_lib import compute_training_plan
    from scripts.train.base_train.uniform_checkpoint_utils import compute_uniform_checkpoint_targets

    total_episodes = 196_536
    full_16 = compute_training_plan(
        total_episodes=total_episodes,
        shard_episodes=total_episodes,
        total_epochs=16,
        shard_epochs=16,
        gpus=4,
        per_gpu_batch=96,
        grad_accum=1,
        current_global_step=0,
    )
    full_2 = compute_training_plan(
        total_episodes=total_episodes,
        shard_episodes=total_episodes,
        total_epochs=2,
        shard_epochs=2,
        gpus=4,
        per_gpu_batch=96,
        grad_accum=1,
        current_global_step=0,
    )

    targets = compute_uniform_checkpoint_targets(total_steps=full_16.total_max_steps, checkpoint_count=8)

    assert full_16.total_max_steps == 409_450
    assert full_2.total_max_steps == 51_182
    assert targets[0] == full_2.total_max_steps


def test_full_6epoch_uniform_archives_include_1epoch_equivalent_checkpoint():
    from scripts.cache_rotation.cache_rotation_lib import compute_training_plan
    from scripts.train.base_train.uniform_checkpoint_utils import compute_uniform_checkpoint_targets

    total_episodes = 196_536
    full_6 = compute_training_plan(
        total_episodes=total_episodes,
        shard_episodes=total_episodes,
        total_epochs=6,
        shard_epochs=6,
        gpus=4,
        per_gpu_batch=96,
        grad_accum=1,
        current_global_step=0,
    )
    full_1 = compute_training_plan(
        total_episodes=total_episodes,
        shard_episodes=total_episodes,
        total_epochs=1,
        shard_epochs=1,
        gpus=4,
        per_gpu_batch=96,
        grad_accum=1,
        current_global_step=0,
    )

    targets = compute_uniform_checkpoint_targets(total_steps=full_6.total_max_steps, checkpoint_count=6)

    assert full_6.total_max_steps == 153_544
    assert full_1.total_max_steps == 25_591
    assert targets[0] == full_1.total_max_steps


def test_full_4epoch_uniform_archives_include_1epoch_equivalent_checkpoint():
    from scripts.cache_rotation.cache_rotation_lib import compute_training_plan
    from scripts.train.base_train.uniform_checkpoint_utils import compute_uniform_checkpoint_targets

    total_episodes = 196_536
    full_4 = compute_training_plan(
        total_episodes=total_episodes,
        shard_episodes=total_episodes,
        total_epochs=4,
        shard_epochs=4,
        gpus=4,
        per_gpu_batch=96,
        grad_accum=1,
        current_global_step=0,
    )
    full_1 = compute_training_plan(
        total_episodes=total_episodes,
        shard_episodes=total_episodes,
        total_epochs=1,
        shard_epochs=1,
        gpus=4,
        per_gpu_batch=96,
        grad_accum=1,
        current_global_step=0,
    )

    targets = compute_uniform_checkpoint_targets(total_steps=full_4.total_max_steps, checkpoint_count=4)

    assert full_4.total_max_steps == 102_363
    assert full_1.total_max_steps == 25_591
    assert targets[0] == full_1.total_max_steps


def test_uniform_checkpoint_targets_can_be_disabled():
    from scripts.train.base_train.uniform_checkpoint_utils import compute_uniform_checkpoint_targets

    assert compute_uniform_checkpoint_targets(total_steps=1000, checkpoint_count=0) == []
