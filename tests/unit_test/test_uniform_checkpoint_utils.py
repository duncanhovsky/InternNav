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


def test_uniform_checkpoint_targets_can_be_disabled():
    from scripts.train.base_train.uniform_checkpoint_utils import compute_uniform_checkpoint_targets

    assert compute_uniform_checkpoint_targets(total_steps=1000, checkpoint_count=0) == []
