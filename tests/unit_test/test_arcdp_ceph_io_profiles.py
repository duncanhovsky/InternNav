from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_8x4090_config_exposes_dataloader_prefetch_factor():
    il_config = (PROJECT_ROOT / "internnav" / "configs" / "trainer" / "il.py").read_text(encoding="utf-8")
    full_config = (
        PROJECT_ROOT
        / "scripts"
        / "train"
        / "base_train"
        / "configs"
        / "bridgedp_full_8x4090.py"
    ).read_text(encoding="utf-8")

    assert "dataloader_prefetch_factor: Optional[int] = None" in il_config
    assert 'BRIDGEDP_PREFETCH_FACTOR", 2' in full_config


def test_training_arguments_receive_validated_prefetch_factor():
    train_entry = (
        PROJECT_ROOT / "scripts" / "train" / "base_train" / "train.py"
    ).read_text(encoding="utf-8")

    assert "dataloader_prefetch_factor = getattr(config.il, 'dataloader_prefetch_factor', None)" in train_entry
    assert "if dataloader_num_workers <= 0:" in train_entry
    assert "dataloader_prefetch_factor=dataloader_prefetch_factor" in train_entry


def test_ceph_io_launcher_defines_two_isolated_full1_profiles():
    launcher = (
        PROJECT_ROOT
        / "scripts"
        / "train"
        / "arcdp_4090"
        / "train_arcdp_full1_8x4090_ceph_io.sh"
    ).read_text(encoding="utf-8")

    assert 'b48_w2)' in launcher
    assert 'PROFILE_BATCH_SIZE=48' in launcher
    assert 'PROFILE_GRAD_ACCUM=1' in launcher
    assert 'PROFILE_NUM_WORKERS=2' in launcher
    assert 'PROFILE_PREFETCH_FACTOR=2' in launcher
    assert 'arcdp_full1_b48_w2_8x4090_ceph' in launcher

    assert 'b24_ga2_w2_pf1)' in launcher
    assert 'PROFILE_BATCH_SIZE=24' in launcher
    assert 'PROFILE_GRAD_ACCUM=2' in launcher
    assert 'PROFILE_PREFETCH_FACTOR=1' in launcher
    assert 'arcdp_full1_b24_ga2_w2_pf1_8x4090_ceph' in launcher

    assert 'BRIDGEDP_NUM_GPUS=8' in launcher
    assert 'BRIDGEDP_EPOCHS=1' in launcher
    assert 'CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"' in launcher
    assert 'effective global batch: ${effective_global_batch}' in launcher
    assert '--model-name bridgedp_full_8x4090' in launcher
    assert 'bridgedp_p0_' not in launcher
    assert 'BRIDGEDP_REPORT_TO="${BRIDGEDP_REPORT_TO:-swanlab}"' in launcher
    assert 'ARCDP_ALLOW_BUSY_GPUS' in launcher
    assert 'ARCDP_DRY_RUN' in launcher
    assert 'findmnt -T "${BRIDGEDP_DATASET_ROOT}"' in launcher


def test_root_wrappers_launch_the_requested_ceph_io_profile():
    expected = {
        "train_arcdp_full1_b48_w2_8x4090_ceph.sh": "b48_w2",
        "train_arcdp_full1_b24_ga2_w2_pf1_8x4090_ceph.sh": "b24_ga2_w2_pf1",
    }

    for filename, profile in expected.items():
        wrapper = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        assert "scripts/train/arcdp_4090/train_arcdp_full1_8x4090_ceph_io.sh" in wrapper
        assert f'"{profile}" "$@"' in wrapper
