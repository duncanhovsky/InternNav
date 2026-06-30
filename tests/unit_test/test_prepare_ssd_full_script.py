from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_prepare_ssd_full_falls_back_when_rsync_is_missing():
    script = PROJECT_ROOT / "scripts" / "prepare_ssd_full.sh"
    text = script.read_text(encoding="utf-8")

    assert "copy_project_to_ssd()" in text
    assert "command -v rsync" in text
    assert "rsync -aH --info=progress2" in text
    assert "rsync not found; falling back to cp -a" in text
    assert "cp -a \"${SRC_PROJECT}/.\" \"${DST_PROJECT}/\"" in text


def test_prepare_ssd_full_reports_bad_archives_before_preload_index():
    script = PROJECT_ROOT / "scripts" / "prepare_ssd_full.sh"
    text = script.read_text(encoding="utf-8")

    assert "BAD_ARCHIVE_LOG" in text
    assert "gzip -t \"${archive}\"" in text
    assert "tar -xzf \"${archive}\" -C \"${dst_dir}\"" in text
    assert "tar -xf \"${archive}\" -C \"${dst_dir}\"" in text
    assert "bad or incomplete archive" in text
    assert "Bad archive list" in text


def test_prepare_ssd_full_checks_ssd_mount_space_and_inodes_before_copy():
    script = PROJECT_ROOT / "scripts" / "prepare_ssd_full.sh"
    text = script.read_text(encoding="utf-8")

    assert "REQUIRE_SSD_MOUNT" in text
    assert "check_ssd_storage()" in text
    assert "findmnt -T \"${SSD_ROOT}\"" in text
    assert "df -h \"${SSD_ROOT}\"" in text
    assert "df -i \"${SSD_ROOT}\"" in text
    assert "SSD_ROOT is not a mount point" in text
    assert "No free bytes on ${SSD_ROOT}" in text
    assert "No free inodes on ${SSD_ROOT}" in text
