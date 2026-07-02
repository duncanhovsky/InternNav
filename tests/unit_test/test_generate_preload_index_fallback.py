import builtins
import importlib.util
import json
from pathlib import Path


def _load_generator_without_jsonlines(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jsonlines":
            raise ImportError("jsonlines intentionally unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "dataset"
        / "generate_preload_index.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_preload_index_without_jsonlines", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_data_unit_reads_jsonl_without_jsonlines(monkeypatch, tmp_path):
    module = _load_generator_without_jsonlines(monkeypatch)

    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    (tmp_path / "meta").mkdir()
    (tmp_path / "videos" / "chunk-000" / "observation.images.rgb").mkdir(parents=True)
    (tmp_path / "videos" / "chunk-000" / "observation.images.depth").mkdir(parents=True)

    (tmp_path / "data" / "chunk-000" / "episode_000000.parquet").touch()
    (tmp_path / "videos" / "chunk-000" / "observation.images.rgb" / "000000.png").touch()
    (tmp_path / "videos" / "chunk-000" / "observation.images.depth" / "000000.png").touch()
    with open(tmp_path / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"episode_index": 0}) + "\n")

    result = module._process_data_unit(tmp_path)

    assert result["trajectory_data_dir"] == [
        str(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    ]
    assert result["trajectory_rgb_path"] == [
        [str(tmp_path / "videos" / "chunk-000" / "observation.images.rgb" / "000000.png")]
    ]
    assert result["trajectory_depth_path"] == [
        [str(tmp_path / "videos" / "chunk-000" / "observation.images.depth" / "000000.png")]
    ]
