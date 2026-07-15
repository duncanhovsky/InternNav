from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_encoder_package_does_not_eagerly_import_longclip_encoders():
    init_path = PROJECT_ROOT / "internnav" / "model" / "encoder" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")

    assert "def __getattr__(name):" in text
    assert '"ImageEncoder": ".image_clip_encoder"' in text
    assert '"InstructionLongCLIPEncoder": ".instruction_longCLIP_encoder"' in text
    assert "from .image_clip_encoder import ImageEncoder" not in text
    assert "from .instruction_longCLIP_encoder import InstructionLongCLIPEncoder" not in text


def test_setup_verification_can_keep_importing_bridgedp_policy():
    setup_script = PROJECT_ROOT / "scripts" / "setup_internnav_pytorch270_cu126.sh"
    text = setup_script.read_text(encoding="utf-8")

    assert 'get_policy("BridgeDP_Policy")' in text
    assert 'get_config("BridgeDP_Policy")' in text
