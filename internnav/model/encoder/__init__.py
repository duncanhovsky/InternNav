"""Encoder exports.

LongCLIP-backed encoders are optional. BridgeDP/ArcDP only need the RGB-D
backbone path, so importing the encoder package should not require the
LongCLIP submodule to exist.
"""

from importlib import import_module

from .bert_backbone import PositionalEncoding
from .distance_encoder import DistanceNetwork
from .instruction_encoder import InstructionEncoder
from .instruction_roberta_encoder import LanguageEncoder
from .vision_language_encoder import VisionLanguageEncoder

_OPTIONAL_EXPORTS = {
    "ImageEncoder": ".image_clip_encoder",
    "InstructionLongCLIPEncoder": ".instruction_longCLIP_encoder",
}

__all__ = [
    "PositionalEncoding",
    "DistanceNetwork",
    "InstructionEncoder",
    "LanguageEncoder",
    "VisionLanguageEncoder",
    "ImageEncoder",
    "InstructionLongCLIPEncoder",
]


def __getattr__(name):
    module_name = _OPTIONAL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        if "LongCLIP" in str(exc):
            raise ModuleNotFoundError(
                f"{name} requires the optional LongCLIP submodule. "
                "ArcDP/BridgeDP training does not need it; install or initialize "
                "LongCLIP only when using LongCLIP/CMA/RDP paths."
            ) from exc
        raise

    value = getattr(module, name)
    globals()[name] = value
    return value
