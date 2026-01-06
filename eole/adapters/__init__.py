"""Module defining adapters / mm projectors."""

from eole.adapters.adapters import VisionLanguageAdapter
from eole.adapters.adapters import Gemma3MultiModalProjector
from eole.adapters.adapters import DeepSeekOCRProjector
from eole.adapters.adapters import HunYuanVisionPatchMerger


str2adapter = {
    "llava": VisionLanguageAdapter,
    "gemma3": Gemma3MultiModalProjector,
    "deepseekocr": DeepSeekOCRProjector,
    "hunyuanocr": HunYuanVisionPatchMerger,
}
