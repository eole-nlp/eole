"""Module defining adapters / mm projectors."""

from eole.adapters.adapters import VisionLanguageAdapter
from eole.adapters.adapters import Gemma3MultiModalProjector
from eole.adapters.adapters import DeepSeekOCRProjector
from eole.adapters.adapters import HunYuanVisionPatchMerger
from eole.adapters.adapters import Qwen3_5VisionMerger


str2adapter = {
    "llava": VisionLanguageAdapter,
    "gemma3": Gemma3MultiModalProjector,
    "deepseekocr": DeepSeekOCRProjector,
    "hunyuanocr": HunYuanVisionPatchMerger,
    "qwen3_5vl": Qwen3_5VisionMerger,
    "qwen3vl": Qwen3_5VisionMerger,  # Qwen3VL uses the same vision merger architecture
}
