"""Define constant values used across the project."""

from enum import Enum
import torch
from eole.modules.rmsnorm import RMSNorm, GemmaRMSNorm
import torch.nn.functional as F


class DefaultTokens(object):
    PAD = "<blank>"
    BOS = "<s>"
    EOS = "</s>"
    UNK = "<unk>"
    MASK = "<mask>"
    VOCAB_PAD = "averyunlikelytoken"
    SENT_FULL_STOPS = [".", "?", "!"]
    PHRASE_TABLE_SEPARATOR = "|||"
    ALIGNMENT_SEPARATOR = " ||| "
    SEP = "｟newline｠"
    MASK_BEFORE = "｟_mask_before_｠"


class CorpusName(str, Enum):
    VALID = "valid"
    TRAIN = "train"
    SAMPLE = "sample"
    INFER = "infer"


class CorpusTask(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    INFER = "infer"


class SubwordMarker(str, Enum):
    SPACER = "▁"
    JOINER = "￭"
    BEGIN_UPPERCASE = "｟mrk_begin_case_region_U｠"
    END_UPPERCASE = "｟mrk_end_case_region_U｠"
    BEGIN_CASED = "｟mrk_case_modifier_C｠"


class ModelType(str, Enum):
    DECODER = "decoder"
    ENCODER_DECODER = "encoder_decoder"
    ENCODER = "encoder"


class PositionEncodingType(str, Enum):
    SinusoidalInterleaved = "SinusoidalInterleaved"
    SinusoidalConcat = "SinusoidalConcat"
    Learned = "Learned"
    Relative = "Relative"
    Rotary = "Rotary"
    Alibi = "Alibi"


class ActivationFunction(str, Enum):
    relu = "relu"
    gelu = "gelu"
    silu = "silu"
    gated_gelu = "gated-gelu"
    gated_silu = "gated-silu"


class TransformType(str, Enum):
    Default = "any"
    Train = "train"
    Predict = "predict"


ACTIVATION_FUNCTIONS = {
    ActivationFunction.relu: F.relu,
    ActivationFunction.gelu: F.gelu,
    ActivationFunction.silu: F.silu,
    ActivationFunction.gated_gelu: F.gelu,
    ActivationFunction.gated_silu: F.silu,
}


class LayerNormFP32(torch.nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        orig_dtype = input.dtype  # Save original dtype (likely bfloat16)
        input = input.to(torch.float32)  # Process in float32
        output = super().forward(input)  # Apply LayerNorm
        return output.to(orig_dtype)  # Convert back to original dtype


LayerNorm = {
    "standardFP32": LayerNormFP32,
    "standard": torch.nn.LayerNorm,
    "rms": RMSNorm,
    "gemma-rms": GemmaRMSNorm,
}


TORCH_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int8": torch.int8,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
