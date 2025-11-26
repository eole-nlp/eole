"""Define constant values used across the project."""

from enum import Enum
import torch
import torch.nn.functional as F
from functools import partial


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


def fused_gated_gelu(x):
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.gelu_and_mul(out, x)
    return out


def fused_gated_silu(x):
    d = x.shape[-1] // 2
    # basically doing return F.silu(x[..., :d]) * x[..., d:]
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


def fused_gated_gelu_tanh(x):
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.gelu_tanh_and_mul(out, x)
    return out


@torch.jit.script
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class ActivationFunction(str, Enum):
    relu = "relu"
    gelu = "gelu"
    gelu_tanh = "gelu-tanh"
    quick_gelu = "quick_gelu"
    silu = "silu"
    gated_gelu = "gated-gelu"
    gated_gelu_tanh = "gated-gelu-tanh"
    gated_silu = "gated-silu"
    fused_gated_gelu = "fused-gated-gelu"
    fused_gated_gelu_tanh = "fused-gated-gelu-tanh"
    fused_gated_silu = "fused-gated-silu"


class TransformType(str, Enum):
    Default = "any"
    Train = "train"
    Predict = "predict"


ACTIVATION_FUNCTIONS = {
    ActivationFunction.relu: F.relu,
    ActivationFunction.gelu: F.gelu,
    ActivationFunction.silu: F.silu,
    ActivationFunction.quick_gelu: quick_gelu,
    ActivationFunction.gated_gelu: F.gelu,
    ActivationFunction.fused_gated_gelu: fused_gated_gelu,
    ActivationFunction.gated_silu: F.silu,
    ActivationFunction.fused_gated_silu: fused_gated_silu,
    ActivationFunction.gelu_tanh: partial(F.gelu, approximate="tanh"),
    ActivationFunction.gated_gelu_tanh: partial(F.gelu, approximate="tanh"),
    ActivationFunction.fused_gated_gelu_tanh: fused_gated_gelu_tanh,
}


class LayerNormFP32(torch.nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        orig_dtype = input.dtype  # Save original dtype (likely bfloat16)
        input = input.to(torch.float32)  # Process in float32
        output = super().forward(input)  # Apply LayerNorm
        return output.to(orig_dtype)  # Convert back to original dtype


class LazyLayerNormDict:
    """Dictionary-like object that lazily imports heavy LayerNorm classes."""

    def __init__(self):
        self._registry = {
            "standardFP32": LayerNormFP32,
            "standard": torch.nn.LayerNorm,
            "rms": None,  # Placeholder
            "gemma-rms": None,  # Placeholder
        }

    def __getitem__(self, key):
        if key not in self._registry:
            raise KeyError(f"Unknown LayerNorm type: {key}")

        # Lazy import only when first accessed
        if key == "rms" and self._registry[key] is None:
            from eole.modules.rmsnorm import RMSNorm

            self._registry[key] = RMSNorm
        elif key == "gemma-rms" and self._registry[key] is None:
            from eole.modules.rmsnorm import GemmaRMSNorm

            self._registry[key] = GemmaRMSNorm

        return self._registry[key]


LayerNorm = LazyLayerNormDict()


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
