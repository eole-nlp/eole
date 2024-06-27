"""Define constant values used across the project."""
from enum import Enum
import torch
from eole.modules.rmsnorm import RMSNorm
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


class ActivationFunction(str, Enum):
    relu = "relu"
    gelu = "gelu"
    silu = "silu"
    gated_gelu = "gated-gelu"
    gated_silu = "gated-silu"


ACTIVATION_FUNCTIONS = {
    ActivationFunction.relu: F.relu,
    ActivationFunction.gelu: F.gelu,
    ActivationFunction.silu: F.silu,
    ActivationFunction.gated_gelu: F.gelu,
    ActivationFunction.gated_silu: F.silu,
}


LayerNorm = {"standard": torch.nn.LayerNorm, "rms": RMSNorm}
