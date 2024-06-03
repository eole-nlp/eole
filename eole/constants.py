"""Define constant values used across the project."""
from enum import Enum


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


class ModelTask(str, Enum):
    LANGUAGE_MODEL = "lm"
    SEQ2SEQ = "seq2seq"
    ENCODER = "encoder"


class PositionEncodingType(str, Enum):
    SinusoidalInterleaved = "SinusoidalInterleaved"
    SinusoidalConcat = "SinusoidalConcat"
