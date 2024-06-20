from typing import List, Literal
from pydantic import Field, model_validator, field_validator

from eole.config.common import RunningConfig, LoRaConfig, QuantizeConfig
from eole.config.config import Config


class DecodingConfig(Config):
    beam_size: int = Field(default=5, description="Beam size.")
    ratio: float = Field(
        default=-0.0, description="Ratio based beam stop condition."
    )  # is the minus sign useful here?
    random_sampling_topk: int = Field(
        default=0,
        description="Set this to -1 to do random sampling from full distribution. "
        "Set this to value k>1 to do random sampling restricted to "
        "the k most likely next tokens. Set this to 1 to use argmax.",
    )
    random_sampling_topp: float = Field(
        default=0.0,
        description="Probability for top-p/nucleus sampling. "
        "Restrict tokens to the most likely until the cumulated probability "
        "is over p. In range [0,1]. (https://arxiv.org/abs/1904.09751)",
        ge=0.0,
        lte=1.0,
    )
    random_sampling_temp: float = Field(
        default=1.0,
        description="If doing random sampling, divide the logits by this "
        "before computing softmax during decoding.",
    )
    length_penalty: Literal["avg", "wu", "none"] = Field(
        default="avg", description="Length penalty to use."
    )
    alpha: float = Field(
        default=1.0, description="Length penalty parameter (higher = longer generation)"
    )
    coverage_penalty: Literal["none", "wu", "summary"] = Field(
        default="none",
        description="Coverage penalty to use. Only available in beam search.",
    )
    beta: float = Field(default=-0.0, description="Coverage penalty parameter.")
    stepwise_penalty: bool = Field(
        default=False,
        description="Apply coverage penalty at every decoding step. Helpful for summary penalty.",
    )
    min_length: int = Field(default=0, description="Minimum prediction length.")
    max_length: int = Field(default=250, description="Maximum prediction length.")
    max_length_ratio: float = Field(
        default=2,
        description="Maximum prediction length ratio. For European languages, "
        "2 is large enough, for target Asian charageters, "
        "need to increase to 2-3, for special languages (Burmese, Amharic) to 10.",
    )
    block_ngram_repeat: int = Field(
        default=0, description="Block repetition of ngrams during decoding."
    )
    ignore_when_blocking: List[str] = Field(
        default=[],
        description="Ignore these strings when blocking repeats. "
        "You want to block sentence delimiters.",
    )
    replace_unk: bool = Field(
        default=False,
        description="Replace the generated UNK tokens with the source token "
        "that had the highest attention weight. If phrase_table is provided, "
        "it will lok up the identified source token and give the "
        "corresponding target token. If it is not provided "
        "(or the identified source token does not exist in the table), "
        "then it will copy the source token.",
    )  # deprecated?
    ban_unk_token: bool = Field(
        default=False,
        description="Prevent unk token generation by setting unk probability to 0.",
    )
    phrase_table: str = Field(
        default="",
        description="If phrase_table is provided (with replace_unk), it will look up "
        "the identified source token and give the corresponding target token.",
    )  # deprecated?
    n_best: int = Field(default=1, description="Output the n_best decoded sentences.")
    # added while testing, might be put elsewhere
    dump_beam: str = Field(
        default="", description="File to dump beam information to."
    )  # not sure it's still working/useful
    verbose: bool = Field(
        default=False, description="Print scores and predictions for each input."
    )
    with_score: bool = Field(
        default=False, description="Add a tab separated score to each output."
    )
    attn_debug: bool = Field(
        default=False, description="Print best attn for each word."
    )
    align_debug: bool = Field(
        default=False, description="Print best align for each word."
    )


# in legacy opts, decoding config is separated (probably to be used elsewhere)
class InferenceConfig(RunningConfig, DecodingConfig, LoRaConfig, QuantizeConfig):
    # TODO: clarify models vs model (model config retrieved from checkpoint)
    model_path: str | List[str] = Field(
        description="Path to model .pt file(s). "
        "Multiple models can be specified for ensemble decoding."
    )  # some specific (mapping to "models") in legacy code, need to investigate
    src: str = Field(description="Source file to decode (one line per sequence).")
    tgt: str | None = Field(
        default=None,
        description="True target sequences, useful for scoring or prefix decoding.",
    )
    tgt_file_prefix: bool = Field(
        default=False, description="Generate predictions using provided tgt as prefix."
    )
    output: str = Field(
        default="pred.txt",
        description="Path to output the predictions (each line will be the decoded sequence).",
    )
    report_align: bool = Field(
        default=False, description="Report alignment for each translation."
    )
    gold_align: bool = Field(
        default=False,
        description="Report alignment between source and gold target. "
        "Useful to test the performance of learnt alignments.",
    )
    report_time: bool = Field(
        default=False, description="Report some translation time metrics."
    )
    profile: bool = Field(default=False, description="Report pytorch profiling stats.")
    batch_size: int = Field(default=30, description="Batch size.")
    batch_type: Literal["sents", "tokens"] = Field(
        default="sents", description="Batch grouping for batch size."
    )
    gpu: int = Field(
        default=-1, description="Device to run on. -1 will default to CPU."
    )
    precision: Literal["", "fp32", "fp16", "int8"] = Field(
        default="",
        description="Precision to run inference. "
        "Default will use model.dtype, "
        "fp32 to force slow fp16 model on gtx1080, "
        "int8 to enable pytorch native 8-bit quantization (cpu only).",
    )
    avg_raw_probs: bool = Field(
        default=False,
        description="If set, during ensembling scores from different models will be combined "
        "by averaging their raw probabilities and then taking the log. "
        "Otherwise, the log probabilities will be averaged directly. "
        "Necessary for models whose output layers can assign zero probability.",
    )
    data_type: str | None = (
        "text"  # deprecated? hopefully will change with input streams logic
    )

    def get_model_path(self):
        return self.model_path[0]

    @field_validator("model_path")
    @classmethod
    def _validate_model_path(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            v = [v]
        return v

    @model_validator(mode="after")
    def _validate_running_config(self):
        super()._validate_running_config()
        if self.gold_align:
            assert self.report_align, "-report_align should be enabled with -gold_align"
            assert (
                not self.replace_unk
            ), "-replace_unk option can not be used with -gold_align enabled"
            assert self.tgt, "-tgt should be specified with -gold_align"
        # originally in validate_translate_opts_dynamic, not sure why
        # self.__dict__["share_vocab"] = False
        return self
