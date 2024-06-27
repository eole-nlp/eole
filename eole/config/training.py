from typing import List, Literal
from pydantic import Field, field_validator, model_validator
import torch

from eole.config.config import Config
from eole.config.common import RunningConfig, LoRaConfig, QuantizeConfig
from eole.utils.logging import logger


class OptimizerConfig(Config):
    """
    Everything related to optimizers.
    Might be split into multiple subclasses later.
    Note: not fully sufficient (yet) to replace full opt namespace in build_torch_optimizer.
    Some other parameters (hidden_size, model_dtype, apex_opt_level, etc.) are accessed.
    """

    optim: Literal[
        "sgd",
        "adagrad",
        "adadelta",
        "adam",
        "sparseadam",
        "adafactor",
        "fusedadam",
        "adamw8bit",
        "pagedadamw8bit",
        "pagedadamw32bit",
    ] = Field(
        default="sgd", description="Optimization method."
    )  # find a better way to list accepted values...
    adagrad_accumulator_init: float = Field(
        default=0,
        description="Initialize the accumulator values in adagrad. "
        "Mirrors initial_accumulator_value flag from tensorflow "
        "adagrad implementation (default 0.1 there).",
    )
    adam_beta1: float = Field(
        default=0.9,
        description="Beta1 parameter used by Adam. "
        "Almost without exception a value of 0.9 is used in the literature, "
        "seemingly giving good results, so we would discourage changing "
        "this value from the default without due consideration.",
    )
    adam_beta2: float = Field(
        default=0.999,
        description="Beta2 parameter used by Adam. "
        "Typically a value of 0.999 is recommended, "
        "as this is the value suggested by the original paper describing Adam, "
        "and is also the value adopted in other frameworks such as Tensorflow "
        "(https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) "
        "and Keras (https://keras.io/optimizers/). "
        "Whereas recently the paper Attention is All You Need suggested "
        "a value of 0.98 for beta2, this parameter may not work well "
        "for normal models / default baselines.",
    )
    learning_rate: float = Field(
        default=1.0,
        description="Starting learning rate. "
        "Recommended settings: "
        "sgd=1, adagrad=0.1, adadelta=1, adam=0.001.",
    )
    learning_rate_decay: float = Field(
        default=0.5,
        description="Decay learning rate by this much if steps have gone past start_decay_steps.",
    )
    start_decay_steps: int = Field(
        default=50000,
        description="Start decaying every decay_steps after this many steps.",
    )
    decay_steps: int = Field(
        default=10000, description="Frequency for learning rate decay, in steps."
    )
    decay_method: Literal["noam", "noamwd", "rsqrt", "none"] = Field(
        default="none", description="Custom decay method to use."
    )
    warmup_steps: int = Field(
        default=4000, description="Number of warmup steps for custom decay."
    )
    reset_optim: Literal["none", "all", "states", "keep_states"] = Field(
        default="none", description="Optimization resetter when using train_from."
    )  # note: this is validated against update_vocab in TrainingConfig


class TrainingConfig(
    RunningConfig,
    OptimizerConfig,
    LoRaConfig,
    QuantizeConfig,
):
    # Init stuff
    param_init: float = Field(
        default=0.1,
        description="Support value for uniform distribution parameters initialization. "
        "Set to 0 not to use initialization.",
    )
    param_init_glorot: bool = Field(
        default=False,
        description="Initialize parameters with xavier_uniform. Required for transformer.",
    )
    freeze_encoder: bool = Field(
        default=False, description="Freeze parameters in encoder."
    )
    freeze_decoder: bool = Field(
        default=False, description="Freeze parameters in decoder."
    )
    # Pretrained word vectors (might be deprecated)
    pre_word_vecs_enc: str | None = Field(
        default=None,
        description="If a valid path is specified, will load pretrained "
        "word embeddings on the encoder side.",
    )
    pre_word_vecs_dec: str | None = Field(
        default=None,
        description="If a valid path is specified, will load pretrained "
        "word embeddings on the decoder side.",
    )
    data_type: str | None = "text"  # not really useful for now
    bucket_size: int = Field(
        default=262144,
        description="A bucket is a buffer of bucket_size examples "
        "to pick from the various corpora. "
        "The dynamic iterator batches batch_size items "
        "from the bucket and shuffle them.",
    )
    bucket_size_init: int = Field(
        default=-1,
        description="Bucket size is initialized with this amount of examples "
        "(see bucket_size_increment).",
    )
    bucket_size_increment: int = Field(
        default=0,
        description="Bucket size incremented with this amount of examples "
        "at each new bucket (up to bucket_size).",
    )
    prefetch_factor: int = Field(
        default=200,
        description="Number of mini-batches loaded in advance to avoid "
        "the GPU waiting during processing of next bucket.",
    )
    save_format: Literal["pytorch", "safetensors"] = Field(
        default="pytorch", description="Format to save the model weights."
    )
    save_checkpoint_steps: int = Field(
        default=5000, description="Frequency of checkpoint saving (in steps)."
    )
    keep_checkpoint: int = Field(
        default=-1, description="Number of checkpoints to retain. (-1 retains all)"
    )
    train_from: str | None = Field(
        default=None,
        description="Pretrained model/checkpoint weights to continue training from.",
    )
    num_workers: int = Field(
        default=2, description="Number of workers for pytorch.DataLoader objects."
    )  # dataloading related stuff (move to dataloader config ?
    batch_size: int = Field(default=64, description="Maximum batch size for training.")
    batch_size_multiple: int = Field(
        default=1, description="Batch size multiple for token batches."
    )
    batch_type: Literal["sents", "tokens"] = Field(
        default="sents", description="Batch grouping for batch_size."
    )
    normalization: Literal["sents", "tokens"] = Field(
        default="sents", description="Normalization method of the gradient."
    )
    accum_count: List[int] = Field(
        default=[1],
        description="Accumulate gradient this many times. "
        "Approximately equivalent to updating "
        "batch_size * accum_count batches at once. "
        "Recommended for transformer.",
    )
    accum_steps: List[int] = Field(
        default=[0], description="Steps at which accum_count values change."
    )
    valid_steps: int = Field(
        default=10000, description="Frequency of validation, in steps."
    )
    valid_batch_size: int = Field(
        default=32, description="Maximum batch size for validation."
    )
    train_steps: int = Field(default=100000, description="Number of training steps.")
    single_pass: bool = Field(
        default=False, description="Make a single pass over the training dataset."
    )  # deprecated? (not really manageable with weighted datasets)
    early_stopping: int = Field(
        default=0,
        description="Number of validation steps without improving that "
        "will trigger early stop of training.",
    )
    early_stopping_criteria: str | None = Field(
        default=None, description="Criteria to use for early stopping."
    )  # allowed values are not clear
    max_grad_norm: float = Field(
        default=5,
        description="If the norm of the gradient vector exceeds this value, "
        "renormalize it to have the norm equal to max_grad_norm.",
    )
    dropout: List[float] = Field(default=[0.3], description="Dropout probability.")
    attention_dropout: List[float] = Field(
        default=[0.1], description="Attention dropout probability."
    )
    dropout_steps: List[int] = Field(
        default=[0], description="Steps at which dropout changes."
    )
    truncated_decoder: int = Field(
        default=0, description="Truncated bptt."
    )  # deprecated?

    label_smoothing: float = Field(
        default=0.0,
        description="Label smoothing value epsilon. "
        "Probability of all non-true labels will be "
        "smoothed by epsilon/(vocab_size-1). "
        "Set to 0 to turn off label smoothing. (https://arxiv.org/abs/1512.00567)",
    )
    average_decay: float = Field(
        default=0.0,
        description="Exponential moving average decay "
        "(https://en.wikipedia.org/wiki/Moving_average). "
        "Set to other than 0 (e.g. 1e-4) to activate. "
        "Similar to Marian NMT implementation "
        "(http://www.aclweb.org/anthology/P18-4020).",
    )
    average_every: int = Field(
        default=1,
        description="Step for moving average. Default is every update if average_decay is set.",
    )

    model_dtype: Literal["fp32", "fp16"] = Field(
        default="fp16", description="Data type of the model."
    )
    loss_scale: float = Field(
        default=0.0,
        description="For FP16 training, the static loss scale to use. "
        "If not set, the loss scale is dynamically computed.",
    )
    apex_opt_level: Literal["", "O0", "O1", "O2", "O3"] = Field(
        default="",
        description="For FP16 training, the opt_level to use. "
        "See https://nvidia.github.io/apex/amp.html#opt-levels.",
    )
    zero_out_prompt_loss: bool = Field(
        default=False,
        description="Set the prompt loss to zero. Mostly for LLM finetuning. "
        "Will be enabled only if the `insert_mask_before_placeholder` "
        "transform is applied.",
    )
    use_ckpting: List[str] = Field(
        default=[], description="Use gradient checkpointing for those modules."
    )  # validate on ["ffn", "mha", "lora"]
    update_vocab: bool = Field(
        default=False, description="Update source and target existing vocabularies."
    )  # not linked to model per se, rather to the training instance of the model

    lm_prior_model: str | None = Field(
        default=None, description="LM model to use to train the TM."
    )
    lm_prior_lambda: float = Field(default=0.0, description="LM Prior Lambda")
    lm_prior_tau: float = Field(default=1.0, description="LM Prior Tau")
    estim_loss_lambda: List[float] = Field(
        default=[1.0], description="Weight applied to estimator loss"
    )
    estim_loss_lambda_steps: List[int] = Field(
        default=[0], description="Steps at which estimator loss lambda changes"
    )
    score_threshold: float = Field(
        default=0.68, description="Threshold to filterout data"
    )

    @field_validator("use_ckpting", mode="after")
    @classmethod
    def checkpointing_layers(cls, field: List[str]):
        _layers = {"ffn", "mha", "lora"}
        if set(field) | _layers != _layers:
            raise ValueError(f"use_ckpting only supports these layers: {_layers}")
        return field

    def get_model_path(self):
        # TODO: clarify this
        if self.train_from is not None:
            return self.train_from
        else:
            return self.model_path

    @model_validator(mode="after")
    def _validate_running_config(self):
        super()._validate_running_config()
        # self._validate_language_model_compatibilities_opts()
        if (
            torch.cuda.is_available()
            and not self.gpu_ranks
            and self.model_fields_set != set()
        ):
            logger.warn("You have a CUDA device, should run with -gpu_ranks")
        if self.world_size < len(self.gpu_ranks):
            raise AssertionError(
                "parameter counts of -gpu_ranks must be less or equal "
                "than -world_size."
            )
        if (
            self.world_size == len(self.gpu_ranks)
            and self.world_size > 0
            and min(self.gpu_ranks) > 0
        ):
            raise AssertionError(
                "-gpu_ranks should have master(=0) rank "
                "unless -world_size is greater than len(gpu_ranks)."
            )

        assert len(self.dropout) == len(
            self.dropout_steps
        ), "Number of dropout values must match accum_steps values"

        assert len(self.attention_dropout) == len(
            self.dropout_steps
        ), "Number of attention_dropout values must match accum_steps values"

        assert len(self.accum_count) == len(
            self.accum_steps
        ), "Number of accum_count values must match number of accum_steps"

        if self.update_vocab:
            assert self.train_from, "-update_vocab needs -train_from option"
            assert self.reset_optim in [
                "states",
                "all",
            ], '-update_vocab needs -reset_optim "states" or "all"'
        return self
