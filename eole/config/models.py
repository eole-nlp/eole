from typing import Dict, Union, Literal, Any, Annotated
from pydantic import (
    Field,
    field_validator,
    model_validator,
)  # , TypeAdapter

from eole.constants import PositionEncodingType, ActivationFunction, ModelType
from eole.config.config import Config


class EmbeddingsConfig(Config):
    src_word_vec_size: int = Field(
        default=512, description="Word embedding size for src."
    )
    tgt_word_vec_size: int = Field(
        default=512, description="Word embedding size for tgt."
    )
    word_vec_size: int = Field(
        default=-1, description="Word embedding size for src and tgt."
    )
    # Freeze word vectors
    freeze_word_vecs_enc: bool = Field(
        default=False, description="Freeze word embeddings on the encoder side."
    )
    freeze_word_vecs_dec: bool = Field(
        default=False, description="Freeze word embeddings on the encoder side."
    )
    position_encoding: bool = Field(
        default=False,
        description="Absolute position encoding, see position_encoding_type. "
        "Necessary for non-RNN style models.",
    )
    position_encoding_type: PositionEncodingType = Field(
        default=PositionEncodingType.SinusoidalInterleaved,
        description="Type of positional encoding.",
    )
    n_positions: int | None = Field(
        default=None,
        description="Absolute number of positions to learn "
        "position embeddings on (position_encoding_type: Learned)",
    )
    position_shift: int | None = Field(
        default=0,
        description="Positions IDS shift before making position embed "
        "dirty patch to cover for xlm-roberta-xl",
    )

    @model_validator(mode="after")
    def validate_embeddings(self):
        if self.position_encoding_type == PositionEncodingType.Learned:
            assert self.n_positions is not None, (
                "n_positions must be set if position_encoding_type "
                f"is {PositionEncodingType.Learned}"
            )
        return self


class EncoderConfig(Config):
    """Abstract class for all encoders"""

    encoder_type: str | None = Field(
        default="rnn", description="Type of encoder layer(s) to use."
    )
    layers: int = Field(default=2, description="Number of layers in the encoder.")
    hidden_size: int = Field(default=512, description="Size of encoder hidden states.")

    # This field should be set at EmbeddingsConfig level but will be copied here for cases
    # where input size to the rnn is different to the hidden size
    src_word_vec_size: int = Field(
        default=512, description="Word embedding size for src."
    )


class DecoderConfig(Config):
    """Abstract class for all decoders"""

    decoder_type: str | None = Field(
        default="rnn", description="Type of decoder layer(s) to use."
    )
    layers: int = Field(default=2, description="Number of layers in the decoder.")
    hidden_size: int = Field(default=512, description="Size of decoder hidden states.")

    # This field should be set at EmbeddingsConfig level but will be copied here for cases
    # where input size to the rnn is different to the hidden size
    tgt_word_vec_size: int = Field(
        default=512, description="Word embedding size for tgt."
    )
    coverage_attn: bool = Field(
        default=False, description="Train a coverage attention layer."
    )
    lambda_coverage: float = Field(
        default=0.0, description="Lambda value for coverage loss of See et al (2017)"
    )
    global_attention: Literal["dot", "general", "mlp", None] = Field(
        default="general",
        description="The attention type to use. (Luong=general, Bahdanau=MLP)",
    )
    global_attention_function: Literal["softmax", "sparsemax"] = Field(
        default="softmax", description="Global attention function to use."
    )

    @model_validator(mode="after")
    def _validate_decoder_config(self):
        return self


class RnnConfig(Config):
    """Just to facilitate testing discriminator stuff."""

    bridge: bool = Field(
        default=False,
        description="Have an additional layer between the last encoder state "
        "and the first decoder state (RNN specific).",
    )
    rnn_type: Literal["LSTM", "GRU"] = Field(
        default="LSTM", description="The gate type to use in the RNNs."
    )  # this should be homogenized with available encoder classes (use some registry? or at least a clean Enum somewhere) # noqa: E501


class RnnEncoderConfig(RnnConfig, EncoderConfig):
    encoder_type: Literal["rnn", "brnn"] = Field(default="rnn")


class RnnDecoderConfig(RnnConfig, DecoderConfig):
    decoder_type: Literal["rnn"] = Field(default="rnn")
    context_gate: Literal["source", "target", "both", None] = Field(
        default=None, description="Type of context gate to use."
    )
    bidirectional_encoder: bool | None = False


class CnnConfig(Config):
    cnn_kernel_width: int = Field(
        default=3,
        description="Size of windows in the cnn, the kernel_size is "
        "(cnn_kernel_width, 1) in convolution layers.",
    )


class CnnEncoderConfig(CnnConfig, EncoderConfig):
    encoder_type: Literal["cnn"] = Field(default="cnn")


class CnnDecoderConfig(CnnConfig, DecoderConfig):
    decoder_type: Literal["cnn"] = Field(default="cnn")


class MeanEncoderConfig(EncoderConfig):
    encoder_type: Literal["mean"] = Field(default="mean")


class TransformerConfig(Config):
    """
    This base TransformerConfig class regroups parameters than can
    both be set at model level or either encoder/decoder level.
    BaseModelConfig._override_values validator overrides
    encoder/decoder values with model values if relevant.
    """

    sliding_window: int = Field(
        default=0, description="Sliding window for transformer self-attention."
    )
    heads: int = Field(
        default=8, description="Number of heads for transformer self-attention."
    )
    transformer_ff: int = Field(
        default=2048, description="Size of hidden transformer feed-forward."
    )
    max_relative_positions: int = Field(
        default=0,
        description="This setting enables relative position encoding. "
        "We support two types of encoding: "
        "-1 enables Rotary Embeddings (https://arxiv.org/abs/2104.09864), "
        "> 0 (e.g. 16 or 32) enables maximum distance between inputs "
        "in relative positions representations (https://arxiv.org/pdf/1803.02155.pdf)",
    )
    relative_positions_buckets: int = Field(
        default=0,
        description="Enable relative position bias "
        "(https://github.com/google-research/text-to-text-transfer-transformer).",
    )
    mlp_activation_fn: ActivationFunction = Field(
        default=ActivationFunction.relu,
        description="The activation function to use in MLP layer.",
    )
    rotary_interleave: bool = Field(
        default=True,
        description="Interleave the head dimensions when rotary embeddings are applied. "
        "Otherwise the head dimensions are sliced in half. "
        "(True=default Llama from Meta (original), "
        "False= used by all HuggingFace models)",
    )
    rotary_theta: int = Field(
        default=10000,
        description="Rotary theta base length, 1e4 for Llama2.Mistral, 1e6 for Mixtral",
    )
    rotary_dim: int = Field(
        default=0,
        description="Rotary dim when model requires it to be different to head dim.",
    )
    layer_norm: Literal["standard", "rms"] = Field(
        default="standard",
        description="Type of layer normalization in transformer architecture.",
    )
    norm_eps: float = Field(default=1e-6, description="Layer norm epsilon.")

    shared_layer_norm: bool = Field(
        default=False,
        description="Use a shared layer_norm in parallel residual attention. "
        "Note: must be True for Falcon 7B, False for Falcon 40B, "
        "same for GPT-J and GPT-NeoX models.",
    )
    add_qkvbias: bool = Field(
        default=False,
        description="Add bias to nn.Linear of Query/Key/Value in MHA. "
        "Note: this will add bias to output projection layer too.",
    )
    heads_kv: int | None = Field(
        default=None,
        description="Number of heads for KV. heads_kv=heads if None, else number of heads for KV"
        "(e.g. Falcon 40B)",
    )
    add_ffnbias: bool = Field(
        default=False, description="Add bias to nn.Linear of MLP FFN."
    )
    parallel_residual: bool = Field(
        default=False,
        description="Use parallel residual in decoder layer. "
        "Note: this is used by GPT-J / Falcon Architecture.",
    )
    num_experts: int = Field(default=0, description="Number of experts for MoE models.")
    num_experts_per_tok: int = Field(
        default=2, description="Number of experts per token."
    )


# could eole.encoders.TransformerEncoder class inherit from this? (it seems not unfortunately)
class TransformerEncoderConfig(TransformerConfig, EncoderConfig):
    encoder_type: Literal["transformer"] = Field(
        default="transformer"
    )  # not sure it's fully proper use, but inspired from docs -- https://docs.pydantic.dev/latest/concepts/fields/#discriminator # noqa: E501


class TransformerDecoderConfig(TransformerConfig, DecoderConfig):
    decoder_type: Literal["transformer"] = Field(default="transformer")

    aan_useffn: bool = Field(
        default=False, description="Turn on the FFN layer in the AAN decoder."
    )
    # some guided alignment specific opts were omitted here (put back for backcompat)
    alignment_layer: int = Field(
        default=-2, description="Layer number which has to be supervised."
    )
    alignment_heads: int = Field(
        default=0,
        description="Number of cross attention heads per layer to supervise with.",
    )
    full_context_alignment: bool = Field(
        default=False,
        description="Whether alignment is conditioned on full target context.",
    )
    lambda_align: float = Field(
        default=0.0,
        description="Lambda value for alignement loss of Garg et al, 2019 "
        "(https://arxiv.org/abs/1909.02074)",
    )

    @model_validator(mode="after")
    def _validate_transformer_decoder_config(self):
        # move to specific decoder config ?
        # this is not working right now if enc_layers is overridden by model level config
        # if self.lambda_align > 0.0:
        #     assert (
        #         self.decoder_type == "transformer"
        #     ), "Only transformer is supported to joint learn alignment."
        #     assert (
        #         self.alignment_layer < self.dec_layers
        #         and self.alignment_layer >= -self.dec_layers
        #     ), f"N° alignment_layer ({self.alignment_layer}) should be smaller "\
        #         "than number of layers ({self.dec_layers})."
        #     logger.info(
        #         "Joint learn alignment at layer [{}] "
        #         "with {} heads in full_context '{}'.".format(
        #             self.alignment_layer,
        #             self.alignment_heads,
        #             self.full_context_alignment,
        #         )
        #     )

        assert (
            self.hidden_size % self.heads == 0
        ), "Transformer Model dimension {} must be divisible by the number of heads {}".format(
            self.hidden_size, self.heads
        )
        return self


class TransformerLMDecoderConfig(TransformerDecoderConfig):
    """
    Right now just wraps TransformerDecoderConfig for simplicity.
    Might merge in a single class later once TransformerLM path is clarified.
    """

    decoder_type: Literal["transformer_lm"] = Field(default="transformer_lm")


# use Field with default= + description would be more readable
# was inheriting from VocabConfig, but removed for now to facilitate inference tests
# could we have different BaseModelConfig classes (inheriting from a base one)
# for different architectures / model types?
class BaseModelConfig(Config):
    # some experiments around nested config
    embeddings: EmbeddingsConfig = Field(
        default_factory=EmbeddingsConfig,
        description="Contains most of the args useful to build the Embeddings module.",
    )
    encoder: Union[
        TransformerEncoderConfig,
        RnnEncoderConfig,
        CnnEncoderConfig,
        MeanEncoderConfig,
    ] | None = Field(
        default=None,
        discriminator="encoder_type",
        description="Major parameters of an encoder.",
    )  # we shall use discriminators here
    decoder: Union[
        TransformerDecoderConfig,
        TransformerLMDecoderConfig,
        RnnDecoderConfig,
        CnnDecoderConfig,
    ] | None = Field(
        default=None,
        discriminator="decoder_type",
        description="Major parameters of a decoder.",
    )  # we shall use discriminators here

    # model level stuff that may override enc/dec level
    hidden_size: int = Field(
        default=-1,
        description="Size of hidden states. Overwrites [encoder/decoder].hidden_size if set.",
    )
    word_vec_size: int = Field(
        default=-1, description="Word embedding size for src and tgt."
    )
    layers: int = Field(
        default=-1,
        description="Number of layers in both encoder and decoder "
        "(will overwrite enc_layers/dec_layers).",
    )
    transformer_ff: int = Field(
        default=-1, description="Size of hidden transformer feed-forward."
    )

    share_decoder_embeddings: bool = Field(
        default=False,
        description="Use a share weight matrix for the input and output "
        "word embeddings in the decoder.",
    )
    share_embeddings: bool = Field(
        default=False,
        description="Share the word embeddings between encoder and decoder. "
        "Need to use shared vocabulary for this option.",
    )
    input_feed: int = Field(
        default=1,
        description="Feed the context vector at each time step as additional input "
        "(via concatenation with the word embeddings) to the decoder.",
    )  # is this useful / working anymore?
    generator_function: Literal["softmax", "sparsemax"] = Field(
        default="softmax",
        description="Which function to use for generating probabilities "
        "over the target vocabulary.",
    )
    add_estimator: bool = Field(default=False, description="Add estimator layer")

    left_pad: bool = Field(
        default=False, description="Enable left-padding, useful for some LLMs."
    )

    # @computed_field()
    # @property
    # def brnn(self) -> bool:
    #     if self.encoder is not None:
    #         return self.encoder.encoder_type == "brnn"

    @property
    def model_type(self) -> ModelType:
        if self.decoder is None:
            return ModelType.ENCODER
        elif self.encoder is None:
            return ModelType.DECODER
        else:
            return ModelType.ENCODER_DECODER

    @field_validator("encoder", "decoder", "embeddings", mode="before")
    @classmethod
    def str_to_dict(cls, v) -> Dict:
        if isinstance(v, str):
            # not sure about this patch to allow forcing encoder/decoder to None
            if v == "None":
                return None
            w = eval(v)
            return w
        return v

    # @field_validator("decoder", mode="before")
    # @classmethod
    # def str_to_dict(cls, v) -> Dict:
    #     if isinstance(v, str):
    #         return eval(v)
    #     return v

    architecture: str | None = None

    @model_validator(mode="before")
    @classmethod
    def default_architecture(cls, data: Any) -> Any:
        if not (isinstance(data, dict)):
            return data
        if "architecture" not in data.keys():
            data["architecture"] = "custom"
        return data

    def update_model_opts(self):
        if self.embeddings is not None and self.embeddings.word_vec_size > 0:
            self.embeddings.src_word_vec_size = self.embeddings.word_vec_size
            self.embeddings.tgt_word_vec_size = self.embeddings.word_vec_size

        # Backward compatibility with "fix_word_vecs_*" opts
        # We can probably drop this now...
        # if hasattr(self, "fix_word_vecs_enc"):
        #     self.embeddings.freeze_word_vecs_enc = self.embeddings.fix_word_vecs_enc
        # if hasattr(self, "fix_word_vecs_dec"):
        #     self.embeddings.freeze_word_vecs_dec = self.embeddings.fix_word_vecs_dec

        if (
            getattr(self.encoder, "encoder_type", None) == "brnn"
            and self.decoder.decoder_type == "rnn"
        ):
            self.decoder.bidirectional_encoder = True

        if self.encoder is not None:
            self.encoder.src_word_vec_size = self.embeddings.src_word_vec_size
        if self.decoder is not None:
            self.decoder.tgt_word_vec_size = self.embeddings.tgt_word_vec_size

        # causing some weird recursion issue in unit test, to investigate
        # if self.encoder is not None:
        #     self.brnn = self.encoder.encoder_type == "brnn"

    @model_validator(mode="after")
    def _override_values(self):
        # aggregate all fields in a single dict,
        # and update once to prevent validation mismatch
        embeddings_fields = {}
        encoder_fields = {}
        decoder_fields = {}
        for field in self.model_fields_set:
            if hasattr(self.embeddings, field):
                embeddings_fields[field] = getattr(self, field)
            if hasattr(self.encoder, field):
                encoder_fields[field] = getattr(self, field)
            if hasattr(self.decoder, field):
                decoder_fields[field] = getattr(self, field)
        if self.embeddings is not None:
            self.embeddings.update(**embeddings_fields)
        if self.encoder is not None:
            self.encoder.update(**encoder_fields)
        if self.decoder is not None:
            self.decoder.update(**decoder_fields)
        return self

    @model_validator(mode="after")
    def _validate_model_config(self):
        self.update_model_opts()

        # encoder and decoder should be same sizes
        if self.encoder is not None and self.decoder is not None:
            same_size = self.encoder.hidden_size == self.decoder.hidden_size
            assert (
                same_size
            ), "The encoder and decoder rnns must be the same size for now"

        if self.share_embeddings:
            if self.encoder is None or self.decoder is None:
                raise AssertionError(
                    "--share_embeddings is for EncoderDecoder models only."
                )

        return self


class CustomModelConfig(BaseModelConfig):
    """
    Wrap anything that does not fit a set common architecture.
    """

    architecture: Literal["custom"] = Field(default="custom")


class RnnModelConfig(RnnConfig, BaseModelConfig):
    architecture: Literal["rnn"] = Field(default="rnn")

    @model_validator(mode="before")
    @classmethod
    def encoder_decoder_type(cls, data: Any) -> Any:
        # patch to allow transparent setting of encoder/decoder_type
        if not (isinstance(data, dict)):
            return data
        if "encoder" in data.keys():
            data["encoder"]["encoder_type"] = "rnn"
        else:
            data["encoder"] = {"encoder_type": "rnn"}
        if "decoder" in data.keys():
            data["decoder"]["decoder_type"] = "rnn"
        else:
            data["decoder"] = {"decoder_type": "rnn"}
        return data

    @model_validator(mode="before")
    @classmethod
    def default_architecture(cls, data: Any) -> Any:
        if not (isinstance(data, dict)):
            return data
        if "architecture" not in data.keys():
            data["architecture"] = "rnn"
        return data


class CnnModelConfig(CnnConfig, BaseModelConfig):
    architecture: Literal["cnn"] = Field(default="cnn")

    @model_validator(mode="before")
    @classmethod
    def encoder_decoder_type(cls, data: Any) -> Any:
        # we shall probably factorize this in BaseModelConfig
        # patch to allow transparent setting of encoder/decoder_type
        if not (isinstance(data, dict)):
            return data
        if "encoder" in data.keys():
            data["encoder"]["encoder_type"] = "cnn"
        else:
            data["encoder"] = {"encoder_type": "cnn"}
        if "decoder" in data.keys():
            data["decoder"]["decoder_type"] = "cnn"
        else:
            data["decoder"] = {"decoder_type": "cnn"}
        return data

    @model_validator(mode="before")
    @classmethod
    def default_architecture(cls, data: Any) -> Any:
        if not (isinstance(data, dict)):
            return data
        if "architecture" not in data.keys():
            data["architecture"] = "cnn"
        return data


class TransformerModelConfig(TransformerConfig, BaseModelConfig):
    """
    Facilitate setting some transformer specific params at model level.
    """

    architecture: Literal["transformer"] = Field(default="transformer")

    @model_validator(mode="before")
    @classmethod
    def encoder_decoder_type(cls, data: Any) -> Any:
        # patch to allow transparent setting of encoder/decoder_type
        if not (isinstance(data, dict)):
            return data
        if "encoder" in data.keys():
            data["encoder"]["encoder_type"] = "transformer"
        else:
            data["encoder"] = {"encoder_type": "transformer"}
        if "decoder" in data.keys():
            data["decoder"]["decoder_type"] = "transformer"
        else:
            data["decoder"] = {"decoder_type": "transformer"}
        return data

    @model_validator(mode="before")
    @classmethod
    def default_architecture(cls, data: Any) -> Any:
        if not (isinstance(data, dict)):
            return data
        if "architecture" not in data.keys():
            data["architecture"] = "transformer"
        return data

    @model_validator(mode="after")
    def _validate_transformer(self):
        if (
            getattr(self.embeddings, "position_encoding", False)
            and self.max_relative_positions != 0
        ):
            raise ValueError(
                "Cannot use absolute and relative position encoding at the"
                "same time. Use either --position_encoding=true for legacy"
                "absolute position encoding or --max_realtive_positions with"
                " -1 for Rotary, or > 0 for Relative Position Representations"
                "as in https://arxiv.org/pdf/1803.02155.pdf"
            )
        return self


class TransformerLMModelConfig(TransformerConfig, BaseModelConfig):
    """
    Facilitate setting some transformer specific params at model level.
    """

    architecture: Literal["transformer_lm"] = Field(default="transformer_lm")

    # force encoder to None in TransformerLM case
    encoder: None = Field(default=None, description="Major parameters of an encoder.")

    @model_validator(mode="before")
    @classmethod
    def encoder_decoder_type(cls, data: Any) -> Any:
        # patch to allow transparent setting of encoder/decoder_type
        if not (isinstance(data, dict)):
            return data
        if "decoder" in data.keys():
            data["decoder"]["decoder_type"] = "transformer_lm"
        else:
            data["decoder"] = {"decoder_type": "transformer_lm"}
        return data

    @model_validator(mode="before")
    @classmethod
    def default_architecture(cls, data: Any) -> Any:
        if not (isinstance(data, dict)):
            return data
        if "architecture" not in data.keys():
            data["architecture"] = "transformer_lm"
        return data

    @model_validator(mode="after")
    def _validate_transformer(self):
        # duplicate with TransformerModelConfig, might merge at some point
        if (
            getattr(self.embeddings, "position_encoding", False)
            and self.max_relative_positions != 0
        ):
            raise ValueError(
                "Cannot use absolute and relative position encoding at the"
                "same time. Use either --position_encoding=true for legacy"
                "absolute position encoding or --max_realtive_positions with"
                " -1 for Rotary, or > 0 for Relative Position Representations"
                "as in https://arxiv.org/pdf/1803.02155.pdf"
            )
        return self


class TransformerEncoderModelConfig(TransformerConfig, BaseModelConfig):
    """
    Facilitate setting some transformer specific params at model level.
    """

    architecture: Literal["transformer_encoder"] = Field(default="transformer_encoder")

    # force decoder to None in TransformerEncoder case
    decoder: None = Field(default=None, description="Major parameters of a decoder.")

    @model_validator(mode="before")
    @classmethod
    def encoder_decoder_type(cls, data: Any) -> Any:
        # patch to allow transparent setting of encoder/decoder_type
        if not (isinstance(data, dict)):
            return data
        if "encoder" in data.keys():
            data["encoder"]["encoder_type"] = "transformer"
        else:
            data["encoder"] = {"encoder_type": "transformer"}
        return data

    @model_validator(mode="before")
    @classmethod
    def default_architecture(cls, data: Any) -> Any:
        if not (isinstance(data, dict)):
            return data
        if "architecture" not in data.keys():
            data["architecture"] = "transformer_encoder"
        return data

    @model_validator(mode="after")
    def _validate_transformer(self):
        # duplicate with TransformerModelConfig, might merge at some point
        if (
            getattr(self.embeddings, "position_encoding", False)
            and self.max_relative_positions != 0
        ):
            raise ValueError(
                "Cannot use absolute and relative position encoding at the"
                "same time. Use either --position_encoding=true for legacy"
                "absolute position encoding or --max_realtive_positions with"
                " -1 for Rotary, or > 0 for Relative Position Representations"
                "as in https://arxiv.org/pdf/1803.02155.pdf"
            )
        return self


# Facilitate transparent instanciation from dumped fields
# https://stackoverflow.com/a/76538571
ModelConfig = Annotated[
    Union[
        TransformerModelConfig,
        TransformerLMModelConfig,
        TransformerEncoderModelConfig,
        RnnModelConfig,
        CnnModelConfig,
        CustomModelConfig,
    ],
    Field(discriminator="architecture", default_factory=RnnModelConfig),  # noqa: F821
]

# Not used anymore, keeping for reference
# build_model_config = TypeAdapter(ModelConfig).validate_python
