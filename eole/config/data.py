import os
from typing import Dict, List, Literal
from pydantic import Field, field_validator, model_validator
from pydantic import create_model

from eole import constants
from eole.utils.logging import logger
from eole.transforms import AVAILABLE_TRANSFORMS
from eole.config.config import Config


class BaseVocabConfig(Config):
    src_vocab: str | None = Field(
        description="Path to src (or shared) vocabulary file. "
        "Format: one <word> or <word>\t<count> per line."
    )
    tgt_vocab: str | None = Field(
        default=None,
        description="Path to tgt vocabulary file. Format: one <word> or <word>\t<count> per line.",
    )
    share_vocab: bool = Field(
        default=False, description="Share source and target vocabulary."
    )
    decoder_start_token: str = Field(
        default=constants.DefaultTokens.BOS,
        description="Default decoder start token. For most models it is <s> = BOS. "
        "Some fairseq models require </s>.",
    )
    default_specials: list = Field(
        default=[
            constants.DefaultTokens.UNK,
            constants.DefaultTokens.PAD,
            constants.DefaultTokens.BOS,
            constants.DefaultTokens.EOS,
        ],
        description="Default specials used for vocab initialization. "
        "UNK, PAD, BOS, EOS will take IDs 0, 1, 2, 3.",
    )
    # pre trained embeddings stuff, might be put elsewhere
    both_embeddings: str | None = Field(
        default=None,
        description="Path to the embeddings file to use for both source and target tokens.",
    )
    src_embeddings: str | None = Field(
        default=None,
        description="Path to the embeddings file to use for source tokens.",
    )
    tgt_embeddings: str | None = Field(
        default=None,
        description="Path to the embeddings file to use for target tokens.",
    )
    embeddings_type: Literal["GloVe", "word2vec"] | None = Field(
        default=None, description="Type of embeddings file."
    )


class VocabConfig(BaseVocabConfig):
    src_vocab_size: int = Field(
        default=32758, description="Maximum size of the source vocabulary."
    )
    tgt_vocab_size: int = Field(
        default=32768, description="Maximum size of the target vocabulary."
    )
    vocab_size_multiple: int = Field(
        default=8,
        description="Make the vocabulary size a multiple of this value. "
        "(Adds dummy tokens if needed.)",
    )
    src_words_min_frequency: int = Field(
        default=0, description="Discard source words with lower frequency."
    )
    tgt_words_min_frequency: int = Field(
        default=0, description="Discard target words with lower frequency."
    )


class Dataset(Config):
    name: str | None = None  # we could put some auto name generation in validation (from path, uuid ,etc.) # noqa: E501
    weight: int | None = 1
    # streams: List[Stream] # maybe implemented later
    transforms: List[str] | None = None
    path_src: str | None = None
    path_tgt: str | None = None
    path_sco: str | None = None
    path_txt: str | None = (
        None  # not sure we need this one with proper path_src handling
    )
    path_align: str | None = None
    # optional stuff for some transforms
    src_prefix: str | None = None
    tgt_prefix: str | None = None
    src_suffix: str | None = None
    tgt_suffix: str | None = None


# add all opts from all transforms (like in eole.opts._add_transform_opt)
# maybe we want to restrict to the transforms actually used here to prevent overcrowding the config

# nested way of storing transform configs, for clarity
transforms_configs = {}
for name, transform_cls in AVAILABLE_TRANSFORMS.items():
    if hasattr(transform_cls, "config_model"):
        transforms_configs[name] = (
            transform_cls.config_model,
            transform_cls.config_model(),
        )


NestedAllTransformsConfig = create_model(
    "NestedAllTransformsConfig",
    __base__=(Config,),
    **transforms_configs,
)


class DataConfig(VocabConfig):  # , AllTransformsConfig):
    data: Dict[str, Dataset] | None = Field(
        description="All datasets and their specifications. "
        "See examples/*.yaml for further details."
    )
    transforms: List[str] = Field(
        default=[],
        description="Default transform pipeline to apply to data. "
        "Can be specified in each corpus of data to override.",
    )  # this should be validated against AVAILABLE_TRANSFORMS.keys() (Enum ?)
    transforms_configs: NestedAllTransformsConfig | None = Field(
        default_factory=NestedAllTransformsConfig
    )
    _all_transforms: List[str]
    # NameError: Fields must not use names with leading underscores
    # _all_transforms: List[str] = Field(
    #     default=None,
    #     description="Dummy field, programmatically deduced from "
    #                 "general and corpus-specific transforms."
    # )  # might be a cleaner way to do this
    skip_empty_level: Literal["silent", "warning", "error"] = Field(
        default="warning",
        description="Logging level when encoutering empty examples. "
        "(silent: silently ignore/skip empty examples, "
        "warning: warn when ignoring/skipping empty examples, "
        "error: raise an error and stop execution when any empty example)",
    )
    # patch to allow test_data_prepare to pass for now
    n_sample: int | None = 0  # mostly overridden in Vocab/Train configs
    save_data: str | None = Field(
        default=None,
        description="Output base path for objects that will be saved "
        "(vocab, transforms, embeddings, ...)",
    )  # "required" depending on build_vocab_only originally
    overwrite: bool = Field(
        default=False, description="Overwrite existing objects if any."
    )

    @field_validator("transforms_configs", mode="before")
    @classmethod
    def _str_to_dict(cls, v) -> Dict:
        if isinstance(v, str):
            if v == "None":
                return None  # not sure about this patch to allow forcing encoder/decoder to None (e.g. LM) # noqa: E501
            w = eval(v)
            return w
        return v

    def _get_all_transform(self):
        # might be good to rename at some point
        all_transforms = set(self.transforms)
        if self.data is not None:
            for cname, corpus in self.data.items():
                _transforms = set(corpus.transforms)
                if len(_transforms) != 0:
                    all_transforms.update(_transforms)
        if (
            hasattr(self, "model")
            and getattr(getattr(self.model, "decoder", None), "lambda_align", 0.0) > 0.0
        ):
            if not all_transforms.isdisjoint({"sentencepiece", "bpe", "onmt_tokenize"}):
                raise ValueError(
                    "lambda_align is not compatible with" " on-the-fly tokenization."
                )
            if not all_transforms.isdisjoint({"tokendrop", "prefix", "bart"}):
                raise ValueError(
                    "lambda_align is not compatible yet with"
                    " potentiel token deletion/addition."
                )
        self._all_transform = all_transforms

    def _validate_vocab_config(self, build_vocab_only=False):
        # patch for n_sample default value around here?

        if build_vocab_only:
            if not self.share_vocab:
                assert self.tgt_vocab, "-tgt_vocab is required if not -share_vocab."
            return
        # validation when train:
        self.__class__._validate_file(self.src_vocab, info="src vocab")
        if not self.share_vocab:
            self.__class__._validate_file(self.tgt_vocab, info="tgt vocab")

        # Check embeddings stuff
        if self.both_embeddings is not None:
            assert (
                self.src_embeddings is None and self.tgt_embeddings is None
            ), "You don't need -src_embeddings or -tgt_embeddings \
                if -both_embeddings is set."

        if any(
            [
                self.both_embeddings is not None,
                self.src_embeddings is not None,
                self.tgt_embeddings is not None,
            ]
        ):
            assert (
                self.embeddings_type is not None
            ), "You need to specify an -embeddings_type!"
            assert (
                self.save_data
            ), "-save_data should be set if use \
                pretrained embeddings."

    @staticmethod
    def _validate_file(file_path, info):
        """Check `file_path` is valid or raise `IOError`."""
        if file_path == "dummy":
            # hack to allow creating objects with required fields
            pass
        elif not os.path.isfile(file_path):
            raise IOError(f"Please check path of your {info} file! ({file_path})")

    def _validate_data(self):
        """Parse corpora specified in data field of YAML file."""
        default_transforms = self.transforms
        if len(default_transforms) != 0 and len(self.data) > 0:
            # 0 datasets means we're loading a converted model
            logger.info(
                f"Default transforms (might be overridden downstream): {default_transforms}."
            )

        corpora = self.data

        for cname, corpus in corpora.items():
            # Check Transforms
            _transforms = corpus.transforms
            if _transforms is None:
                logger.info(
                    f"Missing transforms field for {cname} data, "
                    f"set to default: {default_transforms}."
                )
                corpus.transforms = default_transforms
            # Check path
            if corpus.path_src is None:
                raise ValueError(
                    f"Corpus {cname} src path is required."
                    "tgt path is also required for non language"
                    " modeling tasks."
                )
            else:
                self.__class__._validate_file(corpus.path_src, info=f"{cname}/path_src")
                if corpus.path_tgt is None:
                    logger.debug(
                        "path_tgt is None, it should be set unless the task"
                        " is language modeling"
                    )
                    # tgt is src for LM task
                    # corpus["path_tgt"] = path_src
                    corpora[cname] = corpus
                    # path_tgt = path_src
                else:
                    self.__class__._validate_file(
                        corpus.path_tgt, info=f"{cname}/path_tgt"
                    )
            if corpus.path_align is None:
                if (
                    hasattr(self, "model")
                    and getattr(
                        getattr(self.model, "decoder", None), "lambda_align", 0.0
                    )
                    > 0.0
                ):
                    raise ValueError(
                        f"Corpus {cname} alignment file path are "
                        "required when lambda_align > 0.0"
                    )
                corpus.path_align = None
            else:
                self.__class__._validate_file(
                    corpus.path_align, info=f"{cname}/path_align"
                )
            # Check weight
            if corpus.weight is None:
                if cname != constants.CorpusName.VALID:
                    logger.warning(
                        f"Corpus {cname}'s weight should be given."
                        " We default it to 1 for you."
                    )
                corpus.weight = 1
        if len(corpora) > 0:
            logger.info(f"Parsed {len(corpora)} corpora from -data.")
        # self.data = corpora
        self.__dict__["data"] = corpora  # skip validation to avoid recursion error

    @model_validator(mode="after")
    def _validate_data_config(self, build_vocab_only=False):
        if self.data is not None:  # patch to allow None data
            self._validate_data()
        self._get_all_transform()
        # this is manually triggered where needed, to allow instanciation of
        # TrainConfig without existing files (e.g. inference)
        # self._validate_vocab_config(build_vocab_only=build_vocab_only)
        return self
