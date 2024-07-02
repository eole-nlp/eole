from typing import Dict, List, Any

from eole.config.config import get_config_dict
from eole.config.training import TrainingConfig
from eole.config.inference import InferenceConfig
from eole.config.common import MiscConfig, LoggingConfig
from eole.config.models import ModelConfig
from eole.config.data import (
    DataConfig,
    BaseVocabConfig,
    VocabConfig,
    NestedAllTransformsConfig,
)
from pydantic import Field, field_validator, model_validator


# Models below somewhat replicate prior opt behavior to facilitate transition
class TrainConfig(
    LoggingConfig, MiscConfig, DataConfig, VocabConfig
):  # ModelConfig, TrainingConfig
    n_sample: int = Field(
        default=0,
        description="Number of transformed samples per corpus to use to build the vocabulary. "
        "Set to -1 to use the full corpora.",
    )  # not sure how to handle the legacy build_vocab_only flag here (different default value in both cases) # noqa: E501
    override_opts: bool = Field(
        default=False, description="Allow to override some checkpoint opts."
    )  # this should probably be clarified down the line
    verbose: bool = Field(
        default=False,
        description="Print data loading and statistics for all process "
        "(default only logs the first process shard).",
    )  # not sure this still works
    model: ModelConfig | None = None  # TypeAdapter handling discrimination directly
    training: TrainingConfig | None = Field(default_factory=TrainingConfig)

    def get_model_path(self):
        return self.training.get_model_path()

    @classmethod
    def get_defaults(cls, architecture):
        return cls(
            src_vocab="dummy",
            tgt_vocab="dummy",
            data={},
            model={"architecture": architecture},
        ).model_dump()

    @field_validator("model", "training", mode="before")
    @classmethod
    def str_to_dict(cls, v) -> Dict:
        if isinstance(v, str):
            if v == "None":
                return None  # not sure about this patch to allow forcing encoder/decoder to None (e.g. LM) # noqa: E501
            w = eval(v)
            return w
        return v

    @model_validator(mode="before")
    @classmethod
    def default_architecture(cls, data: Any) -> Any:
        # only enforce default "custom" if some model settings are passed
        if "model" in data.keys():
            data["model"] = cls.str_to_dict(data["model"])
            if isinstance(data.get("model", {}), dict):
                if data.get("model", {}).get("architecture", None) is None:
                    data["model"]["architecture"] = "custom"
        return data

    @model_validator(mode="after")
    def _validate_train_config(self):
        if self.model is None and self.training.train_from is None:
            raise ValueError(
                "No model architecture is configured. "
                "You should either finetune from an existing model, "
                "or specify a model configuration."
            )
        if self.n_sample != 0:
            assert (
                self.save_data
            ), "-save_data should be set if \
                want save samples."
        return self


class PredictConfig(
    InferenceConfig,
    LoggingConfig,
    MiscConfig,
):
    transforms: List[str] | None = []
    transforms_configs: NestedAllTransformsConfig | None = Field(
        default_factory=NestedAllTransformsConfig
    )
    # patch to make it work
    share_vocab: bool | None = False
    _all_transform: List[str] | None = None  # might be a cleaner way to do this
    src_subword_vocab: str | None = (
        None  # patch for CT2 inference engine (to improve later)
    )
    model: ModelConfig | None = None

    @model_validator(mode="after")
    def _validate_predict_config(self):
        if self._all_transform is None:
            self._all_transform = self.transforms
        return self


class BuildVocabConfig(
    DataConfig, MiscConfig, BaseVocabConfig
):  # , AllTransformsConfig):

    model_config = get_config_dict()
    model_config["extra"] = "ignore"

    # some specific items related to build_vocab_only flag
    dump_samples: bool = Field(
        default=False,
        description="Dump samples when building vocabulary. "
        "Warning: this may slow down the process.",
    )
    num_threads: int = Field(
        default=1, description="Number of parallel threads to build the vocabulary."
    )
    learn_subwords: bool = Field(
        default=False,
        description="Learn subwords (based on defined transforms) prior to building vocabulary.",
    )
    learn_subwords_size: int = Field(
        default=32000, description="Number of subwords operations to learn."
    )
    vocab_sample_queue_size: int = Field(
        default=20, description="Size of queues used for dumping samples."
    )
    n_sample: int = Field(
        default=5000,
        description="Number of transformed samples per corpus to use to build the vocabulary. "
        "Set to -1 to use the full corpora.",
    )  # not sure how to handle the legacy build_vocab_only flag here (different default value in both cases) # noqa: E501

    # We might want to validate that we have onmt_tokenize transform if learn_subwords...
    @model_validator(mode="after")
    def _validate_build_vocab_config(self):
        if self.learn_subwords:
            assert "onmt_tokenize" in self.transforms, (
                "The onmt_tokenizer transform should be set "
                "and configured to enable learn_subwords."
            )
