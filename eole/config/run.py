import os
import json
from typing import Dict, List, Any

from eole.config import recursive_model_fields_set
from eole.config.config import get_config_dict
from eole.config.training import TrainingConfig
from eole.config.inference import InferenceConfig
from eole.config.common import MiscConfig, LoggingConfig
from eole.config.models import ModelConfig, build_model_config
from eole.config.data import (
    DataConfig,
    BaseVocabConfig,
    VocabConfig,
    NestedAllTransformsConfig,
)
from eole.transforms import get_transforms_cls
from eole.constants import TransformType
from eole.utils.logging import logger
from pydantic import Field, field_validator, model_validator
import torch


# Models below somewhat replicate prior opt behavior to facilitate transition
class TrainConfig(LoggingConfig, MiscConfig, DataConfig, VocabConfig):  # ModelConfig, TrainingConfig
    n_sample: int = Field(
        default=0,
        description="Number of transformed samples per corpus to use to build the vocabulary. "
        "Set to -1 to use the full corpora.",
    )  # not sure how to handle the legacy build_vocab_only flag here (different default value in both cases) # noqa: E501
    verbose: bool = Field(
        default=False,
        description="Print data loading and statistics for all process " "(default only logs the first process shard).",
    )  # not sure this still works
    model: ModelConfig | None = None  # TypeAdapter handling discrimination directly
    training: TrainingConfig | None = Field(default_factory=TrainingConfig)
    inference: InferenceConfig | None = Field(default=None)

    def get_model_path(self):
        return self.training.get_model_path()

    @property
    def data_type(self):
        return self.training.data_type

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
        if torch.cuda.is_available() and not self.training.gpu_ranks:
            logger.warn("You have a CUDA device, should run with -gpu_ranks")
        return self


class PredictConfig(
    InferenceConfig,
    LoggingConfig,
    MiscConfig,
):
    transforms: List[str] | None = []
    transforms_configs: NestedAllTransformsConfig | None = Field(default_factory=NestedAllTransformsConfig)
    # patch to make it work
    share_vocab: bool | None = False
    _all_transform: List[str] | None = None  # might be a cleaner way to do this
    src_subword_vocab: str | None = None  # patch for CT2 inference engine (to improve later)
    model: ModelConfig | None = None
    model_path: str | List[str] = Field(
        description="Path to model .pt file(s). " "Multiple models can be specified for ensemble decoding."
    )  # some specific (mapping to "models") in legacy code, need to investigate
    src: str = Field(description="Source file to decode (one line per sequence).")
    tgt: str | None = Field(
        default=None,
        description="True target sequences, useful for scoring or prefix decoding.",
    )
    tgt_file_prefix: bool = Field(default=False, description="Generate predictions using provided tgt as prefix.")
    output: str = Field(
        default="pred.txt",
        description="Path to output the predictions (each line will be the decoded sequence).",
    )
    engine: str = Field(
        default="eole",
        description="engine to run inference: eole or ct2",
    )

    @model_validator(mode="after")
    def _validate_predict_config(self):
        # Not sure we want to call this at every validation
        self._update_with_model_config()
        self.check_self_attn_backend()
        # TODO: do we really need this _all_transform?
        if self._all_transform is None:
            self._all_transform = self.transforms
        if torch.cuda.is_available() and not self.gpu_ranks:
            logger.warn("You have a CUDA device, should run with -gpu_ranks")
        return self

    def _update_with_model_config(self):
        # Note: in case of ensemble decoding, grabbing the first model's
        # config and artifacts by default
        os.environ["MODEL_PATH"] = self.model_path[0]
        config_path = os.path.join(self.model_path[0], "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_dict = json.loads(os.path.expandvars(f.read()))
        else:
            config_dict = {}
        # Filter out Train transforms
        transforms = config_dict.get("transforms", [])
        transforms_cls = get_transforms_cls(transforms)
        transforms = [t for t in transforms if transforms_cls[t].type != TransformType.Train]

        if os.path.exists(config_path):
            # logic from models.BaseModel.inference_logic
            model_config = build_model_config(config_dict.get("model", {}))
            training_config = TrainingConfig(**config_dict.get("training", {}))
            training_config.world_size = self.world_size
            training_config.gpu_ranks = self.gpu_ranks
            # retrieve share_vocab from checkpoint config
            self.__dict__["share_vocab"] = config_dict.get("share_vocab", False)
            # retrieve precision from checkpoint config if not explicitly set
            if "compute_dtype" not in self.model_fields_set:
                self.compute_dtype = training_config.compute_dtype
            # quant logic, might be better elsewhere
            if hasattr(training_config, "quant_type") and training_config.quant_type in [
                "awq_gemm",
                "awq_gemv",
            ]:
                if (
                    hasattr(self, "quant_type")
                    and self.quant_type != ""
                    and self.quant_type != training_config.quant_type
                ):
                    raise ValueError("Model is a awq quantized model, cannot overwrite with another quant method")
                self.update(quant_type=training_config.quant_type)
            elif self.quant_type == "" and training_config.quant_type != "":
                self.update(
                    quant_layers=training_config.quant_layers,
                    quant_type=training_config.quant_type,
                )

            self.update(
                model=model_config,
            )

        update_dict = {}
        if "transforms" not in self.model_fields_set:
            update_dict["transforms"] = transforms
            update_dict["_all_transform"] = transforms
        if "transforms_configs" not in self.model_fields_set:
            update_dict["transforms_configs"] = NestedAllTransformsConfig(**config_dict.get("transforms_configs", {}))
        else:
            # merge model transforms_configs with explicitly set config
            # useful for cases like prefix/suffix that might not be set in the checkpoint
            transforms_configs = recursive_model_fields_set(self.transforms_configs)
            transforms_configs.update(config_dict.get("transforms_configs", {}))
            update_dict["transforms_configs"] = NestedAllTransformsConfig(**transforms_configs)
        if "compute_dtype" not in self.model_fields_set:
            self.compute_dtype = config_dict.get("training", {}).get("compute_dtype", "fp16")
        for key, value in config_dict.get("inference", {}).items():
            if key not in self.model_fields_set:
                update_dict[key] = value
        self.update(**update_dict)


PredictConfig.model_rebuild()


class BuildVocabConfig(DataConfig, MiscConfig, BaseVocabConfig):  # , AllTransformsConfig):

    model_config = get_config_dict()
    model_config["extra"] = "ignore"

    # some specific items related to build_vocab_only flag
    dump_samples: bool = Field(
        default=False,
        description="Dump samples when building vocabulary. " "Warning: this may slow down the process.",
    )
    num_threads: int = Field(default=1, description="Number of parallel threads to build the vocabulary.")
    learn_subwords: bool = Field(
        default=False,
        description="Learn subwords (based on defined transforms) prior to building vocabulary.",
    )
    learn_subwords_size: int = Field(default=32000, description="Number of subwords operations to learn.")
    vocab_sample_queue_size: int = Field(default=20, description="Size of queues used for dumping samples.")
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
                "The onmt_tokenizer transform should be set " "and configured to enable learn_subwords."
            )
