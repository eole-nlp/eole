import torch
from typing import List, Literal
from pydantic import Field, model_validator
from importlib import import_module
from eole.config.config import Config

# from eole.utils.logging import logger


class DistributedConfig(Config):
    gpu_ranks: List[int] = Field(
        default=[], description="List of ranks for each process."
    )
    world_size: int = Field(
        default=1, description="Total number of distributed processes."
    )
    parallel_mode: Literal["data_parallel", "tensor_parallel"] = Field(
        default="data_parallel", description="Distributed mode."
    )
    gpu_backend: str = Field(
        default="nccl", description="Type of torch distributed backend."
    )
    gpu_verbose_level: int = Field(
        default=0, description="Gives more info on each process per GPU."
    )
    master_ip: str = Field(
        default="localhost", description="IP of master for torch.distributed training."
    )
    master_port: int = Field(
        default=10000, description="Port of master for torch.distributed training."
    )
    timeout: int = Field(
        default=60, description="Timeout for one GPU to wait for the others."
    )

    @property
    def parallel_gpu(self) -> int:  # converted to a `property` by `computed_field`
        return self.world_size if self.parallel_mode == "tensor_parallel" else 1

    # This could be a good idea, but requires more safeguards, e.g. for inference
    # @model_validator(mode="after")
    # def _validate_distributed_config(self):
    #     if torch.cuda.is_available() and "gpu_ranks" not in self.model_fields_set:
    #         logger.warning(
    #             "GPUs detected, using them all. "
    #             "Explicitly set gpu_ranks/world_size to change this behaviour.")
    #         num_gpus = torch.cuda.device_count()
    #         self.world_size = num_gpus
    #         self.gpu_ranks = list(range(num_gpus))
    #     return self


class LoRaConfig(Config):
    lora_layers: List[str] = Field(
        default=[],
        description="List of layers to be replaced by LoRa layers. "
        "E.g. ['linear_values', 'linear_query'] "
        "(§4.2 in https://arxiv.org/abs/2106.09685)",
    )  # validate against existing layers?
    lora_embedding: bool = Field(
        default=False, description="Replace embeddings with LoRa Embeddings (§5.1)"
    )
    lora_rank: int = Field(
        default=2, description="r=2 successfully tested with NLLB-200 3.3B"
    )
    lora_alpha: int = Field(
        default=1, description="§4.1 https://arxiv.org/abs/2106.09685"
    )
    lora_dropout: float = Field(
        default=0.0, description="Rule of thumb: same value as in main model."
    )  # should probably be in TrainingConfig in fine to be coherent with dropout/attention_dropout


class QuantizeConfig(Config):
    # Quantization related options (previously in MiscConfig)
    quant_layers: List[str] = Field(
        default=[], description="List of layers to be compressed in 4/8bit."
    )  # validate against list of layers names ?
    quant_type: Literal[
        "", "bnb_9bit", "bnb_FP4", "bnb_NF4", "awq_gemm", "awq_gemv"
    ] = Field(default="", description="Type of compression.")
    w_bit: int = Field(
        default=4, description="W_bit quantization"
    )  # single authorized value for now actually
    group_size: int = Field(default=128, description="Group size quantization.")  # same


class MiscConfig(Config):
    # Some opts not super sure where to put
    # Seed stuff
    seed: int = Field(
        default=-1,
        description="Set random seed used for better reproducibility between experiments.",
    )


class LoggingConfig(Config):
    log_file: str = Field(
        default="", description="Output logs to a file under this path."
    )
    # some "observability" stuff
    report_every: int = Field(
        default=50, description="Print stats at this interval (in steps)."
    )
    valid_metrics: List[str] = Field(
        default=[], description="List of names of additional validation metrics."
    )  # should probably be validated properly (or in some registry/enum)
    scoring_debug: bool = Field(
        default=False, description="Dump src/ref/pred of the current batch."
    )
    dump_preds: str | None = Field(
        default=None, description="Folder to dump predictions to."
    )
    tensorboard: bool = Field(
        default=False, description="Use tensorboard for visualization during training."
    )
    tensorboard_log_dir: str = Field(
        default="runs/eole",
        description="Log directory for tensorboard (also the name of the run).",
    )
    tensorboard_log_dir_dated: str | None = (
        None  # field dynamically populated in build_report_manager, might improve later
    )


class RunningConfig(DistributedConfig):
    """
    Should be used as base class for Training/Inference at some point,
    factorizing common stuff like batch_size etc.
    """

    model_path: str = Field(
        default="model",
        description="Path to directory containing all model components.",
    )
    self_attn_backend: Literal["flash", "pytorch"] = Field(
        default="flash",
        description="Self-attention backend.",
    )

    @model_validator(mode="after")
    def _validate_running_config(self):
        try:
            flash_pack = import_module("flash_attn")
            if (
                hasattr(flash_pack, "flash_attn_func")
                and torch.cuda.get_device_capability()[0] >= 8
                and self.self_attn_backend == "flash"
            ):
                pass
            else:
                self.__dict__["self_attn_backend"] = "pytorch"
        except ImportError:
            self.__dict__["self_attn_backend"] = "pytorch"
        return self
