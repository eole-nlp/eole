#!/usr/bin/env python
"""Training on a single process."""
import torch
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from eole.utils.logging import init_logger, logger
from eole.config.run import TrainConfig
from eole.constants import CorpusTask
from eole.transforms import (
    make_transforms,
    get_specials,
    get_transforms_cls,
)
from eole.inputters.inputter import vocabs_to_dict, build_vocab, dict_to_vocabs
from eole.inputters.dynamic_iterator import build_dynamic_dataset_iter
from eole.inputters.text_corpus import save_transformed_sample
from eole.models.model_saver import get_metadata
from eole.utils.optimizers import Optimizer
from eole.utils.misc import set_random_seed, configure_cuda_backends
from eole.trainer import build_trainer
from eole.models.model_saver import build_model_saver
from eole.models.model import get_model_class
from eole.modules.embeddings import prepare_pretrained_embeddings
from eole.config import (
    recursive_model_fields_set,
    recursive_update_dict,
)


@dataclass
class TrainingContext:
    """Container for training initialization results."""

    metadata: Optional[Dict[str, Any]]
    vocabs: Dict[str, Any]
    transforms: Any
    config: TrainConfig


class TrainingInitializer:
    """Handles training initialization logic."""

    def __init__(self, config: TrainConfig):
        self.config = config

    def initialize(self) -> TrainingContext:
        """Initialize training components."""
        metadata = self._load_metadata()
        config = self._update_config_with_metadata(metadata)
        config._validate_data_config()

        transforms_cls = get_transforms_cls(config._all_transform)
        vocabs, transforms = self._prepare_vocabs_and_transforms(config, transforms_cls, metadata)

        self._log_vocab_info(vocabs, config)

        return TrainingContext(metadata=metadata, vocabs=vocabs, transforms=transforms, config=config)

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load metadata from checkpoint if resuming training."""
        if not self.config.training.train_from:
            return None

        metadata = get_metadata(self.config.training.train_from)
        return metadata

    def _prepare_vocabs_and_transforms(
        self, config: TrainConfig, transforms_cls: Any, metadata: Optional[Dict[str, Any]]
    ) -> Tuple[Dict, Any]:
        """Prepare vocabularies and transforms."""
        specials = get_specials(config, transforms_cls)
        vocabs = build_vocab(config, specials)

        prepare_pretrained_embeddings(config, vocabs)
        transforms = make_transforms(config, transforms_cls, vocabs)

        # Handle sample saving
        if config.n_sample != 0:
            self._save_sample_and_exit(config, transforms)

        # Use metadata vocabs if not updating
        if config.training.train_from and not config.training.update_vocab:
            logger.info("Keeping metadata vocabulary")
            vocabs = dict_to_vocabs(metadata["vocab"])

        return vocabs, transforms

    def _save_sample_and_exit(self, config: TrainConfig, transforms: Any):
        """Save transformed samples and exit."""
        logger.warning(
            f"`-n_sample` != 0: Training will not be started. " f"Stop after saving {config.n_sample} samples/corpus."
        )
        save_transformed_sample(config, transforms, n_sample=config.n_sample)
        logger.info("Sample saved, please check it before restart training.")
        sys.exit()

    def _update_config_with_metadata(self, metadata: Optional[Dict[str, Any]]) -> TrainConfig:
        """Update configuration with metadata from checkpoint."""
        if metadata is None:
            return self.config

        defaults = TrainConfig.get_defaults(architecture=metadata["config"].model.architecture)
        metadata_non_defaults = recursive_model_fields_set(metadata["config"])
        new_config = recursive_model_fields_set(self.config)

        # Override data explicitly
        metadata_non_defaults["data"] = new_config["data"]

        # Handle tensorboard log directory
        if new_config.get("tensorboard_log_dir") == metadata_non_defaults.get("tensorboard_log_dir") and hasattr(
            metadata_non_defaults, "tensorboard_log_dir_dated"
        ):
            new_config.tensorboard_log_dir_dated = metadata_non_defaults["tensorboard_log_dir_dated"]

        updated_config = recursive_update_dict(metadata_non_defaults, new_config, defaults)
        return TrainConfig(**updated_config)

    def _log_vocab_info(self, vocabs: Dict, config: TrainConfig):
        """Log vocabulary information."""
        vocab_dict = vocabs_to_dict(vocabs)
        logger.info(f"The first 10 tokens of the vocabs are: {vocab_dict['src'][0:10]}")
        logger.info(f"The decoder start token is: {config.decoder_start_token}")

        special_tokens = {
            "bos_token": config.bos_token,
            "eos_token": config.eos_token,
            "pad_token": config.pad_token,
            "unk_token": config.unk_token,
        }

        for token_name, token_value in special_tokens.items():
            token_id = vocabs["src"]([token_value])
            logger.info(f"{token_name} token is: {token_value} id: {token_id}")


class ValidationConfigurer:
    """Handles validation-specific configuration."""

    @staticmethod
    def configure_validation_transforms(config: TrainConfig):
        """Copy validation transforms to main config if needed."""
        validset_transforms = getattr(config.data.get("valid", None), "transforms", None)
        if not validset_transforms:
            return

        config.transforms = validset_transforms
        valid_data = config.data.get("valid", {})

        # Configure prefix transforms
        if valid_data.tgt_prefix:
            config.transforms_configs.prefix.tgt_prefix = valid_data.tgt_prefix
        if valid_data.src_prefix:
            config.transforms_configs.prefix.src_prefix = valid_data.src_prefix

        # Configure suffix transforms
        if valid_data.tgt_suffix:
            config.transforms_configs.suffix.tgt_suffix = valid_data.tgt_suffix
        if valid_data.src_suffix:
            config.transforms_configs.suffix.src_suffix = valid_data.src_suffix


def configure_process(config: TrainConfig, device_id: int):
    """Configure the training process for the given device."""
    if device_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    set_random_seed(config.seed, device_id >= 0)


def main(config: TrainConfig, device_id: int):
    """Start training on `device_id`."""
    configure_process(config, device_id)
    if device_id == -1 or config.training.gpu_ranks[device_id] == 0:
        init_logger(config.log_file)
    else:
        import logging

        init_logger(config.log_file, log_level=logging.WARNING)

    # Initialize training components
    initializer = TrainingInitializer(config)
    context = initializer.initialize()

    configure_cuda_backends()

    # Configure validation transforms
    ValidationConfigurer.configure_validation_transforms(context.config)

    # Build model
    model = get_model_class(context.config.model).for_training(
        context.config.model,
        context.vocabs,
        context.config.training,
        metadata=context.metadata,
        device_id=device_id,
    )

    if context.config.training.torch_compile:
        torch._dynamo.config.cache_size_limit = 16
        model = torch.compile(model, dynamic=True)

    model.count_parameters(log=logger.info)

    logger.info(f" * src vocab size = {len(context.vocabs['src'])}")
    logger.info(f" * tgt vocab size = {len(context.vocabs['tgt'])}")

    # Build optimizer
    optim = Optimizer.from_config(model, context.config, metadata=context.metadata)

    # Build model saver
    model_saver = build_model_saver(context.config, model, context.vocabs, optim, device_id, context.transforms)

    # Build trainer
    trainer = build_trainer(context.config, device_id, model, context.vocabs, optim, model_saver=model_saver)

    # Build data iterators
    offset, stride = _calculate_parallel_params(device_id, context.config.training)

    train_iter = build_dynamic_dataset_iter(
        context.config,
        context.transforms,
        context.vocabs,
        task=CorpusTask.TRAIN,
        stride=stride,
        offset=offset,
        device_id=device_id,
    )

    valid_iter = build_dynamic_dataset_iter(
        context.config,
        context.transforms,
        context.vocabs,
        task=CorpusTask.VALID,
        device_id=device_id,
    )

    # Log training start
    _log_training_start(context.config.training)

    # Calculate training steps
    train_steps = _calculate_train_steps(context.config.training.train_steps, context.config.training.single_pass)

    # Start training
    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=context.config.training.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=context.config.training.valid_steps,
    )

    # Cleanup
    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()


def _calculate_parallel_params(device_id: int, training_config: Any) -> Tuple[int, int]:
    """Calculate offset and stride for data parallel training."""
    if training_config.parallel_mode == "data_parallel":
        # same model on all GPU - stride = nb of gpu
        offset = max(0, device_id)
        stride = max(1, len(training_config.gpu_ranks))
    else:
        # single GPU or
        # multi GPU but model is split on all GPUs (tensor parallel)
        offset = 0
        stride = 1
    return offset, stride


def _log_training_start(training_config: Any):
    """Log training initialization message."""
    if len(training_config.gpu_ranks):
        logger.info(f"Starting training on GPU: {training_config.gpu_ranks}")
    else:
        logger.info("Starting training on CPU, could be very slow")


def _calculate_train_steps(train_steps: int, single_pass: bool) -> int:
    """Calculate effective training steps."""
    if single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        return 0
    return train_steps
