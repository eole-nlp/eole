#!/usr/bin/env python
"""Training on a single process."""
import torch
import sys

from eole.utils.logging import init_logger, logger
from eole.config.run import TrainConfig
from eole.constants import CorpusTask
from eole.transforms import (
    make_transforms,
    get_specials,
    get_transforms_cls,
)
from eole.inputters.inputter import vocabs_to_dict, build_vocab  # , dict_to_vocabs
from eole.inputters.dynamic_iterator import build_dynamic_dataset_iter
from eole.inputters.text_corpus import save_transformed_sample
from eole.models.model_saver import load_checkpoint
from eole.utils.optimizers import Optimizer
from eole.utils.misc import set_random_seed
from eole.trainer import build_trainer
from eole.models.model_saver import build_model_saver
from eole.models.model import get_model_class
from eole.modules.embeddings import prepare_pretrained_embeddings

from eole.config import (
    recursive_model_fields_set,
    recursive_update_dict,
)


def prepare_transforms_vocabs(config, transforms_cls):
    """Prepare or dump transforms before training."""
    specials = get_specials(config, transforms_cls)

    vocabs = build_vocab(config, specials)

    # maybe prepare pretrained embeddings, if any
    prepare_pretrained_embeddings(config, vocabs)

    transforms = make_transforms(config, transforms_cls, vocabs)
    if config.n_sample != 0:
        logger.warning(
            "`-n_sample` != 0: Training will not be started. "
            f"Stop after saving {config.n_sample} samples/corpus."
        )
        save_transformed_sample(config, transforms, n_sample=config.n_sample)
        logger.info("Sample saved, please check it before restart training.")
        sys.exit()
    logger.info(
        "The first 10 tokens of the vocabs are:"
        f"{vocabs_to_dict(vocabs)['src'][0:10]}"
    )
    logger.info(f"The decoder start token is: {config.decoder_start_token}")
    return vocabs, transforms


def _init_train(config):
    """Common initilization stuff for all training process.
    We need to build or rebuild the vocab in 3 cases:
    - training from scratch (train_from is false)
    - resume training but transforms have changed
    - resume training but vocab file has been modified
    """
    transforms_cls = get_transforms_cls(config._all_transform)

    if config.training.train_from:
        # Load checkpoint if we resume from a previous training.
        checkpoint = load_checkpoint(config.training.train_from)
        # vocabs = dict_to_vocabs(checkpoint["vocab"])
        # TODO: clarify and adapt this?
        # if (
        #     hasattr(checkpoint["config"], "_all_transform")
        #     and len(
        #         config._all_transform.symmetric_difference(
        #             checkpoint["config"]._all_transform
        #         )
        #     )
        #     != 0
        # ):
        #     _msg = "configured transforms is different from checkpoint:"
        #     new_transf = config._all_transform.difference(
        #         checkpoint["config"]._all_transform
        #     )
        #     old_transf = checkpoint["config"]._all_transform.difference(
        #         config._all_transform
        #     )
        #     if len(new_transf) != 0:
        #         _msg += f" +{new_transf}"
        #     if len(old_transf) != 0:
        #         _msg += f" -{old_transf}."
        #     logger.warning(_msg)
        #     vocabs, transforms = prepare_transforms_vocabs(config, transforms_cls)
        # if config.training.update_vocab:
        #     logger.info("Updating checkpoint vocabulary with new vocabulary")
        #     vocabs, transforms = prepare_transforms_vocabs(config, transforms_cls)
    else:
        checkpoint = None
        # vocabs, transforms = prepare_transforms_vocabs(config, transforms_cls)
    config = update_config_with_checkpoint(config, checkpoint=checkpoint)
    vocabs, transforms = prepare_transforms_vocabs(config, transforms_cls)

    return checkpoint, vocabs, transforms, config


def configure_process(config, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(config.seed, device_id >= 0)


def update_config_with_checkpoint(config, checkpoint=None):
    if checkpoint is not None:
        defaults = TrainConfig.get_defaults(
            architecture=checkpoint["config"].model.architecture
        )
        checkpoint_non_defaults = recursive_model_fields_set(checkpoint["config"])
        new_config = recursive_model_fields_set(config)
        # override data explicitly,
        # might change later if we want to finetune with implicit data config
        checkpoint_non_defaults["data"] = new_config["data"]
        if new_config.get("tensorboard_log_dir", None) == checkpoint_non_defaults.get(
            "tensorboard_log_dir", None
        ) and hasattr(checkpoint_non_defaults, "tensorboard_log_dir_dated"):
            # ensure tensorboard output is written in the directory
            # of previous checkpoints
            new_config.tensorboard_log_dir_dated = checkpoint_non_defaults[
                "tensorboard_log_dir_dated"
            ]
        updated_config = recursive_update_dict(
            checkpoint_non_defaults, new_config, defaults
        )
        config = TrainConfig(**updated_config)
    return config


def main(config, device_id):
    """Start training on `device_id`."""
    # config is the full TrainConfig namespace here

    configure_process(config, device_id)
    init_logger(config.log_file)
    checkpoint, vocabs, transforms, config = _init_train(config)

    # if transform + options set in 'valid' we need to copy in main
    # transform / options for scoring considered as inference
    validset_transforms = getattr(config.data.get("valid", None), "transforms", None)
    if validset_transforms:
        config.transforms = validset_transforms
        if config.data.get("valid", {}).tgt_prefix:
            config.transforms_configs.prefix.tgt_prefix = config.data.get(
                "valid", {}
            ).tgt_prefix
            # config.tgt_file_prefix = True
        if config.data.get("valid", {}).src_prefix:
            config.transforms_configs.prefix.src_prefix = config.data.get(
                "valid", {}
            ).src_prefix
        if config.data.get("valid", {}).tgt_suffix:
            config.transforms_configs.suffix.tgt_suffix = config.data.get(
                "valid", {}
            ).tgt_suffix
        if config.data.get("valid", {}).src_suffix:
            config.transforms_configs.suffix.src_suffix = config.data.get(
                "valid", {}
            ).src_suffix

    # Build model.
    # model = build_model(config, vocabs, checkpoint, device_id)
    model, _, _ = get_model_class(config.model).from_config(
        config.model,
        vocabs,
        checkpoint,
        device_id,
        training=True,
        running_config=config.training,
    )

    model.count_parameters(log=logger.info)
    trainable = {
        "torch.float32": 0,
        "torch.float16": 0,
        "torch.uint8": 0,
        "torch.int8": 0,
    }
    non_trainable = {
        "torch.float32": 0,
        "torch.float16": 0,
        "torch.uint8": 0,
        "torch.int8": 0,
    }
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable[str(p.dtype)] += p.numel()
        else:
            non_trainable[str(p.dtype)] += p.numel()
    logger.info("Trainable parameters = %s" % str(trainable))
    logger.info("Non trainable parameters = %s" % str(non_trainable))
    logger.info(" * src vocab size = %d" % len(vocabs["src"]))
    logger.info(" * tgt vocab size = %d" % len(vocabs["tgt"]))

    # Build optimizer.
    optim = Optimizer.from_config(model, config, checkpoint=checkpoint)
    del checkpoint

    # Build model saver
    model_saver = build_model_saver(config, model, vocabs, optim, device_id, transforms)

    trainer = build_trainer(
        config, device_id, model, vocabs, optim, model_saver=model_saver
    )

    offset = (
        max(0, device_id) if config.training.parallel_mode == "data_parallel" else 0
    )
    stride = (
        max(1, len(config.training.gpu_ranks))
        if config.training.parallel_mode == "data_parallel"
        else 1
    )

    train_iter = build_dynamic_dataset_iter(
        config,
        transforms,
        vocabs,
        task=CorpusTask.TRAIN,
        stride=stride,
        offset=offset,
        device_id=device_id,
    )

    valid_iter = build_dynamic_dataset_iter(
        config,
        transforms,
        vocabs,
        task=CorpusTask.VALID,
        device_id=device_id,
    )

    if len(config.training.gpu_ranks):
        logger.info("Starting training on GPU: %s" % config.training.gpu_ranks)
    else:
        logger.info("Starting training on CPU, could be very slow")
    train_steps = config.training.train_steps
    if config.training.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=config.training.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=config.training.valid_steps,
    )

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
