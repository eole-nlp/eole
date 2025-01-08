""" Pytorch Distributed utils
    This piece of code was heavily inspired by the equivalent of Fairseq-py
    https://github.com/pytorch/fairseq
"""

import torch.distributed
from datetime import timedelta
from eole.predict import build_predictor
from eole.transforms import get_transforms_cls, make_transforms
from eole.constants import CorpusTask
from eole.utils.logging import init_logger, logger
from eole.inputters.dynamic_iterator import build_dynamic_dataset_iter


def is_master(config, device_id):
    return config.gpu_ranks[device_id] == 0


def multi_init(config, device_id):
    # config is a running config here
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip=config.master_ip, master_port=config.master_port
    )
    dist_world_size = config.world_size
    torch.distributed.init_process_group(
        backend=config.gpu_backend,
        init_method=dist_init_method,
        world_size=dist_world_size,
        rank=config.gpu_ranks[device_id],
        timeout=timedelta(seconds=config.timeout),
    )
    gpu_rank = torch.distributed.get_rank()
    if not is_master(config, device_id):
        logger.disabled = True

    return gpu_rank


def spawned_train(process_fn, config, device_id, error_queue):  # noqa: E501
    """Run `process_fn` on `device_id` with data from `batch_queue`."""
    # config -> full config
    try:
        gpu_rank = multi_init(config.training, device_id)
        if gpu_rank != config.training.gpu_ranks[device_id]:
            raise AssertionError(
                "An error occurred in \
                  Distributed initialization"
            )
        process_fn(config, device_id=device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((config.training.gpu_ranks[device_id], traceback.format_exc()))


def spawned_infer(config, device_id, error_queue, queue_instruct, queue_result, queue_settings=None):
    """Run various functions for prediction in spawned process on `device_id`."""
    try:
        running_config = config  # will probably switch to config.inference at some point
        gpu_rank = multi_init(running_config, device_id)
        if gpu_rank != running_config.gpu_ranks[device_id]:
            raise AssertionError(
                "An error occurred in \
                  Distributed initialization"
            )
        torch.cuda.set_device(device_id)
        init_logger(config.log_file)
        predictor = build_predictor(config, device_id, logger=logger, report_score=True)
        transforms_cls = get_transforms_cls(config._all_transform)
        transforms = make_transforms(config, transforms_cls, predictor.vocabs)
        while True:
            instruction = queue_instruct.get()
            if queue_settings is not None:
                settings = queue_settings.get()
                predictor.update_settings(**settings)
            if instruction[0] == "stop":
                break
            elif instruction[0] == "infer_list":
                src = instruction[1]
                infer_iter = build_dynamic_dataset_iter(
                    config,
                    transforms,
                    predictor.vocabs,
                    task=CorpusTask.INFER,
                    src=src,
                    device_id=device_id,
                )
                scores, estims, preds = predictor._predict(
                    infer_iter,
                    infer_iter.transforms,
                    config.attn_debug,
                    config.align_debug,
                )
                queue_result.put(scores)
                queue_result.put(estims)
                queue_result.put(preds)
            elif instruction[0] == "infer_file":
                config.src = instruction[1].src
                infer_iter = build_dynamic_dataset_iter(
                    config,
                    transforms,
                    predictor.vocabs,
                    task=CorpusTask.INFER,
                    device_id=device_id,
                )
                scores, estims, preds = predictor._predict(
                    infer_iter,
                    infer_iter.transforms,
                    config.attn_debug,
                    config.align_debug,
                )
                queue_result.put(scores)
                queue_result.put(estims)
                queue_result.put(preds)
            elif instruction[0] == "score_list":
                tgt = instruction[1]
                infer_iter = build_dynamic_dataset_iter(
                    config,
                    transforms,
                    predictor.vocabs,
                    task=CorpusTask.INFER,
                    src=tgt,
                    tgt=tgt,
                    device_id=device_id,
                )
                score_results = predictor._score(infer_iter)
                queue_result.put(score_results)
            elif instruction[0] == "score_file":
                config.src = instruction[1].src
                config.tgt = instruction[1].src
                infer_iter = build_dynamic_dataset_iter(
                    config,
                    transforms,
                    predictor.vocabs,
                    task=CorpusTask.INFER,
                    device_id=device_id,
                )
                score_results = predictor._score(infer_iter)
                queue_result.put(score_results)

    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((running_config.gpu_ranks[device_id], traceback.format_exc()))
