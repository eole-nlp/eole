#!/usr/bin/env python
"""Get vocabulary coutings from transformed corpora samples."""
import os
import copy
import multiprocessing as mp
import pyonmttok
from functools import partial
from eole.utils.logging import init_logger, logger
from eole.utils.misc import set_random_seed, check_path
from argparse import ArgumentParser

from eole.inputters.text_corpus import build_corpora_iters, get_corpora
from eole.inputters.text_utils import transform_bucket
from eole.transforms import make_transforms, get_transforms_cls
from eole.constants import CorpusName, CorpusTask
from collections import Counter

from eole.config.run import BuildVocabConfig
from eole.bin import register_bin
from eole.bin.run import RunBin


MAXBUCKETSIZE = 256000


def write_files_from_queues(sample_path, queues):
    """
    Standalone process that reads data from
    queues in order and write to sample files.
    """
    os.makedirs(sample_path, exist_ok=True)
    for c_name in queues.keys():
        dest_base = os.path.join(sample_path, "{}.{}".format(c_name, CorpusName.SAMPLE))
        with open(dest_base + ".src", "w", encoding="utf-8") as f_src, open(
            dest_base + ".tgt", "w", encoding="utf-8"
        ) as f_tgt:
            while True:
                _next = False
                for q in queues[c_name]:
                    item = q.get()
                    if item == "blank":
                        continue
                    if item == "break":
                        _next = True
                        break
                    _, src_line, tgt_line = item
                    f_src.write(src_line + "\n")
                    f_tgt.write(tgt_line + "\n")
                if _next:
                    break


def build_sub_vocab(corpora, transforms, config, n_sample, stride, offset):
    """Build vocab on (strided) subpart of the data."""
    sub_counter_src = Counter()
    sub_counter_tgt = Counter()
    datasets_iterables = build_corpora_iters(
        corpora,
        transforms,
        config.data,
        skip_empty_level=config.skip_empty_level,
        stride=stride,
        offset=offset,
    )
    for c_name, c_iter in datasets_iterables.items():
        for i, item in enumerate(c_iter):
            maybe_example = transform_bucket(CorpusTask.TRAIN, [item])
            if maybe_example is not None:
                maybe_example = maybe_example[0]
            else:
                if config.dump_samples:
                    build_sub_vocab.queues[c_name][offset].put("blank")
                continue
            src_line, tgt_line = (
                maybe_example["src"]["src"],
                maybe_example["tgt"]["tgt"],
            )
            sub_counter_src.update(src_line.split(" "))
            sub_counter_tgt.update(tgt_line.split(" "))

            if config.dump_samples:
                src_pretty_line = src_line
                build_sub_vocab.queues[c_name][offset].put(
                    (i, src_pretty_line, tgt_line)
                )
            if n_sample > 0 and ((i + 1) * stride + offset) >= n_sample:
                if config.dump_samples:
                    build_sub_vocab.queues[c_name][offset].put("break")
                break
        if config.dump_samples:
            build_sub_vocab.queues[c_name][offset].put("break")
    return sub_counter_src, sub_counter_tgt


def init_pool(queues):
    """Add the queues as attribute of the pooled function."""
    build_sub_vocab.queues = queues


def build_vocab(config, transforms, n_sample=3):
    """Build vocabulary from data."""

    if n_sample == -1:
        logger.info(f"n_sample={n_sample}: Build vocab on full datasets.")
    elif n_sample > 0:
        logger.info(f"Build vocab on {n_sample} transformed examples/corpus.")
    else:
        raise ValueError(f"n_sample should > 0 or == -1, get {n_sample}.")

    if config.dump_samples:
        logger.info(
            "The samples on which the vocab is built will be "
            "dumped to disk. It may slow down the process."
        )
    corpora = get_corpora(config, task=CorpusTask.TRAIN)
    counter_src = Counter()
    counter_tgt = Counter()

    queues = {
        c_name: [
            mp.Queue(config.vocab_sample_queue_size) for i in range(config.num_threads)
        ]
        for c_name in corpora.keys()
    }
    if config.dump_samples:
        sample_path = os.path.join(os.path.dirname(config.save_data), CorpusName.SAMPLE)
        write_process = mp.Process(
            target=write_files_from_queues, args=(sample_path, queues), daemon=True
        )
        write_process.start()
    with mp.Pool(config.num_threads, init_pool, [queues]) as p:
        func = partial(
            build_sub_vocab, corpora, transforms, config, n_sample, config.num_threads
        )
        for sub_counter_src, sub_counter_tgt in p.imap(
            func, range(0, config.num_threads)
        ):
            counter_src.update(sub_counter_src)
            counter_tgt.update(sub_counter_tgt)
    if config.dump_samples:
        write_process.join()
    return counter_src, counter_tgt


def ingest_tokens(config, transforms, n_sample, learner, stride, offset):
    def _mp_ingest(data):
        func = partial(transform_bucket, CorpusName.TRAIN)
        chunk = len(data) // config.num_threads
        with mp.Pool(config.num_threads) as pool:
            buckets = pool.map(
                func,
                [
                    data[i * chunk : (i + 1) * chunk]
                    for i in range(0, config.num_threads)
                ],
            )
        for bucket in buckets:
            for ex in bucket:
                if ex is not None:
                    src_line, tgt_line = (ex["src"]["src"], ex["tgt"]["tgt"])
                    learner.ingest(src_line)
                    learner.ingest(tgt_line)

    corpora = get_corpora(config, task=CorpusTask.TRAIN)
    datasets_iterables = build_corpora_iters(
        corpora,
        transforms,
        config.data,
        skip_empty_level=config.skip_empty_level,
        stride=stride,
        offset=offset,
    )
    to_ingest = []
    for c_name, c_iter in datasets_iterables.items():
        for i, item in enumerate(c_iter):
            if n_sample >= 0 and i >= n_sample:
                break
            if len(to_ingest) >= MAXBUCKETSIZE:
                _mp_ingest(to_ingest)
                to_ingest = []
            to_ingest.append(item)
        _mp_ingest(to_ingest)


def make_learner(tokenization_type, symbols):
    if tokenization_type == "bpe":
        # BPE training
        learner = pyonmttok.BPELearner(tokenizer=None, symbols=symbols)
    elif tokenization_type == "sentencepiece":
        # SentencePiece training
        learner = pyonmttok.SentencePieceLearner(
            vocab_size=symbols, character_coverage=0.98
        )
    return learner


def build_vocab_main(config):
    """Apply transforms to samples of specified data and build vocab from it.

    Transforms that need vocab will be disabled in this.
    Built vocab is saved in plain text format as following and can be pass as
    `-src_vocab` (and `-tgt_vocab`) when training:
    ```
    <tok_0>\t<count_0>
    <tok_1>\t<count_1>
    ```
    """

    config._validate_data_config(build_vocab_only=True)
    assert (
        config.n_sample == -1 or config.n_sample > 1
    ), f"Illegal argument n_sample={config.n_sample}."

    logger = init_logger()
    set_random_seed(config.seed, False)
    transforms_cls = get_transforms_cls(config._all_transform)

    if config.learn_subwords:
        # if {"bpe", "sentencepiece", "onmt_tokenize"}.intersect(config.transforms):
        _config = config.transforms_configs.onmt_tokenize
        logger.info(f"Ingesting {_config.src_subword_type} model from corpus")
        learner = make_learner(_config.src_subword_type, config.learn_subwords_size)
        if _config.src_subword_model is not None:
            tok_path = _config.src_subword_model
        else:
            data_dir = os.path.split(config.save_data)[0]
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            tok_path = os.path.join(data_dir, f"{_config.src_subword_type}.model")
        save_config = copy.deepcopy(config)
        _config.src_subword_type = "none"
        _config.tgt_subword_type = "none"
        _config.src_eoletok_kwargs["joiner_annotate"] = False
        _config.tgt_eoletok_kwargs["joiner_annotate"] = False
        transforms = make_transforms(config, transforms_cls, None)
        ingest_tokens(config, transforms, config.n_sample, learner, 1, 0)
        logger.info(f"Learning {tok_path} model, patience")
        learner.learn(tok_path)
        config = save_config

    transforms = make_transforms(config, transforms_cls, None)

    logger.info(f"Counter vocab from {config.n_sample} samples.")
    src_counter, tgt_counter = build_vocab(config, transforms, n_sample=config.n_sample)

    logger.info(f"Counters src: {len(src_counter)}")
    logger.info(f"Counters tgt: {len(tgt_counter)}")

    def save_counter(counter, save_path):
        check_path(save_path, exist_ok=config.overwrite, log=logger.warning)
        with open(save_path, "w", encoding="utf8") as fo:
            for tok, count in counter.most_common():
                fo.write(tok + "\t" + str(count) + "\n")

    if config.share_vocab:
        src_counter += tgt_counter
        tgt_counter = src_counter
        logger.info(f"Counters after share:{len(src_counter)}")
        save_counter(src_counter, config.src_vocab)
    else:
        save_counter(src_counter, config.src_vocab)
        save_counter(tgt_counter, config.tgt_vocab)


@register_bin(name="build_vocab")
class BuildVocab(RunBin):
    config_class = BuildVocabConfig
    require_config = True

    @classmethod
    def run(cls, args):
        config = cls.build_config(args)
        build_vocab_main(config)


def _get_parser():
    # Patch to allow autodoc with sphinx-argparse
    parser = ArgumentParser()
    BuildVocab.add_args(parser)
    return parser
