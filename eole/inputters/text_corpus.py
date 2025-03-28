"""Module that contain shard utils for dynamic data."""

import os
from eole.utils.logging import logger
from eole.constants import CorpusName, CorpusTask
from eole.transforms import TransformPipe
from eole.inputters.text_utils import transform_bucket
from eole.inputters.image_utils import process_image
from contextlib import contextmanager
import itertools
import json
from datasets import load_dataset


@contextmanager
def exfile_open(filename, *args, **kwargs):
    """Extended file opener enables open(filename=None).

    This context manager enables open(filename=None) as well as regular file.
    filename None will produce endlessly None for each iterate,
    while filename with valid path will produce lines as usual.

    Args:
        filename (str|None): a valid file path or None;
        *args: args relate to open file using codecs;
        **kwargs: kwargs relate to open file using codecs.

    Yields:
        `None` repeatly if filename==None,
        else yield from file specified in `filename`.
    """
    if filename is None:
        from itertools import repeat

        _file = repeat(None)
    else:
        import codecs

        _file = codecs.open(filename, *args, **kwargs)
    yield _file
    if filename is not None and _file:
        _file.close()


class BlockwiseCorpus(object):
    """A corpus class for reading a single file block by block."""

    def __init__(self, name, file_path, block_size=4096):
        """Initialize file path and block size."""
        self.id = name
        self.file_path = file_path
        self.block_size = block_size

    def load(self, offset=0, stride=1):
        """
        Load file and iterate by blocks.
        `offset` and `stride` allow iterating only on every
        `stride` block, starting from `offset`.
        """

        def make_ex(block_content):
            example = {
                "src": block_content,
                "tgt": block_content,
                "src_original": block_content,
                "tgt_original": block_content,
            }
            return example

        with open(self.file_path, mode="r", encoding="utf-8") as file:
            block_content = ""
            block_index = 0

            while True:
                chunk = file.read(self.block_size)
                if not chunk:
                    break

                if (block_index // stride) % stride == offset:
                    block_content += chunk

                    if len(chunk) < self.block_size:
                        # Reached end of file
                        yield make_ex(block_content)
                        break

                    if len(block_content) >= self.block_size:
                        yield make_ex(block_content)
                block_content = ""
                block_index += 1

    def __str__(self):
        cls_name = type(self).__name__
        return f"{cls_name}({self.id}, {self.file_path}, {self.file_path}" f"align={None}"


class ImageTextCorpus(object):
    """
    Handle vision data looking like this:
    ```
    [
        {
            "text": "List the top 5 countries in Europe with the highest GDP {image1}",
            "images": {
                "image1": "./test_data/gdp.png"
            }
        },
    ...
    ]
    ```
    """

    def __init__(self, name, data, is_train=False):
        self.id = name
        self.data = data
        self.is_train = is_train

    def load(self, offset=0, stride=1):
        def make_ex(item):
            example = {"text": item["text"], "images": item.get("images", {})}
            return example

        if isinstance(self.data, list):
            for i, item in enumerate(self.data):
                if (i // stride) % stride == offset:
                    yield make_ex(item)
        else:
            with exfile_open(self.data, mode="rb") as fs:
                data = json.load(fs)
                for i, item in enumerate(data):
                    if (i // stride) % stride == offset:
                        yield make_ex(item)

    def __str__(self):
        cls_name = type(self).__name__
        return f"{cls_name}({self.id}, {self.path}"


class ParallelCorpus(object):
    """A parallel corpus file pair that can be loaded to iterate."""

    def __init__(self, name, src, tgt, sco=None, align=None):
        """Initialize src & tgt side file path."""
        self.id = name
        self.src = src
        self.tgt = tgt
        self.sco = sco
        self.align = align

    def _load_hf_dataset(self, hf_string):
        """
        Load a Hugging Face dataset from the given identifier.
        Matchs the 'hf://' prefix and assumes the dataset is in streaming mode.
        Match the last '/field' to get the language / score field
        Potentially a config_name field between dataset_name and last field
        """
        parts = hf_string.replace("hf://", "").split("/")
        dataset_name = "/".join(parts[:2])
        remaining_parts = parts[2:]

        if len(remaining_parts) == 1:
            return load_dataset(dataset_name, split="train", streaming=True)
        elif len(remaining_parts) == 2:
            return load_dataset(dataset_name, remaining_parts[0], split="train", streaming=True)

    def load(self, offset=0, stride=1):
        """
        Load file and iterate by lines.
        `offset` and `stride` allow to iterate only on every
        `stride` example, starting from `offset`.
        In the case of local files, all files are open exactly the same way by each worker
        Therefore we need to apply a stride / offset rule to make sure we do not process the same ex.
        In the case of HF streaming mode we need to make sure we have more shards than workers.
        Typically we recommend to have shard being a multiple of workers for instance for big datasets:
        16 shards for 4 workers. The shards will be iterated automatically since HF locks shards when in use.
        """

        def make_ex(sline, tline, scoline, align):
            # 'src_original' and 'tgt_original' store the
            # original line before tokenization.
            example = {
                "src": sline,
                "tgt": tline,
                "sco": scoline,
                "src_original": sline,
                "tgt_original": tline,
            }
            if align is not None:
                example["align"] = align
            return example

        if isinstance(self.src, list):
            fs = self.src
            ft = [] if self.tgt is None else self.tgt
            fsco = [] if self.sco is None else self.sco
            fa = [] if self.align is None else self.align
            for i, (sline, tline, scoline, align) in enumerate(itertools.zip_longest(fs, ft, fsco, fa)):
                if (i // stride) % stride == offset:
                    if scoline is None:
                        scoline = 1.0
                    yield make_ex(sline, tline, scoline, align)

        elif self.src.startswith("hf://"):
            # If `src` is a Hugging Face dataset identifier
            dataset = self._load_hf_dataset(self.src)
            for i, example in enumerate(dataset):
                sline = example.get(self.src.split("/")[-1])
                tline = example.get(self.tgt.split("/")[-1])
                if self.sco is not None:
                    scoline = example.get(self.sco.split("/")[-1], 1.0)
                else:
                    scoline = 1.0
                yield make_ex(sline, tline, scoline, None)

        else:
            with exfile_open(self.src, mode="rb") as fs, exfile_open(self.tgt, mode="rb") as ft, exfile_open(
                self.sco, mode="rb"
            ) as fsco, exfile_open(self.align, mode="rb") as fa:
                for i, (sline, tline, scoline, align) in enumerate(zip(fs, ft, fsco, fa)):
                    if (i // stride) % stride == offset:
                        if tline is not None:
                            tline = tline.decode("utf-8")
                        if scoline is not None:
                            scoline = float(scoline.decode("utf-8"))
                        else:
                            scoline = 1.0
                        if align is not None:
                            align = align.decode("utf-8")
                        yield make_ex(sline.decode("utf-8"), tline, scoline, align)

    def __str__(self):
        cls_name = type(self).__name__
        return f"{cls_name}({self.id}, {self.src}, {self.tgt}, " f"align={self.align}"


def get_corpora(config, task=CorpusTask.TRAIN, src=None, tgt=None, align=None):
    corpora_dict = {}
    if task == CorpusTask.TRAIN:
        for corpus_id, corpus_dict in config.data.items():
            if corpus_id != CorpusName.VALID:
                if corpus_dict.path_txt is None:
                    if corpus_dict.path_tgt is None:
                        path_tgt = corpus_dict.path_src
                    else:
                        path_tgt = corpus_dict.path_tgt
                    corpora_dict[corpus_id] = ParallelCorpus(
                        corpus_id,
                        corpus_dict.path_src,
                        path_tgt,
                        corpus_dict.path_sco,
                        corpus_dict.path_align,
                    )
                elif config.data_type == "text":
                    corpora_dict[corpus_id] = BlockwiseCorpus(
                        corpus_id,
                        corpus_dict.path_txt,
                        block_size=8192,  # number of characters
                    )
                elif config.data_type == "image":
                    corpora_dict[corpus_id] = ImageTextCorpus(
                        corpus_id,
                        corpus_dict.path_txt,
                        is_train=True,
                    )
    elif task == CorpusTask.VALID:
        if CorpusName.VALID in config.data.keys():
            if config.data[CorpusName.VALID].path_tgt is None:
                path_tgt = config.data[CorpusName.VALID].path_src
            else:
                path_tgt = config.data[CorpusName.VALID].path_tgt
            if config.data_type == "text":
                corpora_dict[CorpusName.VALID] = ParallelCorpus(
                    CorpusName.VALID,
                    config.data[CorpusName.VALID].path_src,
                    path_tgt if tgt is None else None,
                    None,
                    config.data[CorpusName.VALID].path_align,
                )
            elif config.data_type == "image":
                corpora_dict[CorpusName.VALID] = ImageTextCorpus(
                    CorpusName.VALID,
                    config.data[CorpusName.VALID].path_txt,
                    is_train=True,
                )
        else:
            return None
    else:
        if config.data_type == "text":
            corpora_dict[CorpusName.INFER] = ParallelCorpus(
                CorpusName.INFER,
                src if src else config.src,
                tgt if tgt else config.tgt,
                align if align else None,
            )
        elif config.data_type == "image":
            corpora_dict[CorpusName.INFER] = ImageTextCorpus(
                CorpusName.INFER, src, is_train=False  # maybe homogenize to some better name
            )
    return corpora_dict


class ParallelCorpusIterator(object):
    """An iterator dedicated to ParallelCorpus.

    Args:
        corpus (ParallelCorpus): corpus to iterate;
        transform (TransformPipe): transforms to be applied to corpus;
        skip_empty_level (str): security level when encouter empty line;
        stride (int): iterate corpus with this line stride;
        offset (int): iterate corpus with this line offset.
    """

    def __init__(self, corpus, transform, skip_empty_level="warning", stride=1, offset=0, is_train=False):
        self.cid = corpus.id
        self.corpus = corpus
        self.transform = transform
        if skip_empty_level not in ["silent", "warning", "error"]:
            raise ValueError(f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level
        self.stride = stride
        self.offset = offset

    def _process(self, stream):
        for i, example in enumerate(stream):
            example["src"] = example["src"].strip().split(" ")
            example["src_original"] = example["src_original"].strip().split(" ")
            line_number = i * self.stride + self.offset
            example["cid_line_number"] = line_number
            example["cid"] = self.cid
            if "align" in example:
                example["align"] = example["align"].strip().split(" ")
            if example["tgt"] is not None:
                example["tgt"] = example["tgt"].strip().split(" ")
                example["tgt_original"] = example["tgt_original"].strip().split(" ")
                if (
                    len(example["src"]) == 0
                    or len(example["tgt"]) == 0
                    or ("align" in example and example["align"] == 0)
                ):
                    # empty example: skip
                    empty_msg = f"Empty line  in {self.cid}#{line_number}."
                    if self.skip_empty_level == "error":
                        raise IOError(empty_msg)
                    elif self.skip_empty_level == "warning":
                        logger.warning(empty_msg)
                    if len(example["src"]) == 0 and len(example["tgt"]) == 0:
                        yield (example, self.transform, self.cid)
                    continue
            yield (example, self.transform, self.cid)
        report_msg = self.transform.stats()
        if report_msg != "":
            logger.info("* Transform statistics for {}({:.2f}%):\n{}\n".format(self.cid, 100 / self.stride, report_msg))

    def __iter__(self):
        corpus_stream = self.corpus.load(stride=self.stride, offset=self.offset)
        corpus = self._process(corpus_stream)
        yield from corpus


class ImageTextCorpusIterator(object):
    def __init__(
        self,
        corpus,
        transform,
        skip_empty_level="warning",
        stride=1,
        offset=0,
        is_train=False,
        image_patch_size=16,
    ):
        self.cid = corpus.id
        self.corpus = corpus
        self.transform = transform
        if skip_empty_level not in ["silent", "warning", "error"]:
            raise ValueError(f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level
        self.stride = stride
        self.offset = offset
        self.is_train = is_train
        self.image_patch_size = image_patch_size

    def _process(self, stream):
        for i, example in enumerate(stream):
            # process images
            processed_images = {
                k: process_image(v, image_patch_size=self.image_patch_size)
                for k, v in example.get("images", {}).items()
            }
            text = example["text"]
            image_tokens = {
                # using a list here would induce unwanted whitespaces between tokens downstream
                k: "".join(v["tokens"])
                for k, v in processed_images.items()
            }
            text = text.format(**image_tokens)
            line_number = i * self.stride + self.offset
            example = {
                "src": text,
                "tgt": text if self.is_train else None,
                "images": {k: v["image"] for k, v in processed_images.items()},
                "cid": self.cid,
                "cid_line_number": line_number,
            }
            yield example, self.transform, self.cid

    def __iter__(self):
        corpus_stream = self.corpus.load(stride=self.stride, offset=self.offset)
        corpus = self._process(corpus_stream)
        yield from corpus


def build_corpora_iters(
    corpora, transforms, corpora_info, skip_empty_level="warning", stride=1, offset=0, image_patch_size=16
):
    """Return `ParallelCorpusIterator` for all corpora defined in opts."""
    corpora_iters = dict()
    for c_id, corpus in corpora.items():
        transform_names = corpora_info[c_id].transforms
        corpus_transform = [transforms[name] for name in transform_names if name in transforms]
        transform_pipe = TransformPipe.build_from(corpus_transform)
        if isinstance(corpus, ParallelCorpus):
            corpus_iter = ParallelCorpusIterator(
                corpus,
                transform_pipe,
                skip_empty_level=skip_empty_level,
                stride=stride,
                offset=offset,
                is_train=getattr(corpus, "is_train", None),
            )
        elif isinstance(corpus, ImageTextCorpus):
            corpus_iter = ImageTextCorpusIterator(
                corpus,
                transform_pipe,
                skip_empty_level=skip_empty_level,
                stride=stride,
                offset=offset,
                is_train=getattr(corpus, "is_train", None),
                image_patch_size=image_patch_size,
            )
        corpora_iters[c_id] = corpus_iter
    return corpora_iters


def save_transformed_sample(config, transforms, n_sample=3):
    """Save transformed data sample as specified in opts."""

    if n_sample == -1:
        logger.info(f"n_sample={n_sample}: Save full transformed corpus.")
    elif n_sample == 0:
        logger.info(f"n_sample={n_sample}: no sample will be saved.")
        return
    elif n_sample > 0:
        logger.info(f"Save {n_sample} transformed example/corpus.")
    else:
        raise ValueError(f"n_sample should >= -1, get {n_sample}.")

    corpora = get_corpora(config, CorpusTask.TRAIN)
    datasets_iterables = build_corpora_iters(corpora, transforms, config.data, skip_empty_level=config.skip_empty_level)
    sample_path = os.path.join(config.save_data, "samples", CorpusName.SAMPLE)
    os.makedirs(sample_path, exist_ok=True)
    for c_name, c_iter in datasets_iterables.items():
        dest_base = os.path.join(sample_path, "{}.{}".format(c_name, CorpusName.SAMPLE))
        with open(dest_base + ".src", "w", encoding="utf-8") as f_src, open(
            dest_base + ".tgt", "w", encoding="utf-8"
        ) as f_tgt:
            bucket = []
            for i, ex in enumerate(c_iter):
                if i > n_sample:
                    break
                else:
                    bucket.append(ex)
            pro_bucket = transform_bucket(CorpusTask.TRAIN, bucket)
            if pro_bucket is not None:
                for maybe_example in pro_bucket:
                    if maybe_example is not None:
                        src_line, tgt_line = (
                            maybe_example["src"]["src"],
                            maybe_example["tgt"]["tgt"],
                        )

                        f_src.write(src_line + "\n")
                        f_tgt.write(tgt_line + "\n")
