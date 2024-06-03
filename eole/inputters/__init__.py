"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of vectors.
"""
from eole.inputters.inputter import build_vocab
from eole.inputters.text_utils import (
    text_sort_key,
    transform_bucket,
    numericalize,
    tensorify,
)
from eole.inputters.text_corpus import ParallelCorpus, ParallelCorpusIterator
from eole.inputters.dynamic_iterator import (
    MixingStrategy,
    SequentialMixer,
    WeightedMixer,
    DynamicDatasetIter,
)


__all__ = [
    "build_vocab",
    "text_sort_key",
    "transform_bucket",
    "numericalize",
    "tensorify",
    "ParallelCorpus",
    "ParallelCorpusIterator",
    "MixingStrategy",
    "SequentialMixer",
    "WeightedMixer",
    "DynamicDatasetIter",
]
