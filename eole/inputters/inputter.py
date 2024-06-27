# -*- coding: utf-8 -*-
import os
from math import ceil
import codecs
import pyonmttok
from eole.constants import DefaultTokens


def build_vocab(config, specials):
    """Build vocabs dict to be stored in the checkpoint
        based on vocab files having each line [token, count]
    Args:
        opt: src_vocab, tgt_vocab
    Return:
        vocabs: {'src': pyonmttok.Vocab, 'tgt': pyonmttok.Vocab,
                 'decoder_start_token': DefaultTokens.BOS
                }
    """

    def _pad_vocab_to_multiple(vocab, multiple):
        vocab_size = len(vocab)
        if vocab_size % multiple == 0:
            return vocab
        target_size = int(ceil(vocab_size / multiple)) * multiple
        for i in range(target_size - vocab_size):
            vocab.add_token(DefaultTokens.VOCAB_PAD + str(i))
        return vocab

    default_specials = config.default_specials
    vocabs = {}
    src_vocab = _read_vocab_file(config.src_vocab, config.src_words_min_frequency)

    src_specials = [
        item for item in (default_specials + specials["src"]) if item not in src_vocab
    ]

    if DefaultTokens.SEP in src_specials and (
        "<0x0A>" in src_vocab or "Ċ" in src_vocab
    ):
        # this is hack: if the special separator ｟newline｠is returned because of the
        # "docify" transform.get_specials we don't add it if the corresponding newline code
        # is already included in the sentencepiece or BPE-with-gpt2-pretok.
        src_specials.remove(DefaultTokens.SEP)

    src_vocab = pyonmttok.build_vocab_from_tokens(
        src_vocab, maximum_size=config.src_vocab_size, special_tokens=src_specials
    )
    src_vocab.default_id = src_vocab[DefaultTokens.UNK]
    if src_vocab.default_id >= len(src_vocab):
        src_vocab.default_id = (
            0  # patch that assigns OOV to id=0 when UNK does not exist
        )

    if config.vocab_size_multiple > 1:
        src_vocab = _pad_vocab_to_multiple(src_vocab, config.vocab_size_multiple)
    vocabs["src"] = src_vocab
    if config.share_vocab:
        vocabs["tgt"] = src_vocab
    else:
        tgt_vocab = _read_vocab_file(config.tgt_vocab, config.tgt_words_min_frequency)
        tgt_specials = [
            item
            for item in (default_specials + specials["tgt"])
            if item not in tgt_vocab
        ]
        if DefaultTokens.SEP in tgt_specials and (
            "<0x0A>" in tgt_vocab or "Ċ" in src_vocab
        ):
            tgt_specials.remove(DefaultTokens.SEP)
        tgt_vocab = pyonmttok.build_vocab_from_tokens(
            tgt_vocab, maximum_size=config.tgt_vocab_size, special_tokens=tgt_specials
        )
        tgt_vocab.default_id = tgt_vocab[DefaultTokens.UNK]
        if tgt_vocab.default_id >= len(tgt_vocab):
            tgt_vocab.default_id = (
                0  # patch that assigns OOV to id=0 when UNK does not exist
            )
        if config.vocab_size_multiple > 1:
            tgt_vocab = _pad_vocab_to_multiple(tgt_vocab, config.vocab_size_multiple)
        vocabs["tgt"] = tgt_vocab

    vocabs["decoder_start_token"] = config.decoder_start_token

    return vocabs


def _read_vocab_file(vocab_path, min_count):
    """Loads a vocabulary from the given path.

    Args:
        vocab_path (str): Path to utf-8 text file containing vocabulary.
            Each token should be on a line, may followed with a count number
            seperate by space if `with_count`. No extra whitespace is allowed.
        min_count (int): retains only tokens with min_count frequency.
    """

    if not os.path.exists(vocab_path):
        raise RuntimeError("Vocabulary not found at {}".format(vocab_path))
    else:
        with codecs.open(vocab_path, "rb") as f:
            lines = [line.decode("utf-8") for line in f.read().split(b"\n")]
            lines = lines[:-1]

            first_line = lines[0].split(None, 1)
            has_count = len(first_line) == 2 and first_line[-1].isdigit()
            if has_count:
                vocab = []
                for line in lines:
                    if int(line.split(None, 1)[1]) >= min_count:
                        vocab.append(line.split(None, 1)[0])
            else:
                vocab = lines
            return vocab


def vocabs_to_dict(vocabs):
    """
    Convert a dict of pyonmttok vocabs
    into a plain text dict to be saved in the checkpoint
    """
    vocabs_dict = {}
    vocabs_dict["src"] = vocabs["src"].ids_to_tokens
    vocabs_dict["tgt"] = vocabs["tgt"].ids_to_tokens

    if "decoder_start_token" in vocabs.keys():
        vocabs_dict["decoder_start_token"] = vocabs["decoder_start_token"]
    else:
        vocabs_dict["decoder_start_token"] = DefaultTokens.BOS
    return vocabs_dict


def dict_to_vocabs(vocabs_dict):
    """
    Convert a dict formatted vocabs (as stored in a checkpoint)
    into a dict of pyonmttok vocabs objects.
    """
    vocabs = {}
    if "decoder_start_token" in vocabs_dict.keys():
        vocabs["decoder_start_token"] = vocabs_dict["decoder_start_token"]
    else:
        vocabs["decoder_start_token"] = DefaultTokens.BOS
    vocabs["src"] = pyonmttok.build_vocab_from_tokens(vocabs_dict["src"])
    if vocabs["src"].default_id >= len(vocabs["src"]):
        vocabs["src"].default_id = 0  # patch that assigns OOV to id=0
    if vocabs_dict["src"] == vocabs_dict["tgt"]:
        vocabs["tgt"] = vocabs["src"]
    else:
        vocabs["tgt"] = pyonmttok.build_vocab_from_tokens(vocabs_dict["tgt"])
        if vocabs["tgt"].default_id >= len(vocabs["src"]):
            vocabs["tgt"].default_id = 0  # patch that assigns OOV to id=0

    return vocabs
