import os

from tokenizers import Tokenizer

from eole.config.config import Config
from eole.config.data import NestedAllTransformsConfig
from eole.transforms import get_transforms_cls, make_transforms
from eole.transforms.tokenize import BaseTokenizerConfig


class CometScorerTransformConfig(Config):
    share_vocab: bool = True
    seed: int = 0
    transforms_configs: NestedAllTransformsConfig


def build_comet_sentencepiece_transform(model, model_dir):
    tokenizer_model = os.path.join(model_dir, "sentencepiece.bpe.model")
    if not os.path.exists(tokenizer_model):
        tokenizer_model = os.path.join(model_dir, "sentencepiece.model")
    if not os.path.exists(tokenizer_model):
        raise FileNotFoundError(f"No sentencepiece model found in {model_dir}")

    full_config = CometScorerTransformConfig(
        share_vocab=True,
        seed=0,
        transforms_configs=NestedAllTransformsConfig(
            sentencepiece=BaseTokenizerConfig(
                src_subword_model=tokenizer_model,
                tgt_subword_model=tokenizer_model,
                src_subword_nbest=1,
                tgt_subword_nbest=1,
                src_subword_alpha=0.0,
                tgt_subword_alpha=0.0,
                src_subword_vocab="",
                tgt_subword_vocab="",
                src_vocab_threshold=0,
                tgt_vocab_threshold=0,
            )
        ),
    )
    transforms = make_transforms(full_config, get_transforms_cls(["sentencepiece"]), model.vocabs)
    transform = transforms["sentencepiece"]
    tokenizer_json = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tokenizer_json):
        transform.comet_tokenizer = Tokenizer.from_file(tokenizer_json)
    return transform


def encode_comet_texts_to_ids(texts, tokenizer, model):
    bos_id = model.bos_id
    eos_id = model.eos_id
    max_length = model.max_length
    vocab = model.vocabs["src"]
    encoded = []
    for text in texts:
        pieces = tokenizer.tokenize_string(text, side="src", is_train=False)
        ids = [vocab.lookup_token(piece) for piece in pieces]
        encoded.append(([bos_id] + ids + [eos_id])[:max_length])
    return encoded


def _piece_offsets(text, pieces):
    offsets = [(0, 0)]
    cursor = 0
    for piece in pieces:
        surface = piece.replace("▁", " ")
        if surface.startswith(" "):
            cursor = min(len(text), cursor + len(surface) - len(surface.lstrip(" ")))
            surface = surface.lstrip(" ")
        if not surface:
            offsets.append((cursor, cursor))
            continue
        start = text.find(surface, cursor)
        if start < 0:
            start = cursor
        end = min(len(text), start + len(surface))
        offsets.append((start, end))
        cursor = end
    offsets.append((0, 0))
    return offsets


def encode_comet_texts_to_ids_and_offsets(texts, tokenizer, model):
    bos_id = model.bos_id
    eos_id = model.eos_id
    max_length = model.max_length
    vocab = model.vocabs["src"]
    encoded = []
    for text in texts:
        comet_tokenizer = getattr(tokenizer, "comet_tokenizer", None)
        if comet_tokenizer is not None:
            tokenized = comet_tokenizer.encode(text)
            input_ids = tokenized.ids[:max_length]
            offsets = tokenized.offsets[: len(input_ids)]
            encoded.append({"ids": input_ids, "offsets": offsets, "decode_token_ids": comet_tokenizer.decode})
            continue
        pieces = tokenizer.tokenize_string(text, side="src", is_train=False)
        ids = [vocab.lookup_token(piece) for piece in pieces]
        input_ids = ([bos_id] + ids + [eos_id])[:max_length]
        offsets = _piece_offsets(text, pieces)[: len(input_ids)]
        encoded.append({"ids": input_ids, "offsets": offsets})
    return encoded
