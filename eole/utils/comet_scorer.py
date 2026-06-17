import os

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
    return transforms["sentencepiece"]


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
