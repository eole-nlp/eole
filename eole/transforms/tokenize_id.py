"""Tokenization transforms that also numericalize, such as HF, or Mistral"""

import os
from eole.utils.logging import logger
from eole.transforms import register_transform
from .transform import Transform, TransformConfig  # , ObservableStats
from pydantic import Field
from eole.constants import DefaultTokens


class BaseIdTokenizerConfig(TransformConfig):
    # model_name: str | None = Field(default=None)  # if tokenizer deduced from model
    path: str | None = Field(default=None)  # if local tokenizer


class HuggingfaceTokenizerConfig(BaseIdTokenizerConfig):
    huggingface_model: str | None = Field(default=None)  # if retrieved from huggingface
    max_length: int | None = Field(default=None)


class IntTokenizerTransform(Transform):
    """IntTokenizer transform abstract class."""

    config_model = BaseIdTokenizerConfig
    output_type = "ids"

    def __init__(self, config):
        super().__init__(config)

    def _parse_config(self):
        # self.model_name = self.config.model_name
        self.path = self.config.path

    def warm_up(self, vocabs=None):
        # TODO: factorize here
        raise NotImplementedError


@register_transform(name="huggingface_tokenize")
class HuggingfaceTokenizer(IntTokenizerTransform):
    config_model = HuggingfaceTokenizerConfig

    def __init__(self, config):
        super().__init__(config)

    def _parse_config(self):
        super()._parse_config()
        self.huggingface_model = self.config.huggingface_model
        self.max_length = self.config.max_length

    def warm_up(self, vocabs=None):
        from tokenizers.processors import TemplateProcessing

        if self.huggingface_model is not None:
            from transformers import AutoTokenizer

            self.tokenizers = {}

            self.tokenizers["src"] = AutoTokenizer.from_pretrained(self.huggingface_model, legacy=False)
            # TODO: this needs to be tested and adapted for various models
            self.tokenizers["tgt"] = AutoTokenizer.from_pretrained(self.huggingface_model, legacy=False)
            logger.info(f"Initialized tokenizers from HF model: {self.huggingface_model}")
        elif self.path is not None:
            if os.path.exists(self.path):
                from tokenizers import Tokenizer

                self.tokenizers["src"] = Tokenizer.from_file(self.path, legacy=False)
                # TODO: might be more efficient to have a single tokenizer,
                # and apply the eos logic on top
                self.tokenizers["tgt"] = Tokenizer.from_file(self.path, legacy=False)
            else:
                raise FileNotFoundError(self.path)
            logger.info(f"Initialized tokenizers from local file: {self.path}")
        else:
            raise RuntimeError(f"Either model_name or path must be configured for {self.name} transform")
        # https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
        # bos = self.tokenizers["tgt"].bos_token
        # bos_id = self.tokenizers["tgt"].bos_token_id
        eos = self.tokenizers["tgt"].eos_token
        eos_id = self.tokenizers["tgt"].eos_token_id
        self.tokenizers["tgt"]._tokenizer.post_processor = TemplateProcessing(
            single=f"$A:0 {eos}:0",
            pair=f"$A:0 {eos}:0 $B:1 {eos}:1",
            special_tokens=[
                # (f"{bos}", bos_id),
                (f"{eos}", eos_id)
            ],
        )

    def tokenize_string(self, string, side="src", is_train=False):
        if self.max_length is not None and is_train:
            max_length = self.max_length
            decoder_start_token = self.full_config.decoder_start_token
            if side == "tgt" and self.full_config.model.encoder is None and decoder_start_token == "":
                max_length += 1
            kwargs = {"max_length": max_length, "truncation": True}
        else:
            kwargs = {}
        tokens = self.tokenizers[side].encode(string.replace(DefaultTokens.SEP, "\n"), **kwargs)
        return tokens

    def apply(self, example, is_train=False, stats=None, **kwargs):
        if isinstance(example["src"], str):
            src_tokens = self.tokenize_string(example["src"], side="src", is_train=is_train)
        elif isinstance(example["src"], list):
            src_tokens = self.tokenize_string(" ".join(example["src"]), side="src", is_train=is_train)
        else:
            raise ValueError(f"Unsupported src type: {type(example['src'])}")
        example["src_ids"] = src_tokens
        if example.get("tgt", None) is not None:
            if isinstance(example["tgt"], str):
                tgt_tokens = self.tokenize_string(example["tgt"], side="tgt", is_train=is_train)
            elif isinstance(example["tgt"], list):
                tgt_tokens = self.tokenize_string(" ".join(example["tgt"]), side="tgt", is_train=is_train)
            example["tgt_ids"] = tgt_tokens
        return example

    def apply_reverse(self, predicted):
        detokenized = self.tokenizers["tgt"].decode(predicted).replace("\n", DefaultTokens.SEP)
        return detokenized
