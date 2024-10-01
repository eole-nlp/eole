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


# TODO: this requires further investigation, and better support
# of chat templates across all paths (training+inference)

# @register_transform(name="mistral_tokenize")
# class MistralTokenizer(IntTokenizerTransform):
#     def __init__(self, config):
#         super().__init__(config)

#     def warm_up(self, vocabs=None):
#         from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

#         if self.model_name is not None:
#             self.tokenizer = MistralTokenizer.from_model(self.model_name)
#         elif self.path is not None:
#             if os.path.exists(self.path):
#                 self.tokenizer = MistralTokenizer.from_file(self.path)
#             else:
#                 raise FileNotFoundError(self.path)
#         else:
#             raise RuntimeError(
#                 f"Either model_name or path must be configured for {self.name} transform"
#             )

#     def apply(self, example, is_train=False, stats=None, **kwargs):
#         return example


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
        if self.huggingface_model is not None:
            from transformers import AutoTokenizer
            from tokenizers.processors import TemplateProcessing

            self.tokenizers = {}

            self.tokenizers["src"] = AutoTokenizer.from_pretrained(
                self.huggingface_model
            )
            # https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
            # TODO: this needs to be tested and adapted for various models
            tgt_tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model)
            # bos = tgt_tokenizer.bos_token
            eos = tgt_tokenizer.eos_token
            tgt_tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=f"$A:0 {eos}:0",
                pair=f"$A:0 {eos}:0 $B:1 {eos}:1",
                special_tokens=[
                    # (f"{bos}", tgt_tokenizer.bos_token_id),
                    (f"{eos}", tgt_tokenizer.eos_token_id)
                ],
            )
            self.tokenizers["tgt"] = tgt_tokenizer

            logger.info(
                f"Initialized tokenizer from HF model: {self.huggingface_model}"
            )

        elif self.path is not None:
            if os.path.exists(self.path):
                from tokenizers import Tokenizer

                self.tokenizers["src"] = Tokenizer.from_file(self.path)
                # TODO: this is not efficient, we shall have a single tokenizer
                tgt_tokenizer = Tokenizer.from_file(self.path)
                # bos = tgt_tokenizer.bos_token
                eos = tgt_tokenizer.eos_token
                tgt_tokenizer._tokenizer.post_processor = TemplateProcessing(
                    single=f"$A:0 {eos}:0",
                    pair=f"$A:0 {eos}:0 $B:1 {eos}:1",
                    special_tokens=[
                        # (f"{bos}", tgt_tokenizer.bos_token_id),
                        (f"{eos}", tgt_tokenizer.eos_token_id)
                    ],
                )
                self.tokenizers["tgt"] = tgt_tokenizer

                logger.info(f"Initialized tokenizer from local file: {self.path}")
            else:
                raise FileNotFoundError(self.path)
        else:
            raise RuntimeError(
                f"Either model_name or path must be configured for {self.name} transform"
            )

    def tokenize_string(self, string, side="src", is_train=False):
        if self.max_length is not None and is_train:
            kwargs = {"max_length": self.max_length, "truncation": True}
        else:
            kwargs = {}
        tokens = self.tokenizers[side].encode(string, **kwargs)
        return tokens

    def apply(self, example, is_train=False, stats=None, **kwargs):
        src_tokens = self.tokenize_string(" ".join(example["src"]), side="src")
        example["src_ids"] = src_tokens
        if example.get("tgt", None) is not None:
            tgt_tokens = self.tokenize_string(" ".join(example["tgt"]), side="tgt")
            example["tgt_ids"] = tgt_tokens
        return example

    def apply_reverse(self, predicted):
        detokenized = (
            self.tokenizers["tgt"].decode(predicted).replace("\n", DefaultTokens.SEP)
        )
        return detokenized
