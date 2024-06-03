"""Transforms relate to tokenization/subword."""
import re
from eole.utils.logging import logger
from eole.transforms import register_transform
from .transform import Transform, ObservableStats, TransformConfig
from eole.constants import DefaultTokens
from typing import Literal
from pydantic import model_validator, Field


class BaseTokenizerConfig(TransformConfig):
    src_subword_model: str | None = Field(
        default=None, description="Path of subword model for src (or shared)."
    )
    tgt_subword_model: str | None = Field(
        default=None, description="Path of subword model for tgt."
    )

    src_subword_nbest: int | None = Field(
        default=1,
        description="Number of candidates in subword regularization. "
        "Valid for unigram sampling, invalid for BPE-dropout. (source side)",
    )
    tgt_subword_nbest: int | None = Field(
        default=1,
        description="Number of candidates in subword regularization. "
        "Valid for unigram sampling, invalid for BPE-dropout. (target side)",
    )
    src_subword_alpha: float | None = Field(
        default=0,
        description="Smoothing parameter for sentencepiece unigram sampling, "
        "and dropout probability for BPE-dropout. (source side)",
    )
    tgt_subword_alpha: float | None = Field(
        default=0,
        description="Smoothing parameter for sentencepiece unigram sampling, "
        "and dropout probability for BPE-dropout. (target side)",
    )

    src_subword_vocab: str | None = Field(
        default="",
        description="Path to the vocabulary file for src subword. "
        "Format: <word>\\t<count> per line.",
    )
    tgt_subword_vocab: str | None = Field(
        default="",
        description="Path to the vocabulary file for tgt subword. "
        "Format: <word>\\t<count> per line.",
    )
    src_vocab_threshold: int | None = Field(
        default=0,
        description="Only produce src subword in src_subword_vocab "
        "with frequency >= src_vocab_threshold.",
    )
    tgt_vocab_threshold: int | None = Field(
        default=0,
        description="Only produce tgt subword in tgt_subword_vocab "
        "with frequency >= tgt_vocab_threshold.",
    )

    @model_validator(mode="after")
    def check_values(self):
        # TODO: use numeric constraints on Field instead?
        # https://docs.pydantic.dev/latest/concepts/fields/#numeric-constraints
        assert (
            0 <= self.src_subword_alpha <= 1
        ), "src_subword_alpha should be in the range [0, 1]"
        assert (
            0 <= self.tgt_subword_alpha <= 1
        ), "tgt_subword_alpha should be in the range [0, 1]"
        return self


class ONMTTokenizerConfig(BaseTokenizerConfig):
    src_subword_type: Literal["none", "sentencepiece", "bpe"] | None = Field(
        default="none",
        description="Type of subword model for src (or shared) in pyonmttok.",
    )
    tgt_subword_type: Literal["none", "sentencepiece", "bpe"] | None = Field(
        default="none", description="Type of subword model for tgt in pyonmttok."
    )
    src_onmttok_kwargs: dict | None = Field(
        default={"mode": "none"},
        description="Other pyonmttok options for src in dict string, "
        "except subword related options listed earlier.",
    )
    tgt_onmttok_kwargs: dict | None = Field(
        default={"mode": "none"},
        description="Other pyonmttok options for tgt in dict string, "
        "except subword related options listed earlier.",
    )
    gpt2_pretok: bool | None = Field(
        default=False, description="Preprocess sentence with byte-level mapping."
    )

    @model_validator(mode="after")
    def check_values(self):
        # TODO: this can probably be enhanced
        if isinstance(self.src_onmttok_kwargs, str):
            self.src_onmttok_kwargs = eval(self.src_onmttok_kwargs)
        if isinstance(self.tgt_onmttok_kwargs, str):
            self.tgt_onmttok_kwargs = eval(self.tgt_onmttok_kwargs)
        if not isinstance(self.src_onmttok_kwargs, dict):
            raise ValueError("-src_onmttok_kwargs isn't a dict valid string.")
        if not isinstance(self.tgt_onmttok_kwargs, dict):
            raise ValueError("-tgt_onmttok_kwargs isn't a dict valid string.")
        return self


class TokenizerTransform(Transform):
    """Tokenizer transform abstract class."""

    config_model = BaseTokenizerConfig

    def __init__(self, config):
        """Initialize necessary options for Tokenizer."""
        super().__init__(config)

    def _parse_config(self):
        self.share_vocab = self.full_config.share_vocab
        self.src_subword_model = self.config.src_subword_model
        self.tgt_subword_model = self.config.tgt_subword_model
        self.src_subword_nbest = self.config.src_subword_nbest
        self.tgt_subword_nbest = self.config.tgt_subword_nbest
        self.src_subword_alpha = self.config.src_subword_alpha
        self.tgt_subword_alpha = self.config.tgt_subword_alpha
        self.src_subword_vocab = self.config.src_subword_vocab
        self.tgt_subword_vocab = self.config.tgt_subword_vocab
        self.src_vocab_threshold = self.config.src_vocab_threshold
        self.tgt_vocab_threshold = self.config.tgt_vocab_threshold
        # define which attributes are artifacts to be saved
        self.artifacts = [
            "src_subword_model",
            "tgt_subword_model",
            "src_subword_vocab",
            "tgt_subword_vocab",
        ]

    def _repr_args(self):
        """Return str represent key arguments for TokenizerTransform."""
        kwargs = {
            "share_vocab": self.share_vocab,
            "src_subword_model": self.src_subword_model,
            "tgt_subword_model": self.tgt_subword_model,
            "src_subword_alpha": self.src_subword_alpha,
            "tgt_subword_alpha": self.tgt_subword_alpha,
            "src_subword_vocab": self.src_subword_vocab,
            "tgt_subword_vocab": self.tgt_subword_vocab,
            "src_vocab_threshold": self.src_vocab_threshold,
            "tgt_vocab_threshold": self.tgt_vocab_threshold,
        }
        return ", ".join([f"{kw}={arg}" for kw, arg in kwargs.items()])

    def tokenize_string(self, string, side="src", is_train=False):
        raise NotImplementedError

    def _tokenize(self, tokens, side="src", is_train=False):
        """Tokenize a list of words."""
        # This method embeds a custom logic to correctly handle certain placeholders
        # in case the tokenizer doesn't preserve them.
        sentence = " ".join(tokens)
        # Locate the end-of-sentence placeholders.
        sent_list = sentence.split(DefaultTokens.EOS)
        # Tokenize each sentence separately.
        segmented = []
        for _sentence in sent_list:
            # Locate the mask-before placeholders
            # (to zero-out the prompt loss during LM finetuning).
            _sentence_chunks = _sentence.split(DefaultTokens.MASK_BEFORE)
            # Tokenize each chunk separately and insert the padding token.
            # between each sequence of tokens.
            _sentence_tokens = []
            for _chunk in _sentence_chunks:
                _sentence_tokens += self.tokenize_string(_chunk, side, is_train) + [
                    DefaultTokens.PAD
                ]
            # Re-insert the eos token.
            segmented += _sentence_tokens[:-1] + [DefaultTokens.EOS]
        return segmented[:-1]

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply subword-based tokenenization to src & tgt."""
        src_out = self._tokenize(example["src"], "src", is_train)
        if example["tgt"] is not None:
            tgt_out = self._tokenize(example["tgt"], "tgt", is_train)
            if stats is not None:
                n_words = len(example["src"]) + len(example["tgt"])
                n_subwords = len(src_out) + len(tgt_out)
                stats.update(SubwordStats(n_subwords, n_words))
        else:
            tgt_out = None
            if stats is not None:
                n_words = len(example["src"])
                n_subwords = len(src_out)
                stats.update(SubwordStats(n_subwords, n_words))
        example["src"], example["tgt"] = src_out, tgt_out
        return example


class SubwordStats(ObservableStats):
    """Runing statistics for counting tokens before/after subword transform."""

    __slots__ = ["subwords", "words"]

    def __init__(self, subwords: int, words: int):
        self.subwords = subwords
        self.words = words

    def update(self, other: "SubwordStats"):
        self.subwords += other.subwords
        self.words += other.words

    def __str__(self) -> str:
        return "{}: {} -> {} tokens".format(self.name(), self.words, self.subwords)


@register_transform(name="sentencepiece")
class SentencePieceTransform(TokenizerTransform):
    """SentencePiece subword transform class."""

    def __init__(self, config):
        """Initialize necessary options for sentencepiece."""
        super().__init__(config)

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        import sentencepiece as spm

        spm.set_random_generator_seed(seed)

    def warm_up(self, vocabs=None):
        """Load subword models."""
        super().warm_up(None)
        import sentencepiece as spm

        load_src_model = spm.SentencePieceProcessor()
        load_src_model.Load(self.src_subword_model)
        _diff_vocab = (
            self.src_subword_vocab != self.tgt_subword_vocab
            or self.src_vocab_threshold != self.tgt_vocab_threshold
        )
        if self.src_subword_vocab != "" and self.src_vocab_threshold > 0:
            load_src_model.LoadVocabulary(
                self.src_subword_vocab, self.src_vocab_threshold
            )
        if self.share_vocab and not _diff_vocab:
            self.load_models = {"src": load_src_model, "tgt": load_src_model}
        else:
            load_tgt_model = spm.SentencePieceProcessor()
            load_tgt_model.Load(self.tgt_subword_model)
            if self.tgt_subword_vocab != "" and self.tgt_vocab_threshold > 0:
                load_tgt_model.LoadVocabulary(
                    self.tgt_subword_vocab, self.tgt_vocab_threshold
                )
            self.load_models = {"src": load_src_model, "tgt": load_tgt_model}

    def tokenize_string(self, string, side="src", is_train=False):
        """Apply subword sampling or deterministic subwording"""
        sp_model = self.load_models[side]
        nbest_size = self.tgt_subword_nbest if side == "tgt" else self.src_subword_nbest
        string = string.replace(DefaultTokens.SEP, "\n")
        if is_train is False or nbest_size in [0, 1]:
            # derterministic subwording
            tokens = sp_model.encode(string, out_type=str)
        else:
            # subword sampling when nbest_size > 1 or -1
            # alpha should be 0.0 < alpha < 1.0
            alpha = self.tgt_subword_alpha if side == "tgt" else self.src_subword_alpha
            tokens = sp_model.encode(
                string,
                out_type=str,
                enable_sampling=True,
                alpha=alpha,
                nbest_size=nbest_size,
            )
        return tokens

    def _detokenize(self, tokens, side="src"):
        """Apply SentencePiece Detokenizer"""
        sp_model = self.load_models[side]
        return sp_model.DecodePieces(tokens).replace("\n", DefaultTokens.SEP)

    def apply_reverse(self, predicted):
        """Apply SentencePiece Detokenizer."""
        if isinstance(predicted, list):
            return self._detokenize(predicted, "tgt")
        else:
            return self._detokenize(predicted.split(" "), "tgt")

    def _repr_args(self):
        """Return str represent key arguments for class."""
        kwargs_str = super()._repr_args()
        additional_str = "src_subword_nbest={}, tgt_subword_nbest={}".format(
            self.src_subword_nbest, self.tgt_subword_nbest
        )
        return kwargs_str + ", " + additional_str


@register_transform(name="bpe")
class BPETransform(TokenizerTransform):
    """subword_nmt: official BPE subword transform class."""

    def __init__(self, config):
        """Initialize necessary options for subword_nmt."""
        super().__init__(config)

    def _parse_config(self):
        super()._parse_config()
        self.dropout = {"src": self.src_subword_alpha, "tgt": self.tgt_subword_alpha}

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        import random

        random.seed(seed)

    def warm_up(self, vocabs=None):
        """Load subword models."""
        super().warm_up(None)
        from subword_nmt.apply_bpe import BPE, read_vocabulary

        # Load vocabulary file if provided and set threshold
        src_vocabulary, tgt_vocabulary = None, None
        if self.src_subword_vocab != "" and self.src_vocab_threshold > 0:
            with open(self.src_subword_vocab, encoding="utf-8") as _sv:
                src_vocabulary = read_vocabulary(_sv, self.src_vocab_threshold)
        if self.tgt_subword_vocab != "" and self.tgt_vocab_threshold > 0:
            with open(self.tgt_subword_vocab, encoding="utf-8") as _tv:
                tgt_vocabulary = read_vocabulary(_tv, self.tgt_vocab_threshold)
        # Load Subword Model
        with open(self.src_subword_model, encoding="utf-8") as src_codes:
            load_src_model = BPE(codes=src_codes, vocab=src_vocabulary)
        if self.share_vocab and (src_vocabulary == tgt_vocabulary):
            self.load_models = {"src": load_src_model, "tgt": load_src_model}
        else:
            with open(self.tgt_subword_model, encoding="utf-8") as tgt_codes:
                load_tgt_model = BPE(codes=tgt_codes, vocab=tgt_vocabulary)
            self.load_models = {"src": load_src_model, "tgt": load_tgt_model}

    def tokenize_string(self, string, side="src", is_train=False):
        """Do bpe subword tokenize."""
        tokens = string.split(" ")
        bpe_model = self.load_models[side]
        dropout = self.dropout[side] if is_train else 0.0
        segmented = bpe_model.segment_tokens(tokens, dropout=dropout)
        return segmented

    def _detokenize(self, tokens, side="src", is_train=False):
        """ "Apply bpe subword detokenizer"""
        detokenized = re.sub(r"(@@ )|(@@ ?$)", r"", " ".join(tokens))
        return detokenized

    def apply_reverse(self, predicted):
        """Apply bpe subword detokenizer"""
        if isinstance(predicted, list):
            return self._detokenize(predicted, "tgt")
        else:
            return self._detokenize(predicted.split(" "), "tgt")


@register_transform(name="onmt_tokenize")
class ONMTTokenizerTransform(TokenizerTransform):
    """OpenNMT Tokenizer transform class."""

    # config model as class attribute
    config_model = ONMTTokenizerConfig

    def __init__(self, config):
        """Initialize necessary options for OpenNMT Tokenizer."""
        super().__init__(config)

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        import pyonmttok

        pyonmttok.set_random_seed(seed)

    def _parse_config(self):
        super()._parse_config()
        self.src_subword_type = self.config.src_subword_type
        self.tgt_subword_type = self.config.tgt_subword_type
        logger.debug(
            "Parsed pyonmttok kwargs for src: {}".format(self.config.src_onmttok_kwargs)
        )
        logger.debug(
            "Parsed pyonmttok kwargs for tgt: {}".format(self.config.tgt_onmttok_kwargs)
        )
        self.src_other_kwargs = self.config.src_onmttok_kwargs
        self.tgt_other_kwargs = self.config.tgt_onmttok_kwargs
        self.gpt2_pretok = self.config.gpt2_pretok
        self.preserve_placeholders = self.config.tgt_onmttok_kwargs.get(
            "preserve_placeholders", False
        )

    @classmethod
    def get_specials(cls, config):
        config = getattr(config.transforms_configs, cls.name)
        src_specials, tgt_specials = [], []
        if config.src_onmttok_kwargs.get("case_markup", False):
            _case_specials = [
                "｟mrk_case_modifier_C｠",
                "｟mrk_begin_case_region_U｠",
                "｟mrk_end_case_region_U｠",
            ]
            for src_spec in _case_specials:
                src_specials.append(src_spec)
        if config.tgt_onmttok_kwargs.get("case_markup", False):
            _case_specials = [
                "｟mrk_case_modifier_C｠",
                "｟mrk_begin_case_region_U｠",
                "｟mrk_end_case_region_U｠",
            ]
            for tgt_spec in _case_specials:
                tgt_specials.append(tgt_spec)
        return (src_specials, tgt_specials)

    def _get_subword_kwargs(self, side="src"):
        """Return a dict containing kwargs relate to `side` subwords."""
        subword_type = self.tgt_subword_type if side == "tgt" else self.src_subword_type
        subword_model = (
            self.tgt_subword_model if side == "tgt" else self.src_subword_model
        )
        subword_nbest = (
            self.tgt_subword_nbest if side == "tgt" else self.src_subword_nbest
        )
        subword_alpha = (
            self.tgt_subword_alpha if side == "tgt" else self.src_subword_alpha
        )
        kwargs = dict()
        if subword_type == "bpe":
            kwargs["bpe_model_path"] = subword_model
            kwargs["bpe_dropout"] = subword_alpha
        elif subword_type == "sentencepiece":
            kwargs["sp_model_path"] = subword_model
            kwargs["sp_nbest_size"] = subword_nbest
            kwargs["sp_alpha"] = subword_alpha
        else:
            logger.debug("No subword method will be applied.")
        vocabulary_threshold = (
            self.tgt_vocab_threshold if side == "tgt" else self.src_vocab_threshold
        )
        vocabulary_path = (
            self.tgt_subword_vocab if side == "tgt" else self.src_subword_vocab
        )
        if vocabulary_threshold > 0 and vocabulary_path != "":
            kwargs["vocabulary_path"] = vocabulary_path
            kwargs["vocabulary_threshold"] = vocabulary_threshold
        return kwargs

    def warm_up(self, vocabs=None):
        """Initialize Tokenizer models."""
        super().warm_up(None)
        import pyonmttok

        src_subword_kwargs = self._get_subword_kwargs(side="src")
        src_tokenizer = pyonmttok.Tokenizer(
            **src_subword_kwargs, **self.src_other_kwargs
        )
        tgt_subword_kwargs = self._get_subword_kwargs(side="tgt")
        _diff_vocab = src_subword_kwargs.get(
            "vocabulary_path", ""
        ) != tgt_subword_kwargs.get("vocabulary_path", "") or src_subword_kwargs.get(
            "vocabulary_threshold", 0
        ) != tgt_subword_kwargs.get(
            "vocabulary_threshold", 0
        )
        if self.share_vocab and not _diff_vocab:
            self.load_models = {"src": src_tokenizer, "tgt": src_tokenizer}
        else:
            tgt_subword_kwargs = self._get_subword_kwargs(side="tgt")
            tgt_tokenizer = pyonmttok.Tokenizer(
                **tgt_subword_kwargs, **self.tgt_other_kwargs
            )
            self.load_models = {"src": src_tokenizer, "tgt": tgt_tokenizer}
        if self.gpt2_pretok:
            """
            Returns list of utf-8 byte and a corresponding list of unicode
            strings. The reversible bpe codes work on unicode strings.
            code taken from openai/gpt2
            """
            bs = (
                list(range(ord("!"), ord("~") + 1))
                + list(range(ord("¡"), ord("¬") + 1))
                + list(range(ord("®"), ord("ÿ") + 1))
            )
            cs = bs[:]
            n = 0
            for b in range(2**8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2**8 + n)
                    n += 1
            cs = [chr(n) for n in cs]
            self.maptable = dict(zip(bs, cs))
            self.revtable = {v: k for k, v in self.maptable.items()}

    def tokenize_string(self, sentence, side="src", is_train=False):
        tokenizer = self.load_models[side]
        if self.gpt2_pretok:
            sentence = "".join(
                self.maptable[b]
                for b in sentence.replace(DefaultTokens.SEP, "\n").encode("utf-8")
            )
            segmented1 = tokenizer(sentence)
            segmented = []
            # ugly patch to make sure "\n\n" is split in two items
            for s in segmented1:
                if s == "ĊĊ":
                    segmented.extend(["Ċ", "Ċ"])
                else:
                    segmented.append(s)
        elif (
            self.src_subword_type == "sentencepiece" and not self.preserve_placeholders
        ):
            sentence = sentence.replace(DefaultTokens.SEP, "\n")
            segmented = tokenizer(sentence)
        else:
            segmented = tokenizer(sentence)
        return segmented

    def _detokenize(self, tokens, side="src", is_train=False):
        """Do OpenNMT Tokenizer's detokenize."""
        tokenizer = self.load_models[side]
        if self.gpt2_pretok:
            sentence = "".join(tokens)
            detokenized = bytearray([self.revtable[c] for c in sentence]).decode(
                "utf-8", errors="replace"
            )
        else:
            detokenized = tokenizer.detokenize(tokens)
        return detokenized.replace("\n", DefaultTokens.SEP)

    def apply_reverse(self, predicted):
        """Apply OpenNMT Tokenizer to src & tgt."""
        if isinstance(predicted, list):
            return self._detokenize(predicted, "tgt")
        else:
            return self._detokenize(predicted.split(" "), "tgt")

    def _repr_args(self):
        """Return str represent key arguments for class."""
        repr_str = "{}={}".format("share_vocab", self.share_vocab)
        repr_str += ", src_subword_kwargs={}".format(
            self._get_subword_kwargs(side="src")
        )
        repr_str += ", src_onmttok_kwargs={}".format(self.src_other_kwargs)
        repr_str += ", tgt_subword_kwargs={}".format(
            self._get_subword_kwargs(side="tgt")
        )
        repr_str += ", tgt_onmttok_kwargs={}".format(self.tgt_other_kwargs)
        return repr_str
