from eole.transforms import register_transform
from .transform import Transform, TransformConfig
from pydantic import Field
import unicodedata
import random


class UpperCaseConfig(TransformConfig):
    upper_corpus_ratio: float | None = Field(default=0.01, description="Corpus ratio to apply uppercasing.")


@register_transform(name="uppercase")
class UpperCaseTransform(Transform):
    """
    Convert source and target examples to uppercase.

    This transform uses `unicodedata` to normalize the converted
    uppercase strings as this is needed for some languages (e.g. Greek).
    One issue is that the normalization removes all diacritics and
    accents from the uppercased strings, even though in few occasions some
    diacritics should be kept even in the uppercased form.
    """

    config_model = UpperCaseConfig

    def __init__(self, config):
        super().__init__(config)

    def _parse_config(self):
        self.upper_corpus_ratio = self.config.upper_corpus_ratio

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Convert source and target examples to uppercase."""
        assert isinstance(example["src"], str), "UpperCaseTransform must be placed before Tokenize"
        if random.random() > self.upper_corpus_ratio:
            return example

        _src = example["src"]
        _src = "".join(c for c in unicodedata.normalize("NFD", _src.upper()) if unicodedata.category(c) != "Mn")
        example["src"] = _src

        if example["tgt"] is not None:
            _tgt = example["tgt"]
            _tgt = "".join(c for c in unicodedata.normalize("NFD", _tgt.upper()) if unicodedata.category(c) != "Mn")
            example["tgt"] = _tgt

        return example
