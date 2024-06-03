from .scorer import Scorer
from eole.scorers import register_scorer
from sacrebleu import corpus_bleu


@register_scorer(metric="BLEU")
class BleuScorer(Scorer):
    """BLEU scorer class."""

    def __init__(self, config):
        """Initialize necessary options for sentencepiece."""
        super().__init__(config)

    def compute_score(self, preds, texts_refs):
        if len(preds) > 0:
            score = corpus_bleu(preds, [texts_refs]).score
        else:
            score = 0
        return score
