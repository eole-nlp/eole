from .scorer import Scorer
from eole.scorers import register_scorer
from sacrebleu import corpus_chrf


@register_scorer(metric="CHRF")
class ChrFScorer(Scorer):
    """chrF scorer class."""

    def __init__(self, config):
        """Initialize the chrF scorer with the given configuration."""
        super().__init__(config)

    def compute_score(self, preds, texts_refs, texts_srcs=None):
        if len(preds) > 0:
            score = corpus_chrf(preds, [texts_refs]).score
        else:
            score = 0
        return score
