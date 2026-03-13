from .scorer import Scorer
from eole.scorers import register_scorer
from sacrebleu import corpus_bleu


@register_scorer(metric="BLEU_zh")
class BleuZhScorer(Scorer):
    """BLEU scorer class for Chinese."""

    def __init__(self, config):
        """Initialize the BLEU_zh scorer with the given configuration."""
        super().__init__(config)

    def compute_score(self, preds, texts_refs, texts_srcs=None):
        if len(preds) > 0:
            score = corpus_bleu(preds, [texts_refs], tokenize="zh").score
        else:
            score = 0
        return score
