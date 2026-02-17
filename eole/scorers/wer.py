from .scorer import Scorer
from eole.scorers import register_scorer

try:
    from jiwer import wer
except ImportError:
    wer = None


@register_scorer(metric="WER")
class WerScorer(Scorer):
    """WER (Word Error Rate) scorer class.

    Requires the ``jiwer`` package: ``pip install eole[wer]``
    """

    def __init__(self, config):
        super().__init__(config)

    def compute_score(self, preds, texts_refs):
        if wer is None:
            raise ImportError("jiwer is required for the WER scorer: pip install -e .[wer]")
        if len(preds) > 0:
            score = wer(texts_refs, preds) * 100
        else:
            score = 0
        return score
