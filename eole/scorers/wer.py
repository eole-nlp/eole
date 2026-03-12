from eole.utils.logging import logger
from .scorer import Scorer
from eole.scorers import register_scorer

try:
    from jiwer import wer
except ImportError:
    wer = None


def _build_normalizer(mode):
    """Build a text normalizer function for the given mode."""
    if mode == "none":
        return None
    elif mode == "lowercase":
        import jiwer

        transform = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.Strip(),
                jiwer.RemoveMultipleSpaces(),
            ]
        )
        return transform
    elif mode == "whisper_en":
        from whisper_normalizer.english import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        return normalizer
    elif mode == "whisper_basic":
        from whisper_normalizer.basic import BasicTextNormalizer

        normalizer = BasicTextNormalizer()
        return normalizer
    else:
        raise ValueError(f"Unknown wer_normalize mode: {mode}")


@register_scorer(metric="WER")
class WerScorer(Scorer):
    """WER (Word Error Rate) scorer class.

    Requires the ``jiwer`` package: ``pip install eole[wer]``
    """

    def __init__(self, config):
        super().__init__(config)
        normalize_mode = getattr(config, "wer_normalize", "none")
        self._normalizer = _build_normalizer(normalize_mode)

    def compute_score(self, preds, texts_refs, texts_srcs=None):
        if wer is None:
            raise ImportError("jiwer is required for the WER scorer: pip install -e .[wer]")
        if len(preds) > 0:
            if self._normalizer is not None:
                preds = [self._normalizer(p) for p in preds]
                texts_refs = [self._normalizer(r) for r in texts_refs]
            pairs = [(p, r) for p, r in zip(preds, texts_refs) if r.strip()]
            if not pairs:
                return 0
            preds, texts_refs = zip(*pairs)
            preds = list(preds)
            texts_refs = list(texts_refs)
            corpus_score = wer(texts_refs, preds) * 100
            sent_wers = [min(wer([r], [p]), 1.0) for r, p in zip(texts_refs, preds)]
            mean_sent_score = sum(sent_wers) / len(sent_wers) * 100
            catastrophic = sum(1 for s in sent_wers if s >= 1.0)
            logger.info(
                "WER: corpus=%.2f%% mean_sent=%.2f%% catastrophic=%d/%d",
                corpus_score,
                mean_sent_score,
                catastrophic,
                len(sent_wers),
            )
            score = corpus_score
        else:
            score = 0
        return score
