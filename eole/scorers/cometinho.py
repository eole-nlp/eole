from .scorer import Scorer
from eole.scorers import register_scorer
from comet import download_model, load_from_checkpoint


@register_scorer(metric="COMETINHO")
class CometinhoScorer(Scorer):
    """Cometinho scorer class."""

    def __init__(self, config):
        """Download if needed and load Cometinho model."""
        super().__init__(config)
        comet_model_name = "Unbabel/eamt22-cometinho-da"
        comet_model_path = download_model(comet_model_name)
        self.comet_model = load_from_checkpoint(comet_model_path)

    def compute_score(self, preds, texts_refs, texts_srcs):
        data = []
        for _src, _tgt, _hyp in zip(texts_srcs, texts_refs, preds):
            curr = {"src": _src, "mt": _hyp, "ref": _tgt}
            data.append(curr)
        if len(preds) > 0:
            score = self.comet_model.predict(
                data, batch_size=64, gpus=0, num_workers=0, progress_bar=True
            )
            score = score.system_score
        else:
            score = 0
        return score
