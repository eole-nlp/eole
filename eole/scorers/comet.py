from .scorer import Scorer
from eole.scorers import register_scorer
from comet import download_model, load_from_checkpoint


@register_scorer(metric=["COMET", "COMET-KIWI", "COMETINHO"])
class CometScorer(Scorer):
    """Comet scorer class."""

    def __init__(self, config):
        """Download if needed and load Comet model."""
        super().__init__(config)
        if self.metric == "COMET":
            comet_model_name = "Unbabel/wmt22-comet-da"
        elif self.metric == "COMETINHO":
            comet_model_name = "Unbabel/eamt22-cometinho-da"
        elif self.metric == "COMET-KIWI":
            comet_model_name = "Unbabel/wmt22-cometkiwi-da"
        try:
            comet_model_path = download_model(comet_model_name)
        except KeyError:
            raise PermissionError(
                "Use your hugginface token to download comet-kiwi:\n"
                "huggingface-cli login --token hf_token"
            )
        self.comet_model = load_from_checkpoint(comet_model_path)

    def compute_score(self, preds, texts_refs, texts_srcs):
        data = []
        if self.metric == "COMET-KIWI":
            for _src, _hyp in zip(texts_srcs, preds):
                current_segment = {"src": _src, "mt": _hyp}
                data.append(current_segment)
        else:
            for _src, _tgt, _hyp in zip(texts_srcs, texts_refs, preds):
                current_segment = {"src": _src, "mt": _hyp, "ref": _tgt}
                data.append(current_segment)
        if len(preds) > 0:
            score = self.comet_model.predict(
                data, batch_size=64, gpus=1, num_workers=0, progress_bar=True
            )
            score = score.system_score
        else:
            score = 0
        return score
