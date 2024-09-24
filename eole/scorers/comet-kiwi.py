from .scorer import Scorer
from eole.scorers import register_scorer
from comet import download_model, load_from_checkpoint


@register_scorer(metric="COMET-KIWI")
class CometKiwiScorer(Scorer):
    """COMET scorer class."""

    def __init__(self, config):
        """Initialize necessary options for sentencepiece."""
        super().__init__(config)
        comet_model_name = "Unbabel/wmt22-cometkiwi-da"
        try:
            comet_model_path = download_model(comet_model_name)
        except KeyError:
            raise PermissionError("Use your hugginface token to download comet-kiwi:\n"
                                  "huggingface-cli login --token hf_token")
        self.comet_model = load_from_checkpoint(comet_model_path)

    def compute_score(self, preds, texts_refs, texts_srcs):
        data = []
        for _src, _hyp in zip(texts_srcs, preds):
            curr = {"src": _src, "mt": _hyp}
            data.append(curr)
        if len(preds) > 0:
            score = self.comet_model.predict(
                data, batch_size=64, gpus=0, num_workers=0, progress_bar=True)
            score = score.system_score
        else:
            score = 0
        return score
