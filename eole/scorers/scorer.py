"""Base Scorer class and relate utils."""


class Scorer(object):
    """A Base class that every scoring method should derived from."""

    def __init__(self, config, model_name=None):
        # not used by any scorer for now it seems, but why not
        self.config = config
        self.model_name = model_name

    def compute_score(self, preds, texts_refs, texts_srcs):
        raise NotImplementedError


def build_scorers(config, scorers_cls):
    """Build scorers in `scorers_cls`."""
    scorers = {}
    for metric, scorer_cls in scorers_cls.items():
        scorer_obj = scorer_cls(config)
        scorers[metric] = {"scorer": scorer_obj, "value": 0}
    return scorers
