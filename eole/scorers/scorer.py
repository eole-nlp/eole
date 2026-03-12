"""Base Scorer class and relate utils."""


class Scorer(object):
    """A Base class that every scoring method should derived from."""

    uses_gpu = False  # Set to True for scorers that require a GPU (e.g. COMET)

    def __init__(self, config):
        # not used by any scorer for now it seems, but why not
        self.config = config

    def compute_score(self, preds, texts_refs, texts_srcs=None):
        raise NotImplementedError


def build_scorers(config, scorers_cls):
    """Build scorers in `scorers_cls`."""
    scorers = {}
    for metric, scorer_cls in scorers_cls.items():
        scorer_obj = scorer_cls(config)
        scorers[metric] = {"scorer": scorer_obj, "value": 0}
    return scorers
