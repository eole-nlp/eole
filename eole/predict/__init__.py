""" Modules for prediction """
from eole.predict.translator import Translator
from eole.predict.generator import GeneratorLM
from eole.predict.encoder import Encoder

from eole.predict.beam_search import BeamSearch, GNMTGlobalScorer
from eole.predict.beam_search import BeamSearchLM
from eole.predict.decode_strategy import DecodeStrategy
from eole.predict.greedy_search import GreedySearch, GreedySearchLM
from eole.predict.penalties import PenaltyBuilder
from eole.decoders.ensemble import load_test_model as ensemble_load_test_model
from eole.models import BaseModel
import codecs


__all__ = [
    "Translator",
    "BeamSearch",
    "GNMTGlobalScorer",
    "PenaltyBuilder",
    "DecodeStrategy",
    "GreedySearch",
    "GreedySearchLM",
    "BeamSearchLM",
    "GeneratorLM",
]


def get_infer_class(model_config):
    # might have more cases later
    if model_config.decoder is None:
        return Encoder
    elif model_config.encoder is None:
        return GeneratorLM
    else:
        return Translator


def build_predictor(config, device_id=0, report_score=True, logger=None, out_file=None):
    # right now config is a full (non nested) PredictConfig
    if out_file is None:
        out_file = codecs.open(config.output, "w+", "utf-8")

    load_test_model = (
        ensemble_load_test_model
        if len(config.model_path) > 1
        else BaseModel.load_test_model
    )

    vocabs, model, model_config = load_test_model(config, device_id)
    config.model = model_config

    scorer = GNMTGlobalScorer.from_config(config)

    infer_class = get_infer_class(model_config)
    if infer_class is not None:
        translator = infer_class.from_config(
            model,
            vocabs,
            config,
            model_config,
            global_scorer=scorer,
            out_file=out_file,
            report_align=config.report_align,
            report_score=report_score,
            logger=logger,
        )
    else:
        raise NotImplementedError(f"No infer_class found for {model_config.__class__}")
    return translator
