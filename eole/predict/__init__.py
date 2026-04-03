""" Modules for prediction """

import os

from eole.predict.translator import Translator
from eole.predict.generator import GeneratorLM
from eole.predict.encoder import Encoder
from eole.predict.audio_predictor import AudioPredictor
from eole.predict.streamer import GenerationStreamer  # noqa: F401

from eole.predict.beam_search import GNMTGlobalScorer
from eole.decoders.ensemble import EnsembleModel
from eole.models.model import get_model_class
from eole import EOLE_TORCH_COMPILE


def get_infer_class(model_config):
    # might have more cases later
    if model_config.decoder is None:
        return Encoder
    elif model_config.encoder is None:
        return GeneratorLM
    elif model_config.encoder.encoder_type == "vision":
        return GeneratorLM
    elif getattr(model_config.encoder, "data_type", "text") == "audio":
        return AudioPredictor
    else:
        return Translator


def _is_hf_model_id(model_path: str) -> bool:
    """Return True if *model_path* looks like a HuggingFace model ID.

    A path is treated as an HF model ID when it satisfies all of:

    * it is not an absolute filesystem path (does not start with ``/``),
    * it is not a relative path that begins with ``.`` (e.g. ``./model``),
    * it contains exactly one ``/`` separator (``owner/model-name`` format),
    * and it does **not** exist as a local filesystem path.

    This avoids misclassifying local directories that happen to contain a
    ``/`` separator (e.g. ``/mnt/models/llama``).
    """
    s = str(model_path)
    if os.path.isabs(s) or s.startswith("."):
        return False
    return "/" in s and not os.path.exists(s)


def build_predictor(config, device_id=0, report_score=True, logger=None):
    # right now config is a full (non nested) PredictConfig

    if len(config.model_path) > 1:
        # Ensemble case
        model, vocabs, model_config = EnsembleModel.for_inference(config, device_id)
    elif _is_hf_model_id(config.model_path[0]):
        # Direct HuggingFace inference — no pre-conversion required
        from eole.models.hf_loader import load_hf_model

        model, vocabs, model_config = load_hf_model(config, device_id)
    else:
        # Single EOLE model case
        model_class = get_model_class(config.model)
        model, vocabs, model_config = model_class.for_inference(config, device_id)

    if EOLE_TORCH_COMPILE:
        import torch._dynamo.config as dynamo_config

        dynamo_config.cache_size_limit = 256  # default is 8
        dynamo_config.accumulated_cache_size_limit = 2048  # default is 256
        dynamo_config.skip_nnmodule_hook_guards = True
        dynamo_config.assume_static_by_default = True

        import torch._inductor.config as inductor_config

        inductor_config.compile_threads = 24  # default is 24
        inductor_config.fx_graph_cache = True  # default is True
        inductor_config.unsafe_skip_cache_dynamic_shape_guards = True  # default is False - much faster compilation
        inductor_config.max_autotune = False  # default is False
        inductor_config.triton.cudagraph_skip_dynamic_graphs = False  # default is False - when True much slower
        inductor_config.force_disable_caches = False  # Keep caches enabled

    config.update(model=model_config)

    scorer = GNMTGlobalScorer.from_config(config)

    infer_class = get_infer_class(model_config)
    if infer_class is not None:
        predictor = infer_class(
            model,
            vocabs,
            config,
            model_config,
            device_id=device_id,
            global_scorer=scorer,
            report_score=report_score,
            logger=logger,
        )
    else:
        raise NotImplementedError(f"No infer_class found for {model_config.__class__}")
    return predictor
