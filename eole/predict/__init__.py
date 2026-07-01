""" Modules for prediction """

from eole.predict.translator import Translator
from eole.predict.generator import GeneratorLM
from eole.predict.encoder import Encoder
from eole.predict.audio_predictor import AudioPredictor
from eole.predict.streamer import GenerationStreamer  # noqa: F401

from eole.predict.beam_search import GNMTGlobalScorer
from eole.decoders.ensemble import EnsembleModel
from eole.models.model import get_model_class
from eole.models.hf_resolver import is_hf_model_id, resolve_preconverted_eole_hf_repo
from eole import EOLE_TORCH_COMPILE


def get_infer_class(model_config):
    if model_config is None:
        raise ValueError(
            "Model config is missing. Cannot select an inference class without a valid EOLE model config. "
            "Check that model_path resolves to a valid EOLE model directory or supported Hugging Face model."
        )
    if getattr(model_config, "architecture", None) == "transformer_encoder_scorer":
        from eole.predict.encoder_scorer import EncoderScorer

        return EncoderScorer
    if getattr(model_config, "architecture", None) == "transformer_encoder_decoder_scorer":
        from eole.predict.encoder_decoder_scorer import EncoderDecoderScorer

        return EncoderDecoderScorer
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


def _load_local_eole_model(config, device_id):
    model_class = get_model_class(config.model)
    return model_class.for_inference(config, device_id)


def build_predictor(config, device_id=0, report_score=True, logger=None):
    # right now config is a full (non nested) PredictConfig

    if len(config.model_path) > 1:
        # Ensemble case
        model, vocabs, model_config = EnsembleModel.for_inference(config, device_id)
    elif is_hf_model_id(config.model_path[0]):
        local_model_path = resolve_preconverted_eole_hf_repo(
            config.model_path[0], token=getattr(config, "hf_token", None)
        )
        if local_model_path is not None:
            config.model_path = [local_model_path]
            config._update_with_model_config()
            model, vocabs, model_config = _load_local_eole_model(config, device_id)
        else:
            # Direct HuggingFace inference — no pre-conversion required
            from eole.models.hf_loader import load_hf_model

            model, vocabs, model_config = load_hf_model(config, device_id)
    else:
        # Single EOLE model case
        model, vocabs, model_config = _load_local_eole_model(config, device_id)

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
