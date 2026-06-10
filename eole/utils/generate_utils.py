"""
Utilities for generating text from in-memory batches using a model.

This module provides a lightweight generation interface that can be used
during training (e.g. for RL rollouts, validation scoring) without
requiring file-based I/O or full PredictConfig construction.
"""

import torch
from eole.predict import GNMTGlobalScorer, get_infer_class
from eole.predict.prediction import PredictionBuilder


class GenerationConfig:
    """Minimal generation configuration.

    Provides the subset of PredictConfig attributes needed by the
    Inference/Translator/GeneratorLM classes for batch-level generation.
    """

    def __init__(
        self,
        beam_size=1,
        n_best=1,
        max_length=256,
        max_length_ratio=0,
        context_length=0,
        min_length=0,
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        ban_unk_token=False,
        block_ngram_repeat=0,
        ignore_when_blocking=None,
        replace_unk=False,
        tgt_file_prefix=False,
        phrase_table="",
        data_type="text",
        verbose=False,
        report_time=False,
        dump_beam="",
        stepwise_penalty=False,
        ratio=0.0,
        report_align=False,
        gold_align=False,
        report_score=False,
        with_score=False,
        estim_only=False,
        seed=-1,
        optional_eos=None,
        transforms=None,
        self_attn_backend="",
        dynamic_shapes=None,
        fuse_kvq=False,
        fuse_gate=False,
        attn_debug=False,
        align_debug=False,
    ):
        self.beam_size = beam_size
        self.n_best = n_best
        self.max_length = max_length
        self.max_length_ratio = max_length_ratio
        self.context_length = context_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.ban_unk_token = ban_unk_token
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking or []
        self.replace_unk = replace_unk
        self.tgt_file_prefix = tgt_file_prefix
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_time = report_time
        self.dump_beam = dump_beam
        self.stepwise_penalty = stepwise_penalty
        self.ratio = ratio
        self.report_align = report_align
        self.gold_align = gold_align
        self.report_score = report_score
        self.with_score = with_score
        self.estim_only = estim_only
        self.seed = seed
        self.optional_eos = optional_eos or []
        self.transforms = transforms or []
        self.self_attn_backend = self_attn_backend
        self.dynamic_shapes = dynamic_shapes
        self.fuse_kvq = fuse_kvq
        self.fuse_gate = fuse_gate
        self.attn_debug = attn_debug
        self.align_debug = align_debug


def build_generator(model, vocabs, model_config, device_id=0, gen_config=None):
    """Build a lightweight predictor/generator for in-memory batch generation.

    This is a simpler alternative to ``build_predictor`` that doesn't require
    a full PredictConfig or model loading from disk — it uses an already-loaded
    model (e.g. the training model).

    Args:
        model: The model to use for generation (already loaded, can be in train mode).
        vocabs: Vocabulary dict.
        model_config: The model configuration object.
        device_id: GPU device ID (-1 for CPU).
        gen_config: A GenerationConfig instance. If None, uses defaults (greedy, beam=1).

    Returns:
        An Inference subclass instance (Translator/GeneratorLM/etc.) ready to
        call ``predict_batch()`` on in-memory batches.
    """
    if gen_config is None:
        gen_config = GenerationConfig()

    scorer = GNMTGlobalScorer(alpha=0.0, beta=0.0, length_penalty="none", coverage_penalty="none")

    infer_class = get_infer_class(model_config)
    predictor = infer_class(
        model,
        vocabs,
        gen_config,
        model_config,
        device_id=device_id,
        global_scorer=scorer,
        report_score=False,
        logger=None,
    )
    return predictor


def generate_from_batch(predictor, batch, return_token_ids=False):
    """Generate predictions from an in-memory batch using a predictor.

    This is the core utility for RL and other training-time generation needs.
    It runs generation on a single batch without file I/O.

    Args:
        predictor: An Inference subclass (from ``build_generator`` or ``build_predictor``).
        batch: A batch dict with at minimum 'src' and 'srclen' keys.
            If 'ind_in_bucket' is present, it will be used for stable ordering;
            otherwise sequential indices are assigned automatically.
        return_token_ids: If True, also return raw token ID sequences (before detokenization).

    Returns:
        A dict with:
            - 'predictions': list of list of strings (batch_size x n_best)
            - 'scores': list of list of floats (batch_size x n_best prediction scores)
            - 'estim': list of list of floats (batch_size x n_best estimator scores, if available)
            - 'token_ids': (optional) list of list of LongTensor (batch_size x n_best token sequences)
    """
    was_training = predictor.model.training
    predictor.model.eval()

    try:
        with torch.no_grad():
            batch_data = predictor.predict_batch(batch, attn_debug=False)
    finally:
        if was_training:
            predictor.model.train()

    # Ensure batch has 'ind_in_bucket' for PredictionBuilder.from_batch()
    # If not present (e.g. when called from training), add sequential indices.
    if "ind_in_bucket" not in batch_data:
        batch_size = batch["src"].size(0)
        batch_data["ind_in_bucket"] = torch.arange(batch_size, device=batch["src"].device)

    # Build predictions from raw batch data
    prediction_builder = PredictionBuilder(
        predictor.vocabs,
        predictor.n_best,
        predictor.replace_unk,
        predictor.phrase_table,
        predictor._tgt_eos_idx,
        predictor.id_tokenization,
    )
    translations = prediction_builder.from_batch(batch_data)
    translations = sorted(translations, key=lambda x: x.ind_in_bucket)

    results = {
        "predictions": [],
        "scores": [],
        "estim": [],
    }

    if return_token_ids:
        results["token_ids"] = []

    for trans in translations:
        # Predictions as strings
        if predictor.id_tokenization:
            preds = trans.pred_sents[: predictor.n_best]
        else:
            preds = [" ".join(pred) for pred in trans.pred_sents[: predictor.n_best]]
        results["predictions"].append(preds)
        results["scores"].append(trans.pred_scores[: predictor.n_best])
        results["estim"].append(trans.estim[: predictor.n_best])

        if return_token_ids:
            # Use the original index to correctly align with raw predictions
            raw_preds = batch_data["predictions"]
            results["token_ids"].append(raw_preds[trans.ind_in_bucket][: predictor.n_best])

    return results


def generate_and_score(predictor, batch, scorers, texts_ref=None, texts_src=None, return_token_ids=False):
    """Generate predictions and score them with provided scorers.

    Combines generation and scoring in a single call — useful for RL
    where you need both the generated text and its reward signal.

    Args:
        predictor: An Inference subclass (from ``build_generator``).
        batch: A batch dict with 'src' and 'srclen' keys.
        scorers: Dict of {metric_name: scorer_obj} (eole Scorer instances).
        texts_ref: Reference texts for scoring (list of strings). Required by most scorers.
        texts_src: Source texts for scoring (list of strings). Optional.
        return_token_ids: If True, also return raw token ID sequences.

    Returns:
        A dict with generation results plus:
            - 'rewards': dict of {metric_name: list of scores} for each generated prediction
    """
    gen_results = generate_from_batch(predictor, batch, return_token_ids=return_token_ids)

    # Flatten predictions for scoring (take first/best prediction per sample)
    preds_flat = [p[0] for p in gen_results["predictions"]]

    rewards = {}
    for metric_name, scorer in scorers.items():
        if isinstance(scorer, dict):
            # Handle the {"scorer": obj, "value": float} dict format from build_scorers
            scorer_obj = scorer["scorer"]
        else:
            scorer_obj = scorer
        score = scorer_obj.compute_score(preds_flat, texts_ref, texts_src)
        rewards[metric_name] = score

    gen_results["rewards"] = rewards
    return gen_results
