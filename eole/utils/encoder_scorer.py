import torch

from eole.utils.misc import clear_gpu_cache


def build_segment_rows(preds, texts_srcs, texts_refs, tokenizer, model, encode_texts_fn):
    if len(texts_srcs) != len(preds):
        raise ValueError(f"preds and texts_srcs lengths differ: {len(preds)} vs {len(texts_srcs)}")
    if texts_refs is not None and len(texts_refs) != len(preds):
        raise ValueError(f"preds and texts_refs lengths differ: {len(preds)} vs {len(texts_refs)}")

    requires_reference = bool(getattr(model, "requires_reference", False))
    input_segments = list(getattr(model, "input_segments", []))
    needs_ref_ids = requires_reference or "ref" in input_segments
    src_ids = encode_texts_fn(texts_srcs, tokenizer, model)
    mt_ids = encode_texts_fn(preds, tokenizer, model)
    ref_ids = encode_texts_fn(texts_refs, tokenizer, model) if needs_ref_ids and texts_refs is not None else None

    def _ids(item):
        return item["ids"] if isinstance(item, dict) else item

    rows = []
    for idx in range(len(preds)):
        mt = mt_ids[idx]
        row = {"src_ids": _ids(src_ids[idx]), "mt_ids": _ids(mt), "mt_text": preds[idx]}
        if isinstance(mt, dict):
            if "offsets" in mt:
                row["mt_offsets"] = mt["offsets"]
            if "decode_token_ids" in mt:
                row["decode_token_ids"] = mt["decode_token_ids"]
        if ref_ids is not None:
            row["ref_ids"] = _ids(ref_ids[idx])
        rows.append(row)
    return rows


def predict_scores_with_oom_retry(model, rows, batch_size, logger=None, progress_callback=None):
    batch_size = max(1, int(batch_size))
    all_scores = []
    i = 0
    while i < len(rows):
        current_bs = min(batch_size, len(rows) - i)
        chunk = rows[i : i + current_bs]
        try:
            scores = model.predict_scores(chunk)
            all_scores.extend(scores.tolist())
            i += current_bs
            if progress_callback is not None:
                progress_callback(i)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e).lower():
                raise
            if current_bs == 1:
                raise
            clear_gpu_cache()
            batch_size = max(1, current_bs // 2)
            if logger is not None:
                logger.warning("OOM at batch %d; halving to %d", current_bs, batch_size)
    return all_scores
