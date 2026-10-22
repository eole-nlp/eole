"""Target sampling for deterministic (one-hot) MTP proposals."""

import torch


def apply_token_penalties(log_probs, history, presence_penalty=0.0, frequency_penalty=0.0):
    """Penalize generated history only; prompt/BOS tokens are not counted."""
    if not presence_penalty and not frequency_penalty:
        return log_probs
    counts = torch.zeros_like(log_probs, dtype=torch.float32)
    counts.scatter_add_(1, history, torch.ones_like(history, dtype=torch.float32))
    return log_probs.float() - presence_penalty * counts.gt(0) - frequency_penalty * counts


def sample_target_chunk(
    log_probs, drafts, history, temperature, top_k, top_p, presence_penalty=0.0, frequency_penalty=0.0
):
    """Draw target tokens conditional on each hypothetical accepted draft prefix.

    Accept the longest prefix matching these samples. At the first mismatch the
    sampled token is already the target correction, distributed as p conditioned
    on not selecting the deterministic proposal. Discard later samples. This is
    equivalent in distribution to one-hot speculative rejection sampling, not
    necessarily in seeded output or RNG consumption to sequential sampling.
    """
    from eole.predict.greedy_search import sample_with_temperature

    rows = []
    for index in range(log_probs.size(1)):
        prefix = torch.cat((history, drafts[:, :index]), dim=1)
        rows.append(apply_token_penalties(log_probs[:, index], prefix, presence_penalty, frequency_penalty))
    scores = torch.stack(rows, dim=1)
    tokens, _ = sample_with_temperature(scores.flatten(0, 1), temperature, top_k, top_p)
    return tokens.view(log_probs.shape[:2])
