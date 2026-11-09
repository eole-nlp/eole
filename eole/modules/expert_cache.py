"""Bounded expert residency policy, independent of weight format and execution."""

import torch


class FrequencyExpertCache:
    """Decayed demand counts with bounded, deterministic admission.

    Observations do not change residency. Call commit only after the replacement
    weights have been uploaded successfully. Equal scores never cause churn.
    """

    def __init__(self, experts, capacity, interval=32, swaps=4):
        if not 1 <= capacity <= experts or interval < 1 or swaps < 1:
            raise ValueError("Invalid expert cache capacity or update interval")
        self.scores = torch.zeros(experts, dtype=torch.float32)
        self.resident = set(range(capacity))
        self.interval, self.swaps, self.calls = interval, swaps, 0

    def observe(self, ids):
        self.scores.add_(torch.bincount(ids.reshape(-1), minlength=self.scores.numel()))
        self.calls += 1

    def replacements(self):
        if self.calls % self.interval:
            return []
        scores = self.scores.tolist()
        cold = sorted(self.resident, key=lambda i: (scores[i], i))
        hot = sorted(set(range(len(scores))) - self.resident, key=lambda i: (-scores[i], i))
        result = []
        for old, new in zip(cold[: self.swaps], hot[: self.swaps]):
            if scores[new] <= scores[old] * 1.1:
                break
            result.append((old, new))
        self.scores.mul_(0.9)
        return result

    def commit(self, old, new):
        if old not in self.resident or new in self.resident:
            raise ValueError("Stale expert cache replacement")
        self.resident.remove(old)
        self.resident.add(new)
