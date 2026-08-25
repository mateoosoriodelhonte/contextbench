"""Standard ranked retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence, Set
from typing import Any


def _hits(ranked: Sequence[Any], relevant: Set[Any], k: int) -> list[Any]:
    return [item for item in ranked[:k] if item in relevant]


def recall_at_k(ranked: Sequence[Any], relevant: Set[Any], k: int) -> float:
    return len(_hits(ranked, relevant, k)) / len(relevant) if relevant else 0.0


def precision_at_k(ranked: Sequence[Any], relevant: Set[Any], k: int) -> float:
    return len(_hits(ranked, relevant, k)) / min(k, len(ranked)) if ranked else 0.0


def hit_rate(ranked: Sequence[Any], relevant: Set[Any], k: int) -> float:
    return 1.0 if _hits(ranked, relevant, k) else 0.0


def mrr(ranked: Sequence[Any], relevant: Set[Any]) -> float:
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked: Sequence[Any], relevant: Set[Any], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(index + 2) for index, item in enumerate(ranked[:k]) if item in relevant
    )
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0
