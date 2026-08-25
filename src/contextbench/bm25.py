"""Small, real Okapi BM25 implementation for local corpora."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


class BM25:
    def __init__(
        self, documents: Iterable[tuple[str, str]], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.term_counts: list[Counter[str]] = []
        self.df: Counter[str] = Counter()
        for identifier, text in documents:
            terms = tokenize(text)
            counts = Counter(terms)
            self.ids.append(identifier)
            self.term_counts.append(counts)
            self.df.update(counts.keys())
        self.avgdl = (
            sum(sum(c.values()) for c in self.term_counts) / len(self.term_counts)
            if self.term_counts
            else 0.0
        )

    def score(self, query: str, index: int) -> float:
        if not self.term_counts or not (0 <= index < len(self.term_counts)):
            return 0.0
        query_terms = tokenize(query)
        counts = self.term_counts[index]
        length = sum(counts.values())
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            df = self.df.get(term, 0)
            idf = math.log(1 + (len(self.ids) - df + 0.5) / (df + 0.5))
            denominator = (
                frequency + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                if self.avgdl
                else 1.0
            )
            score += idf * frequency * (self.k1 + 1) / denominator
        return score

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        return sorted(
            ((identifier, self.score(query, i)) for i, identifier in enumerate(self.ids)),
            key=lambda pair: (-pair[1], pair[0]),
        )[:limit]


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], *, k: int = 60, limit: int | None = None
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings.values():
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)
    result = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return result[:limit] if limit is not None else result
