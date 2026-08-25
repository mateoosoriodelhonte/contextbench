"""Experiment execution helpers and stable export format."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from .evaluation import hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k


class Span(Protocol):
    id: str
    document_id: str
    start_char: int
    end_char: int


def map_ranked_chunks_to_relevance_anchors(
    ranked: Sequence[Span], anchors: Sequence[Span]
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    """Map ranked chunks to judged spans without letting fragmentation add relevance units."""
    anchor_ids = {anchor.id for anchor in anchors}
    candidates: dict[str, list[str]] = {anchor.id: [] for anchor in anchors}
    overlaps_by_rank: list[list[str]] = []
    for chunk in ranked:
        overlaps = sorted(
            anchor.id
            for anchor in anchors
            if anchor.document_id == chunk.document_id
            and chunk.start_char < anchor.end_char
            and chunk.end_char > anchor.start_char
        )
        overlaps_by_rank.append(overlaps)
        for anchor_id in overlaps:
            candidates[anchor_id].append(chunk.id)

    anchor_to_rank: dict[str, int] = {}

    def assign(rank: int, seen: set[str]) -> bool:
        for anchor_id in overlaps_by_rank[rank]:
            if anchor_id in seen:
                continue
            seen.add(anchor_id)
            owner = anchor_to_rank.get(anchor_id)
            if owner is None or assign(owner, seen):
                anchor_to_rank[anchor_id] = rank
                return True
        return False

    for rank in range(len(ranked)):
        assign(rank, set())

    rank_to_anchor = {rank: anchor_id for anchor_id, rank in anchor_to_rank.items()}
    metric_ranking = [
        rank_to_anchor.get(rank, f"unmatched:{chunk.id}") for rank, chunk in enumerate(ranked)
    ]
    matched_chunks = {
        ranked[rank].id: anchor_id for rank, anchor_id in sorted(rank_to_anchor.items())
    }
    if not set(matched_chunks.values()) <= anchor_ids:
        raise ValueError("relevance mapping produced an unknown anchor")
    return metric_ranking, matched_chunks, candidates


def evaluate_rankings(
    rankings: list[str], relevant: set[str], k_values: list[int]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(rankings, relevant, k)
        metrics[f"precision@{k}"] = precision_at_k(rankings, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(rankings, relevant, k)
        metrics[f"hitRate@{k}"] = hit_rate(rankings, relevant, k)
    metrics["mrr"] = mrr(rankings, relevant)
    return metrics


def compare_experiments(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    if len(experiments) < 2:
        raise ValueError("at least two experiments are required")
    keys = sorted({key for experiment in experiments for key in experiment.get("metrics", {})})
    return {
        "schema": "contextbench.experiment.v1",
        "experiments": [
            {
                "name": item.get("name"),
                "configuration": item.get("configuration", {}),
                "metrics": item.get("metrics", {}),
            }
            for item in experiments
        ],
        "metricKeys": keys,
    }


def export_experiment(experiment: dict[str, Any]) -> str:
    payload = {
        "schema": "contextbench.experiment.v1",
        "exportedAt": datetime.now(UTC).isoformat(),
        **experiment,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
