"""Experiment execution helpers and stable export format."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .evaluation import hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k


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
            {"name": item.get("name"), "metrics": item.get("metrics", {})} for item in experiments
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
