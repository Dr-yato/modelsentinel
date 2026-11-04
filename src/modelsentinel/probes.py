"""Experimental linear probes on caller-supplied activations.

This follows the public description of Anthropic's defection probes: a linear
direction fit on contrastive activations. ModelSentinel does not ship probe
weights, does not read a model's residual stream unless the caller already
dumped vectors, and does not treat a probe score as proof a model is clean.
"""

from __future__ import annotations

import json
from pathlib import Path


def _mean(rows: list[list[float]]) -> list[float]:
    if not rows:
        raise ValueError("probe needs at least one vector")
    width = len(rows[0])
    acc = [0.0] * width
    for row in rows:
        if len(row) != width:
            raise ValueError("activation vectors must share a width")
        for index, value in enumerate(row):
            acc[index] += value
    return [value / len(rows) for value in acc]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(vector: list[float]) -> list[float]:
    length = sum(value * value for value in vector) ** 0.5
    if length == 0:
        return vector
    return [value / length for value in vector]


def auroc(scores: list[float], labels: list[int]) -> float | None:
    positive = [score for score, label in zip(scores, labels) if label == 1]
    negative = [score for score, label in zip(scores, labels) if label == 0]
    if not positive or not negative:
        return None
    correct = 0.0
    for left in positive:
        for right in negative:
            if left > right:
                correct += 1
            elif left == right:
                correct += 0.5
    return correct / (len(positive) * len(negative))


def contrast_probe(baseline: list[list[float]], triggered: list[list[float]]) -> dict:
    """Fit a difference-of-means direction. Triggered vectors are the positive class."""
    negative = _mean(baseline)
    positive = _mean(triggered)
    direction = _norm([p - n for p, n in zip(positive, negative)])
    scores = [_dot(row, direction) for row in baseline + triggered]
    labels = [0] * len(baseline) + [1] * len(triggered)
    score = auroc(scores, labels)
    return {
        "method": "difference-of-means",
        "experimental": True,
        "auroc": score,
        "dimensions": len(direction),
        "baseline_n": len(baseline),
        "triggered_n": len(triggered),
        "direction_preview": [round(value, 4) for value in direction[:8]],
        "limitation": "A high AUROC on supplied vectors is not evidence about any model whose activations were not in this file, and a low AUROC does not clear a model.",
    }


def load_vectors(path: str) -> list[list[float]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("vectors") or raw.get("activations")
    return [[float(value) for value in row] for row in raw]
