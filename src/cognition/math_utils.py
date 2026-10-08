"""Continuous dynamics primitives — smooth gates and saturating responses."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def sigmoid(x: float, scale: float = 1.0) -> float:
    z = clamp(x * scale, -20.0, 20.0)
    return 1.0 / (1.0 + math.exp(-z))


def softplus(x: float, beta: float = 1.0) -> float:
    z = beta * x
    if z > 20:
        return x
    return math.log1p(math.exp(z)) / beta


def tanh_bounded(x: float, scale: float = 1.0) -> float:
    return math.tanh(x * scale)


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    if not values:
        return np.array([])
    t = max(temperature, 1e-6)
    arr = np.array(values, dtype=float) / t
    arr -= arr.max()
    exp = np.exp(arr)
    return exp / exp.sum()


def recency_kernel(age_rounds: float, half_life: float = 12.0) -> float:
    """Exponential recency — always in (0, 1], no cutoff."""
    age = max(0.0, age_rounds)
    return math.exp(-math.log(2) * age / max(half_life, 1e-6))


def precision_weighted_update(
    prior: float,
    observation: float,
    prior_precision: float,
    obs_precision: float,
) -> float:
    """Bayesian precision-weighted fusion — replaces fixed learning rate."""
    pi_p = max(prior_precision, 1e-6)
    pi_o = max(obs_precision, 0.0)
    return clamp((pi_p * prior + pi_o * observation) / (pi_p + pi_o))


def competitive_inhibition(positive: float, negative: float, coupling: float = 1.2) -> tuple[float, float]:
    """Soft mutual suppression between antagonistic channels."""
    pos = clamp(positive)
    neg = clamp(negative)
    pos_new = clamp(pos - coupling * neg * sigmoid(neg * 3))
    neg_new = clamp(neg - coupling * pos * sigmoid(pos * 3))
    return pos_new, neg_new


def truth_status_precision(truth_status: str) -> float:
    """Evidence precision as a smooth function of an ordered status coordinate."""
    coordinate = {
        "false": -1.5,
        "disputed": -0.4,
        "rumored": 0.2,
        "verified": 1.4,
    }.get(str(truth_status), 0.0)
    return logistic_gate(coordinate, center=0.0, steepness=1.6)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / denom)


def entropy(probs: Sequence[float]) -> float:
    arr = np.array([max(p, 1e-12) for p in probs], dtype=float)
    arr /= arr.sum()
    return float(-np.sum(arr * np.log(arr)))


def normalize_simplex(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in values.values())
    if total <= 1e-9:
        n = len(values)
        return {k: 1.0 / n for k in values}
    return {k: max(v, 0.0) / total for k, v in values.items()}


def blend(old: float, new: float, alpha: float) -> float:
    return clamp((1.0 - alpha) * old + alpha * new)


def impulse_response(intensity: float, sensitivity: float, saturation: float = 2.5) -> float:
    """Saturating response curve — smooth, always differentiable in practice."""
    return intensity * sensitivity / (1.0 + abs(intensity) * saturation)


def logistic_gate(x: float, center: float = 0.5, steepness: float = 6.0) -> float:
    """Smooth gate in (0,1) for graded activation."""
    return sigmoid(x - center, scale=steepness)
