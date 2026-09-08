"""Planted SCM used as a regression oracle for the decompiler.

Y = X0 AND X1, with X2 unused. Contrastive leave-one-out assigns 0 to both
real causes (the AND lie). Shapley splits the total effect 0.5 / 0.5 / 0.
"""

from __future__ import annotations

from math import factorial
from itertools import combinations
from typing import Callable, Iterable

Factor = str


def planted_factors() -> tuple[Factor, ...]:
    return ("promise", "draft", "decoy")


def planted_outcome(active: Iterable[str]) -> float:
    present = set(active)
    return 1.0 if "promise" in present and "draft" in present else 0.0


def shapley_weight(size: int, n: int) -> float:
    return factorial(size) * factorial(n - size - 1) / factorial(n)


def exact_shapley(value_fn: Callable[[Iterable[str]], float], factors: tuple[Factor, ...]) -> dict[str, float]:
    n = len(factors)
    if n == 0:
        return {}
    phi = {f: 0.0 for f in factors}
    for size in range(n):
        for combo in combinations(factors, size):
            s = frozenset(combo)
            v_s = value_fn(s)
            for factor in factors:
                if factor in s:
                    continue
                v_si = value_fn(s | {factor})
                phi[factor] += shapley_weight(size, n) * (v_si - v_s)
    return phi


def contrastive_leave_one_out(factual: Iterable[str], factors: tuple[Factor, ...]) -> dict[str, float]:
    """Single-factor knockout from the factual world. Lies on AND causes."""
    full = set(factual)
    y0 = planted_outcome(full)
    return {f: y0 - planted_outcome(full - {f}) for f in factors}
