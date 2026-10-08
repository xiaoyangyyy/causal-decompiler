"""Budgeted intervention oracle.

The search procedure in this module sees factors and query() results only.
It does not read planted labels or import scenario dynamics.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable


def default_budget(n_factors: int) -> int:
    """Enough for empty, singletons, the full set, and one leave-one-out pass."""
    n = max(0, int(n_factors))
    return 2 * n + 4


class SetOracle:
    """Caches unique set interventions. Repeated queries do not spend budget."""

    def __init__(self, value_fn: Callable[[frozenset[str]], float], factors: list[str], budget: int) -> None:
        self._value_fn = value_fn
        self.factors = list(factors)
        self.budget = int(budget)
        self.calls = 0
        self._cache: dict[frozenset[str], float] = {}

    def query(self, active: frozenset[str]) -> float | None:
        key = frozenset(active)
        if key in self._cache:
            return self._cache[key]
        if self.calls >= self.budget:
            return None
        self.calls += 1
        value = float(self._value_fn(key))
        self._cache[key] = value
        return value


class TimedOracle:
    """Clocked interventions: when a factor is deleted, not only whether it is present."""

    def __init__(self, step_fn: Callable[..., float], budget: int) -> None:
        self._step_fn = step_fn
        self.budget = int(budget)
        self.calls = 0
        self._cache: dict[tuple[Any, Any], float] = {}

    def query(self, *, delete_early: int | None, delete_mediator: int | None) -> float | None:
        key = (delete_early, delete_mediator)
        if key in self._cache:
            return self._cache[key]
        if self.calls >= self.budget:
            return None
        self.calls += 1
        value = float(self._step_fn(delete_early=delete_early, delete_mediator=delete_mediator))
        self._cache[key] = value
        return value


def budgeted_recover(
    oracle: SetOracle,
    *,
    alpha: float = 0.9,
) -> dict[str, Any]:
    """Smallest sufficient sets from set queries.

    Monotone AND / OR / suppressor fit in `default_budget`. Non-monotone
    targets (exactly-k) exhaust this procedure and return failed=True.
    """
    factors = list(oracle.factors)
    empty = oracle.query(frozenset())
    if empty is None:
        return _fail(oracle, exhausted=True)
    singles: dict[str, float] = {}
    for factor in factors:
        value = oracle.query(frozenset({factor}))
        if value is None:
            return _fail(oracle, exhausted=True)
        singles[factor] = value
    full = oracle.query(frozenset(factors))
    if full is None:
        return _fail(oracle, exhausted=True)
    reference = max([empty, full, *singles.values()])
    span = reference - empty
    if span <= 1e-12:
        return _fail(oracle, exhausted=False)
    threshold = empty + alpha * span
    singles_hit = [factor for factor in factors if singles[factor] >= threshold - 1e-9]
    if singles_hit:
        return {
            "set": [singles_hit[0]],
            "alternatives": [[factor] for factor in singles_hit],
            "calls": oracle.calls,
            "exhausted": False,
            "failed": False,
        }
    if full < threshold - 1e-9:
        return _fail(oracle, exhausted=False)
    necessary: list[str] = []
    for factor in factors:
        held_out = oracle.query(frozenset(factors) - {factor})
        if held_out is None:
            return _fail(oracle, exhausted=True)
        if held_out < threshold - 1e-9:
            necessary.append(factor)
    restored = oracle.query(frozenset(necessary))
    if restored is None:
        return _fail(oracle, exhausted=True)
    if restored >= threshold - 1e-9 and necessary:
        return {
            "set": list(necessary),
            "alternatives": [list(necessary)],
            "calls": oracle.calls,
            "exhausted": False,
            "failed": False,
        }
    return _fail(oracle, exhausted=oracle.calls >= oracle.budget)


def budgeted_recover_pairs(
    oracle: SetOracle,
    *,
    alpha: float = 0.9,
) -> dict[str, Any]:
    """Monotone recovery, then size-2 queries if that recovery failed.

    This is a different procedure from `budgeted_recover`. Size-3 patterns
    are still out of reach: pairs that do not move Y are not promoted.
    """
    first = budgeted_recover(oracle, alpha=alpha)
    if not first["failed"]:
        return {**first, "procedure": "monotone"}
    factors = list(oracle.factors)
    empty = oracle.query(frozenset())
    if empty is None:
        return {**_fail(oracle, exhausted=True), "procedure": "pairs"}
    best = float(empty)
    pair_values: list[tuple[tuple[str, str], float]] = []
    truncated = False
    for left, right in combinations(factors, 2):
        value = oracle.query(frozenset({left, right}))
        if value is None:
            truncated = True
            break
        pair_values.append(((left, right), float(value)))
        if float(value) > best:
            best = float(value)
    span = best - float(empty)
    if span <= 1e-12:
        failed = _fail(oracle, exhausted=truncated or oracle.calls >= oracle.budget)
        failed["procedure"] = "pairs"
        return failed
    threshold = float(empty) + alpha * span
    hits = [list(pair) for pair, value in pair_values if value >= threshold - 1e-9]
    if not hits:
        failed = _fail(oracle, exhausted=truncated)
        failed["procedure"] = "pairs"
        return failed
    return {
        "set": hits[0],
        "alternatives": hits,
        "calls": oracle.calls,
        "exhausted": truncated,
        "failed": False,
        "procedure": "pairs",
    }


def recover_timed_effects(oracle: TimedOracle, horizon: int) -> dict[str, Any]:
    """Distinguish deleting the early cause before it writes from deleting the mediator later."""
    base = oracle.query(delete_early=None, delete_mediator=None)
    early_before = oracle.query(delete_early=0, delete_mediator=None)
    early_after = oracle.query(delete_early=int(horizon), delete_mediator=None)
    mediator = oracle.query(delete_early=None, delete_mediator=int(horizon))
    if None in {base, early_before, early_after, mediator}:
        return {"failed": True, "calls": oracle.calls, "exhausted": True}
    base_f = float(base)
    return {
        "failed": False,
        "exhausted": False,
        "calls": oracle.calls,
        "early_only_before_write": float(early_before) < base_f - 1e-9 and abs(float(early_after) - base_f) <= 1e-9,
        "mediator_after_delay": float(mediator) < base_f - 1e-9,
        "y_base": base_f,
        "y_delete_early_before": float(early_before),
        "y_delete_early_after": float(early_after),
        "y_delete_mediator": float(mediator),
    }


def _fail(oracle: SetOracle, *, exhausted: bool) -> dict[str, Any]:
    return {
        "set": [],
        "alternatives": [],
        "calls": oracle.calls,
        "exhausted": exhausted,
        "failed": True,
    }
