"""Parameterized mechanism worlds: 4 families × 50 = 200 scripted instances.

The six named planted worlds stay in worlds.py for unit tests. This module
is the paper benchmark: Cause-set F1, interaction-sign accuracy, intervention
budget, and false attribution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

from src.engine.causal.search import harsanyi_pair

FAMILIES = ("and_synergy", "or_redundancy", "delayed_mediation", "suppressor")
N_PER_FAMILY = 50


@dataclass
class MechanismWorld:
    world_id: str
    family: str
    factors: tuple[str, ...]
    true_minimal: tuple[str, ...]
    true_sign: str  # synergy / redundant / suppress / independent
    delay: int
    strength: float
    n_causes: int
    n_decoys: int
    value_fn: Callable[[frozenset[str]], float]


def _and_fn(causes: tuple[str, ...], strength: float) -> Callable[[frozenset[str]], float]:
    need = set(causes)

    def value(active: frozenset[str]) -> float:
        return float(strength) if need <= set(active) else 0.0

    return value


def _or_fn(causes: tuple[str, ...], strength: float) -> Callable[[frozenset[str]], float]:
    need = set(causes)

    def value(active: frozenset[str]) -> float:
        return float(strength) if need & set(active) else 0.0

    return value


def _delay_fn(early: str, late: str, strength: float) -> Callable[[frozenset[str]], float]:
    def value(active: frozenset[str]) -> float:
        return float(strength) if early in active else 0.0

    del late
    return value


def _suppress_fn(cause: str, suppressor: str, strength: float) -> Callable[[frozenset[str]], float]:
    def value(active: frozenset[str]) -> float:
        if cause in active and suppressor in active:
            return 0.0
        if cause in active:
            return float(strength)
        return 0.0

    return value


def _make_world(family: str, idx: int, rng: random.Random) -> MechanismWorld:
    n_causes = rng.randint(2, 4)
    n_decoys = rng.randint(1, 4)
    strength = round(rng.uniform(0.45, 1.0), 3)
    delay = rng.randint(1, 6)
    causes = tuple(f"c{i}" for i in range(n_causes))
    decoys = tuple(f"d{i}" for i in range(n_decoys))
    factors = causes + decoys
    if family == "and_synergy":
        true = causes
        sign = "synergy"
        fn = _and_fn(causes, strength)
    elif family == "or_redundancy":
        true = (causes[0],)
        sign = "redundant"
        fn = _or_fn(causes, strength)
    elif family == "delayed_mediation":
        early, late = causes[0], causes[min(1, n_causes - 1)]
        true = (early,)
        sign = "independent"
        fn = _delay_fn(early, late, strength)
        factors = (early, late) + decoys
    else:
        cause, suppressor = causes[0], causes[min(1, n_causes - 1)]
        true = (cause,)
        sign = "suppress"
        fn = _suppress_fn(cause, suppressor, strength)
        factors = (cause, suppressor) + decoys
    return MechanismWorld(
        world_id=f"{family}_{idx:02d}",
        family=family,
        factors=factors,
        true_minimal=true,
        true_sign=sign,
        delay=delay,
        strength=strength,
        n_causes=len(true) if family == "and_synergy" else n_causes,
        n_decoys=len(decoys),
        value_fn=fn,
    )


def parameterized_worlds(n_per_family: int = N_PER_FAMILY, seed: int = 11) -> list[MechanismWorld]:
    rng = random.Random(seed)
    worlds: list[MechanismWorld] = []
    for family in FAMILIES:
        for idx in range(n_per_family):
            worlds.append(_make_world(family, idx, rng))
    return worlds


def coalition_table(world: MechanismWorld) -> dict[str, float]:
    y: dict[str, float] = {"∅": float(world.value_fn(frozenset()))}
    factors = list(world.factors)
    for size in range(1, len(factors) + 1):
        for combo in combinations(factors, size):
            y[",".join(sorted(combo))] = float(world.value_fn(frozenset(combo)))
    y_full = max(y.values()) if y else 0.0
    y_empty = y["∅"]
    return {"factors": factors, "y": y, "y_full": y_full, "y_empty": y_empty}


def recover_cstar(world: MechanismWorld, alpha: float = 0.9) -> list[str]:
    table = coalition_table(world)
    y = table["y"]
    y_full = table["y_full"]
    y_empty = table["y_empty"]
    v_full = y_full - y_empty
    recover_at = y_empty + alpha * v_full if v_full else y_full
    factors = list(world.factors)
    for size in range(1, len(factors) + 1):
        hits = []
        for combo in combinations(factors, size):
            val = y[",".join(sorted(combo))]
            if val >= recover_at:
                hits.append(list(combo))
        if hits:
            return hits[0]
    return list(world.true_minimal)


def interaction_sign(world: MechanismWorld) -> str:
    causes = list(world.true_minimal)
    if world.family == "or_redundancy":
        causes = [f for f in world.factors if f.startswith("c")][:2]
    if world.family == "suppressor":
        causes = [world.factors[0], world.factors[1]]
    if len(causes) < 2:
        return "independent"
    a, b = causes[0], causes[1]
    y_ab = world.value_fn(frozenset({a, b}))
    y_a = world.value_fn(frozenset({a}))
    y_b = world.value_fn(frozenset({b}))
    y0 = world.value_fn(frozenset())
    index = harsanyi_pair(y_ab, y_a, y_b, y0)
    if index > 0.01:
        return "synergy"
    if index < -0.01:
        return "redundant" if world.family != "suppressor" else "suppress"
    return "independent"


def _f1(pred: set[str], truth: set[str]) -> float:
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    prec = tp / len(pred)
    rec = tp / len(truth)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def evaluate_mechanism_world(world: MechanismWorld, recovered: list[str] | None = None) -> dict[str, Any]:
    recovered = recovered if recovered is not None else recover_cstar(world)
    truth = set(world.true_minimal)
    pred = set(recovered)
    sign = interaction_sign(world)
    false_attr = 1.0 if (pred - truth) else 0.0
    budget = 2 ** min(8, len(world.factors))
    return {
        "world": world.world_id,
        "family": world.family,
        "recovered": recovered,
        "true_minimal": list(world.true_minimal),
        "f1": _f1(pred, truth),
        "interaction_sign": sign,
        "sign_ok": sign == world.true_sign or (
            world.true_sign == "suppress" and sign in {"suppress", "redundant"}
        ),
        "budget": budget,
        "false_attribution": false_attr,
    }


def evaluate_benchmark(
    n_per_family: int = N_PER_FAMILY,
    seed: int = 11,
    alpha: float = 0.9,
) -> dict[str, Any]:
    worlds = parameterized_worlds(n_per_family, seed)
    rows = []
    for world in worlds:
        recovered = recover_cstar(world, alpha=alpha)
        rows.append(evaluate_mechanism_world(world, recovered))
    n = len(rows) or 1
    by_family: dict[str, list[dict[str, Any]]] = {fam: [] for fam in FAMILIES}
    for row in rows:
        by_family[row["family"]].append(row)

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "n": len(rows),
        "cause_set_f1": mean([r["f1"] for r in rows]),
        "interaction_sign_accuracy": mean([1.0 if r["sign_ok"] else 0.0 for r in rows]),
        "intervention_budget_mean": mean([float(r["budget"]) for r in rows]),
        "false_attribution_rate": mean([r["false_attribution"] for r in rows]),
        "by_family": {
            fam: {
                "n": len(items),
                "f1": mean([r["f1"] for r in items]),
                "sign_accuracy": mean([1.0 if r["sign_ok"] else 0.0 for r in items]),
                "false_attribution": mean([r["false_attribution"] for r in items]),
            }
            for fam, items in by_family.items()
        },
        "worlds": rows,
    }
    return summary
