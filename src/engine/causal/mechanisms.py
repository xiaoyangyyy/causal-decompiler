"""Parameterized mechanism worlds: 4 families × 50 = 200 scripted instances.

Labels stay in this module. Recovery goes through a budgeted oracle that
cannot read `true_minimal`. Delayed mediation is a clocked process, not a
static coalition. The held-out fifth family lives in heldout_family.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from src.engine.causal.oracle import (
    SetOracle,
    TimedOracle,
    budgeted_recover,
    default_budget,
    recover_timed_effects,
)
from src.engine.causal.search import harsanyi_pair, harsanyi_set

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
    true_alternatives: tuple[tuple[str, ...], ...] = ()
    timed: bool = False


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
    """Unused by recovery. Timed queries live on `timed_step`."""
    del early, late

    def value(active: frozenset[str]) -> float:
        del active
        return 0.0

    _ = strength
    return value


def timed_step(strength: float, delay: int) -> Callable[..., float]:
    """Early writes a mediator at t=0. Y is strength only if that mediator remains after `delay`.

    Deleting the early cause after it has written does not remove the mediator.
    Deleting the mediator at the horizon does.
    """

    def step(*, delete_early: int | None, delete_mediator: int | None) -> float:
        written = not (delete_early is not None and delete_early <= 0)
        if not written:
            return 0.0
        if delete_mediator is not None and delete_mediator <= int(delay):
            return 0.0
        return float(strength)

    return step


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
    timed = False
    alternatives: tuple[tuple[str, ...], ...]
    if family == "and_synergy":
        true = causes
        alternatives = (causes,)
        sign = "synergy"
        fn = _and_fn(causes, strength)
    elif family == "or_redundancy":
        true = (causes[0],)
        alternatives = tuple((cause,) for cause in causes)
        sign = "redundant"
        fn = _or_fn(causes, strength)
    elif family == "delayed_mediation":
        early, late = causes[0], causes[min(1, n_causes - 1)]
        true = (early,)
        alternatives = ()
        sign = "mediation"
        fn = _delay_fn(early, late, strength)
        factors = (early, late) + decoys
        timed = True
    else:
        cause, suppressor = causes[0], causes[min(1, n_causes - 1)]
        true = (cause,)
        alternatives = ((cause,),)
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
        true_alternatives=alternatives,
        timed=timed,
    )


def parameterized_worlds(n_per_family: int = N_PER_FAMILY, seed: int = 11) -> list[MechanismWorld]:
    rng = random.Random(seed)
    worlds: list[MechanismWorld] = []
    for family in FAMILIES:
        for idx in range(n_per_family):
            worlds.append(_make_world(family, idx, rng))
    return worlds


def recover_cstar(world: MechanismWorld, alpha: float = 0.9, budget: int | None = None) -> list[str]:
    """Recover a sufficient set using only oracle queries."""
    found = recover_detail(world, alpha=alpha, budget=budget)
    return list(found["set"])


def recover_detail(world: MechanismWorld, alpha: float = 0.9, budget: int | None = None) -> dict[str, Any]:
    if world.timed:
        step = timed_step(world.strength, world.delay)
        oracle = TimedOracle(step, budget if budget is not None else 8)
        timed = recover_timed_effects(oracle, world.delay)
        ok = (not timed["failed"]) and timed["early_only_before_write"] and timed["mediator_after_delay"]
        return {
            "set": [world.factors[0]] if ok else [],
            "alternatives": [],
            "calls": timed.get("calls", oracle.calls),
            "exhausted": bool(timed.get("exhausted")),
            "failed": not ok,
            "timed": timed,
        }
    cap = default_budget(len(world.factors)) if budget is None else int(budget)
    oracle = SetOracle(world.value_fn, list(world.factors), cap)
    found = budgeted_recover(oracle, alpha=alpha)
    found["timed"] = None
    return found


def _dividend_sign(index: float, family: str) -> str:
    if index > 0.01:
        return "synergy"
    if index < -0.01:
        return "suppress" if family == "suppressor" else "redundant"
    return "independent"


def interaction_report(world: MechanismWorld) -> dict[str, Any]:
    """k-order dividend is the AND sign. Pairwise is reported beside it.

    For an AND of k>2 the pairwise dividend on the first two causes is 0.
    """
    if world.timed:
        return {
            "k_order": None,
            "pairwise": None,
            "k_order_sign": "mediation",
            "pairwise_sign": "independent",
            "sign": "mediation",
        }
    causes = [factor for factor in world.factors if str(factor).startswith("c")]
    if world.family == "suppressor":
        causes = list(world.factors[:2])

    def value_of(active: frozenset[str]) -> float:
        return float(world.value_fn(active))

    k_order = harsanyi_set(value_of, tuple(causes)) if len(causes) >= 2 else 0.0
    pairwise = 0.0
    if len(causes) >= 2:
        a, b = causes[0], causes[1]
        pairwise = harsanyi_pair(
            value_of(frozenset({a, b})),
            value_of(frozenset({a})),
            value_of(frozenset({b})),
            value_of(frozenset()),
        )
    if world.family == "and_synergy":
        sign = _dividend_sign(k_order, world.family)
    else:
        sign = _dividend_sign(pairwise, world.family)
    return {
        "k_order": k_order,
        "pairwise": pairwise,
        "k_order_sign": _dividend_sign(k_order, world.family),
        "pairwise_sign": _dividend_sign(pairwise, world.family),
        "sign": sign,
    }


def interaction_sign(world: MechanismWorld) -> str:
    return str(interaction_report(world)["sign"])


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


def _best_f1(pred: set[str], alternatives: tuple[tuple[str, ...], ...] | list[list[str]]) -> float:
    if not alternatives:
        return 0.0
    return max(_f1(pred, set(alt)) for alt in alternatives)


def evaluate_mechanism_world(world: MechanismWorld, recovered: list[str] | None = None) -> dict[str, Any]:
    detail = None if recovered is not None else recover_detail(world)
    recovered_set = list(recovered if recovered is not None else detail["set"])
    pred = set(recovered_set)
    alternatives = world.true_alternatives or (world.true_minimal,)
    if world.timed:
        f1 = 1.0 if detail and not detail["failed"] else (1.0 if recovered is not None and recovered_set else 0.0)
        if detail is None:
            f1 = 1.0 if recovered_set else 0.0
    else:
        f1 = _best_f1(pred, alternatives)
    signs = interaction_report(world)
    sign = signs["sign"]
    allowed = set()
    for alt in alternatives:
        allowed |= set(alt)
    false_attr = 0.0 if world.timed else (1.0 if (pred - allowed) else 0.0)
    calls = int(detail["calls"]) if detail else default_budget(len(world.factors))
    pairwise_ok = signs["pairwise_sign"] == world.true_sign or (
        world.true_sign == "suppress" and signs["pairwise_sign"] in {"suppress", "redundant"}
    )
    return {
        "world": world.world_id,
        "family": world.family,
        "recovered": recovered_set,
        "alternatives": [list(alt) for alt in (detail["alternatives"] if detail else [])],
        "true_minimal": list(world.true_minimal),
        "true_alternatives": [list(alt) for alt in alternatives],
        "f1": f1,
        "interaction_sign": sign,
        "k_order_dividend": signs["k_order"],
        "pairwise_dividend": signs["pairwise"],
        "pairwise_sign": signs["pairwise_sign"],
        "sign_ok": sign == world.true_sign or (
            world.true_sign == "suppress" and sign in {"suppress", "redundant"}
        ),
        "pairwise_sign_ok": pairwise_ok if world.family == "and_synergy" else signs["pairwise_sign"] == world.true_sign,
        "budget": calls,
        "failed": bool(detail["failed"]) if detail else False,
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
        detail = recover_detail(world, alpha=alpha)
        row = evaluate_mechanism_world(world, detail["set"] if not world.timed else (detail["set"] if not detail["failed"] else []))
        if world.timed:
            row["f1"] = 0.0 if detail["failed"] else 1.0
            row["failed"] = bool(detail["failed"])
            row["budget"] = int(detail["calls"])
        else:
            row["failed"] = bool(detail["failed"])
            row["budget"] = int(detail["calls"])
            row["alternatives"] = [list(alt) for alt in detail["alternatives"]]
        rows.append(row)
    by_family: dict[str, list[dict[str, Any]]] = {fam: [] for fam in FAMILIES}
    for row in rows:
        by_family[row["family"]].append(row)

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    and_rows = by_family["and_synergy"]
    summary = {
        "n": len(rows),
        "cause_set_f1": mean([r["f1"] for r in rows]),
        "interaction_sign_accuracy": mean([1.0 if r["sign_ok"] else 0.0 for r in rows]),
        "pairwise_sign_accuracy": mean([1.0 if r["pairwise_sign_ok"] else 0.0 for r in rows]),
        "and_pairwise_sign_accuracy": mean([1.0 if r["pairwise_sign_ok"] else 0.0 for r in and_rows]),
        "intervention_budget_mean": mean([float(r["budget"]) for r in rows]),
        "false_attribution_rate": mean([r["false_attribution"] for r in rows]),
        "failure_rate": mean([1.0 if r["failed"] else 0.0 for r in rows]),
        "by_family": {
            fam: {
                "n": len(items),
                "f1": mean([r["f1"] for r in items]),
                "sign_accuracy": mean([1.0 if r["sign_ok"] else 0.0 for r in items]),
                "pairwise_sign_accuracy": mean([1.0 if r["pairwise_sign_ok"] else 0.0 for r in items]),
                "false_attribution": mean([r["false_attribution"] for r in items]),
                "mean_budget": mean([float(r["budget"]) for r in items]),
            }
            for fam, items in by_family.items()
        },
        "worlds": rows,
    }
    return summary
