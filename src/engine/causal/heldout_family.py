"""Held-out fifth family: Y is on only when exactly two of four causes are active.

search.py and oracle.py must not import this module. The monotone budgeted
procedure does not propose the pair queries this family needs, so recovery
fails. That failure is a result, not a tuning target.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from src.engine.causal.mechanisms import MechanismWorld, evaluate_mechanism_world, recover_detail
from src.engine.causal.oracle import default_budget

FAMILY = "exactly_two"
N_CAUSES = 4
N_DECOYS = 2


def _exactly_two(causes: tuple[str, ...], strength: float) -> Callable[[frozenset[str]], float]:
    need = set(causes)

    def value(active: frozenset[str]) -> float:
        return float(strength) if len(need & set(active)) == 2 else 0.0

    return value


def exactly_two_worlds(n: int = 20, seed: int = 11) -> list[MechanismWorld]:
    rng = random.Random(seed)
    worlds: list[MechanismWorld] = []
    causes = tuple(f"c{i}" for i in range(N_CAUSES))
    decoys = tuple(f"d{i}" for i in range(N_DECOYS))
    alternatives = tuple(tuple(sorted(pair)) for i, a in enumerate(causes) for b in causes[i + 1:] for pair in [(a, b)])
    for idx in range(n):
        strength = round(rng.uniform(0.45, 1.0), 3)
        worlds.append(MechanismWorld(
            world_id=f"exactly_two_{idx:02d}",
            family=FAMILY,
            factors=causes + decoys,
            true_minimal=causes[:2],
            true_sign="exactly_two",
            delay=1,
            strength=strength,
            n_causes=N_CAUSES,
            n_decoys=N_DECOYS,
            value_fn=_exactly_two(causes, strength),
            true_alternatives=alternatives,
            timed=False,
        ))
    return worlds


def evaluate_heldout(
    n: int = 20,
    seed: int = 11,
    budgets: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    worlds = exactly_two_worlds(n, seed)
    width = default_budget(len(worlds[0].factors)) if worlds else 0
    curve_budgets = budgets or (width, width * 2, width * 4)

    def at_budget(budget: int) -> dict[str, float]:
        rows = []
        for world in worlds:
            detail = recover_detail(world, budget=budget)
            row = evaluate_mechanism_world(world, list(detail["set"]))
            row["failed"] = bool(detail["failed"])
            rows.append(row)
        count = len(rows) or 1
        return {
            "budget": float(budget),
            "f1": sum(r["f1"] for r in rows) / count,
            "failure_rate": sum(1.0 if r["failed"] else 0.0 for r in rows) / count,
        }

    curve = [at_budget(int(b)) for b in curve_budgets]
    primary = curve[0]
    return {
        "family": FAMILY,
        "n": len(worlds),
        "default_budget": width,
        "cause_set_f1": primary["f1"],
        "failure_rate": primary["failure_rate"],
        "budget_curve": curve,
        "note": (
            "The monotone leave-one-out procedure never queries the size-2 sets "
            "this family requires. A larger budget inside the same procedure does not repair that."
        ),
    }
