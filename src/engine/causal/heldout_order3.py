"""Unseen family for the pair-query procedure: Y is on for exactly three causes.

oracle.py must not import this module. Pairs never reach a count of three, so
`budgeted_recover_pairs` fails. That failure is the boundary of the pair
procedure. It is not a reason to add triple queries to the same function.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Any, Callable

from src.engine.causal.mechanisms import MechanismWorld, evaluate_mechanism_world
from src.engine.causal.oracle import SetOracle, budgeted_recover_pairs, default_budget

FAMILY = "exactly_three"
N_CAUSES = 5
N_DECOYS = 1


def _exactly_three(causes: tuple[str, ...], strength: float) -> Callable[[frozenset[str]], float]:
    need = set(causes)

    def value(active: frozenset[str]) -> float:
        return float(strength) if len(need & set(active)) == 3 else 0.0

    return value


def exactly_three_worlds(n: int = 12, seed: int = 11) -> list[MechanismWorld]:
    rng = random.Random(seed)
    causes = tuple(f"c{i}" for i in range(N_CAUSES))
    decoys = tuple(f"d{i}" for i in range(N_DECOYS))
    alternatives = tuple(tuple(combo) for combo in combinations(causes, 3))
    worlds: list[MechanismWorld] = []
    for idx in range(n):
        strength = round(rng.uniform(0.45, 1.0), 3)
        worlds.append(MechanismWorld(
            world_id=f"exactly_three_{idx:02d}",
            family=FAMILY,
            factors=causes + decoys,
            true_minimal=causes[:3],
            true_sign="exactly_three",
            delay=1,
            strength=strength,
            n_causes=N_CAUSES,
            n_decoys=N_DECOYS,
            value_fn=_exactly_three(causes, strength),
            true_alternatives=alternatives,
            timed=False,
        ))
    return worlds


def evaluate_order3(n: int = 12, seed: int = 11) -> dict[str, Any]:
    worlds = exactly_three_worlds(n, seed)
    rows = []
    for world in worlds:
        budget = default_budget(len(world.factors)) + len(world.factors) * (len(world.factors) - 1) // 2
        oracle = SetOracle(world.value_fn, list(world.factors), budget)
        found = budgeted_recover_pairs(oracle, alpha=0.9)
        row = evaluate_mechanism_world(world, list(found["set"]))
        row["failed"] = bool(found["failed"])
        row["procedure"] = found.get("procedure")
        rows.append(row)
    count = len(rows) or 1
    return {
        "family": FAMILY,
        "n": len(rows),
        "cause_set_f1": sum(r["f1"] for r in rows) / count,
        "failure_rate": sum(1.0 if r["failed"] else 0.0 for r in rows) / count,
        "note": "Pair queries do not generate a size-3 set. Failure here is the scope of budgeted_recover_pairs.",
    }
