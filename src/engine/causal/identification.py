"""Identification statements for C*_alpha and the counterexamples that bound them.

Propositions are only claimed inside the stated scope. Outside that scope the
same statements stay assumptions or are recorded as counterexamples.
"""

from __future__ import annotations

from typing import Any

from src.engine.causal.baselines import knockout_baseline
from src.engine.causal.heldout_family import evaluate_heldout
from src.engine.causal.mechanisms import (
    MechanismWorld,
    _and_fn,
    _or_fn,
    interaction_report,
    recover_detail,
)
from src.engine.causal.search import harsanyi_set

# Strings consumed by certificates. Status is in the prefix so a reader can
# tell a proved scope from an assumption that was not identified.
IDENTIFICATION_STATEMENTS = (
    "A1 assumption: CRN identity twin on frozen U",
    "A2 assumption: LLM prompt replay until the first prompt mismatch",
    "A3 assumption: Memory IRF is an interventional analogue, not a natural indirect effect",
    "A4 assumption: Minimal cause set is searched over evaluated ops, not a full do-algebra",
    "A5 assumption: Reported fork is earliest meaningful (semantic/behavioral), not string mismatch",
    "P1 proposition: On a deterministic monotone set-oracle, budgeted_recover with budget >= 2n+3 returns one minimal sufficient set and, for OR, the list of size-1 alternatives. Non-monotone exactly-k families are outside P1.",
)


def _world(
    family: str,
    factors: tuple[str, ...],
    true: tuple[str, ...],
    alternatives: tuple[tuple[str, ...], ...],
    sign: str,
    value_fn,
) -> MechanismWorld:
    return MechanismWorld(
        world_id=f"counterexample_{family}",
        family=family,
        factors=factors,
        true_minimal=true,
        true_sign=sign,
        delay=1,
        strength=1.0,
        n_causes=len(true),
        n_decoys=0,
        value_fn=value_fn,
        true_alternatives=alternatives,
    )


def suppressor_counterexample() -> dict[str, Any]:
    """Knockout from the full set credits the suppressor. Recovery must not."""
    factors = ("cause", "suppressor", "decoy")

    def value(active: frozenset[str]) -> float:
        if "cause" in active and "suppressor" in active:
            return 0.0
        if "cause" in active:
            return 1.0
        return 0.0

    world = _world(
        "suppressor",
        factors,
        ("cause",),
        (("cause",),),
        "suppress",
        value,
    )
    knocked = knockout_baseline(world)
    found = recover_detail(world)
    return {
        "name": "suppressor",
        "knockout": knocked,
        "recovered": found["set"],
        "knockout_is_suppressor": knocked == ["suppressor"],
        "recovered_is_cause": found["set"] == ["cause"],
        "holds": knocked == ["suppressor"] and found["set"] == ["cause"] and not found["failed"],
    }


def or_counterexample() -> dict[str, Any]:
    causes = ("c0", "c1", "c2")
    world = _world(
        "or_redundancy",
        causes,
        ("c0",),
        tuple((c,) for c in causes),
        "redundant",
        _or_fn(causes, 1.0),
    )
    found = recover_detail(world)
    alternatives = [tuple(item) for item in found["alternatives"]]
    recovered = tuple(found["set"])
    return {
        "name": "or_redundancy",
        "recovered": found["set"],
        "alternatives": [list(item) for item in found["alternatives"]],
        "holds": (
            len(alternatives) == 3
            and recovered in set(alternatives)
            and not found["failed"]
        ),
    }


def and_order_k_counterexample() -> dict[str, Any]:
    """Pairwise dividend is 0 for k=3. The k-order dividend is the joint effect.

    Summing single deletions overcounts the total effect.
    """
    causes = ("c0", "c1", "c2")
    value = _and_fn(causes, 1.0)
    world = _world("and_synergy", causes, causes, (causes,), "synergy", value)
    signs = interaction_report(world)
    full = value(frozenset(causes))
    empty = value(frozenset())
    deletion_sum = 0.0
    for factor in causes:
        held = value(frozenset(causes) - {factor})
        deletion_sum += full - held
    k_order = harsanyi_set(value, causes)
    found = recover_detail(world)
    return {
        "name": "and_k3",
        "pairwise_dividend": signs["pairwise"],
        "k_order_dividend": k_order,
        "deletion_sum": deletion_sum,
        "total_effect": full - empty,
        "recovered": found["set"],
        "holds": (
            abs(float(signs["pairwise"])) <= 0.01
            and k_order > 0.5
            and deletion_sum > (full - empty) + 0.5
            and set(found["set"]) == set(causes)
        ),
    }


def heldout_failure() -> dict[str, Any]:
    report = evaluate_heldout(n=8, seed=11)
    return {
        "name": "exactly_two",
        "f1": report["cause_set_f1"],
        "failure_rate": report["failure_rate"],
        "budget_curve": report["budget_curve"],
        "holds": report["failure_rate"] == 1.0 and all(point["f1"] == 0.0 for point in report["budget_curve"]),
    }


def alpha_sensitivity(seed: int = 11) -> list[dict[str, Any]]:
    """Monotone families are insensitive to alpha in {0.8, 0.9, 0.95}."""
    from src.engine.causal.mechanisms import evaluate_benchmark

    rows = []
    for alpha in (0.8, 0.9, 0.95):
        bench = evaluate_benchmark(n_per_family=4, seed=seed, alpha=alpha)
        rows.append({
            "alpha": alpha,
            "cause_set_f1": bench["cause_set_f1"],
            "interaction_sign_accuracy": bench["interaction_sign_accuracy"],
        })
    return rows


def identification_report(seed: int = 11) -> dict[str, Any]:
    cases = [
        suppressor_counterexample(),
        or_counterexample(),
        and_order_k_counterexample(),
        heldout_failure(),
    ]
    return {
        "statements": list(IDENTIFICATION_STATEMENTS),
        "counterexamples": cases,
        "all_hold": all(case["holds"] for case in cases),
        "alpha_sensitivity": alpha_sensitivity(seed),
    }
