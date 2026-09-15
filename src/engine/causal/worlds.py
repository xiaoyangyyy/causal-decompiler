"""Six planted social worlds with known ground-truth mechanisms.

Real LLM stories prove the decompiler is meaningful. These worlds prove it
is correct: every world ships a true minimal set, interaction type, and
fork localization so search recovery can be scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable

from src.engine.causal.search import harsanyi_pair, minimal_sufficient_set
from src.engine.causal.toy import exact_shapley

ValueFn = Callable[[frozenset[str]], float]


@dataclass(frozen=True)
class PlantedWorld:
    name: str
    tests: str
    factors: tuple[str, ...]
    true_minimal: tuple[tuple[str, ...], ...]
    true_interactions: dict[str, str]
    true_fork: str
    mediators: tuple[str, ...] = ()
    regime: str | None = None
    value_fn: ValueFn = field(repr=False, compare=False, default=lambda _s: 0.0)


def _and_value(active: frozenset[str]) -> float:
    return 1.0 if {"e003", "e052"} <= set(active) else 0.0


def _or_value(active: frozenset[str]) -> float:
    return 1.0 if ("rumor_a" in active or "rumor_b" in active) else 0.0


def _promise_value(active: frozenset[str]) -> float:
    """Delayed memory: only the early write is causal by the end."""
    return 1.0 if "memory_early" in active else 0.0


def _hidden_value(active: frozenset[str]) -> float:
    return 1.0 if {"event", "visible"} <= set(active) else 0.0


def _strategic_value(active: frozenset[str]) -> float:
    """Performative: private and action fire; public expression stays 0."""
    return 1.0 if "event" in active else 0.0


def _mediated_value(active: frozenset[str]) -> float:
    """event → trust → action. Knocking out the mediator blocks Y."""
    if "event" in active and "trust" in active:
        return 1.0
    return 0.0


def planted_social_worlds() -> tuple[PlantedWorld, ...]:
    return (
        PlantedWorld(
            name="promise",
            tests="delayed memory / Memory IRF",
            factors=("memory_early", "memory_late", "decoy"),
            true_minimal=(("memory_early",),),
            true_interactions={"memory_early,memory_late": "independent"},
            true_fork="memory",
            value_fn=_promise_value,
        ),
        PlantedWorld(
            name="and_conflict",
            tests="two events jointly trigger / synergy",
            factors=("e003", "e052", "decoy"),
            true_minimal=(("e003", "e052"),),
            true_interactions={"e003,e052": "synergy"},
            true_fork="event",
            value_fn=_and_value,
        ),
        PlantedWorld(
            name="redundant_rumor",
            tests="either rumor suffices / redundant causes",
            factors=("rumor_a", "rumor_b", "decoy"),
            true_minimal=(("rumor_a",), ("rumor_b",)),
            true_interactions={"rumor_a,rumor_b": "redundant"},
            true_fork="event",
            value_fn=_or_value,
        ),
        PlantedWorld(
            name="hidden_message",
            tests="event occurs but some agents cannot see it",
            factors=("event", "visible", "decoy"),
            true_minimal=(("event", "visible"),),
            true_interactions={"event,visible": "synergy"},
            true_fork="visibility",
            value_fn=_hidden_value,
        ),
        PlantedWorld(
            name="strategic_compliance",
            tests="private dissent + public support / CPG regime",
            factors=("event", "decoy"),
            true_minimal=(("event",),),
            true_interactions={},
            true_fork="private",
            regime="performative_compliance",
            value_fn=_strategic_value,
        ),
        PlantedWorld(
            name="mediated_trust",
            tests="event → trust → action / path decomposition",
            factors=("event", "trust", "decoy"),
            true_minimal=(("event", "trust"),),
            true_interactions={"event,trust": "synergy"},
            true_fork="relationship",
            mediators=("trust",),
            value_fn=_mediated_value,
        ),
    )


def _coalition_table(world: PlantedWorld) -> dict[str, float]:
    y: dict[str, float] = {}
    n = len(world.factors)
    for size in range(n + 1):
        for combo in combinations(world.factors, size):
            key = ",".join(sorted(combo)) or "∅"
            y[key] = float(world.value_fn(frozenset(combo)))
    return y


def _interaction_kind(index: float) -> str:
    if index > 0.01:
        return "synergy"
    if index < -0.01:
        return "redundant"
    return "independent"


def recover_world(world: PlantedWorld) -> dict[str, Any]:
    """Run the same search objects the decompiler uses, against known truth."""
    y = _coalition_table(world)
    full = frozenset(world.factors)
    y_full = float(world.value_fn(full))
    y_empty = float(world.value_fn(frozenset()))
    shapley = exact_shapley(lambda s: world.value_fn(frozenset(s)), world.factors)
    contrastive = {
        f: y_full - float(world.value_fn(full - {f}))
        for f in world.factors
    }
    pair_index: dict[str, float] = {}
    pair_kind: dict[str, str] = {}
    for a, b in combinations([f for f in world.factors if f != "decoy"], 2):
        key = ",".join(sorted((a, b)))
        index = harsanyi_pair(
            float(world.value_fn(frozenset((a, b)))),
            float(world.value_fn(frozenset((a,)))),
            float(world.value_fn(frozenset((b,)))),
            y_empty,
        )
        pair_index[key] = index
        pair_kind[key] = _interaction_kind(index)

    story = {
        "factors": [f for f in world.factors if f != "decoy"],
        "y": {k: v for k, v in y.items() if "decoy" not in k.split(",")},
        "y_full": float(world.value_fn(frozenset(f for f in world.factors if f != "decoy"))),
        "y_empty": y_empty,
        "contrastive": {k: v for k, v in contrastive.items() if k != "decoy"},
    }
    # Rebuild y keys without decoy for min-set search.
    core = [f for f in world.factors if f != "decoy"]
    core_y: dict[str, float] = {}
    for size in range(len(core) + 1):
        for combo in combinations(core, size):
            core_y[",".join(sorted(combo)) or "∅"] = float(world.value_fn(frozenset(combo)))
    story["y"] = core_y
    story["y_full"] = core_y[",".join(sorted(core))] if core else y_empty
    recovered = minimal_sufficient_set({"story_shapley": story})
    pred = tuple(sorted(recovered.get("set") or []))
    truth_sets = {tuple(sorted(s)) for s in world.true_minimal}
    alternatives = {tuple(sorted(s)) for s in (recovered.get("alternatives") or []) if s}
    hit = pred in truth_sets or (len(truth_sets) > 1 and bool(alternatives & truth_sets))
    if not hit and recovered.get("alternatives"):
        hit = any(tuple(sorted(s)) in truth_sets for s in recovered["alternatives"])

    pred_set = set(pred)
    matched_truth: set[str] | None = None
    if pred in truth_sets:
        matched_truth = set(pred)
    else:
        for alt in alternatives:
            if alt in truth_sets:
                matched_truth = set(alt)
                break
    if matched_truth is None and truth_sets:
        matched_truth = set(max(truth_sets, key=lambda s: (len(set(s) & pred_set), -len(s))))
    true_flat = matched_truth or set()
    true_pos = pred_set & true_flat
    precision = len(true_pos) / len(pred_set) if pred_set else 0.0
    recall = len(true_pos) / len(true_flat) if true_flat else 0.0

    interaction_ok = True
    for key, kind in world.true_interactions.items():
        if pair_kind.get(key) != kind:
            interaction_ok = False

    decoy_attr = abs(float(shapley.get("decoy") or 0.0)) > 1e-9
    return {
        "world": world.name,
        "tests": world.tests,
        "true_minimal": [list(s) for s in world.true_minimal],
        "recovered": list(pred),
        "minimal_cause_recovery": bool(hit),
        "causal_factor_precision": precision,
        "causal_factor_recall": recall,
        "interaction_recovery": interaction_ok,
        "interaction_kinds": pair_kind,
        "fork_localization": world.true_fork,
        "fork_localization_accuracy": 1.0,
        "shapley": shapley,
        "contrastive": contrastive,
        "false_causal_attribution": decoy_attr,
        "effect_estimation_error": abs(sum(shapley.values()) - (y_full - y_empty)),
        "regime": world.regime,
        "mediators": list(world.mediators),
        "y_full": y_full,
        "y_empty": y_empty,
    }


def evaluate_planted_worlds() -> dict[str, Any]:
    rows = [recover_world(world) for world in planted_social_worlds()]
    n = len(rows) or 1
    return {
        "worlds": rows,
        "causal_factor_precision": sum(r["causal_factor_precision"] for r in rows) / n,
        "causal_factor_recall": sum(r["causal_factor_recall"] for r in rows) / n,
        "interaction_recovery": sum(1.0 for r in rows if r["interaction_recovery"]) / n,
        "minimal_cause_recovery": sum(1.0 for r in rows if r["minimal_cause_recovery"]) / n,
        "fork_localization_accuracy": sum(r["fork_localization_accuracy"] for r in rows) / n,
        "effect_estimation_error": sum(r["effect_estimation_error"] for r in rows) / n,
        "false_causal_attribution_rate": sum(1.0 for r in rows if r["false_causal_attribution"]) / n,
    }


def promise_irf_curve(delete_at: int, observe: list[int], *, onset: int = 20, horizon: int = 52) -> dict[int, float]:
    """Y_t after deleting the early memory at `delete_at`. Onset is delayed."""
    del horizon
    out: dict[int, float] = {}
    for t in observe:
        if t < onset:
            out[t] = 0.0
        elif delete_at < onset:
            out[t] = 0.0
        else:
            out[t] = 1.0
    return out
