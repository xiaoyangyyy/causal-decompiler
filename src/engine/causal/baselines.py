"""Baselines and ablations for RQ1/RQ2 on mechanism worlds.

Baselines: recency, text similarity, LLM-direct ranking, single-event knockout.
Ablations: no typed IR, no slice, knockout-only, independent vs shared Gumbel.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from src.engine.causal.mechanisms import (
    evaluate_mechanism_world,
    parameterized_worlds,
    recover_cstar,
)
from src.engine.causal.noise import STREAM_ACTION_GUMBEL, keyed_gumbel, keyed_uniform


def recency_baseline(world: MechanismWorld, k: int = 1) -> list[str]:
    return list(world.factors[-k:])


def text_similarity_baseline(world: MechanismWorld) -> list[str]:
    target = world.family + "_" + world.true_sign
    scored = []
    for factor in world.factors:
        blob = f"{factor} {world.family}"
        overlap = sum(1 for ch in target if ch in blob)
        scored.append((overlap, factor))
    scored.sort(reverse=True)
    return [scored[0][1]] if scored else []


def llm_direct_baseline(world: MechanismWorld) -> list[str]:
    """Deterministic stand-in for an LLM that names one salient factor."""
    digest = hashlib.sha256(world.world_id.encode("utf-8")).digest()
    idx = digest[0] % len(world.factors)
    return [world.factors[idx]]


def knockout_baseline(world: MechanismWorld) -> list[str]:
    full = world.value_fn(frozenset(world.factors))
    best = None
    best_delta = -1.0
    for factor in world.factors:
        without = frozenset(f for f in world.factors if f != factor)
        delta = abs(full - world.value_fn(without))
        if delta > best_delta:
            best_delta = delta
            best = factor
    return [best] if best is not None else []


def random_candidates(world: MechanismWorld, rng: random.Random, k: int = 2) -> list[str]:
    picks = list(world.factors)
    rng.shuffle(picks)
    return picks[:k]


def knockout_only_recovery(world: MechanismWorld) -> list[str]:
    """Ablation: no restoration / no coalitions, just the largest knockout."""
    return knockout_baseline(world)


def shared_gumbel_choice(seed: int, round_num: int, agent: str, actions: list[str], probs: list[float]) -> str:
    best, best_score = actions[0], float("-inf")
    for action, p in zip(actions, probs):
        g = keyed_gumbel(seed, round_num, STREAM_ACTION_GUMBEL, agent, action)
        score = math.log(max(1e-12, p)) + g
        if score > best_score:
            best_score, best = score, action
    return best


def independent_resample_choice(seed: int, round_num: int, agent: str, actions: list[str], probs: list[float], salt: int) -> str:
    u = keyed_uniform(seed + salt, round_num, STREAM_ACTION_GUMBEL, agent, "needle")
    total = 0.0
    for action, p in zip(actions, probs):
        total += p
        if u <= total:
            return action
    return actions[-1]


def gumbel_agreement(n: int = 40, seed: int = 11) -> dict[str, float]:
    """Shared Gumbel should be identical across two 'worlds'; independent resample should not."""
    actions = ["a", "b", "c"]
    probs = [0.5, 0.3, 0.2]
    shared_match = 0
    independent_match = 0
    for i in range(n):
        s1 = shared_gumbel_choice(seed, i + 1, "agent", actions, probs)
        s2 = shared_gumbel_choice(seed, i + 1, "agent", actions, probs)
        i1 = independent_resample_choice(seed, i + 1, "agent", actions, probs, salt=1)
        i2 = independent_resample_choice(seed, i + 1, "agent", actions, probs, salt=2)
        shared_match += int(s1 == s2)
        independent_match += int(i1 == i2)
    return {
        "shared_agreement": shared_match / n,
        "independent_agreement": independent_match / n,
        "n": float(n),
    }


BASELINES = {
    "recency": recency_baseline,
    "text_similarity": text_similarity_baseline,
    "llm_direct": llm_direct_baseline,
    "knockout": knockout_baseline,
}

ABLATIONS = {
    "no_slice": lambda w, rng: random_candidates(w, rng, k=len(w.true_minimal) or 1),
    "knockout_only": lambda w, rng: knockout_only_recovery(w),
    "oracle_cstar": lambda w, rng: recover_cstar(w),
}


def evaluate_baselines(n_per_family: int = 8, seed: int = 11) -> dict[str, Any]:
    worlds = parameterized_worlds(n_per_family, seed)
    rng = random.Random(seed)
    out: dict[str, Any] = {"baselines": {}, "ablations": {}, "gumbel": gumbel_agreement(seed=seed)}
    for name, fn in BASELINES.items():
        rows = [evaluate_mechanism_world(w, fn(w)) for w in worlds]
        out["baselines"][name] = {
            "f1": sum(r["f1"] for r in rows) / len(rows),
            "false_attribution": sum(r["false_attribution"] for r in rows) / len(rows),
        }
    for name, fn in ABLATIONS.items():
        rows = [evaluate_mechanism_world(w, fn(w, rng)) for w in worlds]
        out["ablations"][name] = {
            "f1": sum(r["f1"] for r in rows) / len(rows),
            "false_attribution": sum(r["false_attribution"] for r in rows) / len(rows),
        }
    oracle = [evaluate_mechanism_world(w) for w in worlds]
    out["oracle"] = {
        "f1": sum(r["f1"] for r in oracle) / len(oracle),
        "sign_accuracy": sum(1.0 if r["sign_ok"] else 0.0 for r in oracle) / len(oracle),
    }
    return out
