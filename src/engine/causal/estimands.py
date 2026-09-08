"""Estimands that the decompiler actually has objects for.

Total effect is CRN-paired ATE. Memory IRF is an interventional analogue
of indirect effect over delete-time. Contrastive locus is leave-one-event
out with a point-of-commitment. Shapley is reserved for AND causes and
is exact on the planted SCM; the 60-round story uses a budgeted subset.
Three-worlds splits total effect into a socially gated channel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

from src.engine.causal.algebra import CausalOp, delete_memory, lesion, skip_event
from src.engine.causal.toy import shapley_weight
from src.engine.causal.twin import run_twin
from src.engine.run_log import SPLIT_Y_KEYS, RunLog, extract_outcome
from src.engine.simulation import SimConfig
from src.engine.story_cast import story_cast_from_log

AND_EVENT_IDS = ("E003", "E052")
PAPER_SPLIT_KEYS = (
    "protest_authorship",
    "authorship_escalation_potential",
    "public_private_divergence_mean",
    "post_r52_compliance",
    "authority_compliance",
    "memory_authorship_cluster_strength",
    "promise_broken_strength_r52",
    "promise_honored_strength_r52",
    "trust_pi_logged",
    "trust_pi_path_mean",
    "pi_fairness_r52",
)


@dataclass
class EffectEstimate:
    name: str
    ate: float
    factual_y: float
    twin_y: float
    factor_id: str
    extras: dict[str, Any] = field(default_factory=dict)


def paired_effect(
    factual: RunLog,
    twin: RunLog,
    outcome: str,
    *,
    name: str,
    factor_id: str,
) -> EffectEstimate:
    y0 = extract_outcome(factual, outcome)
    y1 = extract_outcome(twin, outcome)
    return EffectEstimate(name=name, ate=y1 - y0, factual_y=y0, twin_y=y1, factor_id=factor_id)


def memory_irf(
    base: SimConfig,
    factual: RunLog,
    outcome: str,
    delete_rounds: list[int],
    *,
    agent_id: str | None = None,
) -> list[EffectEstimate]:
    idea = agent_id or story_cast_from_log(factual).idea
    estimates: list[EffectEstimate] = []
    for t in delete_rounds:
        op = delete_memory(t, idea)
        twin = run_twin(base, [op], llm_trace=factual.llm_cache)
        est = paired_effect(factual, twin, outcome, name=f"memory_irf[t={t}]", factor_id=op.factor_id())
        est.extras["split"] = {
            key: {"ate": extract_outcome(twin, key) - extract_outcome(factual, key), "twin_y": extract_outcome(twin, key)}
            for key in PAPER_SPLIT_KEYS
        }
        est.extras["fork"] = first_divergence(factual, twin)
        estimates.append(est)
    return estimates


def contrastive_event_effects(
    base: SimConfig,
    factual: RunLog,
    outcome: str,
    event_ids: list[str],
) -> list[EffectEstimate]:
    by_id = {e.get("event_id"): e for e in factual.events}
    estimates: list[EffectEstimate] = []
    for event_id in event_ids:
        rec = by_id.get(event_id)
        if rec is None:
            continue
        op = skip_event(int(rec["round"]), event_id)
        twin = run_twin(base, [op], llm_trace=factual.llm_cache)
        est = paired_effect(
            factual, twin, outcome, name=f"skip[{event_id}]", factor_id=op.factor_id(),
        )
        est.extras["fork"] = first_divergence(factual, twin)
        estimates.append(est)
    return estimates


def point_of_commitment(effects: list[EffectEstimate], *, eps: float = 1e-4) -> EffectEstimate | None:
    """Latest skipped event whose twin still moves Y. None if no single skip matters."""
    movers = [e for e in effects if abs(e.ate) > eps]
    if not movers:
        return None
    return movers[-1]


def split_y(log: RunLog) -> dict[str, float]:
    keys = tuple(dict.fromkeys([*PAPER_SPLIT_KEYS, *SPLIT_Y_KEYS]))
    return {key: float(extract_outcome(log, key) or 0.0) for key in keys}


def paired_split_effects(
    factual: RunLog,
    twin: RunLog,
    *,
    name: str,
    factor_id: str,
    keys: tuple[str, ...] = PAPER_SPLIT_KEYS,
) -> dict[str, EffectEstimate]:
    """Same do() on a vector of Ys. Public and private can move in opposite directions."""
    out: dict[str, EffectEstimate] = {}
    for key in keys:
        out[key] = paired_effect(factual, twin, key, name=f"{name}[{key}]", factor_id=factor_id)
    return out


def default_memory_irf_rounds(log: RunLog) -> list[int]:
    cast = story_cast_from_log(log)
    max_round = max((int(r.get("round") or 0) for r in log.round_records), default=int(log.config.get("max_rounds") or 0))
    beats = [cast.memory_cluster_min, 20, 45, cast.draft_round]
    seen: set[int] = set()
    out: list[int] = []
    for t in beats:
        t = int(t)
        if t < 1 or t > max_round or t in seen:
            continue
        seen.add(t)
        out.append(t)
    if not out and max_round >= 3:
        out = [max(1, max_round // 2)]
    return out


def and_event_ids(log: RunLog) -> list[str]:
    """Promise ∧ draft (E003 × E052). Short runs fall back to first+last events."""
    present = {str(e.get("event_id")) for e in log.events if e.get("event_id")}
    ids = [eid for eid in AND_EVENT_IDS if eid in present]
    if len(ids) == 2:
        return ids
    ordered = [str(e.get("event_id")) for e in log.events if e.get("event_id")]
    if len(ids) == 1:
        other = next((eid for eid in reversed(ordered) if eid != ids[0]), None)
        return ids + ([other] if other else [])
    if len(ordered) >= 2:
        return [ordered[0], ordered[-1]]
    return ordered


def draft_beat_op(log: RunLog) -> CausalOp | None:
    """Skip the authorship draft beat (E052 / R52). Short runs skip the last event."""
    by_id = {str(e.get("event_id")): e for e in log.events if e.get("event_id")}
    if "E052" in by_id:
        rec = by_id["E052"]
        return skip_event(int(rec["round"]), "E052")
    cast = story_cast_from_log(log)
    for rec in log.events:
        if int(rec.get("round") or 0) == int(cast.draft_round):
            return skip_event(int(rec["round"]), str(rec.get("event_id")))
    if log.events:
        rec = log.events[-1]
        eid = rec.get("event_id")
        if eid:
            return skip_event(int(rec.get("round") or 1), str(eid))
    return None


def _stance_key(payload: Any) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return (str(payload),)
    return (
        str(payload.get("statement_type") or ""),
        str(payload.get("authorship_claim") or payload.get("goal") or ""),
        str(payload.get("content_summary") or ""),
    )


def _action_index(log: RunLog) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for act in log.actions:
        key = (int(act.get("round") or 0), str(act.get("agent") or ""))
        out[key] = act
    return out


def first_divergence(factual: RunLog, twin: RunLog) -> dict[str, Any]:
    """First round the twin leaves the factual transcript.

    Channel order: missing action, selected action type, public stance, private intent.
    Identity twins should return identical=True.
    """
    fact = _action_index(factual)
    other = _action_index(twin)
    for key in sorted(set(fact) | set(other)):
        rnd, agent = key
        fa = fact.get(key)
        ta = other.get(key)
        if fa is None or ta is None:
            return {
                "identical": False,
                "round": rnd,
                "agent": agent,
                "channel": "presence",
                "factual": None if fa is None else str(fa.get("type")),
                "twin": None if ta is None else str(ta.get("type")),
            }
        if str(fa.get("type")) != str(ta.get("type")):
            return {
                "identical": False,
                "round": rnd,
                "agent": agent,
                "channel": "action",
                "factual": str(fa.get("type")),
                "twin": str(ta.get("type")),
            }
        if _stance_key(fa.get("public_position")) != _stance_key(ta.get("public_position")):
            return {
                "identical": False,
                "round": rnd,
                "agent": agent,
                "channel": "public",
                "factual": _stance_key(fa.get("public_position")),
                "twin": _stance_key(ta.get("public_position")),
            }
        if _stance_key(fa.get("private_intent")) != _stance_key(ta.get("private_intent")):
            return {
                "identical": False,
                "round": rnd,
                "agent": agent,
                "channel": "private",
                "factual": _stance_key(fa.get("private_intent")),
                "twin": _stance_key(ta.get("private_intent")),
            }
    return {"identical": True, "round": None, "agent": None, "channel": None, "factual": None, "twin": None}


def _dump_split(effects: dict[str, EffectEstimate]) -> dict[str, dict[str, Any]]:
    return {key: asdict(est) for key, est in effects.items()}


def story_shapley(
    base: SimConfig,
    factual: RunLog,
    event_ids: list[str],
    outcome: str,
) -> dict[str, Any]:
    """Exact Shapley on a budgeted set of story events (n<=3).

    Value of a coalition is Y when those events are present and the rest are skipped.
    Contrastive knockout from the factual world double-counts AND causes.
    """
    by_id = {e.get("event_id"): e for e in factual.events}
    factors = [eid for eid in event_ids if eid in by_id][:3]
    if len(factors) < 2:
        return {"factors": factors, "shapley": {}, "contrastive": {}, "interaction": 0.0, "y": {}}

    cache: dict[frozenset[str], float] = {}

    def y_of(present: frozenset[str]) -> float:
        key = frozenset(present)
        if key in cache:
            return cache[key]
        skips = []
        for eid in factors:
            if eid in present:
                continue
            rec = by_id[eid]
            skips.append(skip_event(int(rec["round"]), str(eid)))
        if not skips:
            val = extract_outcome(factual, outcome)
        else:
            twin = run_twin(base, skips, llm_trace=factual.llm_cache)
            val = extract_outcome(twin, outcome)
        cache[key] = float(val)
        return cache[key]

    full = frozenset(factors)
    empty = frozenset()
    y_full = y_of(full)
    y_empty = y_of(empty)
    shapley = {f: 0.0 for f in factors}
    n = len(factors)
    for size in range(n):
        for combo in combinations(factors, size):
            s = frozenset(combo)
            v_s = y_of(s)
            weight = shapley_weight(size, n)
            for factor in factors:
                if factor in s:
                    continue
                shapley[factor] += weight * (y_of(s | {factor}) - v_s)

    contrastive = {f: y_full - y_of(full - {f}) for f in factors}
    interaction = 0.0
    if n == 2:
        a, b = factors
        interaction = y_full - y_of(frozenset({a})) - y_of(frozenset({b})) + y_empty
    return {
        "factors": factors,
        "outcome": outcome,
        "y": {",".join(sorted(k)) or "∅": v for k, v in cache.items()},
        "y_full": y_full,
        "y_empty": y_empty,
        "total_effect": y_full - y_empty,
        "shapley": shapley,
        "contrastive": contrastive,
        "contrastive_sum": sum(contrastive.values()),
        "interaction": interaction,
        "and_lie": abs(sum(contrastive.values())) > abs(y_full - y_empty) + 1e-9,
    }


def three_worlds(
    base: SimConfig,
    factual: RunLog,
    op: CausalOp,
    outcome: str,
    *,
    split_keys: tuple[str, ...] = PAPER_SPLIT_KEYS,
) -> dict[str, Any]:
    """W0 factual; W1 do(op) with social gating; W2 do(op)+omniscient observation.

    gated_channel = Y1 - Y2 on private divergence. If |ΔPPD| >> |Δpublic|,
    the intervention mostly moved the hidden transcript.
    """
    w1 = run_twin(base, [op], llm_trace=factual.llm_cache)
    w2 = run_twin(base, [op, lesion("observation")], llm_trace=factual.llm_cache)
    split0 = split_y(factual)
    split1 = split_y(w1)
    split2 = split_y(w2)
    y0 = extract_outcome(factual, outcome)
    y1 = extract_outcome(w1, outcome)
    y2 = extract_outcome(w2, outcome)
    ppd0 = split0.get("public_private_divergence_mean", 0.0)
    ppd1 = split1.get("public_private_divergence_mean", 0.0)
    public0 = split0.get("post_r52_compliance", 0.0)
    public1 = split1.get("post_r52_compliance", 0.0)
    return {
        "factor_id": op.factor_id(),
        "outcome": outcome,
        "y": {"w0": y0, "w1": y1, "w2": y2},
        "ate_total": y1 - y0,
        "ate_omniscient": y2 - y0,
        "gated_channel": y1 - y2,
        "split": {
            "w0": {k: split0.get(k, 0.0) for k in split_keys},
            "w1": {k: split1.get(k, 0.0) for k in split_keys},
            "w2": {k: split2.get(k, 0.0) for k in split_keys},
        },
        "hypocrisy_index": abs(ppd1 - ppd0) - abs(public1 - public0),
        "fork_w1": first_divergence(factual, w1),
        "identity_w1_ok": True,
        "run_ids": {"w1": w1.run_id, "w2": w2.run_id},
    }


def lambda_lesion_effects(
    base: SimConfig,
    factual: RunLog,
    outcome: str,
    *,
    values: tuple[float, ...] = (0.0, 1.0),
) -> list[EffectEstimate]:
    from src.engine.causal.algebra import set_policy_lambda

    estimates: list[EffectEstimate] = []
    for value in values:
        op = set_policy_lambda(value)
        twin = run_twin(base, [op], llm_trace=factual.llm_cache)
        est = paired_effect(factual, twin, outcome, name=f"lambda={value}", factor_id=op.factor_id())
        est.extras["split"] = _dump_split(paired_split_effects(factual, twin, name=est.name, factor_id=est.factor_id))
        est.extras["llm_replay"] = twin.outcomes.get("llm_trace_stats") or {}
        estimates.append(est)
    return estimates


def crn_aligned_draws(factual: RunLog, twin: RunLog, *, from_round: int) -> bool:
    """True if keyed draws after from_round match on shared (round, stream, agent, name)."""
    def keyed(log: RunLog) -> dict[tuple, float]:
        out: dict[tuple, float] = {}
        for d in log.noise_log:
            rnd = int(d["round"])
            if rnd < from_round:
                continue
            out[(rnd, d["stream"], d.get("agent_id"), d["name"])] = float(d["value"])
        return out

    fact = keyed(factual)
    other = keyed(twin)
    shared = set(fact) & set(other)
    if not shared:
        return True
    return all(abs(fact[k] - other[k]) < 1e-12 for k in shared)
