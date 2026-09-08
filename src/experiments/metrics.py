"""Metrics computation for Agent MRI reports."""

from __future__ import annotations

import math
from typing import Any

from src.engine.run_log import RunLog
from src.engine.story_cast import story_cast_from_log

SNAPSHOT_ROUNDS = (1, 20, 40, 60)


def compute_run_metrics(log: RunLog) -> dict[str, Any]:
    """Aggregate per-run metrics for reports and CSV export."""
    cast = story_cast_from_log(log)
    idea = cast.idea
    outcomes = dict(log.outcomes)
    timeline = _causal_memory_timeline(log, agent_id=idea)
    trust_curve = _trust_fragmentation_curve(log)
    authorship_curve = _authorship_dispute_curve(log)
    divergence_peaks = _divergence_ranked_rounds(log)

    return {
        "run_id": log.run_id,
        "experiment_id": log.config.get("experiment_id"),
        "condition_id": log.config.get("condition_id"),
        "seed": log.config.get("seed"),
        "outcomes": outcomes,
        "timeline": timeline,
        "trust_fragmentation_curve": trust_curve,
        "authorship_dispute_curve": authorship_curve,
        "trust_snapshots": _trust_snapshots(log, source=idea, rounds=cast.snapshot_rounds),
        "divergence_peaks": divergence_peaks,
        "critic_count": len(log.critic_violations),
        "intervention_count": len(log.interventions_applied),
        "round_count": len(log.round_records),
        "behavioral_trace_metrics": _behavioral_trace_metrics(log, agent_id=idea),
        "action_field_explanations": _action_field_explanations(log),
        "llm_scoring_influence": _llm_scoring_influence(log),
        "llm_influence_footprint": _llm_influence_footprint(log),
        "critic_audit_metrics": _critic_audit_metrics(log),
        "power_surface_final": _power_surface_from_log(log),
        "path_level_causal_chain": _path_level_causal_chain(log, agent_id=idea),
    }





def _field_force_items(decomp: dict[str, Any], n: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values: dict[str, float] = {}
    values["baseline"] = float(decomp.get("baseline", 0.0))
    for name, value in (decomp.get("motive_contributions") or {}).items():
        values[str(name)] = float(value)
    for name in ("event_affinity", "repetition_penalty", "avoidance_penalty", "timing_adjustment", "emotional_bonus", "noise", "floor_adjustment"):
        values[name] = float(decomp.get(name, 0.0))
    positives = sorted(
        ({"force": k, "contribution": round(v, 4)} for k, v in values.items() if v > 0),
        key=lambda item: item["contribution"],
        reverse=True,
    )[:n]
    negatives = sorted(
        ({"force": k, "contribution": round(v, 4)} for k, v in values.items() if v < 0),
        key=lambda item: item["contribution"],
    )[:n]
    return positives, negatives


def _action_field_explanations(log: RunLog, limit: int = 8) -> list[dict[str, Any]]:
    explanations: list[dict[str, Any]] = []
    for action in log.actions:
        selected = action.get("selected_action") or {}
        decomp = selected.get("field_decomposition") or {}
        if not decomp:
            continue
        positives, negatives = _field_force_items(decomp)
        explanations.append({
            "round": action.get("round"),
            "agent": action.get("agent"),
            "action": selected.get("type") or action.get("type"),
            "target": selected.get("target") or action.get("target"),
            "field_tendency": selected.get("social_physics_tendency", selected.get("tendency", 0.0)),
            "fused_tendency": selected.get("fused_tendency", selected.get("tendency", 0.0)),
            "probability": selected.get("probability", 0.0),
            "positive_forces": positives,
            "negative_forces": negatives,
            "memory_trigger": decomp.get("memory_trigger") or {},
        })
    explanations.sort(key=lambda item: (int(item.get("round") or 0), str(item.get("agent") or "")))
    return explanations[:limit]

def _ranking(items: list[dict[str, Any]], key: str, reverse: bool = True) -> list[str]:
    ranked = sorted(items, key=lambda item: float(item.get(key, 0.0)), reverse=reverse)
    return [str(item.get("type", "unknown")) for item in ranked]


def _top_rank_items(items: list[dict[str, Any]], key: str, n: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(items, key=lambda item: float(item.get(key, 0.0)), reverse=True)[:n]
    return [{"type": str(item.get("type", "unknown")), "score": round(float(item.get(key, 0.0)), 4)} for item in ranked]


def _rank_of(action_type: str, ranking: list[str]) -> int:
    try:
        return ranking.index(action_type) + 1
    except ValueError:
        return len(ranking) + 1


def _kendall_rank_shift(a: list[str], b: list[str]) -> float:
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return 0.0
    pos_a = {x: i for i, x in enumerate(a)}
    pos_b = {x: i for i, x in enumerate(b)}
    inversions = 0
    total = 0
    for i, left in enumerate(common):
        for right in common[i + 1:]:
            total += 1
            if (pos_a[left] - pos_a[right]) * (pos_b[left] - pos_b[right]) < 0:
                inversions += 1
    return inversions / total if total else 0.0


def _llm_scoring_influence(log: RunLog, limit: int = 8) -> dict[str, Any]:
    """Summarize how much LLM candidate scoring changed action rankings."""
    examples: list[dict[str, Any]] = []
    pressures: list[float] = []
    selected_rank_shifts: list[float] = []

    for action in log.actions:
        candidates = action.get("action_candidates") or []
        selected = action.get("selected_action") or {}
        audit = action.get("llm_action_scoring") or {}
        if not candidates or audit.get("source") not in {"field_llm_fused", "dual_engine_fused", "llm_native_generated", "llm_native_fallback"}:
            continue
        if not all("llm_score" in c for c in candidates):
            continue

        field_rank = _ranking(candidates, "tendency")
        llm_rank = _ranking(candidates, "llm_score")
        fused_rank = _ranking(candidates, "fused_tendency")
        selected_type = str(selected.get("type") or action.get("type") or "unknown")
        field_selected_rank = _rank_of(selected_type, field_rank)
        fused_selected_rank = _rank_of(selected_type, fused_rank)
        llm_selected_rank = _rank_of(selected_type, llm_rank)
        pressure = _kendall_rank_shift(field_rank, fused_rank)
        selected_shift = max(0, field_selected_rank - fused_selected_rank)
        pressures.append(pressure)
        selected_rank_shifts.append(float(selected_shift))

        if pressure > 0 or selected_shift > 0 or len(examples) < 2:
            examples.append({
                "round": action.get("round"),
                "agent": action.get("agent"),
                "selected_action": selected_type,
                "field_top3": _top_rank_items(candidates, "tendency"),
                "llm_top3": _top_rank_items(candidates, "llm_score"),
                "fused_top3": _top_rank_items(candidates, "fused_tendency"),
                "selected_ranks": {
                    "field": field_selected_rank,
                    "llm": llm_selected_rank,
                    "fused": fused_selected_rank,
                },
                "override_pressure": round(pressure, 4),
                "selected_rank_lift": selected_shift,
            })

    examples.sort(key=lambda item: (float(item.get("override_pressure", 0.0)), float(item.get("selected_rank_lift", 0.0))), reverse=True)
    return {
        "mean_override_pressure": round(sum(pressures) / len(pressures), 4) if pressures else 0.0,
        "max_override_pressure": round(max(pressures), 4) if pressures else 0.0,
        "mean_selected_rank_lift": round(sum(selected_rank_shifts) / len(selected_rank_shifts), 4) if selected_rank_shifts else 0.0,
        "scored_action_count": len(pressures),
        "examples": examples[:limit],
    }


def _llm_influence_footprint(log: RunLog, limit: int = 8) -> dict[str, Any]:
    """Measure field-vs-LLM perceived-intention shift before final fusion."""
    field_to_llm: list[float] = []
    selected_lifts: list[float] = []
    examples: list[dict[str, Any]] = []
    for action in log.actions:
        candidates = action.get("action_candidates") or []
        selected = action.get("selected_action") or {}
        audit = action.get("llm_action_scoring") or {}
        if not candidates or audit.get("source") not in {"field_llm_fused", "dual_engine_fused", "llm_native_generated", "llm_native_fallback"}:
            continue
        if not all("llm_score" in c for c in candidates):
            continue
        field_rank = _ranking(candidates, "tendency")
        llm_rank = _ranking(candidates, "llm_score")
        fused_rank = _ranking(candidates, "fused_tendency")
        selected_type = str(selected.get("type") or action.get("type") or "unknown")
        field_rank_selected = _rank_of(selected_type, field_rank)
        llm_rank_selected = _rank_of(selected_type, llm_rank)
        pressure = _kendall_rank_shift(field_rank, llm_rank)
        lift = max(0, field_rank_selected - llm_rank_selected)
        field_to_llm.append(pressure)
        selected_lifts.append(float(lift))
        private = action.get("private_intent") or {}
        public = action.get("public_position") or {}
        examples.append({
            "round": action.get("round"),
            "agent": action.get("agent"),
            "selected_action": selected_type,
            "field_rank": field_rank_selected,
            "llm_rank": llm_rank_selected,
            "fused_rank": _rank_of(selected_type, fused_rank),
            "field_to_llm_pressure": round(pressure, 4),
            "selected_llm_rank_lift": lift,
            "private_strategy": private.get("strategy"),
            "public_statement_type": public.get("statement_type"),
            "field_top3": _top_rank_items(candidates, "tendency"),
            "llm_top3": _top_rank_items(candidates, "llm_score"),
        })
    examples.sort(key=lambda item: (float(item.get("field_to_llm_pressure", 0.0)), float(item.get("selected_llm_rank_lift", 0.0))), reverse=True)
    return {
        "mean_field_to_llm_pressure": round(sum(field_to_llm) / len(field_to_llm), 4) if field_to_llm else 0.0,
        "max_field_to_llm_pressure": round(max(field_to_llm), 4) if field_to_llm else 0.0,
        "mean_selected_llm_rank_lift": round(sum(selected_lifts) / len(selected_lifts), 4) if selected_lifts else 0.0,
        "scored_action_count": len(field_to_llm),
        "examples": examples[:limit],
    }


def _behavioral_drift_curve(log: RunLog, agent_id: str | None = None) -> list[dict[str, float]]:
    agent_id = agent_id or story_cast_from_log(log).idea
    protest = {"ask_for_authorship", "privately_lobby_pi", "confront", "challenge_claim", "withdraw", "rebel", "document_contribution"}
    curve: list[dict[str, float]] = []
    for rec in log.round_records:
        rnd = int(rec.get("round", 0))
        recent = [a for a in log.actions if a.get("agent") == agent_id and max(1, rnd - 4) <= int(a.get("round", 0)) <= rnd]
        pressure = 0.0
        if recent:
            pressure = sum(float(a.get("intensity", 0.0)) for a in recent if a.get("type") in protest) / max(len(recent), 1)
        metrics = rec.get("metrics", {})
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written") or {}
        memory_signal = 0.0
        if mem.get("content_type") in {"authorship_signal", "promise_broken", "promise_fulfilled", "betrayal_signal"}:
            memory_signal = abs(float(mem.get("valence", 0.0))) * float(mem.get("strength", 0.0))
        curve.append({
            "round": rnd,
            "memory_cluster_increment": round(memory_signal, 4),
            "behavioral_drift": round(pressure, 4),
            "authorship_dispute_index": round(float(metrics.get("authorship_dispute_index", 0.0)), 4),
            "trust_fragmentation": round(float(metrics.get("trust_fragmentation", 0.0)), 4),
        })
    return curve


def _phase_transition_summary(log: RunLog, agent_id: str | None = None) -> dict[str, Any]:
    curve = _behavioral_drift_curve(log, agent_id)
    if not curve:
        return {"phase_transition_round": None, "phases": []}
    combined = [
        {
            **point,
            "phase_pressure": round(
                0.35 * point["authorship_dispute_index"]
                + 0.25 * point["trust_fragmentation"]
                + 0.25 * point["behavioral_drift"]
                + 0.15 * point["memory_cluster_increment"],
                4,
            ),
        }
        for point in curve
    ]
    peak = max(combined, key=lambda point: point["phase_pressure"])
    phases = [
        {"name": "event_chain", "round_range": "R1-R20", "signal": "early authorship and credit events create latent pressure"},
        {"name": "memory_cluster_formation", "round_range": "R20-R40", "signal": "authorship/promise memories accumulate and become recallable"},
        {"name": "behavioral_drift", "round_range": "R40-R51", "signal": "documentation, lobbying, delay, and confrontational candidates gain mass"},
        {"name": "phase_transition", "round_range": f"R{peak['round']}", "signal": "combined memory, trust, authorship dispute, and action pressure peaks"},
    ]
    return {
        "phase_transition_round": peak["round"],
        "phase_pressure_peak": peak["phase_pressure"],
        "curve_sample": combined[:: max(1, len(combined) // 8)],
        "phases": phases,
    }

def _path_level_causal_chain(log: RunLog, agent_id: str | None = None) -> dict[str, Any]:
    """Extract a readable long-horizon causal path from events, memory writes, and actions.

    This is a report/explanation artifact, not a simulation trigger: it summarizes the
    continuous trajectory after the run has already happened.
    """
    agent_id = agent_id or story_cast_from_log(log).idea
    salient_event_types = {
        "authorship_promise",
        "authorship_ambiguity",
        "authorship_draft",
        "credit_dispute",
        "private_lobbying",
        "narrative_change",
        "alumni_warning",
        "submission_decision",
    }
    protest_actions = {
        "ask_for_authorship",
        "privately_lobby_pi",
        "confront",
        "challenge_claim",
        "withdraw",
        "rebel",
        "document_contribution",
    }
    nodes: list[dict[str, Any]] = []

    event_by_id = {e.get("event_id"): e for e in log.events}
    for event in log.events:
        if event.get("type") in salient_event_types or "authorship" in str(event.get("type", "")):
            nodes.append({
                "round": int(event.get("round", 0)),
                "kind": "event",
                "label": str(event.get("type")),
                "event_id": event.get("event_id"),
                "detail": event.get("payload", {}).get("summary") or event.get("payload", {}).get("generator") or "state event",
            })

    for rec in log.round_records:
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if not mem:
            continue
        ctype = str(mem.get("content_type", ""))
        if "authorship" not in ctype and "promise" not in ctype:
            continue
        event_ref = mem.get("event_ref")
        event = event_by_id.get(event_ref, {})
        nodes.append({
            "round": int(rec.get("round", 0)),
            "kind": "memory",
            "label": ctype,
            "event_id": event_ref,
            "strength": float(mem.get("strength", 0.0)),
            "valence": float(mem.get("valence", 0.0)),
            "detail": mem.get("interpretation") or event.get("type") or "memory write",
        })

    for action in log.actions:
        if action.get("agent") != agent_id or action.get("type") not in protest_actions:
            continue
        nodes.append({
            "round": int(action.get("round", 0)),
            "kind": "action",
            "label": str(action.get("type")),
            "event_id": action.get("event_id"),
            "intensity": float(action.get("intensity", 0.0)),
            "detail": action.get("public_position", {}).get("authorship_claim") or action.get("target") or "action",
        })

    nodes.sort(key=lambda n: (n.get("round", 0), {"event": 0, "memory": 1, "action": 2}.get(n.get("kind"), 3)))
    compact: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for node in nodes:
        key = (int(node.get("round", 0)), str(node.get("kind")), str(node.get("label")))
        if key in seen:
            continue
        seen.add(key)
        compact.append(node)

    outcomes = log.outcomes
    return {
        "agent": agent_id,
        "nodes": compact[:18],
        "trajectory_phases": _phase_transition_summary(log, agent_id),
        "finding": (
            "Final authorship escalation is summarized as a mediated path through authorship memory, "
            "trust erosion, and late-stage authorship pressure rather than a single draft event."
        ),
        "outcome_summary": {
            "protest_authorship": outcomes.get("protest_authorship", 0.0),
            "authorship_escalation_score": outcomes.get("authorship_escalation_score", 0.0),
            "memory_authorship_cluster_strength": outcomes.get("memory_authorship_cluster_strength", 0.0),
            "promise_broken_strength_r52": outcomes.get("promise_broken_strength_r52", 0.0),
            "authority_compliance": outcomes.get("authority_compliance", 0.0),
        },
        "counterfactual_hint": "Compare against memory-delete or reframing conditions to estimate how much of the path is mediated by the authorship memory cluster.",
    }

def _causal_memory_timeline(log: RunLog, agent_id: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
    agent_id = agent_id or story_cast_from_log(log).idea
    nodes: list[dict[str, Any]] = []
    for rec in log.round_records:
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if not mem:
            continue
        nodes.append({
            "round": rec.get("round"),
            "agent": agent_id,
            "event_ref": mem.get("event_ref"),
            "content_type": mem.get("content_type"),
            "strength": mem.get("strength"),
            "valence": mem.get("valence"),
            "interpretation": mem.get("interpretation"),
        })
    nodes.sort(key=lambda n: float(n.get("strength") or 0), reverse=True)
    return nodes[:limit]


def _trust_fragmentation_curve(log: RunLog) -> list[dict[str, float]]:
    return [
        {"round": r.get("round", 0), "trust_fragmentation": r.get("metrics", {}).get("trust_fragmentation", 0.0)}
        for r in log.round_records
    ]


def _authorship_dispute_curve(log: RunLog) -> list[dict[str, float]]:
    return [
        {"round": r.get("round", 0), "authorship_dispute_index": r.get("metrics", {}).get("authorship_dispute_index", 0.0)}
        for r in log.round_records
    ]


def _trust_snapshots(
    log: RunLog,
    source: str | None = None,
    rounds: tuple[int, ...] | None = None,
) -> dict[int, dict[str, float]]:
    cast = story_cast_from_log(log)
    source = source or cast.idea
    snap_rounds = rounds or cast.snapshot_rounds
    snaps: dict[int, dict[str, float]] = {}
    for rnd in snap_rounds:
        for rec in log.round_records:
            if rec.get("round") != rnd:
                continue
            metrics = rec.get("metrics", {})
            prefix = f"trust_{source}_"
            edge_trust = {k: v for k, v in metrics.items() if k.startswith(prefix)}
            if edge_trust:
                snaps[rnd] = edge_trust
    return snaps


def _divergence_ranked_rounds(log: RunLog, limit: int = 8) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for rec in log.round_records:
        div = rec.get("metrics", {}).get("public_private_divergence", 0.0)
        activation = 1.0 / (1.0 + math.exp(-(float(div) - 0.35) * 6.0))
        ranked.append({
            "round": rec.get("round"),
            "divergence": div,
            "activation": round(activation, 4),
            "event_id": rec.get("event_id"),
        })
    ranked.sort(key=lambda p: p["activation"], reverse=True)
    return ranked[:limit]


def _entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = [v / total for v in counts.values() if v > 0]
    raw = -sum(p * math.log(p) for p in probs)
    norm = math.log(len(probs)) if len(probs) > 1 else 1.0
    return round(raw / norm, 4)


def _motive_entropy(motives: dict[str, float]) -> float:
    vals = [max(float(v), 0.0) for v in motives.values()]
    total = sum(vals)
    if total <= 0:
        return 0.0
    probs = [v / total for v in vals if v > 0]
    raw = -sum(p * math.log(p) for p in probs)
    norm = math.log(len(probs)) if len(probs) > 1 else 1.0
    return raw / norm


def _delayed_reaction_lag(log: RunLog, agent_id: str | None = None) -> float:
    agent_id = agent_id or story_cast_from_log(log).idea
    signal_rounds = [
        int(e.get("round", 0))
        for e in log.events
        if e.get("type") in {"authorship_ambiguity", "authorship_draft", "private_lobbying", "narrative_change"}
    ]
    reaction_rounds = [
        int(a.get("round", 0))
        for a in log.actions
        if a.get("agent") == agent_id
        and a.get("type") in {"ask_for_authorship", "confront", "withdraw", "challenge_claim", "privately_lobby_pi"}
    ]
    lags: list[int] = []
    for signal in signal_rounds:
        future = [r - signal for r in reaction_rounds if r > signal]
        if future:
            lags.append(min(future))
    return round(sum(lags) / len(lags), 4) if lags else 0.0


def _behavioral_trace_metrics(log: RunLog, agent_id: str | None = None) -> dict[str, float]:
    action_counts: dict[str, int] = {}
    motive_scores: list[float] = []
    candidate_counts: list[float] = []
    for action in log.actions:
        atype = str(action.get("type", "unknown"))
        action_counts[atype] = action_counts.get(atype, 0) + 1
        motives = action.get("private_motives") or action.get("private_intent", {}).get("private_motives") or {}
        if isinstance(motives, dict):
            motive_scores.append(_motive_entropy(motives))
        candidates = action.get("action_candidates") or []
        if isinstance(candidates, list):
            candidate_counts.append(float(len(candidates)))

    state_events = [e for e in log.events if e.get("payload", {}).get("generator") == "state_event_field"]
    return {
        "action_entropy": _entropy_from_counts(action_counts),
        "mean_motive_diversity": round(sum(motive_scores) / len(motive_scores), 4) if motive_scores else 0.0,
        "mean_candidate_count": round(sum(candidate_counts) / len(candidate_counts), 4) if candidate_counts else 0.0,
        "state_generated_event_fraction": round(len(state_events) / len(log.events), 4) if log.events else 0.0,
        "delayed_reaction_lag": _delayed_reaction_lag(log, agent_id),
    }

def _critic_audit_metrics(log: RunLog) -> dict[str, float]:
    if not log.critic_violations:
        return {"llm_drift_count": 0.0, "hard_violation_count": 0.0, "soft_violation_count": 0.0}
    llm = sum(1 for v in log.critic_violations if "llm_" in str(v.get("code", v.get("message", ""))))
    hard = sum(1 for v in log.critic_violations if v.get("severity") == "hard")
    soft = sum(1 for v in log.critic_violations if v.get("severity") == "soft")
    return {
        "llm_drift_count": float(llm),
        "hard_violation_count": float(hard),
        "soft_violation_count": float(soft),
    }


def _power_surface_from_log(log: RunLog) -> dict[str, float]:
    # RunLog does not store the final WorldState, so expose logged proxy metrics here.
    last = log.round_records[-1].get("metrics", {}) if log.round_records else {}
    return {
        "career_hostage_index": float(log.outcomes.get("career_hostage_index", 0.0)),
        "trust_pi_final": float(log.outcomes.get("trust_pi_final", 0.0)),
        "authorship_dispute_index": float(last.get("authorship_dispute_index", 0.0)),
    }
