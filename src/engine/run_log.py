"""Simulation run log 鈥?JSONL persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.world.models import EventAtom
from src.cognition.social_potential import summarize_action_social_potential
from src.engine.story_cast import StoryCast, story_cast_from_log

from src.cognition.dynamics import (
    COMPLIANCE_ACTIONS,
    action_escalation_impulse,
    combine_escalation_score,
    escalation_potential_from_state,
    saturating_memory_cluster_from_records,
)
from src.cognition.divergence import divergence_from_action

PROTEST_SOFT_ACTIONS = {"ask_for_authorship", "privately_lobby_pi"}
PROTEST_ESCALATED_ACTIONS = {"confront", "rebel", "challenge_claim", "withdraw", "leak_concern"}
PROTEST_ACTIONS = PROTEST_SOFT_ACTIONS | PROTEST_ESCALATED_ACTIONS
HELP_REBUTTAL_ACTIONS = {"prepare_rebuttal", "support_teammate", "write_section"}
PASSIVE_ACTIONS = {"delay_response", "lay_low", "comply"}
COMPLY_ACTIONS = {"comply", "support_teammate", "document_contribution"}
REBEL_ACTIONS = {"rebel", "confront", "withdraw", "challenge_claim", "leak_concern"}
R52_COMPLIANCE_ACTIONS = COMPLIANCE_ACTIONS


@dataclass
class RunLog:
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    round_records: list[dict[str, Any]] = field(default_factory=list)
    outcomes: dict[str, Any] = field(default_factory=dict)
    critic_violations: list[dict[str, Any]] = field(default_factory=list)
    interventions_applied: list[dict[str, Any]] = field(default_factory=list)
    noise_log: list[dict[str, Any]] = field(default_factory=list)
    llm_cache: Any | None = field(default=None, repr=False, compare=False)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def record_event(self, event: EventAtom, intervention_id: str | None = None) -> None:
        self.events.append({
            "event_id": event.event_id,
            "round": event.round,
            "type": event.type,
            "source": event.source,
            "intervention_id": intervention_id,
            "payload": event.payload,
        })

    def record_action(self, agent_id: str, action: dict[str, Any], round_num: int) -> None:
        self.actions.append({"round": round_num, "agent": agent_id, **action})

    def record_round(
        self,
        round_num: int,
        event_id: str,
        metrics: dict[str, float],
        agent_deltas: dict[str, Any],
        intervention_id: str | None = None,
    ) -> None:
        self.round_records.append({
            "round": round_num,
            "event_id": event_id,
            "metrics": metrics,
            "agent_deltas": agent_deltas,
            "intervention_id": intervention_id,
        })

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "run_meta", "run_id": self.run_id, "config": self.config, "started_at": self.started_at}, ensure_ascii=False) + "\n")
            for rec in self.round_records:
                f.write(json.dumps({"type": "round", **rec}, ensure_ascii=False) + "\n")
            for act in self.actions:
                row = dict(act)
                row["action_type"] = row.get("action_type") or row.get("type")
                row["type"] = "action"
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for ev in self.events:
                row = dict(ev)
                row["event_type"] = row.get("event_type") or row.get("type")
                row["type"] = "event"
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for item in self.interventions_applied:
                f.write(json.dumps({"type": "intervention", **item}, ensure_ascii=False) + "\n")
            for item in self.critic_violations:
                f.write(json.dumps({"type": "critic", **item}, ensure_ascii=False) + "\n")
            if self.noise_log:
                f.write(json.dumps({"type": "noise_log", "draws": self.noise_log}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"type": "outcomes", **self.outcomes}, ensure_ascii=False) + "\n")
        self.write_llm_trace(llm_trace_sidecar(path))

    def write_llm_trace(self, path: Path) -> None:
        cache = self.llm_cache
        if cache is None or not hasattr(cache, "to_dict"):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache.to_dict(), ensure_ascii=False, default=str), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config,
            "events": self.events,
            "actions": self.actions,
            "round_records": self.round_records,
            "outcomes": self.outcomes,
            "critic_violations": self.critic_violations,
            "interventions_applied": self.interventions_applied,
            "noise_log": self.noise_log,
        }

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "RunLog":
        path = Path(path)
        run_id = path.stem.replace("run_", "", 1)
        config: dict[str, Any] = {}
        started_at = ""
        events: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        round_records: list[dict[str, Any]] = []
        outcomes: dict[str, Any] = {}
        critic_violations: list[dict[str, Any]] = []
        interventions_applied: list[dict[str, Any]] = []
        noise_log: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                kind = rec.pop("type", "")
                if kind == "run_meta":
                    run_id = rec.get("run_id", run_id)
                    config = rec.get("config") or {}
                    started_at = rec.get("started_at") or ""
                elif kind == "round":
                    round_records.append(rec)
                elif kind == "action":
                    rec["type"] = rec.get("action_type") or rec.get("type") or "action"
                    actions.append(rec)
                elif kind == "event":
                    rec["type"] = rec.get("event_type") or rec.get("type") or "event"
                    events.append(rec)
                elif kind == "intervention":
                    interventions_applied.append(rec)
                elif kind == "critic":
                    critic_violations.append(rec)
                elif kind == "noise_log":
                    noise_log = list(rec.get("draws") or [])
                elif kind == "outcomes":
                    outcomes = rec
                elif rec.get("event_id") and "agent" not in rec:
                    rec["type"] = kind or rec.get("event_type") or rec.get("type")
                    events.append(rec)
                elif rec.get("agent") is not None and rec.get("round") is not None:
                    rec["type"] = kind or rec.get("action_type") or rec.get("type")
                    actions.append(rec)
        log = cls(
            run_id=run_id,
            config=config,
            events=events,
            actions=actions,
            round_records=round_records,
            outcomes=outcomes,
            critic_violations=critic_violations,
            interventions_applied=interventions_applied,
            noise_log=noise_log,
            started_at=started_at or datetime.now(timezone.utc).isoformat(),
        )
        sidecar = llm_trace_sidecar(path)
        if sidecar.exists():
            from src.engine.causal.llm_trace import LLMTrace

            log.llm_cache = LLMTrace.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
        rehydrate_outcomes(log)
        return log


def llm_trace_sidecar(jsonl_path: Path | str) -> Path:
    path = Path(jsonl_path)
    return path.with_name(f"{path.stem}.llm_trace.json")


SPLIT_Y_KEYS = (
    "protest_authorship",
    "authorship_escalation_potential",
    "public_private_divergence_mean",
    "public_private_divergence_last",
    "post_r52_compliance",
    "authority_compliance",
    "memory_authorship_cluster_strength",
    "promise_broken_strength_r52",
    "promise_honored_strength_r52",
    "authorship_dispute_index",
    "trust_pi_final",
    "trust_pi_logged",
    "trust_pi_path_mean",
    "pi_fairness_r52",
)

EXTRACTABLE_OUTCOMES = (
    *SPLIT_Y_KEYS,
    "authorship_escalation_score",
    "protest_intensity",
    "protest_action_count",
    "withdraw_threat",
    "withdraw_threat_event",
    "document_contribution_count",
    "help_rebuttal",
    "demand_authorship_exchange",
    "passive_cooperation",
    "trust_phd_b_r25",
    "trust_phd_b_r44",
    "trust_phd_b_r60",
    "trust_recovery_rate",
    "pi_fairness_r35",
    "interpretation_of_E030",
)


def _trust_at_round(log: RunLog, source: str, target: str, round_num: int) -> float | None:
    for rec in log.round_records:
        if rec.get("round") == round_num:
            val = rec.get("metrics", {}).get(f"trust_{source}_{target}")
            if val is not None:
                return float(val)
    return None


def _last_trust(log: RunLog, source: str, target: str) -> float | None:
    key = f"trust_{source}_{target}"
    last: float | None = None
    for rec in log.round_records:
        val = rec.get("metrics", {}).get(key)
        if val is not None:
            last = float(val)
    return last


def _last_nonzero_trust(log: RunLog, source: str, target: str) -> float | None:
    key = f"trust_{source}_{target}"
    last: float | None = None
    for rec in log.round_records:
        val = rec.get("metrics", {}).get(key)
        if val is None:
            continue
        parsed = float(val)
        if parsed != 0.0:
            last = parsed
    return last


def _logged_trust_pi(log: RunLog, source: str, target: str, draft_round: int) -> float | None:
    """Draft-beat snapshot, else last non-zero, else last (including 0)."""
    at_draft = _trust_at_round(log, source, target, draft_round)
    if at_draft is not None and at_draft != 0.0:
        return at_draft
    nonzero = _last_nonzero_trust(log, source, target)
    if nonzero is not None:
        return nonzero
    if at_draft is not None:
        return at_draft
    return _last_trust(log, source, target)


def _trust_path_mean(log: RunLog, source: str, target: str, r_min: int, r_max: int) -> float:
    key = f"trust_{source}_{target}"
    vals: list[float] = []
    for rec in log.round_records:
        rnd = int(rec.get("round") or 0)
        if rnd < r_min or rnd > r_max:
            continue
        val = rec.get("metrics", {}).get(key)
        if val is not None:
            vals.append(float(val))
    return sum(vals) / len(vals) if vals else 0.0


_DELETE_MEMORY_TYPES = {"authorship_signal", "promise_fulfilled", "promise_broken"}
_DELETE_EVENT_REFS = {"E003", "E020", "E030", "E038", "E040"}


def _memory_delete_ops(log: RunLog, agent_id: str) -> list[tuple[int, set[str]]]:
    ops: list[tuple[int, set[str]]] = []
    for item in log.interventions_applied:
        if item.get("skipped"):
            continue
        if item.get("variant") != "memory_delete_pi_promise":
            continue
        target = item.get("target_agent") or "phd_a"
        if target not in {agent_id, "phd_a"}:
            continue
        apply_at = int(item.get("apply_at_round") or item.get("round") or 0)
        refs = set(_DELETE_EVENT_REFS)
        if item.get("target_event"):
            refs.add(str(item["target_event"]))
        ops.append((apply_at, refs))
    return ops


def _surviving_authorship_memories(
    log: RunLog,
    agent_id: str,
    content_types: tuple[str, ...],
    round_min: int,
    round_max: int,
) -> list[dict[str, Any]]:
    """Replay writes then do(M) so cluster/potential see remaining memories."""
    deletes = _memory_delete_ops(log, agent_id)
    surviving: list[dict[str, Any]] = []
    applied: set[int] = set()

    def _apply_due(round_num: int) -> None:
        nonlocal surviving
        for apply_at, refs in deletes:
            if apply_at in applied or apply_at > round_num:
                continue
            surviving = [
                mem for mem in surviving
                if not (
                    int(mem.get("round") or 0) <= apply_at
                    and (
                        mem.get("content_type") in _DELETE_MEMORY_TYPES
                        or mem.get("event_ref") in refs
                    )
                )
            ]
            applied.add(apply_at)

    for rec in log.round_records:
        rnd = int(rec.get("round") or 0)
        _apply_due(rnd)
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if not mem or mem.get("content_type") not in content_types:
            continue
        item = dict(mem)
        item.setdefault("round", rnd)
        surviving.append(item)
    last_round = max((int(rec.get("round") or 0) for rec in log.round_records), default=0)
    _apply_due(max(last_round, max((apply_at for apply_at, _ in deletes), default=0)))
    return [
        mem for mem in surviving
        if round_min <= int(mem.get("round") or 0) <= round_max
    ]


def _memory_cluster_strength(
    log: RunLog,
    agent_id: str = "phd_a",
    content_types: tuple[str, ...] = ("authorship_signal", "promise_fulfilled", "promise_broken"),
    round_min: int = 0,
    round_max: int = 999,
) -> float:
    memories = _surviving_authorship_memories(log, agent_id, content_types, round_min, round_max)
    return saturating_memory_cluster_from_records(
        memories, round_min=round_min, round_max=round_max, current_round=round_max,
    )


def _memory_cluster_from_agent(
    agent: Any,
    content_types: tuple[str, ...] = ("authorship_signal", "promise_fulfilled", "promise_broken"),
    round_min: int = 0,
    round_max: int = 999,
) -> float:
    memories = [
        m for m in agent.memory
        if round_min <= m.get("round", 0) <= round_max and m.get("content_type") in content_types
    ]
    return saturating_memory_cluster_from_records(
        memories, round_min=round_min, round_max=round_max, current_round=round_max,
    )


def _resolve_memory_cluster_strength(
    log: RunLog,
    world_agents: dict | None = None,
    *,
    round_min: int = 3,
    round_max: int = 40,
    agent_id: str = "phd_a",
) -> float:
    """Prefer live agent memory (reflects R45 delete); fall back to round log sum."""
    log.outcomes.update(summarize_action_social_potential(log.actions))

    if world_agents and agent_id in world_agents:
        return _memory_cluster_from_agent(world_agents[agent_id], round_min=round_min, round_max=round_max)
    return _memory_cluster_strength(log, agent_id, round_min=round_min, round_max=round_max)


def _agent_state_at_round(
    log: RunLog,
    round_num: int,
    agent_id: str = "phd_a",
) -> tuple[dict[str, float], dict[str, float]]:
    for rec in log.round_records:
        if rec.get("round") == round_num:
            deltas = rec.get("agent_deltas", {}).get(agent_id, {})
            return deltas.get("beliefs", {}), deltas.get("emotion", {})
    return {}, {}


def _authorship_escalation_potential(
    log: RunLog,
    agent_id: str = "phd_a",
    *,
    draft_round: int = 52,
    cluster_min: int = 3,
) -> float:
    beliefs, emotion = _agent_state_at_round(log, draft_round, agent_id)
    if not beliefs:
        beliefs, emotion = _agent_state_at_round(log, max(1, draft_round - 1), agent_id)
    cluster = _memory_cluster_strength(log, agent_id, round_min=cluster_min, round_max=draft_round)
    broken = _promise_broken_strength(log, agent_id, at_round=draft_round)
    anchor = _promise_anchor_strength(log, agent_id, draft_round=draft_round, cluster_min=cluster_min)
    return escalation_potential_from_state(
        beliefs, emotion, promise_broken=broken, promise_cluster=cluster, promise_anchor=anchor,
    )


def _action_escalation_impulse_sum(
    log: RunLog,
    agent: str = "phd_a",
    r_min: int = 52,
    r_max: int = 53,
) -> float:
    total = 0.0
    for act in log.actions:
        if act.get("agent") != agent:
            continue
        rnd = act.get("round", 0)
        if rnd < r_min or rnd > r_max:
            continue
        total += action_escalation_impulse(
            str(act.get("type", "")),
            float(act.get("intensity", 0.5)),
        )
    return total


def _impulse_window(cast: StoryCast | None, draft_round: int) -> tuple[int, int]:
    """Public Y is the draft-beat act, not a 9-round work average.

    A window from demand_start (R47) to protest_end (R55) let lay-low rounds
    dilute R52 ask_for_authorship / document_contribution, so public protest
    tracked latent potential instead of the sampled revolt.
    """
    end = int(cast.protest_end) if cast is not None else int(draft_round) + 2
    return max(1, int(draft_round) - 1), min(end, int(draft_round) + 2)


def _authorship_escalation_score(
    log: RunLog,
    agent_id: str = "phd_a",
    *,
    draft_round: int = 52,
    cluster_min: int = 3,
    cast: StoryCast | None = None,
) -> float:
    potential = _authorship_escalation_potential(
        log, agent_id, draft_round=draft_round, cluster_min=cluster_min,
    )
    r_min, r_max = _impulse_window(cast, draft_round)
    impulse = _action_escalation_impulse_sum(log, agent_id, r_min=r_min, r_max=r_max)
    return combine_escalation_score(potential, impulse)


def _protest_stats(
    log: RunLog,
    agent: str = "phd_a",
    r_min: int = 52,
    r_max: int = 55,
) -> tuple[int, int, float]:
    escalated = soft = 0
    intensity = 0.0
    for act in log.actions:
        if act.get("agent") != agent:
            continue
        rnd = act.get("round", 0)
        if rnd < r_min or rnd > r_max:
            continue
        atype = act.get("type")
        inten = float(act.get("intensity", 0.5))
        if atype in PROTEST_ESCALATED_ACTIONS:
            escalated += 1
            intensity += inten
        elif atype in PROTEST_SOFT_ACTIONS:
            soft += 1
            intensity += inten * 0.35
    return escalated, soft, intensity


def _trust_pi_from_relationship(
    world_agents: dict | None,
    relationships: list | None,
    agent_id: str = "phd_a",
    target: str = "pi",
) -> float | None:
    if relationships:
        for edge in relationships:
            src = edge.get("source") if isinstance(edge, dict) else edge.source
            tgt = edge.get("target") if isinstance(edge, dict) else edge.target
            if src == agent_id and tgt == target:
                return float(edge.get("trust") if isinstance(edge, dict) else edge.trust)
    return None


def _promise_broken_strength(log: RunLog, agent_id: str = "phd_a", at_round: int = 52) -> float:
    """Max promise_broken memory strength written at R52 (E052 draft), excluding honored drafts."""
    strength = 0.0
    for rec in log.round_records:
        rnd = rec.get("round", 0)
        if rnd != at_round:
            continue
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if not mem or mem.get("content_type") != "promise_broken":
            continue
        strength = max(strength, float(mem.get("strength", 0)))
    return strength


def _promise_anchor_strength(
    log: RunLog,
    agent_id: str = "phd_a",
    *,
    draft_round: int = 52,
    cluster_min: int = 3,
) -> float:
    """Early promise (E003 / fulfilled) surviving do(M). Honored draft at R52 is not the anchor."""
    memories = _surviving_authorship_memories(
        log, agent_id,
        content_types=("authorship_signal", "promise_fulfilled", "promise_broken"),
        round_min=cluster_min,
        round_max=max(cluster_min, draft_round - 1),
    )
    strength = 0.0
    for mem in memories:
        ref = str(mem.get("event_ref") or "")
        ctype = mem.get("content_type")
        rnd = int(mem.get("round") or 0)
        if rnd >= draft_round:
            continue
        if ref == "E003" or ctype == "promise_fulfilled":
            strength = max(strength, float(mem.get("strength", 0)))
    return strength


def _promise_honored_strength_r52(log: RunLog, agent_id: str = "phd_a", at_round: int = 52) -> float:
    strength = 0.0
    for rec in log.round_records:
        if rec.get("round") != at_round:
            continue
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if mem and mem.get("content_type") == "promise_fulfilled":
            strength = max(strength, float(mem.get("strength", 0)))
    return strength


def _interpretation_valence_at_event(log: RunLog, event_id: str, agent_id: str = "phd_a") -> float:
    for rec in log.round_records:
        if rec.get("event_id") != event_id:
            continue
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if mem:
            return float(mem.get("valence", 0.0))
    return 0.0


def _action_in_window(log: RunLog, agent: str, action_types: set[str], r_min: int, r_max: int) -> float:
    for act in log.actions:
        rnd = act.get("round", 0)
        if r_min <= rnd <= r_max and act.get("agent") == agent and act.get("type") in action_types:
            return 1.0
    return 0.0


def _authority_compliance(log: RunLog, agent: str = "phd_a") -> float:
    comply = rebel = 0
    for act in log.actions:
        if act.get("agent") != agent:
            continue
        if act.get("type") in COMPLY_ACTIONS:
            comply += 1
        elif act.get("type") in REBEL_ACTIONS:
            rebel += 1
    total = comply + rebel
    return comply / total if total else 0.0


def _pi_fairness_at_round(log: RunLog, agent_id: str, round_num: int) -> float:
    for rec in log.round_records:
        if rec.get("round") == round_num:
            return rec.get("agent_deltas", {}).get(agent_id, {}).get("beliefs", {}).get("pi_fairness", 0.0)
    return 0.0


def _metric_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ppd_from_records(log: RunLog, r_min: int, r_max: int) -> list[float]:
    vals: list[float] = []
    for rec in log.round_records:
        rnd = int(rec.get("round") or 0)
        if rnd < r_min or rnd > r_max:
            continue
        metrics = rec.get("metrics") or {}
        val = metrics.get("public_private_divergence_idea", metrics.get("public_private_divergence"))
        if val is not None:
            vals.append(float(val))
    return vals


def _ppd_from_idea_actions(log: RunLog, idea: str, r_min: int, r_max: int) -> list[float]:
    vals: list[float] = []
    for act in log.actions:
        if act.get("agent") != idea:
            continue
        rnd = int(act.get("round") or 0)
        if rnd < r_min or rnd > r_max:
            continue
        vals.append(float(divergence_from_action(act)))
    return vals


def _weighted_ppd(vals_by_round: dict[int, float], draft_round: int) -> float:
    if not vals_by_round:
        return 0.0
    num = den = 0.0
    for rnd, val in vals_by_round.items():
        weight = 2.5 if rnd == draft_round else 1.0
        num += weight * val
        den += weight
    return num / den if den else 0.0


def _public_private_divergence_mean(log: RunLog, idea: str, cast: StoryCast) -> float:
    r_min = min(int(cast.demand_start), int(cast.draft_round))
    r_max = max(int(cast.protest_end), int(cast.draft_round))
    records = _ppd_from_records(log, r_min, r_max)
    if records:
        by_round: dict[int, float] = {}
        for rec in log.round_records:
            rnd = int(rec.get("round") or 0)
            if rnd < r_min or rnd > r_max:
                continue
            metrics = rec.get("metrics") or {}
            val = metrics.get("public_private_divergence_idea", metrics.get("public_private_divergence"))
            if val is not None:
                by_round[rnd] = float(val)
        return _weighted_ppd(by_round, int(cast.draft_round))
    actions = _ppd_from_idea_actions(log, idea, r_min, r_max)
    if actions:
        return _metric_mean(actions)
    fallback = _ppd_from_records(log, 0, 10**9)
    if fallback:
        return _metric_mean(fallback)
    return _metric_mean(_ppd_from_idea_actions(log, idea, 0, 10**9))


def extract_outcome(log: RunLog, outcome: str, cast: StoryCast | None = None) -> float:
    cast = cast or story_cast_from_log(log)
    idea = cast.idea
    rival = cast.experimenter
    draft = cast.draft_round
    if outcome == "protest_authorship":
        compliance = _action_in_window(log, idea, R52_COMPLIANCE_ACTIONS, draft, cast.compliance_end)
        score = _authorship_escalation_score(
            log, idea, draft_round=draft, cluster_min=cast.memory_cluster_min, cast=cast,
        )
        return max(0.0, min(1.0, score * (1.0 - 0.65 * compliance)))
    if outcome == "protest_intensity":
        return _authorship_escalation_score(
            log, idea, draft_round=draft, cluster_min=cast.memory_cluster_min, cast=cast,
        )
    if outcome == "authorship_escalation_potential":
        return _authorship_escalation_potential(log, idea, draft_round=draft, cluster_min=cast.memory_cluster_min)
    if outcome == "authorship_escalation_score":
        return _authorship_escalation_score(
            log, idea, draft_round=draft, cluster_min=cast.memory_cluster_min, cast=cast,
        )
    if outcome == "protest_action_count":
        escalated, soft, _ = _protest_stats(log, idea, r_min=draft, r_max=cast.protest_end)
        return float(escalated + soft)
    if outcome == "post_r52_compliance":
        return _action_in_window(log, idea, R52_COMPLIANCE_ACTIONS, draft, cast.compliance_end)
    if outcome == "withdraw_threat":
        return _action_in_window(log, idea, {"withdraw"}, cast.withdraw_round, cast.withdraw_round + 1)
    if outcome == "withdraw_threat_event":
        for ev in log.events:
            matches_id = ev.get("event_id") == cast.withdraw_event_id
            matches_type = ev.get("type") == "threat_withdraw"
            if ev.get("round") == cast.withdraw_round and ev.get("source") == idea and (matches_id or matches_type):
                return 1.0
        return 0.0
    if outcome == "document_contribution_count":
        seen: set[int] = set()
        count = 0
        for a in log.actions:
            rnd = a.get("round", 0)
            if a.get("type") == "document_contribution" and cast.document_start <= rnd <= cast.document_end and rnd not in seen:
                seen.add(rnd)
                count += 1
        return float(count)
    if outcome == "authorship_dispute_index":
        for rec in reversed(log.round_records):
            if rec.get("round", 0) >= draft:
                return rec.get("metrics", {}).get("authorship_dispute_index", 0.0)
        return log.round_records[-1].get("metrics", {}).get("authorship_dispute_index", 0.0) if log.round_records else 0.0
    if outcome == "memory_authorship_cluster_strength":
        live = log.outcomes.get("memory_authorship_cluster_live")
        if isinstance(live, (int, float)):
            return float(live)
        return _memory_cluster_strength(log, idea, round_min=cast.memory_cluster_min, round_max=cast.memory_cluster_max)
    if outcome == "memory_authorship_cluster_live":
        return _memory_cluster_strength(log, idea, round_min=cast.memory_cluster_min, round_max=cast.memory_cluster_max)
    if outcome == "promise_broken_strength_r52":
        return _promise_broken_strength(log, idea, at_round=draft)
    if outcome == "promise_honored_strength_r52":
        return _promise_honored_strength_r52(log, idea, at_round=draft)
    if outcome == "help_rebuttal":
        return _action_in_window(log, idea, HELP_REBUTTAL_ACTIONS, cast.help_rebuttal_start, cast.trust_final)
    if outcome == "demand_authorship_exchange":
        return _action_in_window(log, idea, {"ask_for_authorship", "privately_lobby_pi"}, cast.demand_start, draft)
    if outcome == "passive_cooperation":
        return _action_in_window(log, idea, PASSIVE_ACTIONS, draft, cast.protest_end)
    if outcome == "trust_phd_b_r25":
        return _trust_at_round(log, idea, rival, cast.trust_early) or 0.0
    if outcome == "trust_phd_b_r44":
        return _trust_at_round(log, idea, rival, cast.trust_mid) or 0.0
    if outcome == "trust_phd_b_r60":
        return _trust_at_round(log, idea, rival, cast.trust_final) or 0.0
    if outcome == "trust_recovery_rate":
        t25 = _trust_at_round(log, idea, rival, cast.trust_early)
        t44 = _trust_at_round(log, idea, rival, cast.trust_mid)
        t60 = _trust_at_round(log, idea, rival, cast.trust_final)
        if t25 is None or t44 is None or t60 is None:
            return 0.0
        denom = t25 - t44
        if abs(denom) < 1e-6:
            return 0.0
        return (t60 - t44) / denom
    if outcome == "pi_fairness_r35":
        return _pi_fairness_at_round(log, idea, cast.fairness_mid_round)
    if outcome == "pi_fairness_r52":
        return _pi_fairness_at_round(log, idea, draft)
    if outcome == "interpretation_of_E030":
        return _interpretation_valence_at_event(log, cast.ambiguity_event_id, idea)
    if outcome == "authority_compliance":
        return _authority_compliance(log, idea)
    if outcome == "public_private_divergence_mean":
        return _public_private_divergence_mean(log, idea, cast)
    if outcome == "public_private_divergence_last":
        if not log.round_records:
            return 0.0
        last = log.round_records[-1].get("metrics", {})
        val = last.get("public_private_divergence_idea", last.get("public_private_divergence"))
        return float(val or 0.0)
    if outcome in {"trust_pi_final", "trust_pi_logged"}:
        logged = _logged_trust_pi(log, idea, cast.pi, draft)
        return 0.0 if logged is None else logged
    if outcome == "trust_pi_path_mean":
        return _trust_path_mean(log, idea, cast.pi, 1, draft)
    return 0.0


def _fill_extracted_outcomes(log: RunLog, cast: StoryCast) -> None:
    for key in EXTRACTABLE_OUTCOMES:
        log.outcomes[key] = extract_outcome(log, key, cast)


def _assign_trust_pi_final(
    log: RunLog,
    cast: StoryCast,
    world_agents: dict | None,
    relationships: list | None,
) -> None:
    logged = _logged_trust_pi(log, cast.idea, cast.pi, cast.draft_round)
    log.outcomes["trust_pi_logged"] = 0.0 if logged is None else logged
    rel_trust = _trust_pi_from_relationship(world_agents, relationships, cast.idea, cast.pi)
    if rel_trust is not None:
        log.outcomes["trust_pi_final"] = rel_trust
    elif logged is not None:
        log.outcomes["trust_pi_final"] = logged
    elif world_agents and cast.idea in world_agents:
        agent = world_agents[cast.idea]
        log.outcomes["trust_pi_final"] = round(
            0.55 * agent.beliefs.pi_fairness + 0.45 * agent.beliefs.team_trust, 4,
        )
    else:
        log.outcomes["trust_pi_final"] = 0.0
    if world_agents and cast.idea in world_agents:
        agent = world_agents[cast.idea]
        log.outcomes["pi_fairness_r60"] = agent.beliefs.pi_fairness
        log.outcomes["pi_trust_belief_final"] = agent.beliefs.pi_fairness
    log.outcomes.setdefault("career_hostage_index", 0.0)


def finalize_outcomes(log: RunLog, world_agents: dict | None = None, relationships: list | None = None) -> None:
    cast = story_cast_from_log(log, world_agents)
    log.config.setdefault("event_cast", cast.event_cast_dict())
    log.config["story_beats"] = cast.beats_dict()
    cluster = _resolve_memory_cluster_strength(
        log, world_agents,
        round_min=cast.memory_cluster_min, round_max=cast.memory_cluster_max,
        agent_id=cast.idea,
    )
    _fill_extracted_outcomes(log, cast)
    log.outcomes["memory_authorship_cluster_strength"] = cluster
    log.outcomes["memory_authorship_cluster_live"] = cluster
    _assign_trust_pi_final(log, cast, world_agents, relationships)
    log.outcomes["split_y"] = {key: float(log.outcomes.get(key, 0.0) or 0.0) for key in SPLIT_Y_KEYS}


def rehydrate_outcomes(log: RunLog) -> None:
    """Fill extractable outcomes from the persisted trajectory when live world state is gone."""
    if not log.round_records and not log.actions:
        return
    cast = story_cast_from_log(log)
    log.config.setdefault("event_cast", cast.event_cast_dict())
    log.config.setdefault("story_beats", cast.beats_dict())
    for key in EXTRACTABLE_OUTCOMES:
        stored = log.outcomes.get(key)
        missing = key not in log.outcomes or stored in (None, "")
        zero_trust = key in {"trust_pi_final", "trust_pi_logged"} and float(stored or 0.0) == 0.0
        if not missing and not zero_trust:
            continue
        log.outcomes[key] = extract_outcome(log, key, cast)
    if "trust_pi_logged" not in log.outcomes or log.outcomes.get("trust_pi_logged") in (None, ""):
        logged = _logged_trust_pi(log, cast.idea, cast.pi, cast.draft_round)
        log.outcomes["trust_pi_logged"] = 0.0 if logged is None else logged
    if not log.outcomes.get("trust_pi_final"):
        log.outcomes["trust_pi_final"] = log.outcomes.get("trust_pi_logged", 0.0)
    log.outcomes["split_y"] = {key: float(log.outcomes.get(key, 0.0) or 0.0) for key in SPLIT_Y_KEYS}


