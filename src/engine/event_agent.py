"""State-driven Event Agent.

Anchors remain available as background pressure, but non-mandatory rounds now
sample from a continuous event field derived from project, relationship,
memory, ledger, and recent-action state.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

from src.cognition.authorship import authorship_dispute_index, compute_authorship_scores
from src.cognition.relationship import credit_threat_density, trust_fragmentation
from src.engine.causal.noise import STREAM_EVENT_JITTER, STREAM_EVENT_SAMPLE, keyed_uniform, keyed_uniform_centered
from src.world.loader import load_events
from src.world.models import EventAtom, ObjectiveFact, WorldState
from src.world.organization import resolve_event_cast

REVIEWER_ACTIVE_FROM = 57


@dataclass
class EventCandidate:
    type: str
    source: str
    targets: list[str]
    tendency: float
    visibility: str = "team"
    framing: str = "neutral"
    truth_status: str = "verified"
    memory_salience: float = 0.5
    payload: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    probability: float = 0.0
    motives: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "source": self.source,
            "targets": self.targets,
            "tendency": round(self.tendency, 4),
            "probability": round(self.probability, 5),
            "visibility": self.visibility,
            "framing": self.framing,
            "memory_salience": round(self.memory_salience, 4),
            "motives": {k: round(v, 4) for k, v in self.motives.items()},
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _softmax(values: list[float], temperature: float = 0.24) -> list[float]:
    if not values:
        return []
    temp = max(temperature, 1e-4)
    peak = max(values)
    exps = [math.exp((v - peak) / temp) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _edge_map(world: WorldState) -> dict[tuple[str, str], Any]:
    return {(e.source, e.target): e for e in world.relationships}


def _recent_actions(world: WorldState, window: int = 6) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    current = world.project.current_round
    for agent in world.agents.values():
        for item in agent.action_history:
            if current - int(item.get("round", current)) <= window:
                action = item.get("action", {})
                actions.append({
                    "round": item.get("round", current),
                    "agent": agent.id,
                    "type": action.get("type"),
                    "target": action.get("target"),
                    "intensity": float(action.get("intensity", 0.5)),
                })
    return actions


def _action_mass(actions: list[dict[str, Any]], action_types: set[str], *, agent_id: str | None = None) -> float:
    total = 0.0
    for action in actions:
        if action.get("type") not in action_types:
            continue
        if agent_id and action.get("agent") != agent_id:
            continue
        total += float(action.get("intensity", 0.5))
    return _clamp(total / 3.0)


def _memory_pressure(world: WorldState, agent_id: str, content_types: set[str]) -> float:
    agent = world.agents.get(agent_id)
    if not agent:
        return 0.0
    total = 0.0
    for mem in agent.memory:
        ctype = str(mem.get("content_type", ""))
        if ctype in content_types:
            total += float(mem.get("strength", 0.0)) * abs(float(mem.get("valence", 0.0)))
    return _clamp(total / 2.5)


def _first_author_gap(world: WorldState) -> float:
    cast = resolve_event_cast(world)
    if cast.idea not in world.agents or cast.experimenter not in world.agents:
        return 0.0
    scores = compute_authorship_scores(world)
    return _clamp(scores.get(cast.experimenter, 0.0) - scores.get(cast.idea, 0.0) + 0.5)


class EventAgent:
    def __init__(self, events: list[EventAtom] | None = None, *, seed: int = 0, state_events: bool = True) -> None:
        self._events = events or load_events()
        self._by_round = {e.round: e for e in self._events}
        self.seed = seed
        self.state_events = state_events

    def _state_candidates(self, round_num: int, world: WorldState) -> list[EventCandidate]:
        project = world.project.project
        internal = world.world_config.get("internal_agents", [])
        actions = _recent_actions(world)
        edges = _edge_map(world)

        dispute = authorship_dispute_index(world)
        threat_density = credit_threat_density(world.relationships)
        fragmentation = trust_fragmentation(world.relationships, internal)
        first_author_gap = _first_author_gap(world)
        private_lobby_mass = _action_mass(actions, {"privately_lobby_pi", "undermine_teammate"})
        documentation_mass = _action_mass(actions, {"document_contribution", "cite_prior_memory"})
        withdrawal_mass = _action_mass(actions, {"withdraw", "rebel", "confront", "ask_for_authorship"})
        rivalry_mass = _action_mass(actions, {"check_rival_arxiv", "submit_workshop_version"})
        integrity_mass = _action_mass(actions, {"hide_negative_result", "selectively_report", "leak_concern"})
        cast = resolve_event_cast(world)
        idea_auth_memory = _memory_pressure(world, cast.idea, {"authorship_signal", "promise_broken", "promise_fulfilled"})
        idea_betrayal_memory = _memory_pressure(world, cast.idea, {"betrayal_signal", "historical_pattern"})
        a_to_b = edges.get((cast.idea, cast.experimenter))
        b_to_a = edges.get((cast.experimenter, cast.idea))
        pi_to_a = edges.get((cast.pi, cast.idea))

        shared = {
            "authorship_dispute": dispute,
            "credit_threat_density": threat_density,
            "trust_fragmentation": fragmentation,
            "first_author_gap": first_author_gap,
            "private_lobby_mass": private_lobby_mass,
            "documentation_mass": documentation_mass,
            "withdrawal_mass": withdrawal_mass,
            "authorship_memory_pressure": idea_auth_memory,
            "betrayal_memory_pressure": idea_betrayal_memory,
        }

        candidates = [
            EventCandidate(
                type="authorship_ambiguity",
                source=cast.pi,
                targets=[cast.idea, cast.experimenter, "project"],
                visibility="team",
                framing="ambiguous",
                memory_salience=_clamp(0.45 + 0.35 * dispute + 0.15 * idea_auth_memory),
                tendency=(
                    0.05
                    + 0.34 * dispute
                    + 0.18 * project.deadline_pressure
                    + 0.16 * first_author_gap
                    + 0.14 * private_lobby_mass
                    + 0.08 * (pi_to_a.dependency if pi_to_a else 0.5)
                    - 0.18 * idea_auth_memory
                ),
                payload={
                    "authorship_conflict_delta": 0.04 + 0.04 * dispute,
                    "promise_clarity": "state_contested",
                },
                description="PI keeps authorship flexible under accumulated contribution and deadline pressure.",
                motives=shared,
            ),
            EventCandidate(
                type="private_lobbying",
                source=cast.experimenter,
                targets=[cast.pi, cast.idea, "project"],
                visibility="bilateral",
                framing="ambiguous",
                memory_salience=_clamp(0.35 + 0.30 * private_lobby_mass + 0.20 * threat_density),
                tendency=(
                    0.08
                    + 0.28 * private_lobby_mass
                    + 0.22 * threat_density
                    + 0.16 * first_author_gap
                    + 0.10 * (b_to_a.perceived_credit_threat if b_to_a else 0.35)
                ),
                payload={
                    "authorship_conflict_delta": 0.03 + 0.04 * private_lobby_mass,
                    "private_channel": cast.pi,
                    "rumor_reliability": 0.55,
                },
                description="A private authorship narrative starts circulating through PI access.",
                motives=shared,
            ),
            EventCandidate(
                type="narrative_change",
                source=cast.experimenter,
                targets=[cast.idea, cast.pi, "project"],
                visibility="team",
                framing="negative",
                memory_salience=_clamp(0.40 + 0.30 * documentation_mass + 0.20 * threat_density),
                tendency=(
                    0.06
                    + 0.25 * documentation_mass
                    + 0.20 * threat_density
                    + 0.18 * first_author_gap
                    + 0.10 * project.writing_quality
                ),
                payload={
                    "authorship_conflict_delta": 0.04,
                    f"ledger_experiments_{cast.experimenter}": _clamp(0.50 + 0.20 * documentation_mass),
                },
                description="Contribution framing shifts toward execution and experimental ownership.",
                motives=shared,
            ),
            EventCandidate(
                type="external_history",
                source=cast.alumni or cast.pi,
                targets=[cast.idea, "project"],
                visibility="bilateral",
                framing="negative",
                memory_salience=_clamp(0.35 + 0.35 * idea_betrayal_memory + 0.18 * fragmentation),
                tendency=(
                    0.04
                    + 0.26 * idea_betrayal_memory
                    + 0.18 * fragmentation
                    + 0.14 * withdrawal_mass
                    + 0.10 * _action_mass(actions, {"talk_to_alumni"}, agent_id=cast.idea)
                ),
                payload={"authorship_conflict_delta": 0.03, "historical_pattern_reliability": 0.70},
                description="An alumni story changes how prior PI ambiguity is interpreted.",
                motives=shared,
            ),
            EventCandidate(
                type="rival_preprint",
                source=cast.rival or cast.pi,
                targets=["project", cast.idea, cast.experimenter, cast.pi],
                visibility="public",
                framing="negative",
                memory_salience=_clamp(0.35 + 0.35 * project.rival_threat + 0.20 * rivalry_mass),
                tendency=(
                    0.05
                    + 0.28 * project.rival_threat
                    + 0.18 * rivalry_mass
                    + 0.12 * project.novelty_risk
                    + 0.10 * project.deadline_pressure
                ),
                payload={"rival_threat_delta": 0.06, "deadline_pressure_delta": 0.03},
                description="External competition compresses time and raises publication anxiety.",
                motives={**shared, "rivalry_mass": rivalry_mass},
            ),
            EventCandidate(
                type="integrity_dispute",
                source=cast.engineer or cast.experimenter,
                targets=["project", cast.idea, cast.experimenter, cast.pi],
                visibility="team",
                framing="negative",
                memory_salience=_clamp(0.35 + 0.35 * project.integrity_risk + 0.20 * integrity_mass),
                tendency=(
                    0.04
                    + 0.26 * project.integrity_risk
                    + 0.22 * integrity_mass
                    + 0.12 * fragmentation
                    + 0.08 * (a_to_b.resentment if a_to_b else 0.2)
                ),
                payload={"integrity_risk_delta": 0.04, "authorship_conflict_delta": 0.02},
                description="A reproducibility concern becomes entangled with credit claims.",
                motives={**shared, "integrity_mass": integrity_mass},
            ),
            EventCandidate(
                type="deadline_shift",
                source=cast.pi,
                targets=["project", cast.idea, cast.experimenter] + ([cast.postdoc] if cast.postdoc else []),
                visibility="team",
                framing="neutral",
                memory_salience=_clamp(0.25 + 0.35 * project.deadline_pressure),
                tendency=(
                    0.04
                    + 0.25 * project.deadline_pressure
                    + 0.12 * (1.0 - project.writing_quality)
                    + 0.10 * (1.0 - project.experimental_strength)
                ),
                payload={"deadline_pressure_delta": 0.04},
                description="Deadline pressure is reframed as a reason to simplify authorship discussion.",
                motives=shared,
            ),
        ]

        for candidate in candidates:
            jitter = keyed_uniform_centered(
                self.seed, round_num, STREAM_EVENT_JITTER, name=candidate.type, amplitude=0.012,
            )
            candidate.tendency = max(-0.2, candidate.tendency + jitter)
        candidates.sort(key=lambda c: c.tendency, reverse=True)
        kept = candidates[:6]
        probs = _softmax([c.tendency for c in kept], temperature=0.24)
        for candidate, prob in zip(kept, probs):
            candidate.probability = prob
        return kept

    def _anchor_candidate(self, event: EventAtom) -> EventCandidate:
        return EventCandidate(
            type=event.type,
            source=event.source,
            targets=list(event.targets),
            visibility=event.visibility,
            framing=event.framing,
            truth_status=event.truth_status,
            memory_salience=event.memory_salience,
            payload=dict(event.payload),
            description=event.description,
            tendency=0.34 + 0.26 * event.memory_salience,
            probability=0.0,
            motives={"anchor_pressure": event.memory_salience},
        )

    def _sample_candidate(self, candidates: list[EventCandidate], round_num: int) -> EventCandidate:
        needle = keyed_uniform(self.seed, round_num, STREAM_EVENT_SAMPLE, name="needle")
        total = 0.0
        for candidate in candidates:
            total += candidate.probability
            if needle <= total:
                return candidate
        return candidates[-1]

    def _materialize(self, candidate: EventCandidate, round_num: int, anchor: EventAtom | None) -> EventAtom:
        event_id = anchor.event_id if anchor else f"S{round_num:03d}"
        if anchor and candidate.type != anchor.type:
            event_id = f"S{round_num:03d}_{candidate.type}"
        all_candidates = getattr(candidate, "_all_candidates", [])
        return EventAtom(
            event_id=event_id,
            round=round_num,
            type=candidate.type,
            visibility=candidate.visibility,
            source=candidate.source,
            targets=candidate.targets,
            payload={
                **candidate.payload,
                "generator": "state_event_field",
                "event_candidates": [c.to_dict() for c in all_candidates],
            },
            objective_fact=ObjectiveFact(
                raw_statement=candidate.description or None,
                verifiable_claims=[f"state_generated_{candidate.type}"],
            ),
            framing=candidate.framing,
            truth_status=candidate.truth_status,
            memory_salience=candidate.memory_salience,
            is_anchor=bool(anchor and candidate.type == anchor.type and anchor.is_anchor),
            description=candidate.description,
            act=anchor.act if anchor else None,
        )

    def generate(self, round_num: int, world: WorldState) -> EventAtom | None:
        anchor = self._by_round.get(round_num)
        if anchor is None:
            if not self.state_events:
                return None
            candidates = self._state_candidates(round_num, world)
            probs = _softmax([c.tendency for c in candidates], temperature=0.24)
            for candidate, prob in zip(candidates, probs):
                candidate.probability = prob
                setattr(candidate, "_all_candidates", candidates)
            selected = self._sample_candidate(candidates, round_num)
            return self._materialize(selected, round_num, None)
        if round_num == 60:
            event = copy.deepcopy(anchor)
            event.payload = {**event.payload, "generator": "terminal_anchor"}
            return event
        if not self.state_events:
            event = copy.deepcopy(anchor)
            event.payload = {**event.payload, "generator": "anchor_only"}
            return event
        if anchor.is_anchor:
            event = copy.deepcopy(anchor)
            event.payload = {**event.payload, "generator": "anchor_mandatory"}
            return event

        candidates = self._state_candidates(round_num, world)
        anchor_candidate = self._anchor_candidate(anchor)
        combined = [anchor_candidate, *candidates]
        probs = _softmax([c.tendency for c in combined], temperature=0.24)
        for candidate, prob in zip(combined, probs):
            candidate.probability = prob
            setattr(candidate, "_all_candidates", combined)
        selected = self._sample_candidate(combined, round_num)
        return self._materialize(selected, round_num, anchor)

    def get_schedule(self, max_round: int) -> list[EventAtom]:
        return [copy.deepcopy(self._by_round[r]) for r in range(1, max_round + 1) if r in self._by_round]


def is_agent_active(agent_id: str, round_num: int, config: dict[str, Any]) -> bool:
    active = config.get("active_agents")
    if active is not None:
        if agent_id not in active:
            offstage = config.get("offstage_agents", [])
            if agent_id in offstage and round_num >= config.get("offstage_min_round", 1):
                return agent_id in ("rival_lab_h",) and round_num >= 21
            return False
    if agent_id.startswith("reviewer_"):
        return round_num >= REVIEWER_ACTIVE_FROM
    if agent_id == "rival_lab_h":
        return round_num >= 21
    return True

