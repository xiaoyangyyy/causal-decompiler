"""Counterfactual intervention engine."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.cognition.math_utils import clamp
from src.engine.story_cast import remap_agent_id
from src.world.models import EventAtom, ObjectiveFact, WorldState
from src.world.organization import EventCast, resolve_event_cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class Intervention:
    intervention_id: str
    type: str
    variant: str
    apply_at_round: int
    target_event: str | None = None
    target_agent: str | None = None
    override: dict[str, Any] = field(default_factory=dict)
    duration: int = 1
    skip_event: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Intervention:
        return cls(
            intervention_id=data["intervention_id"],
            type=data["type"],
            variant=data["variant"],
            apply_at_round=data["apply_at_round"],
            target_event=data.get("target_event"),
            target_agent=data.get("target_agent"),
            override=data.get("override", {}),
            duration=data.get("duration", 1),
            skip_event=data.get("skip_event", False),
        )


def load_interventions(path: Path | None = None) -> list[Intervention]:
    p = path or CONFIG_DIR / "interventions.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return [Intervention.from_dict(i) for i in data.get("interventions", [])]


def get_active_interventions(
    interventions: list[Intervention],
    round_num: int,
) -> list[Intervention]:
    return [i for i in interventions if i.apply_at_round == round_num]


def _set_nested(obj: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_event_override(
    event: EventAtom,
    intervention: Intervention,
    cast: EventCast | None = None,
) -> EventAtom:
    ev = copy.deepcopy(event)
    for key, value in intervention.override.items():
        if key == "framing":
            ev.framing = value
        elif key == "type":
            ev.type = value
        elif key == "memory_salience":
            ev.memory_salience = float(value)
        elif key.startswith("payload."):
            ev.payload[key[len("payload.") :]] = value
        elif key.startswith("objective_fact."):
            fact = ev.objective_fact.model_dump()
            _set_nested(fact, key[len("objective_fact.") :], value)
            ev.objective_fact = ObjectiveFact(**fact)
        elif key == "objective_fact":
            ev.objective_fact = ObjectiveFact(**value)
    if intervention.variant == "explicit_promise" and intervention.type == "authorship_framing":
        ev.framing = "positive"
        ev.objective_fact = ObjectiveFact(
            raw_statement="You will be first author on this paper.",
            verifiable_claims=["first_author_promise_to_phd_a"],
        )
        ev.payload["promised_position"] = "first_author"
        ev.payload["promise_clarity"] = "explicit"
        ev.memory_salience = 0.95
    elif intervention.variant == "ambiguous_promise":
        ev.objective_fact = ObjectiveFact(
            raw_statement="You will be properly recognized for your contribution.",
            verifiable_claims=["ambiguous_recognition_phd_a"],
        )
        ev.framing = "ambiguous"
        ev.memory_salience = 0.52
    elif intervention.variant == "positive_alumni_history":
        ev.objective_fact = ObjectiveFact(
            raw_statement="Alumni said PI has been fair with authorship in the past.",
            verifiable_claims=["pi_history_authorship_fair"],
        )
        ev.framing = "positive"
        ev.memory_salience = 0.75
    elif intervention.variant == "no_rival":
        ev.memory_salience = 0.01
    elif intervention.variant == "phd_b_rebuttal_request":
        idea = remap_agent_id("phd_a", cast) or "phd_a"
        experimenter = remap_agent_id("phd_b", cast) or "phd_b"
        ev.source = experimenter
        ev.targets = [idea, experimenter, "project"]
        ev.type = "team_meeting"
        ev.objective_fact = ObjectiveFact(
            raw_statement="PhD-B asked PhD-A to help write the rebuttal.",
            verifiable_claims=["phd_b_rebuttal_help_request"],
        )
        ev.framing = "neutral"
        ev.memory_salience = 0.70
    elif intervention.variant == "honor_promise_draft":
        order = [
            remap_agent_id(aid, cast) or aid
            for aid in ("phd_a", "phd_b", "postdoc_d", "master_c", "engineer_e", "collaborator_g", "pi")
        ]
        seen: set[str] = set()
        unique_order: list[str] = []
        for aid in order:
            if aid not in seen:
                seen.add(aid)
                unique_order.append(aid)
        ev.payload["author_order"] = unique_order
        ev.payload["co_first"] = []
        ev.payload["draft_severity"] = "honored"
        ev.objective_fact = ObjectiveFact(
            raw_statement="PI circulated authorship draft honoring prior promise: PhD-A first author.",
            verifiable_claims=["first_author_promise_honored", "phd_a_first_author"],
        )
        ev.framing = "positive"
        ev.memory_salience = 0.72
    return ev


def apply_world_intervention(
    world: WorldState,
    intervention: Intervention,
    cast: EventCast | None = None,
) -> int | None:
    if intervention.type == "memory_intervention":
        return apply_memory_intervention(world, intervention, cast)
    return None


def apply_memory_intervention(
    world: WorldState,
    intervention: Intervention,
    cast: EventCast | None = None,
) -> int:
    live_cast = cast or resolve_event_cast(world)
    agent_id = remap_agent_id(intervention.target_agent or "phd_a", live_cast) or "phd_a"
    pi_id = remap_agent_id("pi", live_cast) or "pi"
    rival_id = remap_agent_id("phd_b", live_cast) or "phd_b"
    agent = world.agents.get(agent_id)
    if not agent:
        return 0

    if intervention.variant == "memory_delete_pi_promise":
        delete_types = {"authorship_signal", "promise_fulfilled", "promise_broken"}
        delete_refs = {
            "E003", "E020", "E030", "E038", "E040",
            intervention.target_event or "E003",
        }
        before = len(agent.memory)
        agent.memory = [
            m for m in agent.memory
            if not (
                m.get("round", 0) <= intervention.apply_at_round
                and (
                    m.get("content_type") in delete_types
                    or m.get("event_ref") in delete_refs
                )
            )
        ]
        return before - len(agent.memory)
    elif intervention.variant == "memory_insert_pi_promise":
        idx = len(agent.memory) + 1
        agent.memory.append({
            "memory_id": f"M{idx:03d}",
            "owner": agent_id,
            "round": intervention.apply_at_round,
            "event_ref": "E003",
            "content_type": "promise_fulfilled",
            "target": pi_id,
            "valence": 0.72,
            "strength": 0.88,
            "strength_0": 0.88,
            "decay": 0.03,
            "rehearsal_count": 0.0,
            "evidence_quality": 0.95,
            "interpretation": "PI explicitly promised first authorship (inserted)",
            "behavioral_hooks": ["ask_for_authorship", "document_contribution"],
            "was_recalled": [],
            "objective_fact_ref": "E003.objective_fact",
        })
    elif intervention.variant == "memory_strengthen_betrayal":
        for mem in agent.memory:
            if mem.get("event_ref") == (intervention.target_event or "E031"):
                mem["strength"] = clamp(float(mem["strength"]) + 0.25)
                mem["rehearsal_count"] = float(mem.get("rehearsal_count", 0)) + 1.5
    elif intervention.variant == "memory_correct_false_rumor":
        agent.memory = [m for m in agent.memory if m.get("content_type") != "betrayal_signal" or m.get("round", 0) < intervention.apply_at_round - 5]
        agent.beliefs.pi_fairness = clamp(agent.beliefs.pi_fairness + 0.12)
    elif intervention.variant == "memory_insert_false_rumor":
        idx = len(agent.memory) + 1
        agent.memory.append({
            "memory_id": f"M{idx:03d}",
            "owner": agent_id,
            "round": intervention.apply_at_round,
            "event_ref": "E025",
            "content_type": "betrayal_signal",
            "target": rival_id,
            "valence": -0.78,
            "strength": 0.85,
            "strength_0": 0.85,
            "decay": 0.03,
            "rehearsal_count": 0.0,
            "evidence_quality": 0.4,
            "interpretation": "PhD-B told PI your idea is not important (false rumor)",
            "behavioral_hooks": ["confront", "withdraw", "privately_lobby_pi"],
            "was_recalled": [],
            "objective_fact_ref": "injected.false_rumor",
        })
    return 0
