"""Cognitive dynamics pipeline 鈥?one round state evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.world.events import apply_payload_to_project
from src.world.models import Agent, EventAtom, WorldState

from .authorship import (
    authorship_dispute_index,
    update_ledger_from_action,
    update_ledger_from_event,
)
from .belief import apply_action_belief_feedback, update_beliefs
from .dynamics import COMPLIANCE_ACTIONS, ESCALATED_ACTIONS
from .math_utils import clamp, impulse_response
from .divergence import compute_divergence, mean_divergence
from .emotion import update_emotion
from .memory import (
    RecallResult,
    apply_rehearsal,
    decay_memories,
    recall_memories,
    reconsolidate_memories,
    write_memory,
)
from .relationship import (
    coalition_strength,
    credit_threat_density,
    trust_fragmentation,
    update_relationships,
)
from .reputation import update_reputation_from_action
from src.world.organization import (
    default_private_intent,
    observation_channel,
    perceived_event,
    resolve_event_cast,
    rumor_recipients,
)


@dataclass
class CognitiveStepResult:
    round: int
    event_id: str
    recalls: dict[str, RecallResult] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    agent_deltas: dict[str, dict[str, Any]] = field(default_factory=dict)


def pre_decision_recall(
    world: WorldState,
    event: EventAtom,
    *,
    disable_memory: bool = False,
    omniscient_observation: bool = False,
) -> dict[str, RecallResult]:
    """Phase A: decay + recall before agent policy (no memory write yet)."""
    leaked = set() if omniscient_observation else set(rumor_recipients(world, event))
    if disable_memory:
        empty: dict[str, RecallResult] = {}
        for agent_id, agent in world.agents.items():
            channel = observation_channel(
                agent, event, leaked=leaked, omniscient=omniscient_observation
            )
            empty[agent_id] = RecallResult(
                {}, [], 0.0, 0.0, {"disabled": True, "observation_channel": channel}
            )
        return empty

    current_round = event.round
    recalls: dict[str, RecallResult] = {}
    for agent_id, agent in world.agents.items():
        decay_memories(agent, current_round, same_event=event)
        channel = observation_channel(
            agent, event, leaked=leaked, omniscient=omniscient_observation
        )
        cue = perceived_event(event, channel)
        recall = recall_memories(agent, cue, current_round)
        recall.audit["observation_channel"] = channel
        recalls[agent_id] = recall
    return recalls


def commit_cognition_phase(
    world: WorldState,
    event: EventAtom,
    recalls: dict[str, RecallResult],
    actions: list[dict[str, Any]] | None = None,
    *,
    disable_memory: bool = False,
    llm_adapter: Any | None = None,
    omniscient_observation: bool = False,
) -> CognitiveStepResult:
    """Phase B: after decisions 鈥?write memory, update states, apply action effects."""
    current_round = event.round
    world.project.current_round = current_round

    project_dict = world.project.project.model_dump()
    project_dict = apply_payload_to_project(project_dict, event.payload)
    from src.world.models import ProjectMetrics

    world.project.project = ProjectMetrics(**project_dict)
    update_ledger_from_event(world, event.type, event.payload)

    leaked = set() if omniscient_observation else set(rumor_recipients(world, event))
    channels: dict[str, str] = {}
    agent_deltas: dict[str, dict[str, Any]] = {}

    for agent_id, agent in world.agents.items():
        recall = recalls.get(agent_id)
        channel = (recall.audit or {}).get("observation_channel") if recall else None
        if channel not in {"direct", "rumor", "none"}:
            channel = observation_channel(
                agent, event, leaked=leaked, omniscient=omniscient_observation
            )
        channels[agent_id] = channel
        if recall and not disable_memory:
            apply_rehearsal(agent, recall.attention_weights, current_round)

        reconsolidation = {"updated": []}
        mem = None
        if not disable_memory and channel == "direct":
            reconsolidation = reconsolidate_memories(agent, event, recall, current_round)
            mem = write_memory(agent, event, current_round, llm_adapter=llm_adapter, world=world)
        cue = perceived_event(event, channel)
        emotion = update_emotion(agent, cue, world.project.project, recall, channel=channel)
        beliefs = update_beliefs(agent, cue, world.project.project, recall, channel=channel)

        agent_deltas[agent_id] = {
            "memory_written": mem.to_dict() if mem else None,
            "emotion": emotion,
            "beliefs": beliefs,
            "recall_audit": recall.audit if recall else {},
            "memory_reconsolidation": reconsolidation,
            "observation_channel": channel,
        }

    if not disable_memory:
        for agent_id, channel in channels.items():
            if channel != "rumor":
                continue
            rumor_mem = write_memory(
                world.agents[agent_id],
                event,
                current_round,
                llm_adapter=llm_adapter,
                channel="rumor",
                world=world,
            )
            if rumor_mem:
                agent_deltas.setdefault(agent_id, {})
                agent_deltas[agent_id]["rumor_memory"] = rumor_mem.to_dict()

    if actions:
        for act in actions:
            aid = act.get("agent")
            if aid and aid in world.agents:
                apply_action_cognition(world, aid, act, current_round)

    update_relationships(world, event, recalls, actions=actions)

    internal = world.world_config.get("internal_agents", [])
    n = max(1, len(channels))
    cast = resolve_event_cast(world)
    idea_div = 0.0
    if cast.idea in world.agents:
        idea_div = compute_divergence(world.agents[cast.idea])
    metrics = {
        "authorship_dispute_index": authorship_dispute_index(world),
        "trust_fragmentation": trust_fragmentation(world.relationships, internal),
        "coalition_strength": coalition_strength(world.relationships),
        "credit_threat_density": credit_threat_density(world.relationships),
        "public_private_divergence": idea_div,
        "public_private_divergence_idea": idea_div,
        "public_private_divergence_lab": mean_divergence(world.agents, internal),
        "observation_direct_share": round(sum(1 for c in channels.values() if c == "direct") / n, 4),
        "observation_rumor_share": round(sum(1 for c in channels.values() if c == "rumor") / n, 4),
        "observation_blind_share": round(sum(1 for c in channels.values() if c == "none") / n, 4),
    }

    return CognitiveStepResult(
        round=current_round,
        event_id=event.event_id,
        recalls=recalls,
        metrics=metrics,
        agent_deltas=agent_deltas,
    )


def process_event_phase(world: WorldState, event: EventAtom, llm_adapter: Any | None = None) -> CognitiveStepResult:
    """Full single-phase cognition (recall + commit). Used by Part 2 tests."""
    if llm_adapter is None:
        from src.engine.llm_adapter import get_adapter

        llm_adapter = get_adapter()
    recalls = pre_decision_recall(world, event)
    return commit_cognition_phase(world, event, recalls, llm_adapter=llm_adapter)


def apply_action_cognition(
    world: WorldState,
    agent_id: str,
    action: dict[str, Any],
    current_round: int,
) -> None:
    """Apply cognitive consequences of an agent action."""
    agent = world.agents[agent_id]
    atype = action.get("type")
    intensity = float(action.get("intensity", 0.5))

    update_ledger_from_action(world, agent_id, intensity, str(atype or ""))

    apply_action_belief_feedback(agent, str(atype or ""), intensity)

    if atype in ESCALATED_ACTIONS:
        shock = impulse_response(intensity, sensitivity=0.36, saturation=2.4)
        agent.emotion.resentment = clamp(agent.emotion.resentment + shock * 0.22)
        agent.emotion.anger = clamp(agent.emotion.anger + shock * 0.16)
    elif atype in COMPLIANCE_ACTIONS:
        relief = impulse_response(intensity, sensitivity=0.18, saturation=3.0)
        agent.emotion.resentment = clamp(agent.emotion.resentment - relief * 0.12)
        agent.emotion.anxiety = clamp(agent.emotion.anxiety - relief * 0.08)

    agent.public_position = action.get("public_position", agent.public_position)
    agent.private_intent = action.get("private_intent", agent.private_intent)
    if not agent.private_intent:
        agent.private_intent = default_private_intent(agent, str(action.get("type", "lay_low")))
    update_reputation_from_action(world, action)

    divergence = compute_divergence(agent)
    agent.action_history.append(
        {
            "round": current_round,
            "action": action,
            "public_private_divergence": divergence,
        }
    )


def _noop_event(round_num: int) -> EventAtom:
    from src.world.models import ObjectiveFact

    return EventAtom(
        event_id="E000",
        round=round_num,
        type="team_meeting",
        visibility="team",
        source="pi",
        targets=["project"],
        payload={},
        objective_fact=ObjectiveFact(raw_statement=None, verifiable_claims=[]),
        framing="neutral",
        truth_status="verified",
        memory_salience=0.01,
        is_anchor=False,
    )


def simulate_memory_decay_only(agent: Agent, rounds: int, emotional_arousal: float = 0.0) -> list[float]:
    """Helper for tests: pure decay trajectory without events."""
    from src.world.models import Emotion

    agent.emotion = Emotion(
        confidence=0.5,
        anxiety=0.3,
        anger=emotional_arousal * 0.5,
        resentment=emotional_arousal * 0.5,
        guilt=0.0,
        hope=0.5,
        burnout=0.2,
    )
    strengths: list[float] = []
    if not agent.memory:
        return strengths
    mem = agent.memory[0]
    s0 = float(mem.get("strength_0", mem["strength"]))
    mem["strength_0"] = s0
    strengths.append(float(mem["strength"]))
    for r in range(2, rounds + 1):
        decay_memories(agent, r)
        strengths.append(float(agent.memory[0]["strength"]))
    return strengths


