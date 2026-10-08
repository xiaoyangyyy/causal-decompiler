"""Organization identity, observation, and credit helpers.

Hardcoded phd_a / phd_b / pi lookups make the canonical 14-agent story work
and make scaled populations silently incoherent. These helpers keep the
canonical cast when it exists, and infer equivalent roles otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.world.models import Agent, AgentRole, EventAtom, RelationshipEdge, WorldState, clamp

EXTERNAL_ROLES = {
    AgentRole.REVIEWER,
    AgentRole.PROGRAM_OFFICER,
    AgentRole.RIVAL_LAB,
    AgentRole.ALUMNI,
    AgentRole.COLLABORATOR,
}

OBSERVATION_GAIN = {
    "direct": 1.0,
    "rumor": 0.45,
    "none": 0.0,
}

ROLE_DEFAULT_GOALS = {
    AgentRole.PI: "protect_lab_and_arbitrate_credit",
    AgentRole.IDEA_ORIGINATOR: "secure_first_author",
    AgentRole.EXPERIMENTER: "convert_execution_into_authorship",
    AgentRole.MASTER_STUDENT: "gain_visible_credit",
    AgentRole.POSTDOC: "protect_independence_and_credit",
    AgentRole.ENGINEER: "protect_reproducibility",
    AgentRole.VISITING_STUDENT: "avoid_career_damage",
    AgentRole.COLLABORATOR: "extract_shared_credit",
    AgentRole.RIVAL_LAB: "outcompete_on_priority",
    AgentRole.REVIEWER: "evaluate_paper",
    AgentRole.PROGRAM_OFFICER: "protect_funding_integrity",
    AgentRole.ALUMNI: "warn_or_reinterpret_history",
    AgentRole.OPERATOR: "keep_the_operation_stable",
}


@dataclass(frozen=True)
class EventCast:
    """Role mapping used by the state-driven event field."""

    pi: str
    idea: str
    experimenter: str
    postdoc: str | None = None
    engineer: str | None = None
    alumni: str | None = None
    rival: str | None = None


def internal_ids(world: WorldState) -> list[str]:
    configured = [aid for aid in world.world_config.get("internal_agents", []) if aid in world.agents]
    if configured:
        return configured
    return [aid for aid, agent in world.agents.items() if agent.role not in EXTERNAL_ROLES]


def authority_ids(world: WorldState) -> list[str]:
    return [aid for aid, agent in world.agents.items() if agent.role == AgentRole.PI]


def lab_id_of(agent: Agent | None) -> int:
    if agent is None:
        return 1
    return int(agent.extra_traits.get("lab_id", 1) or 1)


def primary_authority(world: WorldState, agent: Agent | None = None) -> str | None:
    authorities = authority_ids(world)
    if not authorities:
        return None
    if "pi" in world.agents and (agent is None or lab_id_of(agent) == lab_id_of(world.agents["pi"])):
        return "pi"
    if agent is None:
        return authorities[0]
    same_lab = [aid for aid in authorities if lab_id_of(world.agents[aid]) == lab_id_of(agent)]
    return same_lab[0] if same_lab else authorities[0]


def _first_with_role(world: WorldState, role: AgentRole, preferred_ids: Iterable[str] = ()) -> str | None:
    for aid in preferred_ids:
        agent = world.agents.get(aid)
        if agent and agent.role == role:
            return aid
    for aid, agent in world.agents.items():
        if agent.role == role:
            return aid
    return None


def _edge_map(world: WorldState) -> dict[tuple[str, str], RelationshipEdge]:
    return {(edge.source, edge.target): edge for edge in world.relationships}


def agent_contribution_share(world: WorldState, agent_id: str) -> float:
    """Mean ledger share across contribution dimensions.

    The ledger is keyed dimension -> agent, not agent -> dimension.
    """
    ledger = world.project.contribution_ledger or {}
    shares: list[float] = []
    for bucket in ledger.values():
        if isinstance(bucket, dict) and agent_id in bucket:
            shares.append(float(bucket.get(agent_id, 0.0)))
    if shares:
        return clamp(sum(shares) / len(shares))
    agent = world.agents.get(agent_id)
    if agent is None:
        return 0.0
    controls = agent.resources.code_control + agent.resources.data_control + agent.resources.writing_control
    return clamp(controls / 3.0)


def credit_rival_pair(world: WorldState) -> tuple[str, str] | None:
    if "phd_a" in world.agents and "phd_b" in world.agents:
        return "phd_a", "phd_b"
    internal = [aid for aid in internal_ids(world) if world.agents[aid].role != AgentRole.PI]
    if len(internal) < 2:
        return None
    edges = _edge_map(world)
    best: tuple[str, str] | None = None
    best_score = -1.0
    for src in internal:
        for tgt in internal:
            if src >= tgt:
                continue
            ab = edges.get((src, tgt))
            ba = edges.get((tgt, src))
            threat = 0.0
            if ab:
                threat += ab.perceived_credit_threat + ab.resentment
            if ba:
                threat += ba.perceived_credit_threat + ba.resentment
            share_gap = abs(agent_contribution_share(world, src) - agent_contribution_share(world, tgt))
            score = threat + share_gap
            if score > best_score:
                best_score = score
                best = (src, tgt)
    return best


def resolve_event_cast(world: WorldState) -> EventCast:
    pi = primary_authority(world) or next(iter(world.agents))
    rivals = credit_rival_pair(world)
    idea = "phd_a" if "phd_a" in world.agents else None
    experimenter = "phd_b" if "phd_b" in world.agents else None
    if rivals:
        left, right = rivals
        if idea is None:
            idea = left if world.agents[left].role == AgentRole.IDEA_ORIGINATOR else left
        if experimenter is None:
            experimenter = right if right != idea else left
        if idea == experimenter:
            experimenter = right if idea == left else left
    if idea is None:
        idea = _first_with_role(world, AgentRole.IDEA_ORIGINATOR) or next(
            (aid for aid in internal_ids(world) if aid != pi),
            pi,
        )
    if experimenter is None:
        experimenter = _first_with_role(world, AgentRole.EXPERIMENTER) or next(
            (aid for aid in internal_ids(world) if aid not in {pi, idea}),
            idea,
        )
    return EventCast(
        pi=pi,
        idea=idea,
        experimenter=experimenter,
        postdoc=_first_with_role(world, AgentRole.POSTDOC, ("postdoc_d",)),
        engineer=_first_with_role(world, AgentRole.ENGINEER, ("engineer_e",)),
        alumni=_first_with_role(world, AgentRole.ALUMNI, ("lab_alumni",)),
        rival=_first_with_role(world, AgentRole.RIVAL_LAB, ("rival_lab_h",)),
    )


def default_private_intent(agent: Agent, strategy: str = "lay_low") -> dict[str, object]:
    return {
        "goal": ROLE_DEFAULT_GOALS.get(agent.role, "survive_the_lab"),
        "strategy": strategy,
        "trust_pi": agent.beliefs.pi_fairness,
    }


def can_directly_observe(agent: Agent, event: EventAtom) -> bool:
    """Whether the agent perceives the event as a first-hand social fact."""
    if agent.id == event.source or agent.id in event.targets:
        return True
    visibility = str(event.visibility or "team")
    if visibility == "public":
        return True
    if visibility == "team":
        return agent.role not in EXTERNAL_ROLES
    return False


def rumor_leaks(world: WorldState, event: EventAtom) -> dict[str, float]:
    """Gossip weight for each non-participant. Zero weight means no path."""
    if str(event.visibility or "team") not in {"bilateral", "private"}:
        return {}
    participants = {event.source, *[t for t in event.targets if t in world.agents]}
    edges = _edge_map(world)
    leaked: dict[str, float] = {}
    for agent_id, agent in world.agents.items():
        if agent_id in participants or agent.role in EXTERNAL_ROLES:
            continue
        leak = 0.0
        for participant in participants:
            forward = edges.get((participant, agent_id))
            backward = edges.get((agent_id, participant))
            for edge in (forward, backward):
                if edge is None:
                    continue
                leak = max(leak, edge.communication_frequency * edge.information_access)
        if leak > 0.0:
            leaked[agent_id] = leak
    return leaked


def rumor_recipients(world: WorldState, event: EventAtom) -> list[str]:
    """Agents with a positive gossip weight."""
    return list(rumor_leaks(world, event))


def observation_channel(
    agent: Agent,
    event: EventAtom,
    *,
    leaked: Iterable[str] = (),
    omniscient: bool = False,
) -> str:
    """Return direct, rumor, or none. Lesion `omniscient` removes the filter."""
    if omniscient:
        return "direct"
    if can_directly_observe(agent, event):
        return "direct"
    if agent.id in set(leaked):
        return "rumor"
    return "none"


def observation_gain(channel: str, *, leak: float = 1.0) -> float:
    base = float(OBSERVATION_GAIN.get(channel, 0.0))
    if channel == "rumor":
        return base * clamp(float(leak))
    return base


def perceived_event(event: EventAtom, channel: str, *, leak: float = 1.0) -> EventAtom:
    """Event as this observer actually received it — weaker cue if rumor or blind."""
    if channel == "direct":
        return event
    ev = event.model_copy(deep=True)
    if channel == "rumor":
        ev.memory_salience = clamp(float(event.memory_salience) * 0.55 * clamp(float(leak)))
        ev.truth_status = "rumored"
        return ev
    ev.memory_salience = 0.0
    ev.truth_status = "unobserved"
    return ev
