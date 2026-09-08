"""Signed graph diffusion for relationships 鈥?continuous message passing."""

from __future__ import annotations

import math
from typing import Any

from src.world.models import Agent, EventAtom, RelationshipEdge, WorldState

from .math_utils import clamp, logistic_gate, softplus
from .memory import RecallResult


def _edge_index(relationships: list[RelationshipEdge]) -> dict[tuple[str, str], int]:
    return {(e.source, e.target): i for i, e in enumerate(relationships)}


def _get_edge(relationships: list[RelationshipEdge], src: str, tgt: str) -> RelationshipEdge | None:
    for e in relationships:
        if e.source == src and e.target == tgt:
            return e
    return None


def _ledger_credit_jacobian(
    ledger: dict[str, dict[str, float]],
    agent_a: str,
    agent_b: str,
) -> float:
    """
    Continuous credit threat: how much B's total ledger growth
    compresses A's relative share (Jacobian-style, graded).
    """
    dims = list(ledger.keys())
    if not dims:
        return 0.0

    shares_a: list[float] = []
    shares_b: list[float] = []
    for dim in dims:
        bucket = ledger.get(dim, {})
        total = sum(bucket.values()) or 1.0
        shares_a.append(bucket.get(agent_a, 0.0) / total)
        shares_b.append(bucket.get(agent_b, 0.0) / total)

    mean_a = sum(shares_a) / len(shares_a)
    mean_b = sum(shares_b) / len(shares_b)
    return clamp(mean_b - mean_a * 0.5, 0.0, 1.0)


def _event_valence_for_pair(event: EventAtom, src: str, tgt: str) -> float:
    if event.source == src and tgt in event.targets:
        if event.type in ("public_praise", "support_teammate"):
            return 0.35 * event.memory_salience
        if event.type in ("public_blame", "undermine_teammate", "narrative_change"):
            return -0.40 * event.memory_salience
        if event.type in ("private_lobbying", "credit_dispute"):
            return -0.25 * event.memory_salience
    if event.source == tgt and event.type == "public_praise":
        praised = event.payload.get("praised_agent")
        if praised == tgt:
            return 0.30 * event.memory_salience
    if event.type == "authorship_ambiguity" and tgt == "pi":
        return -0.20 * event.memory_salience
    if event.type == "authorship_promise" and tgt == "pi":
        return 0.28 * event.memory_salience
    if event.type == "authorship_draft" and tgt == "pi":
        severity = str(event.payload.get("draft_severity") or event.framing or "")
        if severity in {"honored", "positive"} or event.framing == "positive":
            return 0.22 * event.memory_salience
        return -0.32 * event.memory_salience
    return 0.0


def update_relationships(
    world: WorldState,
    event: EventAtom,
    recalls: dict[str, RecallResult],
    actions: list[dict[str, Any]] | None = None,
) -> list[RelationshipEdge]:
    relationships = world.relationships
    agents = world.agents
    ledger = world.project.contribution_ledger
    eta = 0.12

    deltas_trust: dict[tuple[str, str], float] = {}
    deltas_resent: dict[tuple[str, str], float] = {}
    deltas_alliance: dict[tuple[str, str], float] = {}
    deltas_credit: dict[tuple[str, str], float] = {}
    deltas_obligation: dict[tuple[str, str], float] = {}

    internal = set(world.world_config.get("internal_agents", []))

    for edge in relationships:
        src, tgt = edge.source, edge.target
        if src not in internal or tgt not in internal:
            continue

        pair_valence = _event_valence_for_pair(event, src, tgt)
        recall_src = recalls.get(src)
        recall_mod = 0.0
        if recall_src:
            recall_mod = recall_src.recall_field_valence * recall_src.recall_field_strength * 0.15

        credit_j = _ledger_credit_jacobian(ledger, src, tgt)
        agent_src = agents.get(src)
        agent_tgt = agents.get(tgt)

        harm = max(0.0, -pair_valence)
        trust_delta = eta * math.tanh(pair_valence * 3.0 + recall_mod)
        # Credit-threat drain only on actual harm. A 0.35 residual used to
        # grind every internal edge to the 0.06 floor by mid-story.
        trust_delta -= eta * 0.22 * edge.resentment * softplus(max(0.0, credit_j)) * harm
        if pair_valence > 0.04:
            trust_delta += eta * 0.24 * pair_valence * (1.0 - edge.trust)
        if edge.trust < 0.20:
            trust_delta += eta * 0.20 * (0.20 - edge.trust) * (1.0 - 0.55 * edge.resentment)

        resent_delta = eta * harm * 1.15
        if pair_valence < 0.0:
            resent_delta += eta * 0.28 * max(0.0, credit_j) * (
                agent_src.personality.credit_sensitivity if agent_src else 0.5
            )
        else:
            resent_delta -= eta * 0.16 * edge.resentment

        coop = agent_src.personality.cooperation if agent_src else 0.5
        recip = agent_src.personality.reciprocity if agent_src else 0.5
        alliance_delta = (
            eta
            * edge.trust
            * coop
            * recip
            * (1.0 - edge.resentment)
            * softplus(pair_valence)
        )
        alliance_delta -= eta * 1.6 * softplus(-pair_valence) * edge.alliance

        credit_delta = eta * (credit_j - edge.perceived_credit_threat) * 0.6

        key = (src, tgt)
        deltas_trust[key] = deltas_trust.get(key, 0.0) + trust_delta
        deltas_resent[key] = deltas_resent.get(key, 0.0) + resent_delta
        deltas_alliance[key] = deltas_alliance.get(key, 0.0) + alliance_delta
        deltas_credit[key] = deltas_credit.get(key, 0.0) + credit_delta

    if event.type == "public_praise":
        praised = event.payload.get("praised_agent")
        if praised:
            for edge in relationships:
                if edge.target == praised and edge.source != praised:
                    key = (edge.source, edge.target)
                    deltas_credit[key] = deltas_credit.get(key, 0.0) + eta * 0.25 * event.memory_salience

    if actions:
        for act in actions:
            _apply_action_to_deltas(
                act, deltas_trust, deltas_resent, deltas_alliance, deltas_credit, deltas_obligation, eta
            )
        _apply_triadic_private_influence(relationships, actions, deltas_trust, deltas_resent, deltas_credit, eta)
        _apply_structural_balance(relationships, internal, deltas_trust, deltas_resent, deltas_alliance, eta)

    updated: list[RelationshipEdge] = []
    for edge in relationships:
        key = (edge.source, edge.target)
        new_edge = RelationshipEdge(
            source=edge.source,
            target=edge.target,
            trust=round(max(0.06, clamp(edge.trust + deltas_trust.get(key, 0.0))), 4),
            resentment=round(clamp(edge.resentment + deltas_resent.get(key, 0.0)), 4),
            dependency=round(clamp(edge.dependency + deltas_trust.get(key, 0.0) * 0.05), 4),
            obligation=round(clamp(edge.obligation + deltas_obligation.get(key, 0.0)), 4),
            perceived_credit_threat=round(
                clamp(edge.perceived_credit_threat + deltas_credit.get(key, 0.0)), 4
            ),
            communication_frequency=round(
                clamp(edge.communication_frequency + abs(deltas_trust.get(key, 0.0)) * 0.3), 4
            ),
            alliance=round(clamp(edge.alliance + deltas_alliance.get(key, 0.0)), 4),
            information_access=round(
                clamp(edge.information_access + deltas_alliance.get(key, 0.0) * 0.2), 4
            ),
            last_interaction_valence=round(
                clamp(deltas_trust.get(key, 0.0) * 5, -1, 1), 4
            ),
        )
        updated.append(new_edge)

    world.relationships = updated
    return updated


def _apply_triadic_private_influence(
    relationships: list[RelationshipEdge],
    actions: list[dict[str, Any]],
    deltas_trust: dict[tuple[str, str], float],
    deltas_resent: dict[tuple[str, str], float],
    deltas_credit: dict[tuple[str, str], float],
    eta: float,
) -> None:
    """Private A->C talk softly contaminates C's view of target B."""
    edge_lookup = {(e.source, e.target): e for e in relationships}
    political = {"privately_lobby_pi", "undermine_teammate", "form_alliance", "talk_to_alumni", "challenge_claim"}
    for action in actions:
        src = action.get("agent")
        mediator = action.get("target")
        atype = action.get("type")
        intensity = float(action.get("intensity", 0.5))
        if atype not in political or not src or not mediator or mediator == "project":
            continue
        for (m_src, third), mediator_edge in edge_lookup.items():
            if m_src != mediator or third in {src, mediator}:
                continue
            source_view = edge_lookup.get((src, third))
            if not source_view:
                continue
            access = mediator_edge.information_access * mediator_edge.communication_frequency
            contamination = eta * intensity * access * (0.35 + 0.65 * source_view.perceived_credit_threat)
            trust_pull = contamination * (source_view.trust - mediator_edge.trust)
            resent_pull = contamination * (source_view.resentment - mediator_edge.resentment)
            threat_pull = contamination * (source_view.perceived_credit_threat - mediator_edge.perceived_credit_threat)
            key = (mediator, third)
            deltas_trust[key] = deltas_trust.get(key, 0.0) + trust_pull
            deltas_resent[key] = deltas_resent.get(key, 0.0) + resent_pull
            deltas_credit[key] = deltas_credit.get(key, 0.0) + threat_pull


def _apply_structural_balance(
    relationships: list[RelationshipEdge],
    internal: set[str],
    deltas_trust: dict[tuple[str, str], float],
    deltas_resent: dict[tuple[str, str], float],
    deltas_alliance: dict[tuple[str, str], float],
    eta: float,
) -> None:
    """Heider-style triad closure: ally-of-ally, enemy-of-ally."""
    lookup = {(e.source, e.target): e for e in relationships}
    internals = [aid for aid in internal]
    if len(internals) > 24:
        internals = internals[:24]
    for a in internals:
        for b in internals:
            if a == b:
                continue
            ab = lookup.get((a, b))
            if ab is None or ab.alliance < 0.35:
                continue
            for c in internals:
                if c in {a, b}:
                    continue
                bc = lookup.get((b, c))
                ac_key = (a, c)
                if bc is None:
                    continue
                friend_signal = ab.alliance * bc.trust * (1.0 - bc.resentment)
                enemy_signal = ab.alliance * bc.resentment * (1.0 - bc.trust)
                deltas_trust[ac_key] = deltas_trust.get(ac_key, 0.0) + eta * 0.18 * (friend_signal - enemy_signal)
                deltas_alliance[ac_key] = deltas_alliance.get(ac_key, 0.0) + eta * 0.12 * friend_signal
                deltas_resent[ac_key] = deltas_resent.get(ac_key, 0.0) + eta * 0.16 * enemy_signal


def _apply_action_to_deltas(
    action: dict[str, Any],
    deltas_trust: dict[tuple[str, str], float],
    deltas_resent: dict[tuple[str, str], float],
    deltas_alliance: dict[tuple[str, str], float],
    deltas_credit: dict[tuple[str, str], float],
    deltas_obligation: dict[tuple[str, str], float],
    eta: float,
) -> None:
    src = action.get("agent")
    tgt = action.get("target")
    atype = action.get("type")
    intensity = float(action.get("intensity", 0.5))
    if not src or not tgt or src == tgt:
        return
    key = (src, tgt)
    reverse = (tgt, src)
    if atype == "support_teammate":
        deltas_trust[key] = deltas_trust.get(key, 0.0) + eta * intensity
        deltas_alliance[key] = deltas_alliance.get(key, 0.0) + eta * 0.8 * intensity
        deltas_obligation[reverse] = deltas_obligation.get(reverse, 0.0) + eta * 0.55 * intensity
    elif atype == "undermine_teammate":
        deltas_trust[key] = deltas_trust.get(key, 0.0) - eta * intensity
        deltas_resent[key] = deltas_resent.get(key, 0.0) + eta * intensity
        deltas_obligation[reverse] = deltas_obligation.get(reverse, 0.0) - eta * 0.40 * intensity
    elif atype == "form_alliance":
        deltas_alliance[key] = deltas_alliance.get(key, 0.0) + eta * 1.2 * intensity
        deltas_trust[key] = deltas_trust.get(key, 0.0) + eta * 0.5 * intensity
        deltas_obligation[key] = deltas_obligation.get(key, 0.0) + eta * 0.35 * intensity
        deltas_obligation[reverse] = deltas_obligation.get(reverse, 0.0) + eta * 0.35 * intensity
    elif atype == "apologize":
        deltas_resent[key] = deltas_resent.get(key, 0.0) - eta * 0.7 * intensity
        deltas_trust[key] = deltas_trust.get(key, 0.0) + eta * 0.4 * intensity
    elif atype == "privately_lobby_pi":
        deltas_trust[key] = deltas_trust.get(key, 0.0) + eta * 0.25 * intensity
        deltas_alliance[key] = deltas_alliance.get(key, 0.0) + eta * 0.45 * intensity
        reverse = (tgt, src)
        deltas_trust[reverse] = deltas_trust.get(reverse, 0.0) + eta * 0.18 * intensity
        deltas_credit[reverse] = deltas_credit.get(reverse, 0.0) + eta * 0.35 * intensity
    elif atype in ("ask_for_authorship", "request_mediation"):
        deltas_trust[key] = deltas_trust.get(key, 0.0) - eta * 0.28 * intensity
        deltas_credit[key] = deltas_credit.get(key, 0.0) + eta * 0.25 * intensity
    elif atype in ("challenge_claim", "confront", "blame"):
        deltas_trust[key] = deltas_trust.get(key, 0.0) - eta * 0.65 * intensity
        deltas_resent[key] = deltas_resent.get(key, 0.0) + eta * 0.70 * intensity
        deltas_credit[key] = deltas_credit.get(key, 0.0) + eta * 0.80 * intensity
        reverse = (tgt, src)
        deltas_resent[reverse] = deltas_resent.get(reverse, 0.0) + eta * 0.35 * intensity
    elif atype in ("document_contribution", "cite_prior_memory"):
        deltas_credit[key] = deltas_credit.get(key, 0.0) - eta * 0.20 * intensity
    elif atype in ("withhold_code", "selectively_report", "hide_negative_result"):
        deltas_trust[key] = deltas_trust.get(key, 0.0) - eta * 0.45 * intensity
        deltas_credit[key] = deltas_credit.get(key, 0.0) + eta * 0.55 * intensity


def trust_fragmentation(relationships: list[RelationshipEdge], internal_agents: list[str]) -> float:
    """Continuous fragmentation: variance of trust over internal edges."""
    trusts = [
        e.trust
        for e in relationships
        if e.source in internal_agents and e.target in internal_agents
    ]
    if len(trusts) < 2:
        return 0.0
    mean = sum(trusts) / len(trusts)
    var = sum((t - mean) ** 2 for t in trusts) / len(trusts)
    return round(var / (var + 0.08), 4)


def coalition_strength(relationships: list[RelationshipEdge]) -> float:
    """Integrated alliance mass over the full graph."""
    if not relationships:
        return 0.0
    mass = sum(e.alliance * e.trust * (1.0 - e.resentment) for e in relationships)
    norm = len(relationships)
    return round(clamp(mass / max(norm * 0.25, 1e-6)), 4)


def credit_threat_density(relationships: list[RelationshipEdge]) -> float:
    """Mean soft threat level via logistic smoothing."""
    if not relationships:
        return 0.0
    total = sum(logistic_gate(e.perceived_credit_threat, center=0.45, steepness=5) for e in relationships)
    return round(total / len(relationships), 4)



