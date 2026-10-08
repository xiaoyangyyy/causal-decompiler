"""Precision-weighted predictive coding for beliefs — nonlinear observation shocks."""

from __future__ import annotations

from dataclasses import dataclass

from src.world.models import Agent, AgentRole, Beliefs, EventAtom, ProjectMetrics

from .dynamics import authorship_memory_cluster, draft_rank_shock, nonlinear_belief_target, nonlinear_recall_shift
from .math_utils import clamp, impulse_response, logistic_gate, precision_weighted_update, truth_status_precision
from .memory import RecallResult


BELIEF_KEYS = [
    "pi_fairness",
    "project_publishability",
    "rival_lab_threat",
    "my_first_author_probability",
    "team_trust",
    "deadline_feasibility",
    "my_contribution_recognized",
    "others_are_free_riding",
    "academic_integrity_risk",
]


@dataclass
class BeliefObservation:
    pi_fairness: float | None = None
    project_publishability: float | None = None
    rival_lab_threat: float | None = None
    my_first_author_probability: float | None = None
    team_trust: float | None = None
    deadline_feasibility: float | None = None
    my_contribution_recognized: float | None = None
    others_are_free_riding: float | None = None
    academic_integrity_risk: float | None = None


def _prior_precision(agent: Agent, belief_key: str) -> float:
    base = 2.5
    if belief_key == "pi_fairness":
        base += agent.personality.authority_dependence * 1.5
    if belief_key == "my_first_author_probability":
        base += agent.personality.credit_sensitivity * 1.2
    if belief_key == "my_contribution_recognized":
        base += agent.personality.credit_sensitivity
    return base


def _cluster_amplification(agent: Agent, event: EventAtom) -> float:
    if agent.id != "phd_a" and agent.role != AgentRole.IDEA_ORIGINATOR:
        return 0.0
    cluster = authorship_memory_cluster(agent, round_min=1, round_max=event.round, current_round=event.round)
    return cluster * 0.85


def _observation_from_event(agent: Agent, event: EventAtom) -> BeliefObservation:
    obs = BeliefObservation()
    sal = event.memory_salience
    cluster_amp = _cluster_amplification(agent, event)

    if event.type == "authorship_ambiguity":
        obs.pi_fairness = nonlinear_belief_target(
            agent.beliefs.pi_fairness, -0.28, sal, cluster_amp=cluster_amp * 0.9,
        )
        obs.my_first_author_probability = nonlinear_belief_target(
            agent.beliefs.my_first_author_probability, -0.20, sal, cluster_amp=cluster_amp * 0.5,
        )
    elif event.type == "authorship_promise":
        clarity = event.payload.get("promise_clarity", "implied")
        if clarity == "explicit":
            obs.pi_fairness = nonlinear_belief_target(
                agent.beliefs.pi_fairness, 0.58, sal, positive_boost=1.35,
            )
            obs.my_first_author_probability = nonlinear_belief_target(
                agent.beliefs.my_first_author_probability, 0.42, sal, positive_boost=1.2,
            )
        else:
            obs.pi_fairness = nonlinear_belief_target(agent.beliefs.pi_fairness, 0.18, sal)
            obs.my_first_author_probability = nonlinear_belief_target(
                agent.beliefs.my_first_author_probability, 0.14, sal,
            )
    elif event.type == "authorship_draft":
        cluster = authorship_memory_cluster(
            agent, round_min=1, round_max=event.round, current_round=event.round,
        )
        fairness_shock, author_shock = draft_rank_shock(agent, event, cluster)
        negative_gate = logistic_gate(-fairness_shock, center=0.0, steepness=12.0)
        positive_gate = 1.0 - negative_gate
        amp = cluster_amp * negative_gate
        boost = 1.0 + 0.4 * positive_gate
        obs.pi_fairness = nonlinear_belief_target(
            agent.beliefs.pi_fairness, fairness_shock, sal, cluster_amp=amp, positive_boost=boost,
        )
        obs.my_first_author_probability = nonlinear_belief_target(
            agent.beliefs.my_first_author_probability,
            author_shock,
            sal,
            cluster_amp=amp * 0.5,
            positive_boost=boost,
        )
    elif event.type == "public_praise":
        praised = event.payload.get("praised_agent")
        if praised == agent.id:
            obs.my_contribution_recognized = nonlinear_belief_target(
                agent.beliefs.my_contribution_recognized, 0.22, sal,
            )
        elif praised and praised != agent.id:
            obs.my_contribution_recognized = nonlinear_belief_target(
                agent.beliefs.my_contribution_recognized, -0.14, sal,
            )
            obs.others_are_free_riding = nonlinear_belief_target(
                agent.beliefs.others_are_free_riding, 0.12, sal,
            )
    elif event.type == "rival_preprint":
        obs.rival_lab_threat = nonlinear_belief_target(agent.beliefs.rival_lab_threat, 0.32, sal)
        obs.project_publishability = nonlinear_belief_target(agent.beliefs.project_publishability, -0.12, sal)
        obs.deadline_feasibility = nonlinear_belief_target(agent.beliefs.deadline_feasibility, -0.10, sal)
    elif event.type == "external_history":
        obs.pi_fairness = nonlinear_belief_target(
            agent.beliefs.pi_fairness, -0.30, sal, cluster_amp=cluster_amp * 0.5,
        )
        obs.my_first_author_probability = nonlinear_belief_target(
            agent.beliefs.my_first_author_probability, -0.14, sal,
        )
    elif event.type == "credit_dispute":
        obs.my_contribution_recognized = nonlinear_belief_target(
            agent.beliefs.my_contribution_recognized, -0.12, sal,
        )
        obs.team_trust = nonlinear_belief_target(agent.beliefs.team_trust, -0.14, sal)
    elif event.type == "integrity_dispute":
        obs.academic_integrity_risk = nonlinear_belief_target(
            agent.beliefs.academic_integrity_risk, 0.20, sal,
        )
    elif event.type == "experiment_success":
        obs.project_publishability = nonlinear_belief_target(agent.beliefs.project_publishability, 0.10, sal)
        obs.deadline_feasibility = nonlinear_belief_target(agent.beliefs.deadline_feasibility, 0.08, sal)
    elif event.type == "experiment_failure":
        obs.project_publishability = nonlinear_belief_target(agent.beliefs.project_publishability, -0.10, sal)
        obs.deadline_feasibility = nonlinear_belief_target(agent.beliefs.deadline_feasibility, -0.08, sal)
    elif event.type == "funding_pressure":
        obs.deadline_feasibility = nonlinear_belief_target(agent.beliefs.deadline_feasibility, -0.12, sal)

    return obs


def _recall_belief_observation(agent: Agent, recall: RecallResult | None) -> BeliefObservation:
    if not recall:
        return BeliefObservation()
    v = recall.recall_field_valence
    s = recall.recall_field_strength
    obs = BeliefObservation()
    obs.pi_fairness = nonlinear_recall_shift(agent.beliefs.pi_fairness, v, s)
    obs.my_first_author_probability = nonlinear_recall_shift(
        agent.beliefs.my_first_author_probability, v * 0.85, s,
    )
    obs.team_trust = nonlinear_recall_shift(agent.beliefs.team_trust, v * 0.6, s)
    return obs


def _project_belief_observation(project: ProjectMetrics) -> BeliefObservation:
    return BeliefObservation(
        project_publishability=project.experimental_strength * 0.4 + project.writing_quality * 0.3 + 0.3 * (1 - project.novelty_risk),
        rival_lab_threat=project.rival_threat,
        deadline_feasibility=1.0 - project.deadline_pressure * 0.6 - project.novelty_risk * 0.2,
        academic_integrity_risk=project.integrity_risk,
        team_trust=project.team_morale,
    )


def update_beliefs(
    agent: Agent,
    event: EventAtom,
    project: ProjectMetrics,
    recall: RecallResult | None = None,
    *,
    channel: str = "direct",
    lab_authorship: bool = True,
) -> dict[str, float]:
    from src.world.organization import observation_gain

    leak = 1.0
    if channel == "rumor" and recall is not None:
        leak = float((recall.audit or {}).get("rumor_leak") or 0.0)
    gain = observation_gain(channel, leak=leak)
    if gain <= 0.0:
        return agent.beliefs.model_dump()

    obs_precision_base = truth_status_precision(event.truth_status) * (0.5 + 0.5 * event.memory_salience)
    obs_precision_base *= 0.7 + 0.3 * (1.0 - agent.personality.conflict_avoidance)
    obs_precision_base *= max(gain, 0.15)

    beliefs = agent.beliefs.model_dump()
    observations = [
        _observation_from_event(agent, event),
        _recall_belief_observation(agent, recall),
        _project_belief_observation(project),
    ]
    obs_weights = [1.0, 0.55, 0.35]

    for key in BELIEF_KEYS:
        prior = beliefs[key]
        prior_prec = _prior_precision(agent, key)
        fused_obs = prior
        total_prec = 0.0
        for obs, w in zip(observations, obs_weights):
            val = getattr(obs, key, None)
            if val is None:
                continue
            prec = obs_precision_base * w
            fused_obs = precision_weighted_update(fused_obs, val, prior_prec + total_prec, prec)
            total_prec += prec
        beliefs[key] = round(clamp(fused_obs), 4)

    if lab_authorship:
        cluster = authorship_memory_cluster(agent, round_min=1, round_max=event.round, current_round=event.round)
        cluster_gate = logistic_gate(cluster, center=0.45, steepness=5.0)
        calm_gate = 1.0 - logistic_gate(agent.emotion.resentment, center=0.58, steepness=5.0)
        unfair_gate = logistic_gate(0.52 - beliefs["pi_fairness"], center=0.0, steepness=5.0)
        anchor = 0.30 + agent.personality.reciprocity * 0.14 + 0.08 * logistic_gate(cluster, center=1.0, steepness=4.0)
        pull = 0.012 * cluster_gate * calm_gate * unfair_gate
        beliefs["pi_fairness"] = round(
            clamp(beliefs["pi_fairness"] + (anchor - beliefs["pi_fairness"]) * pull),
            4,
        )

    agent.beliefs = Beliefs(**beliefs)
    return beliefs


def apply_action_belief_feedback(agent: Agent, action_type: str, intensity: float) -> None:
    """Nonlinear feedback: escalated actions erode pi_fairness; compliance softens resentment."""
    from .dynamics import COMPLIANCE_ACTIONS, ESCALATED_ACTIONS

    inten = max(0.0, min(1.0, intensity))
    if action_type in ESCALATED_ACTIONS:
        shock = impulse_response(inten, sensitivity=0.24, saturation=2.8)
        agent.beliefs.pi_fairness = clamp(agent.beliefs.pi_fairness - shock * 0.10)
        agent.beliefs.team_trust = clamp(agent.beliefs.team_trust - shock * 0.06)
    elif action_type in COMPLIANCE_ACTIONS:
        relief = impulse_response(inten, sensitivity=0.20, saturation=3.5)
        agent.beliefs.pi_fairness = clamp(agent.beliefs.pi_fairness + relief * 0.06)


def behavior_tendency_multipliers(agent: Agent) -> dict[str, float]:
    unfairness = 1.0 - agent.beliefs.pi_fairness
    credit_pressure = agent.personality.credit_sensitivity * unfairness
    sensitivity = 0.8 + agent.personality.resentment_sensitivity
    cluster = authorship_memory_cluster(agent, round_min=1, round_max=999)
    escalation_bias = 1.0 + impulse_response(cluster, sensitivity=0.35, saturation=2.5) * unfairness

    return {
        "ask_for_authorship": 1.0 + credit_pressure * sensitivity,
        "document_contribution": 1.0 + credit_pressure * 0.75,
        "comply": clamp(1.0 - unfairness * sensitivity * 0.9, 0.05, 2.0),
        "confront": escalation_bias + agent.emotion.resentment * 0.8,
        "withdraw": 1.0 + agent.emotion.anxiety * 0.5 + unfairness * 0.35,
    }
