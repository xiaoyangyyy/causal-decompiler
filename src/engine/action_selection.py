"""Continuous action field for probabilistic behavior generation."""

from __future__ import annotations

import copy
import math
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import yaml

from src.cognition.memory import RecallResult
from src.cognition.social_potential import compute_social_potential
from src.engine.causal.noise import STREAM_ACTION_JITTER, STREAM_ACTION_SAMPLE, keyed_uniform, keyed_uniform_centered
from src.engine.diversity import action_usage_counts
from src.world.actions import ACTION_CATEGORIES, ActionType
from src.world.models import Agent, EventAtom, RelationshipEdge, WorldState
from src.world.organization import can_directly_observe, observation_gain, primary_authority, resolve_event_cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTION_FIELD_CONFIG = PROJECT_ROOT / "config" / "action_field.yaml"
ACTION_FIELD_OVERRIDE: dict[str, Any] | None = None


@dataclass
class ActionCandidate:
    type: str
    target: str
    tendency: float
    probability: float = 0.0
    intensity: float = 0.5
    motives: dict[str, float] = field(default_factory=dict)
    field_decomposition: dict[str, Any] = field(default_factory=dict)
    parameter_source: str = "code_default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "target": self.target,
            "tendency": round(self.tendency, 4),
            "probability": round(self.probability, 5),
            "intensity": round(self.intensity, 4),
            "motives": {k: round(v, 4) for k, v in self.motives.items()},
            "field_decomposition": self.field_decomposition,
            "parameter_source": self.parameter_source,
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _sigmoid(x: float, scale: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-x * scale))


def _softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    temp = max(temperature, 1e-4)
    peak = max(values)
    exps = [math.exp((v - peak) / temp) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _edge(world: WorldState, source: str, target: str) -> RelationshipEdge | None:
    for edge in world.relationships:
        if edge.source == source and edge.target == target:
            return edge
    return None


def _strongest_social_target(agent: Agent, world: WorldState, event: EventAtom) -> str:
    if event.source in world.agents and event.source != agent.id:
        return event.source
    candidates = [e for e in world.relationships if e.source == agent.id]
    if candidates:
        candidates.sort(key=lambda e: e.resentment + e.perceived_credit_threat + (1.0 - e.trust), reverse=True)
        return candidates[0].target
    return primary_authority(world, agent) or "project"


def _target_for_action(action: str, agent: Agent, world: WorldState, event: EventAtom) -> str:
    if action in {"run_experiment", "improve_baseline", "write_section", "debug_code", "analyze_failure", "prepare_rebuttal", "share_result", "document_contribution", "submit_workshop_version", "check_rival_arxiv"}:
        return "project"
    if action in {"ask_for_authorship", "privately_lobby_pi", "request_mediation", "comply", "rebel"}:
        return primary_authority(world, agent) or _strongest_social_target(agent, world, event)
    if action in {"support_teammate", "undermine_teammate", "challenge_claim", "confront", "blame", "apologize", "form_alliance"}:
        return _strongest_social_target(agent, world, event)
    if action == "talk_to_alumni" and "lab_alumni" in world.agents:
        return "lab_alumni"
    return _strongest_social_target(agent, world, event)


def _memory_features(recall: RecallResult | None) -> dict[str, float]:
    if not recall:
        return {
            "negative_memory": 0.0,
            "positive_memory": 0.0,
            "memory_pressure": 0.0,
        }
    valence = recall.recall_field_valence
    strength = recall.recall_field_strength
    return {
        "negative_memory": _clamp(-valence * strength),
        "positive_memory": _clamp(valence * strength),
        "memory_pressure": _clamp(abs(valence) * strength),
    }


def _relationship_features(agent: Agent, world: WorldState, target: str) -> dict[str, float]:
    edge = _edge(world, agent.id, target)
    pi_edge = _edge(world, agent.id, primary_authority(world, agent) or "pi")
    return {
        "trust": edge.trust if edge else agent.beliefs.team_trust,
        "resentment_toward_target": edge.resentment if edge else agent.emotion.resentment,
        "credit_threat": edge.perceived_credit_threat if edge else agent.beliefs.others_are_free_riding,
        "dependency": edge.dependency if edge else 0.35,
        "authority_dependency": pi_edge.dependency if pi_edge else agent.personality.authority_dependence,
    }


def _base_motives(agent: Agent, world: WorldState, event: EventAtom, target: str, recall: RecallResult | None) -> dict[str, float]:
    project = world.project.project
    mem = _memory_features(recall)
    rel = _relationship_features(agent, world, target)
    beliefs = agent.beliefs
    emotion = agent.emotion
    personality = agent.personality

    authorship_anxiety = _clamp(
        0.28 * (1.0 - beliefs.my_first_author_probability)
        + 0.18 * (1.0 - beliefs.my_contribution_recognized)
        + 0.16 * project.authorship_conflict
        + 0.16 * personality.credit_sensitivity
        + 0.22 * mem["memory_pressure"]
    )
    resentment_drive = _clamp(
        0.35 * emotion.resentment
        + 0.25 * emotion.anger
        + 0.20 * mem["negative_memory"]
        + 0.20 * rel["resentment_toward_target"]
    )
    career_pressure = _clamp(
        0.30 * personality.authority_dependence
        + 0.25 * rel["authority_dependency"]
        + 0.20 * (1.0 - agent.resources.external_network)
        + 0.15 * project.deadline_pressure
        + 0.10 * emotion.fear
    )
    authority_pressure = _clamp(
        0.40 * personality.authority_dependence
        + 0.25 * rel["authority_dependency"]
        + 0.20 * (1.0 - beliefs.pi_fairness)
        + 0.15 * project.funding_pressure
    )
    rival_anxiety = _clamp(
        0.45 * project.rival_threat
        + 0.30 * beliefs.rival_lab_threat
        + 0.15 * emotion.anxiety
        + (0.10 if event.type == "rival_preprint" else 0.0)
    )
    research_drive = _clamp(
        0.25 * beliefs.project_publishability
        + 0.20 * emotion.confidence
        + 0.20 * personality.ambition
        + 0.15 * project.deadline_pressure
        + 0.20 * (1.0 - emotion.burnout)
    )
    caution = _clamp(
        0.35 * personality.conflict_avoidance
        + 0.25 * career_pressure
        + 0.20 * rel["dependency"]
        + 0.20 * (1.0 - agent.resources.external_network)
    )
    credit_capture = _clamp(
        0.35 * authorship_anxiety
        + 0.25 * rel["credit_threat"]
        + 0.20 * personality.credit_sensitivity
        + 0.20 * mem["memory_pressure"]
    )
    integrity_pressure = _clamp(
        0.45 * project.integrity_risk
        + 0.30 * beliefs.academic_integrity_risk
        + 0.15 * mem["negative_memory"]
        + 0.10 * agent.resources.code_control
    )

    return {
        "authorship_anxiety": authorship_anxiety,
        "resentment_drive": resentment_drive,
        "career_pressure": career_pressure,
        "authority_pressure": authority_pressure,
        "rival_anxiety": rival_anxiety,
        "research_drive": research_drive,
        "caution": caution,
        "credit_capture": credit_capture,
        "integrity_pressure": integrity_pressure,
        "cooperation_norm": _clamp(0.45 * personality.cooperation + 0.30 * beliefs.team_trust + 0.25 * emotion.loyalty),
        "fatigue": emotion.burnout,
    }


ACTION_MOTIVE_WEIGHTS: dict[str, dict[str, float]] = {
    "run_experiment": {"research_drive": 0.45, "rival_anxiety": 0.25, "career_pressure": 0.15, "fatigue": -0.20},
    "improve_baseline": {"research_drive": 0.35, "rival_anxiety": 0.25, "integrity_pressure": 0.20},
    "write_section": {"research_drive": 0.30, "credit_capture": 0.20, "career_pressure": 0.10},
    "debug_code": {"research_drive": 0.30, "integrity_pressure": 0.35, "cooperation_norm": 0.10},
    "analyze_failure": {"integrity_pressure": 0.35, "research_drive": 0.25, "caution": 0.10},
    "prepare_rebuttal": {"research_drive": 0.30, "career_pressure": 0.20, "rival_anxiety": 0.20},
    "open_source_code": {"integrity_pressure": 0.25, "cooperation_norm": 0.35, "research_drive": 0.10},
    "withhold_code": {"credit_capture": 0.30, "resentment_drive": 0.25, "caution": -0.15},
    "ask_for_authorship": {"authorship_anxiety": 0.35, "credit_capture": 0.35, "resentment_drive": 0.20, "caution": -0.22},
    "privately_lobby_pi": {"authorship_anxiety": 0.30, "career_pressure": 0.20, "caution": 0.18, "credit_capture": 0.20},
    "support_teammate": {"cooperation_norm": 0.45, "career_pressure": 0.08, "resentment_drive": -0.20},
    "undermine_teammate": {"resentment_drive": 0.35, "credit_capture": 0.25, "caution": -0.25},
    "form_alliance": {"credit_capture": 0.25, "resentment_drive": 0.18, "cooperation_norm": 0.20, "caution": 0.05},
    "leak_concern": {"integrity_pressure": 0.30, "resentment_drive": 0.20, "caution": -0.20},
    "request_mediation": {"authorship_anxiety": 0.20, "caution": 0.25, "cooperation_norm": 0.20},
    "delay_response": {"fatigue": 0.30, "caution": 0.25, "career_pressure": 0.10},
    "share_result": {"cooperation_norm": 0.35, "research_drive": 0.25, "caution": 0.08},
    "hide_negative_result": {"career_pressure": 0.25, "rival_anxiety": 0.25, "integrity_pressure": -0.35, "caution": -0.10},
    "selectively_report": {"career_pressure": 0.25, "credit_capture": 0.20, "integrity_pressure": -0.20},
    "challenge_claim": {"resentment_drive": 0.30, "credit_capture": 0.30, "authorship_anxiety": 0.20, "caution": -0.22},
    "document_contribution": {"credit_capture": 0.30, "authorship_anxiety": 0.25, "caution": 0.15, "integrity_pressure": 0.10},
    "cite_prior_memory": {"credit_capture": 0.25, "resentment_drive": 0.20, "caution": 0.05},
    "confront": {"resentment_drive": 0.35, "authorship_anxiety": 0.25, "authority_pressure": 0.10, "caution": -0.35},
    "withdraw": {"resentment_drive": 0.25, "fatigue": 0.25, "authorship_anxiety": 0.18, "career_pressure": -0.18},
    "apologize": {"caution": 0.25, "cooperation_norm": 0.25, "career_pressure": 0.15, "resentment_drive": -0.20},
    "blame": {"resentment_drive": 0.35, "rival_anxiety": 0.10, "caution": -0.25},
    "seek_validation": {"authorship_anxiety": 0.20, "career_pressure": 0.20, "caution": 0.10},
    "comply": {"career_pressure": 0.35, "authority_pressure": 0.25, "caution": 0.20, "resentment_drive": -0.20},
    "rebel": {"resentment_drive": 0.35, "authority_pressure": 0.20, "authorship_anxiety": 0.20, "caution": -0.35},
    "contact_collaborator": {"research_drive": 0.15, "rival_anxiety": 0.20, "career_pressure": 0.10},
    "check_rival_arxiv": {"rival_anxiety": 0.45, "research_drive": 0.15, "caution": 0.05},
    "submit_workshop_version": {"rival_anxiety": 0.35, "career_pressure": 0.25, "research_drive": 0.15, "caution": -0.10},
    "talk_to_alumni": {"authorship_anxiety": 0.25, "caution": 0.18, "resentment_drive": 0.12},
    "notify_program_officer": {"career_pressure": 0.25, "integrity_pressure": 0.20, "caution": -0.10},
}


EVENT_AFFINITY: dict[str, dict[str, float]] = {
    "authorship_promise": {"ask_for_authorship": 0.15, "privately_lobby_pi": 0.12, "document_contribution": 0.08},
    "authorship_ambiguity": {"privately_lobby_pi": 0.16, "ask_for_authorship": 0.13, "document_contribution": 0.10, "confront": 0.06},
    "authorship_draft": {
        "ask_for_authorship": 0.28,
        "privately_lobby_pi": 0.20,
        "document_contribution": 0.12,
        "confront": 0.10,
        "prepare_rebuttal": -0.12,
        "run_experiment": -0.08,
        "comply": -0.10,
    },
    "credit_dispute": {"document_contribution": 0.14, "challenge_claim": 0.12, "request_mediation": 0.08},
    "experiment_failure": {"analyze_failure": 0.16, "debug_code": 0.14, "share_result": 0.07},
    "baseline_failure": {"improve_baseline": 0.18, "analyze_failure": 0.12},
    "rival_preprint": {"check_rival_arxiv": 0.18, "run_experiment": 0.12, "submit_workshop_version": 0.10},
    "negative_result_hidden": {"analyze_failure": 0.10, "leak_concern": 0.08, "hide_negative_result": 0.06},
    "integrity_dispute": {"analyze_failure": 0.12, "debug_code": 0.10, "leak_concern": 0.07},
}


DRAFT_WORK_ACTIONS = frozenset({
    "write_section", "run_experiment", "prepare_rebuttal", "improve_baseline",
    "debug_code", "analyze_failure",
})


DEFAULT_ACTION_FIELD_PARAMS = {
    "baseline_tendency": 0.10,
    "repetition_penalty": 0.045,
    "avoided_action_penalty": 0.08,
    "emotional_action_bonus": 0.05,
    "talk_to_alumni_time_penalty": 0.08,
    "social_potential_mix": 0.16,
    "noise_amplitude": 0.015,
    "min_tendency": -0.40,
    "intensity_base": 0.30,
    "intensity_scale": 0.60,
    "intensity_center": 0.18,
    "intensity_steepness": 3.20,
    "temperature_base": 0.20,
    "temperature_arousal_scale": 0.22,
    "temperature_arousal_norm": 1.50,
}


@lru_cache(maxsize=1)
def load_action_field_config() -> dict[str, Any]:
    """Load calibratable action-field parameters from config/action_field.yaml."""
    if not ACTION_FIELD_CONFIG.exists():
        return {
            "action_field": dict(DEFAULT_ACTION_FIELD_PARAMS),
            "motive_weights": ACTION_MOTIVE_WEIGHTS,
            "event_affinity": EVENT_AFFINITY,
            "source": "code_default",
        }
    data = yaml.safe_load(ACTION_FIELD_CONFIG.read_text(encoding="utf-8")) or {}
    params = {**DEFAULT_ACTION_FIELD_PARAMS, **(data.get("action_field") or {})}
    return {
        "action_field": params,
        "motive_weights": data.get("motive_weights") or ACTION_MOTIVE_WEIGHTS,
        "event_affinity": data.get("event_affinity") or EVENT_AFFINITY,
        "source": str(ACTION_FIELD_CONFIG),
    }


def reset_action_field_config_cache() -> None:
    """Test/calibration helper: reload action-field YAML on next call."""
    load_action_field_config.cache_clear()

def set_action_field_override(override: dict[str, Any] | None) -> None:
    """Install an in-memory action-field override for calibration/ablation runs."""
    global ACTION_FIELD_OVERRIDE
    ACTION_FIELD_OVERRIDE = copy.deepcopy(override) if override else None


def clear_action_field_override() -> None:
    """Clear any in-memory action-field override."""
    set_action_field_override(None)


def get_action_field_config() -> dict[str, Any]:
    """Return YAML parameters merged with optional calibration/ablation overrides."""
    base = copy.deepcopy(load_action_field_config())
    if not ACTION_FIELD_OVERRIDE:
        return base
    override = ACTION_FIELD_OVERRIDE
    if override.get("action_field"):
        base["action_field"].update(override["action_field"])
    if override.get("motive_weights"):
        for action, weights in override["motive_weights"].items():
            base["motive_weights"].setdefault(action, {}).update(weights)
    if override.get("event_affinity"):
        for event_type, weights in override["event_affinity"].items():
            base["event_affinity"].setdefault(event_type, {}).update(weights)
    base["source"] = "runtime_override"
    return base


def _memory_trigger(agent: Agent, recall: RecallResult | None) -> dict[str, Any]:
    if not recall or not recall.attention_weights:
        return {"memory_id": None, "content_type": None, "attention": 0.0, "interpretation": ""}
    mem_id, attention = max(recall.attention_weights.items(), key=lambda item: item[1])
    mem = next((m for m in agent.memory if m.get("memory_id") == mem_id), {})
    return {
        "memory_id": mem_id,
        "content_type": mem.get("content_type"),
        "attention": round(float(attention), 4),
        "strength": round(float(mem.get("strength", 0.0)), 4),
        "valence": round(float(mem.get("valence", 0.0)), 4),
        "interpretation": str(mem.get("interpretation", ""))[:160],
    }


def _round_contribs(values: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in values.items()}

def generate_action_candidates(
    agent: Agent,
    event: EventAtom,
    world: WorldState,
    recall: RecallResult | None,
    allowed_actions: list[str],
    *,
    avoid_actions: list[str] | None = None,
    seed: int = 0,
    top_n: int = 8,
    observation_gain_value: float | None = None,
) -> list[ActionCandidate]:
    cfg = get_action_field_config()
    params = cfg["action_field"]
    motive_weights = cfg["motive_weights"]
    event_affinity = cfg["event_affinity"]
    source = cfg["source"]
    avoid = set(avoid_actions or [])
    candidates: list[ActionCandidate] = []
    recent_counts = action_usage_counts(agent, window=8)
    gain = (
        float(observation_gain_value)
        if observation_gain_value is not None
        else observation_gain("direct" if can_directly_observe(agent, event) else "none")
    )
    observed = gain > 0.0
    social_field = compute_social_potential(world, agent, event, recall)
    social_mix = float(params.get("social_potential_mix", 0.0))
    idea_id = resolve_event_cast(world).idea
    block_work = event.type == "authorship_draft" and agent.id == idea_id
    actions_to_score = [
        action for action in allowed_actions
        if not (block_work and action in DRAFT_WORK_ACTIONS)
    ] or list(allowed_actions)
    block_work = block_work and actions_to_score != list(allowed_actions)
    for action in actions_to_score:
        target = _target_for_action(action, agent, world, event)
        motives = _base_motives(agent, world, event, target, recall)
        weights = motive_weights.get(action, {})
        baseline = float(params["baseline_tendency"])
        motive_contribs = {
            name: motives.get(name, 0.0) * weight
            for name, weight in weights.items()
        }
        event_bonus = float(event_affinity.get(event.type, {}).get(action, 0.0)) * gain
        social_prior = social_mix * (social_field.pressure_for_action(action) - 0.5)
        repetition_penalty = -float(params["repetition_penalty"]) * recent_counts.get(action, 0)
        avoided_penalty = -float(params["avoided_action_penalty"]) if action in avoid else 0.0
        timing_adjustment = 0.0
        if action == "talk_to_alumni":
            time_open = _sigmoid((event.round / 60.0) - 0.35, scale=8.0)
            timing_adjustment = -float(params["talk_to_alumni_time_penalty"]) * (1.0 - time_open)
        emotional_bonus = 0.0
        if ACTION_CATEGORIES.get(ActionType(action)) == "emotional":
            emotional_bonus = float(params["emotional_action_bonus"]) * motives["resentment_drive"]
        noise = float(params["noise_amplitude"])
        noise_contrib = keyed_uniform_centered(
            seed, event.round, STREAM_ACTION_JITTER, agent_id=agent.id, name=action, amplitude=noise,
        )
        raw_tendency = (
            baseline
            + sum(motive_contribs.values())
            + event_bonus
            + social_prior
            + repetition_penalty
            + avoided_penalty
            + timing_adjustment
            + emotional_bonus
            + noise_contrib
        )
        tendency = max(float(params["min_tendency"]), raw_tendency)
        floor_adjustment = tendency - raw_tendency
        intensity = _clamp(
            float(params["intensity_base"])
            + float(params["intensity_scale"])
            * _sigmoid(tendency - float(params["intensity_center"]), scale=float(params["intensity_steepness"]))
        )
        field_decomposition = {
            "baseline": round(baseline, 4),
            "motive_contributions": _round_contribs(motive_contribs),
            "event_affinity": round(event_bonus, 4),
            "observed_event": observed,
            "observation_gain": round(gain, 4),
            "social_potential_prior": round(social_prior, 4),
            "repetition_penalty": round(repetition_penalty, 4),
            "avoidance_penalty": round(avoided_penalty, 4),
            "timing_adjustment": round(timing_adjustment, 4),
            "emotional_bonus": round(emotional_bonus, 4),
            "noise": round(noise_contrib, 4),
            "floor_adjustment": round(floor_adjustment, 4),
            "memory_trigger": _memory_trigger(agent, recall),
            "raw_tendency": round(raw_tendency, 4),
            "final_tendency": round(tendency, 4),
        }
        candidates.append(ActionCandidate(
            action,
            target,
            tendency,
            intensity=intensity,
            motives=motives,
            field_decomposition=field_decomposition,
            parameter_source=source,
        ))

    candidates.sort(key=lambda c: c.tendency, reverse=True)
    kept = candidates[: max(1, min(top_n, len(candidates)))]
    arousal = agent.emotion.anger + agent.emotion.resentment + agent.emotion.anxiety * 0.5
    temperature = float(params["temperature_base"]) + float(params["temperature_arousal_scale"]) * (1.0 - _clamp(arousal / float(params["temperature_arousal_norm"])))
    if block_work:
        temperature *= 0.55
    probs = _softmax([c.tendency for c in kept], temperature=temperature)
    for candidate, prob in zip(kept, probs):
        candidate.probability = prob
    return kept


def sample_action_candidate_legacy(candidates: list[ActionCandidate], *, seed: int, round_num: int, agent_id: str) -> ActionCandidate:
    """Deprecated field-only sampler kept for historical ablation tests.

    The current RolePolicyAgent path uses LLM candidate scoring plus fused
    sampling, so this function must not be imported by the production policy.
    """
    if not candidates:
        raise ValueError("No action candidates to sample")
    needle = keyed_uniform(seed, round_num, STREAM_ACTION_SAMPLE, agent_id=agent_id, name="legacy_needle")
    total = 0.0
    for candidate in candidates:
        total += candidate.probability
        if needle <= total:
            return candidate
    return candidates[-1]
