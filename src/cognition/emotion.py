"""Antagonistic homeostasis emotion network — continuous coupling, no step rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.world.models import Agent, Emotion, EventAtom, ProjectMetrics

from .dynamics import apply_saturating_emotion_delta
from .math_utils import clamp, competitive_inhibition, impulse_response, sigmoid, softplus
from .memory import RecallResult


# Emotion channels in fixed order for matrix ops
EMOTION_KEYS = [
    "confidence",
    "anxiety",
    "anger",
    "resentment",
    "guilt",
    "hope",
    "burnout",
    "envy",
    "fear",
    "loyalty",
]


@dataclass
class EmotionImpulse:
    confidence: float = 0.0
    anxiety: float = 0.0
    anger: float = 0.0
    resentment: float = 0.0
    guilt: float = 0.0
    hope: float = 0.0
    burnout: float = 0.0
    envy: float = 0.0
    fear: float = 0.0
    loyalty: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in EMOTION_KEYS}

    def add(self, other: EmotionImpulse, scale: float = 1.0) -> None:
        for k in EMOTION_KEYS:
            setattr(self, k, getattr(self, k) + scale * getattr(other, k))


EVENT_EMOTION_SIGNATURES: dict[str, EmotionImpulse] = {
    "public_praise": EmotionImpulse(confidence=0.12, hope=0.05),
    "authorship_ambiguity": EmotionImpulse(anxiety=0.15, anger=0.10, fear=0.06),
    "authorship_promise": EmotionImpulse(hope=0.12, confidence=0.08, loyalty=0.05),
    "rival_preprint": EmotionImpulse(anxiety=0.12, fear=0.08, hope=-0.05),
    "credit_dispute": EmotionImpulse(anger=0.18, resentment=0.12, anxiety=0.08),
    "external_history": EmotionImpulse(fear=0.10, resentment=0.05, anxiety=0.06),
    "experiment_failure": EmotionImpulse(anxiety=0.10, burnout=0.06, hope=-0.08),
    "integrity_dispute": EmotionImpulse(anxiety=0.12, guilt=0.08, anger=0.05),
    "threat_withdraw": EmotionImpulse(anger=0.10, anxiety=0.15, fear=0.08),
    "authorship_draft": EmotionImpulse(anxiety=0.14, anger=0.08, resentment=0.10),
    "negative_result_hidden": EmotionImpulse(guilt=0.10, anxiety=0.08),
}


def _event_impulse(agent: Agent, event: EventAtom, *, gain: float = 1.0) -> EmotionImpulse:
    if gain <= 0.0:
        return EmotionImpulse()
    base = EVENT_EMOTION_SIGNATURES.get(event.type, EmotionImpulse())
    impulse = EmotionImpulse(**base.as_dict())
    salience = event.memory_salience
    sensitivity = 0.6 + 0.4 * agent.personality.resentment_sensitivity

    if event.type == "public_praise":
        praised = event.payload.get("praised_agent")
        if praised == agent.id:
            impulse.add(EmotionImpulse(confidence=0.22, hope=0.12, loyalty=0.06), scale=salience)
        elif praised and praised != agent.id:
            credit_gap = softplus((agent.personality.credit_sensitivity - 0.5) * 4.0) / 4.0
            impulse.add(
                EmotionImpulse(resentment=0.14 * credit_gap, envy=0.11 * credit_gap, anger=0.05 * credit_gap),
                scale=salience,
            )

    involved = agent.id in event.targets or event.source == agent.id
    witness = 1.0 if involved else 0.55
    scale = salience * sensitivity * gain * witness
    return EmotionImpulse(**{k: v * scale for k, v in impulse.as_dict().items()})


def _memory_field_impulse(recall: RecallResult | None, agent: Agent | None = None) -> EmotionImpulse:
    if not recall or recall.recall_field_strength <= 0:
        return EmotionImpulse()
    v = recall.recall_field_valence
    s = recall.recall_field_strength
    negative_mass = softplus(-v * 2.5) * s
    positive_mass = softplus(v * 2.5) * s
    resentment_amp = 1.0
    if agent is not None:
        resentment_amp += softplus(agent.emotion.resentment - 0.35) * agent.personality.resentment_sensitivity
    return EmotionImpulse(
        anger=0.05 * negative_mass * resentment_amp,
        resentment=0.04 * negative_mass * resentment_amp,
        hope=0.03 * positive_mass,
        confidence=0.02 * positive_mass,
        fear=0.02 * negative_mass,
    )


def _project_pressure_impulse(project: ProjectMetrics) -> EmotionImpulse:
    """Continuous project→emotion coupling via multiplicative gates (no if pressure > X)."""
    return EmotionImpulse(
        anxiety=impulse_response(project.deadline_pressure, 0.11)
        * (1.0 - 0.4 * project.team_morale),
        burnout=impulse_response(project.deadline_pressure, 0.06)
        + impulse_response(1.0 - project.team_morale, 0.10),
        resentment=impulse_response(project.authorship_conflict, 0.07),
        fear=impulse_response(project.rival_threat, 0.08),
        guilt=impulse_response(project.integrity_risk, 0.06),
        hope=-impulse_response(project.integrity_risk, 0.05),
    )


def _apply_emotion_coupling(emotion: dict[str, float]) -> dict[str, float]:
    """Positive feedback between anger and resentment via smooth coupling."""
    e = dict(emotion)
    e["anger"] = clamp(e["anger"] + 0.10 * softplus(e["resentment"] - 0.38))
    e["resentment"] = clamp(e["resentment"] + 0.08 * softplus(e["anger"] - 0.32))
    e["anxiety"] = clamp(e["anxiety"] + 0.06 * softplus(e["resentment"] + e["anger"] - 0.75))
    return e


def _apply_antagonistic_homeostasis(
    emotion: dict[str, float],
    skip_confidence_anxiety: bool = False,
    skip_loyalty_resentment: bool = False,
) -> dict[str, float]:
    """Cross-inhibition pairs relax extremes without cutoffs."""
    e = dict(emotion)
    if not skip_confidence_anxiety:
        e["confidence"], e["anxiety"] = competitive_inhibition(e["confidence"], e["anxiety"], coupling=0.85)
    e["hope"], e["burnout"] = competitive_inhibition(e["hope"], e["burnout"], coupling=0.75)
    if not skip_loyalty_resentment:
        e["loyalty"], e["resentment"] = competitive_inhibition(e["loyalty"], e["resentment"], coupling=0.55)

    arousal = e["anger"] + e["resentment"] + e["anxiety"] * 0.5
    damp = 1.0 - 0.25 * sigmoid(arousal - 0.35, scale=3)
    for k in ("anger", "resentment", "envy"):
        e[k] = clamp(e[k] * damp)
    return e


def _apply_restoration(emotion: dict[str, float], dt: float = 0.12) -> dict[str, float]:
    """Leaky return toward personality baseline."""
    baseline = {
        "confidence": 0.5,
        "anxiety": 0.35,
        "anger": 0.08,
        "resentment": 0.12,
        "guilt": 0.02,
        "hope": 0.55,
        "burnout": 0.25,
        "envy": 0.05,
        "fear": 0.08,
        "loyalty": 0.5,
    }
    return {k: clamp(e + (baseline[k] - e) * dt) for k, e in emotion.items()}


def update_emotion(
    agent: Agent,
    event: EventAtom,
    project: ProjectMetrics,
    recall: RecallResult | None = None,
    *,
    channel: str = "direct",
) -> dict[str, float]:
    from src.world.organization import observation_gain

    e = agent.emotion.model_dump()
    total = EmotionImpulse()
    leak = 1.0
    if channel == "rumor" and recall is not None:
        leak = float((recall.audit or {}).get("rumor_leak") or 0.0)
    gain = observation_gain(channel, leak=leak)

    total.add(_event_impulse(agent, event, gain=gain))
    total.add(_memory_field_impulse(recall, agent), scale=gain if channel != "none" else 0.35)
    total.add(_project_pressure_impulse(project), scale=0.85)

    for k in EMOTION_KEYS:
        e[k] = apply_saturating_emotion_delta(e[k], total.as_dict()[k])

    e = _apply_emotion_coupling(e)

    praised_self = (
        event.type == "public_praise" and event.payload.get("praised_agent") == agent.id
    )
    praised_other = (
        event.type == "public_praise"
        and event.payload.get("praised_agent") not in (None, agent.id)
        and agent.id in event.targets
    )
    e = _apply_antagonistic_homeostasis(
        e,
        skip_confidence_anxiety=praised_self,
        skip_loyalty_resentment=praised_other,
    )
    restore_dt = 0.08 if praised_self else (0.06 if praised_other else 0.12)
    e = _apply_restoration(e, dt=restore_dt)
    agent.emotion = Emotion(**{k: round(e[k], 4) for k in EMOTION_KEYS})
    return agent.emotion.model_dump()


def emotion_vector(agent: Agent) -> list[float]:
    return [getattr(agent.emotion, k) for k in EMOTION_KEYS]
