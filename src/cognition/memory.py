"""Memory resonance field 鈥?continuous recall without cutoffs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.world.models import Agent, AgentRole, EventAtom
from src.world.organization import can_directly_observe, primary_authority

from .math_utils import (
    clamp,
    recency_kernel,
    sigmoid,
    softmax,
    tanh_bounded,
    truth_status_precision,
)


CONTENT_TYPE_BY_EVENT: dict[str, str] = {
    "authorship_promise": "authorship_signal",
    "authorship_ambiguity": "authorship_signal",
    "authorship_draft": "authorship_signal",
    "idea_claim": "credit_claim",
    "public_praise": "credit_claim",
    "credit_dispute": "credit_claim",
    "private_lobbying": "betrayal_signal",
    "narrative_change": "betrayal_signal",
    "negative_result_hidden": "integrity_signal",
    "experiment_failure": "integrity_signal",
    "integrity_dispute": "integrity_signal",
    "funding_pressure": "authority_signal",
    "deadline_shift": "authority_signal",
    "rival_preprint": "rival_threat",
    "external_history": "historical_pattern",
}


FRAMING_VALENCE: dict[str, float] = {
    "neutral": 0.0,
    "positive": 0.55,
    "negative": -0.55,
    "ambiguous": -0.15,
    "team": 0.20,
    "merit": 0.10,
    "execution": 0.25,
    "idea": 0.15,
}


BEHAVIORAL_HOOKS: dict[str, list[str]] = {
    "authorship_signal": ["ask_for_authorship", "document_contribution", "privately_lobby_pi"],
    "credit_claim": ["document_contribution", "challenge_claim"],
    "betrayal_signal": ["withdraw", "confront", "form_alliance"],
    "integrity_signal": ["share_result", "rebel", "leak_concern"],
    "authority_signal": ["comply", "privately_lobby_pi"],
    "rival_threat": ["check_rival_arxiv", "run_experiment", "form_alliance"],
    "historical_pattern": ["talk_to_alumni", "document_contribution", "cite_prior_memory"],
    "promise_fulfilled": ["support_teammate", "comply"],
    "promise_broken": ["ask_for_authorship", "confront", "document_contribution"],
}


@dataclass
class MemoryRecord:
    memory_id: str
    owner: str
    round: int
    event_ref: str
    content_type: str
    target: str
    valence: float
    strength: float
    strength_0: float
    decay: float
    rehearsal_count: float
    evidence_quality: float
    interpretation: str
    behavioral_hooks: list[str]
    was_recalled: list[int] = field(default_factory=list)
    objective_fact_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "owner": self.owner,
            "round": self.round,
            "event_ref": self.event_ref,
            "content_type": self.content_type,
            "target": self.target,
            "valence": round(self.valence, 4),
            "strength": round(self.strength, 4),
            "strength_0": round(self.strength_0, 4),
            "decay": self.decay,
            "rehearsal_count": round(self.rehearsal_count, 4),
            "evidence_quality": self.evidence_quality,
            "interpretation": self.interpretation,
            "behavioral_hooks": self.behavioral_hooks,
            "was_recalled": self.was_recalled,
            "objective_fact_ref": self.objective_fact_ref,
        }


@dataclass
class RecallResult:
    attention_weights: dict[str, float]
    recalled_ids: list[str]
    recall_field_valence: float
    recall_field_strength: float
    audit: dict[str, Any]


def _memory_counter(agent: Agent) -> int:
    return len(agent.memory)


def _infer_target(event: EventAtom, agent: Agent, world: Any | None = None) -> str:
    if event.type in ("authorship_promise", "authorship_ambiguity", "authorship_draft"):
        if world is not None:
            return primary_authority(world, agent) or event.source
        return "pi"
    if event.type == "rival_preprint":
        return event.source if event.source else "rival_lab_h"
    if event.source != agent.id and event.source in event.targets:
        return event.source
    for t in event.targets:
        if t not in ("project", agent.id):
            return t
    return event.source


def compute_valence(agent: Agent, event: EventAtom, content_type: str) -> float:
    """
    Contrastive predictive coding for valence:
    surprise = observed_signal - agent-specific expectation, passed through tanh.
    Same event 鈫?opposite valence for agents with divergent stakes.
    """
    framing_signal = FRAMING_VALENCE.get(event.framing, 0.0)
    salience = event.memory_salience

    role_expectation = 0.0
    if content_type == "authorship_signal":
        role_expectation = agent.beliefs.my_first_author_probability - 0.5
    elif content_type == "credit_claim":
        role_expectation = agent.personality.credit_sensitivity - 0.5
    elif content_type == "betrayal_signal":
        role_expectation = -(agent.personality.reciprocity - 0.5)
    elif content_type == "integrity_signal":
        role_expectation = -(agent.beliefs.academic_integrity_risk - 0.5)

    if event.type == "authorship_ambiguity":
        if agent.id == "phd_a" or agent.role == AgentRole.IDEA_ORIGINATOR:
            role_expectation += 0.35
        elif agent.id == "phd_b" or agent.role == AgentRole.EXPERIMENTER:
            role_expectation -= 0.55
        elif agent.role == AgentRole.PI:
            role_expectation -= 0.10
    praised = event.payload.get("praised_agent")
    if event.type == "public_praise" and praised:
        if agent.id == praised:
            role_expectation -= 0.30
        elif agent.id == "phd_a" or agent.role == AgentRole.IDEA_ORIGINATOR:
            role_expectation += 0.25

    observed = framing_signal * salience
    if event.type in ("authorship_promise",):
        observed += 0.40 * salience
    if event.type in ("authorship_ambiguity", "authorship_draft"):
        observed -= 0.25 * salience

    prediction_error = observed - role_expectation
    sensitivity = 1.0 + agent.personality.resentment_sensitivity * 0.8
    return tanh_bounded(prediction_error * sensitivity, scale=1.6)


def _fallback_interpretation(agent: Agent, event: EventAtom, valence: float, content_type: str) -> str:
    if valence < -0.25:
        tone = "threatened"
    elif valence > 0.25:
        tone = "reassured"
    else:
        tone = "watchful"
    return f"I felt {tone} about {event.type} as {content_type}."[:240]


def _interpretation_via_llm(
    llm: Any,
    agent: Agent,
    event: EventAtom,
    valence: float,
    content_type: str,
) -> str:
    from src.engine.prompts import MEMORY_INTERPRETATION_SYSTEM, build_memory_interpretation_prompt

    user = build_memory_interpretation_prompt(agent, event, valence, content_type)
    try:
        result = llm.complete_json(MEMORY_INTERPRETATION_SYSTEM, user)
        text = result.get("interpretation", "")
        if text and isinstance(text, str):
            return text.strip()[:240]
    except Exception as exc:
        from src.engine.llm_adapter import QuotaExhaustedError

        if isinstance(exc, QuotaExhaustedError):
            raise
    return _fallback_interpretation(agent, event, valence, content_type)


def initial_strength(event: EventAtom, evidence_quality: float, valence: float) -> float:
    return clamp(
        event.memory_salience * evidence_quality * (1.0 + abs(valence) * 0.3),
        0.01,
        1.0,
    )


def write_memory(
    agent: Agent,
    event: EventAtom,
    current_round: int,
    llm_adapter: Any | None = None,
    *,
    channel: str = "direct",
    world: Any | None = None,
) -> MemoryRecord | None:
    if channel == "direct" and not can_directly_observe(agent, event):
        return None

    content_type = CONTENT_TYPE_BY_EVENT.get(event.type, "credit_claim")
    if event.type == "authorship_promise":
        content_type = "promise_fulfilled"
    if event.type == "authorship_ambiguity":
        # State ambiguity is contested credit, not a broken promise.
        content_type = "authorship_signal"
    if event.type == "authorship_draft":
        if event.payload.get("draft_severity") == "honored" or event.framing == "positive":
            content_type = "promise_fulfilled"
        else:
            content_type = "promise_broken"

    valence = compute_valence(agent, event, content_type)
    evidence_quality = truth_status_precision(event.truth_status) * (0.6 + 0.4 * event.memory_salience)
    if channel == "rumor":
        evidence_quality *= 0.55
        valence *= 0.72
    s0 = initial_strength(event, evidence_quality, valence)

    if llm_adapter is None:
        from src.engine.llm_adapter import get_adapter

        llm_adapter = get_adapter()

    interpretation = _interpretation_via_llm(llm_adapter, agent, event, valence, content_type)
    if channel == "rumor":
        interpretation = ("Heard second-hand: " + interpretation)[:240]

    idx = _memory_counter(agent) + 1
    record = MemoryRecord(
        memory_id=f"M{idx:03d}",
        owner=agent.id,
        round=current_round,
        event_ref=event.event_id,
        content_type=content_type,
        target=_infer_target(event, agent, world),
        valence=valence,
        strength=s0,
        strength_0=s0,
        decay=0.03,
        rehearsal_count=0.0,
        evidence_quality=evidence_quality,
        interpretation=interpretation,
        behavioral_hooks=list(BEHAVIORAL_HOOKS.get(content_type, ["document_contribution"])),
        objective_fact_ref=f"{event.event_id}.objective_fact",
    )
    payload = record.to_dict()
    payload["channel"] = channel
    agent.memory.append(payload)
    return record


def _context_vector(event: EventAtom, current_round: int) -> dict[str, float]:
    ctype = CONTENT_TYPE_BY_EVENT.get(event.type, "credit_claim")
    return {
        "target_match": 1.0 if event.source else 0.5,
        "content_type": hash(ctype) % 100 / 100.0,
        "recency": recency_kernel(0),
        "salience": event.memory_salience,
        "framing": FRAMING_VALENCE.get(event.framing, 0.0) + 0.5,
        "round_norm": current_round / 60.0,
    }


def _memory_vector(mem: dict[str, Any], current_round: int) -> list[float]:
    age = current_round - int(mem["round"])
    return [
        float(mem["strength"]),
        float(mem["valence"]) + 0.5,
        recency_kernel(age),
        hash(mem["content_type"]) % 100 / 100.0,
        hash(mem.get("target", "")) % 100 / 100.0,
    ]


def _alignment(ctx: dict[str, float], mem: dict[str, Any], current_round: int) -> float:
    ctx_vec = [
        ctx["salience"],
        ctx["framing"],
        ctx["recency"],
        ctx["content_type"],
        ctx["round_norm"],
    ]
    return max(0.0, cosine_similarity(ctx_vec, _memory_vector(mem, current_round)))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def recall_memories(
    agent: Agent,
    event: EventAtom,
    current_round: int,
    temperature: float = 0.35,
) -> RecallResult:
    """
    Resonance field recall:
    every memory participates via softmax attention 鈥?no cutoff, no hard top-K.
    Temperature modulated by emotional arousal (high arousal 鈫?sharper focus).
    """
    memories = agent.memory
    if not memories:
        audit = {
            "round": current_round,
            "agent": agent.id,
            "recalled_memories": [],
            "attention_weights": {},
            "recall_field_valence": 0.0,
            "recall_field_strength": 0.0,
        }
        agent.memory_recall_log.append(audit)
        return RecallResult({}, [], 0.0, 0.0, audit)

    ctx = _context_vector(event, current_round)
    arousal = clamp(
        agent.emotion.anger + agent.emotion.resentment + agent.emotion.anxiety * 0.5,
        0.0,
        1.5,
    )
    effective_temp = temperature * (1.0 - 0.35 * sigmoid(arousal - 0.4, scale=4))

    resonances: list[float] = []
    for mem in memories:
        age = current_round - int(mem["round"])
        binding = 1.0 + 0.25 * math.log1p(float(mem.get("rehearsal_count", 0)))
        resonance = (
            float(mem["strength"])
            * recency_kernel(age, half_life=14.0)
            * (0.35 + 0.65 * _alignment(ctx, mem, current_round))
            * binding
        )
        resonances.append(max(resonance, 1e-8))

    weights = softmax(resonances, temperature=effective_temp)
    weight_map = {
        memories[i]["memory_id"]: float(weights[i]) for i in range(len(memories))
    }

    field_valence = sum(
        weights[i] * float(memories[i]["valence"]) for i in range(len(memories))
    )
    field_strength = sum(
        weights[i] * float(memories[i]["strength"]) for i in range(len(memories))
    )

    recalled_ids = [memories[i]["memory_id"] for i in range(len(memories))]

    recall_reason = {}
    for i, mem in enumerate(memories):
        if weights[i] > 1e-4:
            recall_reason[mem["memory_id"]] = (
                f"resonance={resonances[i]:.4f}, weight={weights[i]:.4f}, "
                f"content={mem['content_type']}, target={mem.get('target')}"
            )

    not_active = [m["memory_id"] for m in memories if m["memory_id"] not in recalled_ids]

    audit = {
        "round": current_round,
        "agent": agent.id,
        "recalled_memories": recalled_ids,
        "not_recalled_but_active": not_active,
        "attention_weights": {k: round(v, 5) for k, v in weight_map.items()},
        "recall_field_valence": round(field_valence, 4),
        "recall_field_strength": round(field_strength, 4),
        "recall_reason": recall_reason,
        "temperature": round(effective_temp, 4),
    }
    agent.memory_recall_log.append(audit)

    return RecallResult(
        attention_weights=weight_map,
        recalled_ids=recalled_ids,
        recall_field_valence=field_valence,
        recall_field_strength=field_strength,
        audit=audit,
    )


def reconsolidate_memories(
    agent: Agent,
    event: EventAtom,
    recall: RecallResult | None,
    current_round: int,
) -> dict[str, Any]:
    """Continuously rewrite recalled memories through the current interpretive frame.

    Memory is not treated as a fixed fact table. Attention mass, event salience,
    evidence quality, and emotional arousal softly pull earlier memories toward
    the current interpretation, while preserving an audit trail for explanation.
    """
    if not recall or not agent.memory:
        return {"updated": []}

    current_type = CONTENT_TYPE_BY_EVENT.get(event.type, "credit_claim")
    event_valence = compute_valence(agent, event, current_type)
    arousal = clamp(agent.emotion.anger + agent.emotion.resentment + 0.5 * agent.emotion.anxiety, 0.0, 1.5)
    emotional_gain = sigmoid(arousal - 0.35, scale=3.0)
    updates: list[dict[str, Any]] = []

    for mem in agent.memory:
        mem_id = mem.get("memory_id", "")
        attention = float(recall.attention_weights.get(mem_id, 0.0))
        if attention <= 0.0:
            continue

        same_target = 1.0 if mem.get("target") in [event.source, *event.targets] else 0.35
        same_content = 1.0 if mem.get("content_type") == current_type else 0.45
        evidence = float(mem.get("evidence_quality", 0.5))
        plasticity = clamp(
            attention
            * (0.25 + 0.45 * event.memory_salience)
            * (0.45 + 0.35 * emotional_gain + 0.20 * (1.0 - evidence))
            * (0.50 + 0.30 * same_target + 0.20 * same_content),
            0.0,
            0.35,
        )

        old_valence = float(mem.get("valence", 0.0))
        valence_new = old_valence + plasticity * (event_valence - old_valence)
        strength_old = float(mem.get("strength", 0.0))
        strength_new = strength_old + plasticity * event.memory_salience * (0.35 + abs(event_valence))
        mem["valence"] = round(clamp(valence_new, -1.0, 1.0), 4)
        mem["strength"] = round(clamp(strength_new), 4)
        mem["last_reconsolidated_round"] = current_round
        mem["interpretation_bias"] = round(
            float(mem.get("interpretation_bias", 0.0)) + plasticity * event_valence,
            4,
        )
        history = mem.setdefault("reconsolidation_history", [])
        history.append({
            "round": current_round,
            "event_ref": event.event_id,
            "attention": round(attention, 5),
            "plasticity": round(plasticity, 5),
            "valence_before": round(old_valence, 4),
            "valence_after": mem["valence"],
            "strength_after": mem["strength"],
        })
        updates.append(history[-1] | {"memory_id": mem_id})

    return {
        "event_valence": round(event_valence, 4),
        "updated": updates,
    }

def decay_memories(agent: Agent, current_round: int, same_event: EventAtom | None = None) -> None:
    """Continuous decay with emotional binding and fractional rehearsal reinforcement."""
    arousal = agent.emotion.anger + agent.emotion.resentment
    emotional_binding = sigmoid(arousal, scale=2.5)

    for mem in agent.memory:
        s_old = float(mem["strength"])
        s0 = float(mem.get("strength_0", s_old))
        decay = float(mem.get("decay", 0.03))
        effective_decay = decay * (1.0 - 0.45 * emotional_binding)

        salience_boost = 0.0
        if same_event and mem.get("event_ref") == same_event.event_id:
            salience_boost = 0.1 * same_event.memory_salience

        rehearsal = float(mem.get("rehearsal_count", 0))
        rehearsal_boost = 0.08 * math.log1p(rehearsal)

        s_new = s_old * (1.0 - effective_decay) + salience_boost + rehearsal_boost
        floor = s0 * 0.05 * (1.0 - math.exp(-rehearsal))
        mem["strength"] = round(max(s_new, floor), 4)


def apply_rehearsal(agent: Agent, attention_weights: dict[str, float], current_round: int) -> None:
    """Fractional rehearsal from attention mass 鈥?not binary recall."""
    for mem in agent.memory:
        w = attention_weights.get(mem["memory_id"], 0.0)
        if w <= 0:
            continue
        mem["rehearsal_count"] = round(float(mem.get("rehearsal_count", 0)) + w, 4)
        recalled = mem.setdefault("was_recalled", [])
        recalled.append(current_round)

