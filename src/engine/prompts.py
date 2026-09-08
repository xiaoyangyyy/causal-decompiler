"""LLM prompt builders for Role Policy and Memory agents."""

from __future__ import annotations

import json
from typing import Any

from src.cognition.memory import RecallResult
from src.engine.diversity import memory_action_hints, recent_action_history
from src.world.models import Agent, EventAtom, RelationshipEdge, WorldState


ROLE_POLICY_SYSTEM = """You are an agent in a long-horizon academic lab power-struggle simulation (LabWars).
Render public/private stance for the sampled action candidate. Output valid JSON only — no markdown.

Rules:
1. Reflect personality, beliefs, emotions, recalled memories, and the current event.
2. public_position = public stance; private_intent = true goal (may diverge).
3. Do not override sampled_action.type; it was selected after continuous field scoring and LLM plausibility fusion.
4. Use action_candidates only to explain motives and tradeoffs, not to choose a different action in this rendering step.
5. Memory behavioral_hooks are soft context, not rules.
6. Match public/private wording to the sampled action and motive mixture.
7. Prefer stance_prior.statement_type. Credit claims (ask_for_authorship, document_contribution, confront, challenge_claim) must not be rendered as team_support; use self_advocacy. team_support is for comply/support_teammate only."""




LLM_NATIVE_POLICY_SYSTEM = """You propose candidate actions for an agent in LabWars.
Output valid JSON only - no markdown.

Rules:
1. Generate 3-8 plausible action candidates from allowed_actions.
2. Use only action types from allowed_actions. Do not invent schema labels.
3. Each candidate must include type, target, intensity, plausibility, public_reason, private_reason.
4. This is an LLM-native contrast condition: you may propose candidates directly, but the simulator will still validate the schema.
5. Avoid threshold logic; express continuous plausibility in [0, 1].

Output schema:
{"native_candidates":[{"type":"action_type","target":"agent_id|project|shared_doc","intensity":0.5,"plausibility":0.5,"public_reason":"...","private_reason":"..."}]}
"""
ACTION_SCORING_SYSTEM = """You score candidate actions for a long-horizon academic lab power-struggle simulation (LabWars).
Output valid JSON only - no markdown.

Rules:
1. Score every candidate action independently for subjective plausibility from this agent's current state, recalled memories, relationship field, and current event.
2. Return one score per input candidate. Do not add candidates, remove candidates, or write public/private stance.
3. plausibility must be a continuous number in [0, 1]. Avoid threshold logic.
4. These scores will be fused with structural field scores; they are not a free-form action override.
5. reason should be short and explain the psychological fit.

Output schema:
{"candidate_scores":[{"type":"action_type","plausibility":0.0,"reason":"short rationale"}]}
"""
MEMORY_INTERPRETATION_SYSTEM = """You are generating a first-person memory interpretation for an academic lab agent.
One short sentence (max 30 words). Subjective, emotionally colored, consistent with valence.
Vary wording across events — avoid repeating the same sentence template.
Output JSON: {"interpretation": "..."}"""


EVENT_ACTION_HINTS: dict[str, list[str]] = {
    "authorship_promise": ["ask_for_authorship", "privately_lobby_pi", "document_contribution", "comply"],
    "authorship_ambiguity": ["privately_lobby_pi", "ask_for_authorship", "confront", "delay_response"],
    "authorship_draft": ["privately_lobby_pi", "ask_for_authorship", "comply", "delay_response", "confront", "document_contribution"],
    "experiment_success": ["share_result", "write_section", "run_experiment", "support_teammate"],
    "experiment_failure": ["analyze_failure", "debug_code", "share_result", "blame"],
    "public_praise": ["support_teammate", "confront", "document_contribution", "undermine_teammate"],
    "credit_dispute": ["challenge_claim", "confront", "document_contribution", "privately_lobby_pi"],
    "team_meeting": ["share_result", "write_section", "comply", "form_alliance"],
    "deadline_shift": ["run_experiment", "write_section", "delay_response", "privately_lobby_pi"],
}


def _event_action_hints(agent: Agent, event: EventAtom) -> list[str]:
    """Event-compatible options only; probabilities come from the continuous action field."""
    return list(EVENT_ACTION_HINTS.get(event.type, []))
def _top_memories(agent: Agent, recall: RecallResult | None, k: int = 5) -> list[dict[str, Any]]:
    if not recall or not recall.attention_weights:
        return []
    ranked = sorted(recall.attention_weights.items(), key=lambda x: x[1], reverse=True)[:k]
    out = []
    for mem_id, weight in ranked:
        mem = next((m for m in agent.memory if m.get("memory_id") == mem_id), None)
        if mem:
            out.append({
                "memory_id": mem_id,
                "attention": round(weight, 4),
                "event_ref": mem.get("event_ref"),
                "content_type": mem.get("content_type"),
                "strength": mem.get("strength"),
                "valence": mem.get("valence"),
                "interpretation": mem.get("interpretation"),
            })
    return out


def _relationship_summary(agent_id: str, edges: list[RelationshipEdge], top_n: int = 4) -> list[dict[str, Any]]:
    relevant = [e for e in edges if e.source == agent_id or e.target == agent_id]
    relevant.sort(key=lambda e: e.perceived_credit_threat + e.resentment, reverse=True)
    out = []
    for e in relevant[:top_n]:
        other = e.target if e.source == agent_id else e.source
        out.append({
            "other": other,
            "trust": round(e.trust, 3),
            "resentment": round(e.resentment, 3),
            "credit_threat": round(e.perceived_credit_threat, 3),
        })
    return out




def build_llm_native_policy_prompt(
    agent: Agent,
    event: EventAtom,
    world: WorldState,
    recall: RecallResult | None,
    allowed_actions: list[str],
    *,
    avoid_actions: list[str] | None = None,
) -> str:
    state = {
        "agent_id": agent.id,
        "role": agent.role.value,
        "round": event.round,
        "personality": agent.personality.model_dump(),
        "beliefs": agent.beliefs.model_dump(),
        "emotion": agent.emotion.model_dump(),
        "resources": agent.resources.model_dump(),
    }
    event_block = {
        "event_id": event.event_id,
        "type": event.type,
        "source": event.source,
        "targets": event.targets,
        "framing": event.framing,
        "memory_salience": event.memory_salience,
        "objective_fact": event.objective_fact.raw_statement,
        "payload": event.payload,
    }
    payload = {
        "state": state,
        "recent_action_history": recent_action_history(agent, n=5),
        "avoid_actions": avoid_actions or [],
        "memory_soft_hooks": memory_action_hints(agent, recall),
        "event_action_hints": _event_action_hints(agent, event),
        "recalled_memories": _top_memories(agent, recall),
        "recall_field": {
            "valence": recall.recall_field_valence if recall else 0.0,
            "strength": recall.recall_field_strength if recall else 0.0,
        },
        "current_event": event_block,
        "relationship_summary": _relationship_summary(agent.id, world.relationships),
        "allowed_actions": allowed_actions,
        "output_schema": {
            "native_candidates": [
                {
                    "type": "one allowed action type",
                    "target": "agent_id|project|shared_doc",
                    "intensity": "0.0-1.0",
                    "plausibility": "0.0-1.0",
                    "public_reason": "brief public rationale",
                    "private_reason": "brief private rationale",
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def build_action_scoring_prompt(
    agent: Agent,
    event: EventAtom,
    world: WorldState,
    recall: RecallResult | None,
    action_candidates: list[dict[str, Any]],
) -> str:
    state = {
        "agent_id": agent.id,
        "role": agent.role.value,
        "round": event.round,
        "personality": agent.personality.model_dump(),
        "beliefs": agent.beliefs.model_dump(),
        "emotion": agent.emotion.model_dump(),
        "resources": agent.resources.model_dump(),
    }
    event_block = {
        "event_id": event.event_id,
        "type": event.type,
        "source": event.source,
        "targets": event.targets,
        "framing": event.framing,
        "memory_salience": event.memory_salience,
        "objective_fact": event.objective_fact.raw_statement,
        "payload": event.payload,
    }
    payload = {
        "state": state,
        "recent_action_history": recent_action_history(agent, n=5),
        "memory_soft_hooks": memory_action_hints(agent, recall),
        "event_action_hints": _event_action_hints(agent, event),
        "recalled_memories": _top_memories(agent, recall),
        "recall_field": {
            "valence": recall.recall_field_valence if recall else 0.0,
            "strength": recall.recall_field_strength if recall else 0.0,
        },
        "current_event": event_block,
        "relationship_summary": _relationship_summary(agent.id, world.relationships),
        "action_candidates": action_candidates,
        "output_schema": {
            "candidate_scores": [
                {"type": "same as candidate.type", "plausibility": "0.0-1.0", "reason": "short rationale"}
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def build_role_policy_prompt(
    agent: Agent,
    event: EventAtom,
    world: WorldState,
    recall: RecallResult | None,
    allowed_actions: list[str],
    *,
    action_candidates: list[dict[str, Any]] | None = None,
    sampled_action: dict[str, Any] | None = None,
    avoid_actions: list[str] | None = None,
    retry_note: str = "",
    validation_error: str = "",
    stance_prior: dict[str, Any] | None = None,
) -> str:
    state = {
        "agent_id": agent.id,
        "role": agent.role.value,
        "round": event.round,
        "personality": agent.personality.model_dump(),
        "beliefs": agent.beliefs.model_dump(),
        "emotion": agent.emotion.model_dump(),
        "resources": agent.resources.model_dump(),
    }
    event_block = {
        "event_id": event.event_id,
        "type": event.type,
        "source": event.source,
        "targets": event.targets,
        "framing": event.framing,
        "memory_salience": event.memory_salience,
        "objective_fact": event.objective_fact.raw_statement,
        "payload": event.payload,
    }
    payload = {
        "state": state,
        "recent_action_history": recent_action_history(agent, n=5),
        "avoid_actions": avoid_actions or [],
        "memory_soft_hooks": memory_action_hints(agent, recall),
        "event_action_hints": _event_action_hints(agent, event),
        "recalled_memories": _top_memories(agent, recall),
        "recall_field": {
            "valence": recall.recall_field_valence if recall else 0.0,
            "strength": recall.recall_field_strength if recall else 0.0,
        },
        "current_event": event_block,
        "relationship_summary": _relationship_summary(agent.id, world.relationships),
        "allowed_actions": allowed_actions,
        "action_candidates": action_candidates or [],
        "sampled_action": sampled_action or {},
        "stance_prior": stance_prior or {},
        "retry_note": retry_note,
        "validation_error": validation_error,
        "output_schema": {
            "primary_action": {
                "type": "use sampled_action.type",
                "target": "use sampled_action.target",
                "intensity": "0.0-1.0",
            },
            "communication_action": {"type": "string", "target": "agent_id", "content_summary": "string"},
            "public_position": {"statement_type": "team_support|self_advocacy|neutral", "authorship_claim": "string"},
            "private_intent": {
                "goal": "string",
                "strategy": "string",
                "trust_pi": "0.0-1.0",
                "private_motives": "briefly reflect sampled_action.motives",
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_memory_interpretation_prompt(
    agent: Agent,
    event: EventAtom,
    valence: float,
    content_type: str,
) -> str:
    recent_interp = [
        m.get("interpretation", "")[:80]
        for m in agent.memory[-3:]
        if m.get("interpretation")
    ]
    return json.dumps({
        "agent_id": agent.id,
        "role": agent.role.value,
        "event_id": event.event_id,
        "event_type": event.type,
        "content_type": content_type,
        "valence": round(valence, 3),
        "framing": event.framing,
        "objective_fact": event.objective_fact.raw_statement,
        "beliefs": agent.beliefs.model_dump(),
        "emotion": {
            "anger": agent.emotion.anger,
            "resentment": agent.emotion.resentment,
            "anxiety": agent.emotion.anxiety,
        },
        "avoid_repeating_phrases": recent_interp,
    }, ensure_ascii=False, indent=2)
