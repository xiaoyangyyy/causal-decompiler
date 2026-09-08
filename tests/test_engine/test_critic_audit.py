"""Tests for LLM wording/action drift audit."""

from __future__ import annotations

from src.engine.critic import CriticAgent
from src.world.loader import load_world


def test_critic_flags_public_action_drift():
    world = load_world()
    agent = world.agents["phd_a"]
    action = {
        "agent": "phd_a",
        "type": "confront",
        "target": "pi",
        "intensity": 0.7,
        "public_position": {"statement_type": "team_support"},
        "private_intent": {"strategy": "comply"},
        "selected_action": {"type": "confront"},
    }
    violations = CriticAgent().check(action, agent, world)
    codes = {v.code for v in violations}
    assert "llm_public_action_drift" in codes
    assert "llm_private_strategy_drift" in codes


def test_critic_accepts_self_advocacy_on_credit_claims():
    world = load_world()
    agent = world.agents["phd_a"]
    action = {
        "agent": "phd_a",
        "type": "ask_for_authorship",
        "target": "pi",
        "intensity": 0.7,
        "public_position": {"statement_type": "self_advocacy", "authorship_claim": "first_author"},
        "private_intent": {"strategy": "ask_for_authorship"},
        "selected_action": {"type": "ask_for_authorship"},
    }
    codes = {v.code for v in CriticAgent().check(action, agent, world)}
    assert "llm_public_action_drift" not in codes
