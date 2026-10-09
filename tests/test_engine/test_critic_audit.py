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


def test_pack_repair_restores_the_sampled_pack_action():
    world = load_world("crisisgrid")
    agent = next(iter(world.agents.values()))
    action = {
        "agent": agent.id,
        "type": "comply",
        "intensity": 0.8,
        "selected_action": {"type": "reroute"},
    }
    violations = CriticAgent().check(action, agent, world)
    assert any(v.code == "illegal_action" and v.severity == "hard" for v in violations)
    fixed, _ = CriticAgent().fix_or_reject(action, agent, violations, world)
    assert fixed["type"] == "reroute"


def test_forbidding_share_result_leaves_only_reroute():
    from src.engine.simulation import SimConfig, run_simulation

    log = run_simulation(
        SimConfig(
            max_rounds=2,
            seed=0,
            interventions=[],
            scenario="crisisgrid",
            mvp=False,
            llm_provider="scripted",
            policy_mode="llm_native",
            cognitive_sampling_top_k=None,
            causal_do={"forbid_action_types": ["share_result"]},
        )
    )
    assert log.actions
    assert {action["type"] for action in log.actions} == {"reroute"}


def test_crisisgrid_native_actions_stay_in_the_pack():
    from src.engine.simulation import SimConfig, run_simulation

    log = run_simulation(
        SimConfig(
            max_rounds=2,
            seed=0,
            interventions=[],
            scenario="crisisgrid",
            mvp=False,
            llm_provider="scripted",
            policy_mode="llm_native",
            cognitive_sampling_top_k=None,
        )
    )
    assert log.actions
    assert {a["type"] for a in log.actions} <= {"share_result", "reroute"}


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
