"""Tests for policy_mode contrast tracks."""

from __future__ import annotations

from src.engine.event_agent import EventAgent
from src.engine.role_policy import RolePolicyAgent
from src.engine.simulation import SimConfig, run_simulation
from src.world.loader import load_world


def test_llm_native_policy_generates_candidate_space(llm_adapter):
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent(seed=31).generate(30, world)
    policy = RolePolicyAgent(llm_adapter)

    action = policy.decide(agent, event, world, None, {"seed": 31, "policy_mode": "llm_native"})

    assert action is not None
    assert action["llm_action_scoring"]["source"] == "llm_native_generated"
    assert action["selected_action"]["scoring_source"] == "llm_native_generated"
    assert action["selected_action"]["parameter_source"] == "llm_native_policy"
    assert action["action_candidates"]


def test_social_physics_policy_is_field_only():
    log = run_simulation(SimConfig(max_rounds=3, seed=32, interventions=[], policy_mode="social_physics"))
    assert log.actions
    assert all(a.get("llm_action_scoring", {}).get("source") == "field_only" for a in log.actions)


def test_dual_engine_and_native_modes_run():
    field = run_simulation(SimConfig(max_rounds=3, seed=0, interventions=[], policy_mode="social_physics", llm_provider="scripted"))
    dual = run_simulation(SimConfig(max_rounds=3, seed=0, interventions=[], policy_mode="dual_engine", llm_provider="scripted"))
    native = run_simulation(SimConfig(max_rounds=3, seed=0, interventions=[], policy_mode="llm_native", llm_provider="scripted"))
    assert field.actions and dual.actions and native.actions
    assert any(a.get("llm_action_scoring", {}).get("source") == "llm_native_generated" for a in native.actions)
