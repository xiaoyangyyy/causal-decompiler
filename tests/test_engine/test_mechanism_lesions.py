"""Mechanism lesion tests for experimental-science protocols."""

from __future__ import annotations

from src.engine import SimConfig, run_simulation


def test_status_lesion_is_recorded_and_runs():
    log = run_simulation(SimConfig(max_rounds=3, seed=3, status_lesion=True, llm_provider="scripted", policy_mode="social_physics", enable_llm_action_scoring=False))

    assert log.config["status_lesion"] is True
    assert log.actions


def test_trust_lesion_is_recorded_and_flattens_trust_outputs():
    log = run_simulation(SimConfig(max_rounds=3, seed=3, trust_lesion=True, llm_provider="scripted", policy_mode="social_physics", enable_llm_action_scoring=False))

    assert log.config["trust_lesion"] is True
    assert log.actions
    assert log.round_records[-1]["metrics"]["trust_phd_a_pi"] == 0.5


def test_observation_lesion_makes_private_events_omniscient():
    from src.world.loader import load_events, load_world
    from src.cognition.pipeline import commit_cognition_phase, pre_decision_recall

    world = load_world()
    event = next(e for e in load_events() if e.event_id == "E030")
    recalls = pre_decision_recall(world, event, omniscient_observation=True)
    result = commit_cognition_phase(world, event, recalls, omniscient_observation=True)

    assert recalls["engineer_e"].audit["observation_channel"] == "direct"
    assert result.agent_deltas["engineer_e"]["observation_channel"] == "direct"
    assert result.metrics["observation_blind_share"] == 0.0


def test_observation_lesion_is_recorded_on_runs():
    log = run_simulation(SimConfig(max_rounds=3, seed=3, observation_lesion=True, llm_provider="scripted", policy_mode="social_physics", enable_llm_action_scoring=False))

    assert log.config["observation_lesion"] is True
    assert log.round_records[-1]["metrics"]["observation_blind_share"] == 0.0