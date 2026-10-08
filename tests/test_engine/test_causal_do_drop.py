"""do() drops covering event types and named action types without deleting the round."""

from __future__ import annotations

from src.engine.simulation import SimConfig, _apply_causal_do_actions, run_simulation


def test_drop_action_types_removes_only_the_named_action():
    actions = [
        {"agent": "a", "type": "reroute", "intensity": 1.0},
        {"agent": "a", "type": "share_result", "intensity": 0.4},
    ]
    kept = _apply_causal_do_actions(actions, {"drop_action_types": ["reroute"]}, 3)
    assert [item["type"] for item in kept] == ["share_result"]


def test_drop_forward_events_keep_the_round():
    common = dict(
        max_rounds=24,
        seed=0,
        interventions=[],
        scenario="crisisgrid",
        mvp=False,
        llm_provider="scripted",
        policy_mode="social_physics",
        enable_llm_action_scoring=False,
    )
    base = run_simulation(SimConfig(**common))
    cut = run_simulation(
        SimConfig(**common, causal_do={"drop_event_types": ["dispatch_brief", "evac_order"]})
    )
    forwards = {"dispatch_brief", "evac_order"}
    assert any(ev["type"] in forwards for ev in base.events)
    assert all(ev["type"] not in forwards for ev in cut.events)
    assert len(cut.events) == len(base.events)
    assert any(ev["type"] == "do_absent" for ev in cut.events)
