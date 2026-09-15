"""Regression tests for RolePolicyAgent action selection ownership."""

from __future__ import annotations

from pathlib import Path

from src.engine.role_policy import _normalize_action_response
from src.world.actions import ActionType
from src.world.loader import load_events, load_world


def test_role_policy_does_not_import_or_call_legacy_sampler():
    source = Path("src/engine/role_policy.py").read_text(encoding="utf-8")
    assert "sample_action_candidate" not in source
    assert "sample_action_candidate_legacy" not in source
    assert "_score_candidates" in source
    assert "_sample_payload" in source


def test_normalize_accepts_string_communication_action():
    world = load_world()
    agent = world.agents["phd_a"]
    event = load_events()[0]
    act = _normalize_action_response(
        {
            "primary_action": {"type": "share_result", "target": "pi", "intensity": 0.4},
            "communication_action": "share_result",
            "public_position": "neutral",
        },
        agent,
        event,
        world,
        [ActionType.SHARE_RESULT, ActionType.COMPLY],
    )
    assert act["communication_action"]["type"] == "share_result"
    assert isinstance(act["public_position"], dict)
