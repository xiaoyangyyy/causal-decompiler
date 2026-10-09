"""OR covers become the size-1 alternatives of C*_α."""

from __future__ import annotations

from src.engine.causal.action_recovery import recover


def test_two_singletons_are_recorded_as_redundant():
    story = {
        "factors": ["share_result", "reroute"],
        "y": {
            "∅": -7 / 24,
            "share_result": 0.0,
            "reroute": 0.0,
            "reroute,share_result": 0.0,
        },
        "y_full": 0.0,
        "y_empty": -7 / 24,
    }
    found = recover(story)
    assert found["size"] == 1
    assert {tuple(item) for item in found["alternatives"]} == {("share_result",), ("reroute",)}
    assert found["interaction"]["kind"] == "redundant"
    assert found["sensitivity"]["0.8"]["size"] == 1
    assert found["sensitivity"]["0.95"]["size"] == 1


def test_restore_alone_is_the_releaseops_set():
    story = {
        "factors": ["restore_release", "rollback"],
        "y": {
            "∅": -0.05,
            "restore_release": 0.0,
            "rollback": -0.05,
            "restore_release,rollback": 0.0,
        },
        "y_full": 0.0,
        "y_empty": -0.05,
    }
    found = recover(story)
    assert found["set"] == ["restore_release"]
    assert found["alternatives"] == []
    assert found["interaction"]["kind"] == "independent"
