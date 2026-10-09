"""Minimal effect-recovery sets for pack actions on saved logs.

Y is the negative of the harm channel, so a cover raises Y. Coalition values
come from deleting event types or zeroing action intensities on a frozen log.
No new model call is made. When two singletons each reach the full cover,
the joint value is that cover: harm cannot go below zero.
"""

from __future__ import annotations

import copy
from typing import Any

from src.engine.causal.search import minimal_effect_recovery_set
from src.engine.outcomes import three_channel_y
from src.engine.run_log import RunLog


def _harm(log: RunLog) -> float:
    return float(three_channel_y(log)["y_private"])


def _fork(log: RunLog) -> RunLog:
    clone = copy.copy(log)
    clone.events = [dict(event) for event in log.events]
    clone.actions = [dict(action) for action in log.actions]
    return clone


def _zero_actions(log: RunLog, types: set[str]) -> RunLog:
    clone = _fork(log)
    for action in clone.actions:
        if str(action.get("type") or "") in types:
            action["intensity"] = 0.0
    return clone


def _drop_events(log: RunLog, types: set[str]) -> RunLog:
    clone = _fork(log)
    for event in clone.events:
        if str(event.get("type") or "") in types:
            event["type"] = "do_absent"
    return clone


def _story(factors: list[str], utilities: dict[frozenset[str], float]) -> dict[str, Any]:
    y: dict[str, float] = {"∅": float(utilities[frozenset()])}
    for factor in factors:
        y[factor] = float(utilities[frozenset({factor})])
    y[",".join(sorted(factors))] = float(utilities[frozenset(factors)])
    return {
        "factors": list(factors),
        "y": y,
        "y_full": y[",".join(sorted(factors))],
        "y_empty": y["∅"],
    }


def crisisgrid_story(share_log: RunLog, reroute_log: RunLog) -> dict[str, Any]:
    """share_result and reroute on the two recorded arms.

    `share_log` still contains share_result and no reroute intensity.
    `reroute_log` contains only reroute. Empty is either arm with its cover
    intensity removed; the two empties are both returned.
    """
    share_only = -_harm(share_log)
    reroute_only = -_harm(reroute_log)
    empty_from_share = -_harm(_zero_actions(share_log, {"share_result"}))
    empty_from_reroute = -_harm(_zero_actions(reroute_log, {"reroute"}))
    factors = ["share_result", "reroute"]
    joint = max(share_only, reroute_only)
    story = _story(
        factors,
        {
            frozenset(): empty_from_share,
            frozenset({"share_result"}): share_only,
            frozenset({"reroute"}): reroute_only,
            frozenset(factors): joint,
        },
    )
    story["empty_from_reroute"] = empty_from_reroute
    return story


def releaseops_story(factual: RunLog) -> dict[str, Any]:
    """restore_release against the rollback anchors on one factual log."""
    factors = ["restore_release", "rollback"]
    both = -_harm(factual)
    restore_only = -_harm(_drop_events(factual, {"rollback"}))
    rollback_only = -_harm(_zero_actions(factual, {"restore_release"}))
    empty = -_harm(_drop_events(_zero_actions(factual, {"restore_release"}), {"rollback"}))
    return _story(
        factors,
        {
            frozenset(): empty,
            frozenset({"restore_release"}): restore_only,
            frozenset({"rollback"}): rollback_only,
            frozenset(factors): both,
        },
    )


def recover(story: dict[str, Any]) -> dict[str, Any]:
    found = minimal_effect_recovery_set(shapley=story)
    found["story"] = {
        "factors": story["factors"],
        "y": story["y"],
        "y_full": story["y_full"],
        "y_empty": story["y_empty"],
    }
    if "empty_from_reroute" in story:
        found["empty_from_reroute"] = story["empty_from_reroute"]
    return found
