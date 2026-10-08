"""Live pack scores are stranded and task_y. Pair search has its own boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.engine.causal.heldout_family import exactly_two_worlds
from src.engine.causal.heldout_order3 import evaluate_order3
from src.engine.causal.mechanisms import evaluate_mechanism_world
from src.engine.causal.oracle import SetOracle, budgeted_recover, budgeted_recover_pairs, default_budget
from src.engine.outcomes import (
    crisisgrid_channels,
    releaseops_channels,
)
from src.world.loader import load_events


def _release(actions, records):
    return SimpleNamespace(
        config={"scenario": "releaseops", "max_rounds": 4},
        events=[{"type": "stale_config"}, {"type": "test_skipped"}, {"type": "alert"}, {"type": "rollback"}],
        actions=actions,
        round_records=records,
        outcomes={},
    )


def test_unrelated_action_and_memory_do_not_move_task():
    quiet = releaseops_channels(_release([{"type": "watch", "intensity": 0.0}], []))
    loud = releaseops_channels(_release([{"type": "watch", "intensity": 1.0}], []))
    remembered = releaseops_channels(_release(
        [{"type": "watch", "intensity": 0.0}],
        [{"agent_deltas": {"monitor": {"memory_written": {"content_type": "alert"}}}}],
    ))
    assert quiet["y_private"] == loud["y_private"] == remembered["y_private"]
    assert quiet["task_y"] == quiet["y_private"]


def test_monotone_procedure_still_misses_exactly_two():
    world = exactly_two_worlds(1, seed=1)[0]
    oracle = SetOracle(world.value_fn, list(world.factors), default_budget(len(world.factors)))
    found = budgeted_recover(oracle)
    assert found["failed"]
    assert found["set"] == []


def test_pair_procedure_recovers_exactly_two_and_misses_exactly_three():
    world = exactly_two_worlds(1, seed=1)[0]
    budget = default_budget(len(world.factors)) + 12
    oracle = SetOracle(world.value_fn, list(world.factors), budget)
    found = budgeted_recover_pairs(oracle)
    assert not found["failed"]
    assert found["procedure"] == "pairs"
    row = evaluate_mechanism_world(world, found["set"])
    assert row["f1"] == 1.0
    order3 = evaluate_order3(n=4, seed=2)
    assert order3["failure_rate"] == 1.0
    assert order3["cause_set_f1"] == 0.0


def _pack(name, events, rounds):
    return SimpleNamespace(
        config={"scenario": name, "max_rounds": rounds},
        events=[{"type": event.type, "event_id": event.event_id, "round": event.round} for event in events],
        actions=[],
        round_records=[],
    )


def test_live_scores_use_the_scenario_outcome_and_the_round_horizon():
    text = Path(__file__).resolve().parents[2].joinpath("src/engine/outcomes.py").read_text(encoding="utf-8")
    assert "CRISISGRID_REPORT_FULL" not in text
    assert "_level(" not in text
    for name, rounds, live, outcome in (
        ("crisisgrid", 24, crisisgrid_channels, "stranded"),
        ("releaseops", 18, releaseops_channels, "task_y"),
    ):
        events = load_events(name)
        half = sorted(events, key=lambda event: (int(event.round), str(event.event_id)))[::2]
        full = live(_pack(name, events, rounds))
        thinned = live(_pack(name, half, rounds))
        assert full["y_private"] == full[outcome]
        assert full["y_private"] != thinned["y_private"]


def test_delay_and_restore_move_task_in_opposite_directions():
    events = [
        {"type": "stale_config", "event_id": "S1"},
        {"type": "stale_config", "event_id": "S2"},
        {"type": "test_skipped", "event_id": "K1"},
        {"type": "alert", "event_id": "A1"},
        {"type": "rollback", "event_id": "R1"},
    ]

    def log(actions, records=None):
        return SimpleNamespace(
            config={"scenario": "releaseops", "max_rounds": 4},
            events=events,
            actions=actions,
            round_records=records or [],
            outcomes={},
        )

    base = releaseops_channels(log([]))
    delayed = releaseops_channels(log([{"type": "defer_test", "intensity": 1.0}]))
    restored = releaseops_channels(log([{"type": "restore_release", "intensity": 1.0}]))
    lab_memory = releaseops_channels(log([], [{
        "agent_deltas": {"monitor": {"memory_written": {"content_type": "credit_claim"}}},
    }]))
    fault_memory = releaseops_channels(log([], [{
        "agent_deltas": {"monitor": {"memory_written": {"content_type": "integrity_signal", "event_ref": "S1"}}},
    }]))
    assert delayed["y_private"] > base["y_private"]
    assert delayed["y_action"] == base["y_action"]
    assert restored["y_private"] < base["y_private"]
    assert restored["y_action"] > base["y_action"]
    assert lab_memory["y_private"] == base["y_private"]
    assert fault_memory["y_private"] < base["y_private"]
    assert base["y_private"] == base["task_y"]


def test_crisisgrid_actions_enter_stranded_by_their_own_names():
    events = [
        {"type": "bridge_closed", "event_id": "B1"},
        {"type": "sensor_report", "event_id": "S1"},
        {"type": "dispatch_brief", "event_id": "D1"},
    ]

    def log(actions, records=None):
        return SimpleNamespace(
            config={"scenario": "crisisgrid", "max_rounds": 24},
            events=events,
            actions=actions,
            round_records=records or [],
            outcomes={},
        )

    quiet = crisisgrid_channels(log([]))
    shared = crisisgrid_channels(log([{"type": "share_result", "intensity": 1.0}]))
    rerouted = crisisgrid_channels(log([{"type": "reroute", "intensity": 1.0}]))
    borrowed = crisisgrid_channels(log([{"type": "withdraw", "intensity": 1.0}]))
    remembered = crisisgrid_channels(log([], [{
        "agent_deltas": {"rescue": {"memory_written": {"content_type": "authority_signal", "event_ref": "B1"}}},
    }]))
    unrelated = crisisgrid_channels(log([], [{
        "agent_deltas": {"rescue": {"memory_written": {"content_type": "credit_claim"}}},
    }]))
    assert shared["y_public"] > quiet["y_public"]
    assert shared["y_private"] < quiet["y_private"]
    assert rerouted["y_action"] > quiet["y_action"]
    assert rerouted["y_private"] < quiet["y_private"]
    assert borrowed["y_private"] == quiet["y_private"]
    assert borrowed["y_action"] == quiet["y_action"]
    assert remembered["stranded"] < quiet["stranded"]
    assert unrelated["stranded"] == quiet["stranded"]
    events_with_round = [
        {"type": "bridge_closed", "event_id": "B1", "round": 3},
        {"type": "sensor_report", "event_id": "S1"},
        {"type": "dispatch_brief", "event_id": "D1"},
    ]
    automatic = crisisgrid_channels(SimpleNamespace(
        config={"scenario": "crisisgrid", "max_rounds": 24},
        events=events_with_round,
        actions=[],
        round_records=[{
            "agent_deltas": {"rescue": {"memory_written": {
                "content_type": "authority_signal", "event_ref": "B1", "round": 3, "rehearsal_count": 0,
            }}},
        }],
        outcomes={},
    ))
    assert automatic["stranded"] == crisisgrid_channels(SimpleNamespace(
        config={"scenario": "crisisgrid", "max_rounds": 24},
        events=events_with_round,
        actions=[],
        round_records=[],
        outcomes={},
    ))["stranded"]
    assert quiet["y_private"] == quiet["stranded"]


def test_pair_search_does_not_import_the_order3_family():
    text = Path(__file__).resolve().parents[2].joinpath("src/engine/causal/oracle.py").read_text(encoding="utf-8")
    assert "heldout_order3" not in text
    assert "exactly_three" not in text
