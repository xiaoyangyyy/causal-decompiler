"""Blind ranking, budgeted recovery, timed mediation, held-out family, external log."""

from __future__ import annotations

from pathlib import Path

from src.engine.causal.algebra import delete_memory
from src.engine.causal.external_adapter import external_mri
from src.engine.causal.heldout_family import evaluate_heldout
from src.engine.causal.identification import identification_report
from src.engine.causal.mechanisms import parameterized_worlds, recover_detail, timed_step
from src.engine.causal.oracle import TimedOracle, recover_timed_effects
from src.engine.causal.search import _candidate_relevance
from src.engine.simulation import SimConfig
from src.experiments.aamas_protocol import decide, run_grid
from src.world.loader import PROJECT_ROOT


def test_ranking_ignores_planted_event_ids():
    planted = _candidate_relevance({
        "source_event": "E003",
        "type": "event",
        "layer": "E",
        "round": 3,
        "payload": {"event_type": "authorship_promise"},
    })
    other = _candidate_relevance({
        "source_event": "E999",
        "type": "event",
        "layer": "E",
        "round": 3,
        "payload": {},
    })
    assert planted == other


def test_search_modules_do_not_import_heldout_family():
    root = Path(__file__).resolve().parents[2] / "src" / "engine" / "causal"
    for name in ("search.py", "oracle.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "heldout_family" not in text
        assert "E003" not in text


def test_delete_memory_requires_subject():
    try:
        delete_memory(3, "")
    except ValueError:
        return
    raise AssertionError("empty subject was accepted")


def test_timed_mediation_distinguishes_deletion_time():
    step = timed_step(1.0, delay=4)
    oracle = TimedOracle(step, budget=8)
    found = recover_timed_effects(oracle, 4)
    assert found["early_only_before_write"]
    assert found["mediator_after_delay"]
    assert found["y_delete_early_after"] == 1.0
    assert found["y_delete_early_before"] == 0.0
    assert found["y_delete_mediator"] == 0.0


def test_delayed_family_uses_the_clock():
    world = next(w for w in parameterized_worlds(1, seed=0) if w.timed)
    detail = recover_detail(world)
    assert not detail["failed"]
    assert detail["timed"]["early_only_before_write"]
    assert detail["timed"]["mediator_after_delay"]


def test_heldout_exactly_two_is_not_recovered():
    report = evaluate_heldout(n=4, seed=3)
    assert report["failure_rate"] == 1.0
    assert report["cause_set_f1"] == 0.0
    assert all(point["f1"] == 0.0 for point in report["budget_curve"])


def test_identification_counterexamples_hold():
    report = identification_report()
    assert report["all_hold"]
    assert any(line.startswith("P1 proposition:") for line in report["statements"])
    assert any(line.startswith("A3 assumption:") for line in report["statements"])


def test_external_log_identity_and_four_layers():
    path = PROJECT_ROOT / "external_traces" / "agent_trace.json"
    report = external_mri(path)
    assert report["identity_ok"]
    assert report["identity_y"] == 1.0
    assert {item["layer"] for item in report["effects"]} == {
        "do_event", "do_visibility", "do_memory", "do_behavior",
    }
    assert all(item["ate"] == -1.0 for item in report["effects"])
    assert report["minimal_set_size"] == 4
    assert report["claim"] == "interface_check"
    source = Path(PROJECT_ROOT / "src" / "engine" / "causal" / "external_adapter.py").read_text(encoding="utf-8")
    assert "src.cognition" not in source
    assert "src.world" not in source
    assert "src.engine.simulation" not in source


def test_preregistered_decision_does_not_invent_a_model():
    rows = [{
        "provider": "scripted",
        "anchor_keep_fraction": 1.0,
        "channels": {"y_private": 0.1, "y_public": 0.1, "y_action": 0.0},
    }]
    decision = decide(rows)
    assert decision["claim"] == "cross_model_not_identified"
    assert "threshold" not in decision


def test_anchor_half_grid_runs_short():
    payload = run_grid(
        scenarios=("releaseops",),
        seeds=(0,),
        providers=("scripted",),
        anchor_fractions=(1.0, 0.5),
        rounds=2,
    )
    assert len(payload["rows"]) == 2
    assert payload["errors"] == []
    assert {row["anchor_keep_fraction"] for row in payload["rows"]} == {1.0, 0.5}
    assert SimConfig(anchor_keep_fraction=0.5).anchor_keep_fraction == 0.5
