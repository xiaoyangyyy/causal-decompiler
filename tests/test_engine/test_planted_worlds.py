"""Planted social worlds, MIRF PostAUC, information algebra, hierarchical ops."""

from __future__ import annotations

from src.engine.causal import evaluate_planted_worlds, planted_social_worlds
from src.engine.causal.algebra import apply_ops, do_belief, do_private_public, do_visibility
from src.engine.causal.dynamics import assemble_surface, bootstrap_ci, post_auc
from src.engine.causal.search import causal_hypergraph, hierarchical_ops
from src.engine.causal.worlds import promise_irf_curve
from src.engine.simulation import SimConfig
from src.engine.run_log import RunLog
from src.engine.causal.ir import extract_ir


def test_six_planted_worlds_recover_truth():
    result = evaluate_planted_worlds()
    assert len(result["worlds"]) == 6
    assert result["minimal_cause_recovery"] == 1.0
    assert result["interaction_recovery"] == 1.0
    assert result["false_causal_attribution_rate"] == 0.0
    assert result["fork_localization_accuracy"] == 1.0
    names = {row["world"] for row in result["worlds"]}
    assert names == {
        "promise", "and_conflict", "redundant_rumor",
        "hidden_message", "strategic_compliance", "mediated_trust",
    }
    assert result["causal_factor_recall"] == 1.0
    and_row = next(r for r in result["worlds"] if r["world"] == "and_conflict")
    assert set(and_row["recovered"]) == {"e003", "e052"}
    rumor = next(r for r in result["worlds"] if r["world"] == "redundant_rumor")
    assert rumor["interaction_kinds"]["rumor_a,rumor_b"] == "redundant"
    assert rumor["causal_factor_recall"] == 1.0
    assert rumor["minimal_cause_recovery"] is True


def test_promise_irf_is_delayed_then_committed():
    early = promise_irf_curve(3, [10, 20, 40, 52])
    late = promise_irf_curve(45, [10, 20, 40, 52])
    assert early[10] == 0.0
    assert early[52] == 0.0
    assert late[20] == 1.0
    assert late[52] == 1.0
    auc_early = post_auc({t: 1.0 if t >= 20 else 0.0 for t in range(20, 53)}, 3)
    auc_late = post_auc({t: 0.0 for t in range(45, 53)}, 45)
    # remaining-time mean: late delete of a committed path has less room to move.
    assert abs(auc_early) >= abs(auc_late)


def test_do_belief_and_hide_compile_into_causal_do():
    cfg = SimConfig(mvp=True, max_rounds=6, llm_provider="scripted")
    patched, _ = apply_ops(cfg, [do_belief(3, "phd_a", {"pi_fairness": 0.9})])
    assert patched.causal_do["force_belief"]["pi_fairness"] == 0.9
    vis, _ = apply_ops(cfg, [do_visibility("hide_idea", event_id="E052", agent_id="phd_a", round_num=52)])
    assert vis.causal_do["hide_event_id"] == "E052"
    pub, _ = apply_ops(cfg, [do_private_public(52, "phd_a")])
    assert pub.causal_do["force_public"] == "team_support"


def test_bootstrap_ci_and_surface_assemble():
    ci = bootstrap_ci([0.1, 0.2, 0.15])
    assert ci["low"] <= ci["mean"] <= ci["high"]
    surface = assemble_surface([
        {"extras": {"mirf": {"delete_at": 3, "curve": {3: -0.1, 6: -0.2}, "post_auc": -0.15, "terminal": -0.2}}},
        {"extras": {"mirf": {"delete_at": 6, "curve": {6: -0.05}, "post_auc": -0.05, "terminal": -0.05}}},
    ])
    assert surface["delete"] == [3, 6]
    assert surface["matrix"]["3"]["6"] == -0.2


def test_hypergraph_and_hierarchical_ops_from_ir():
    graph = causal_hypergraph(
        {"factors": ["E003", "E052"], "kind": "synergy", "index": 0.059, "edge": {"source": "E003", "target": "E052", "relation": "synergy"}},
        {"set": ["E003", "E052"]},
        {"label": "performative_compliance"},
    )
    assert "synergy" in graph["mermaid"]
    log = RunLog(run_id="h", config={"seed": 1})
    log.events = [{"event_id": "E003", "round": 3, "type": "authorship_promise", "source": "pi"}]
    log.round_records = [{
        "round": 3, "event_id": "E003",
        "metrics": {"trust_phd_a_pi": 0.5},
        "agent_deltas": {"phd_a": {"memory_written": {"memory_id": "m1", "content_type": "authorship_signal", "event_ref": "E003"}}},
    }]
    layers = hierarchical_ops(extract_ir(log), log)
    names = {row["layer"] for row in layers}
    assert names & {"layer", "cluster", "item"}
    assert "cluster" in names or "item" in names


def test_planted_world_count():
    assert len(planted_social_worlds()) == 6
