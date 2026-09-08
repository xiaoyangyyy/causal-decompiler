"""Paper Causal Decompiler battery: split-Y, Shapley vs skip, three-worlds, persist."""

from __future__ import annotations

from src.engine.causal import CausalDecompiler, load_factual, run_factual, skip_event
from src.engine.causal.algebra import lesion
from src.engine.causal.estimands import (
    and_event_ids,
    draft_beat_op,
    first_divergence,
    split_y,
    story_shapley,
    three_worlds,
)
from src.engine.causal.toy import contrastive_leave_one_out, exact_shapley, planted_factors, planted_outcome
from src.engine.simulation import SimConfig
from src.experiments.paper_protocol import run_paper_protocol
from src.experiments.paper_tables import render_paper_markdown, table_forks, table_memory_irf, table_shapley


def _cfg(**kwargs) -> SimConfig:
    defaults = dict(mvp=True, seed=11, max_rounds=6, interventions=[], llm_provider="scripted")
    defaults.update(kwargs)
    return SimConfig(**defaults)


def test_story_shapley_adds_to_total_effect():
    cfg = _cfg()
    factual = run_factual(cfg)
    ids = and_event_ids(factual)
    result = story_shapley(cfg, factual, ids, "protest_authorship")
    if len(result.get("factors") or []) < 2:
        return
    phi = sum(result["shapley"].values())
    assert abs(phi - result["total_effect"]) < 1e-6


def test_three_worlds_returns_split_vector():
    cfg = _cfg()
    factual = run_factual(cfg)
    op = skip_event(int(factual.events[0]["round"]), str(factual.events[0]["event_id"]))
    worlds = three_worlds(cfg, factual, op, "protest_authorship")
    assert set(worlds["y"]) == {"w0", "w1", "w2"}
    assert "hypocrisy_index" in worlds
    assert "public_private_divergence_mean" in worlds["split"]["w0"]


def test_decompile_log_replays_persisted_factual(tmp_path):
    cfg = _cfg(output_dir=tmp_path)
    factual = run_factual(cfg)
    jsonl = tmp_path / f"run_{factual.run_id}.jsonl"
    assert jsonl.exists()
    loaded = load_factual(jsonl)
    report = CausalDecompiler().decompile_log(
        loaded,
        memory_rounds=[3],
        blame_limit=1,
        auto_battery=False,
        include_story_shapley=False,
        include_three_worlds=False,
    )
    assert report.identity_twin_ok
    assert report.llm_replay["identity_run_misses"] == 0
    assert "protest_authorship" in report.split_y


def test_paper_protocol_lite_emits_tables():
    result = run_paper_protocol(
        _cfg(),
        auto_battery=False,
        include_toy_shapley=True,
        write_output=False,
    )
    assert result.report.identity_twin_ok
    assert result.report.findings
    md = render_paper_markdown(result.report.to_dict())
    assert "Split-Y" in md
    assert "AND" in table_shapley(result.report.to_dict())
    assert result.report.shapley_toy["promise"] == 0.5
    md_full = render_paper_markdown({
        **result.report.to_dict(),
        "memory_irf": [{
            "factor_id": "MEMORY_DELETE:r3",
            "ate": 0.01,
            "extras": {
                "split": {
                    "protest_authorship": {"ate": 0.01},
                    "authorship_escalation_potential": {"ate": -0.04},
                    "public_private_divergence_mean": {"ate": -0.2},
                    "post_r52_compliance": {"ate": 0.0},
                    "memory_authorship_cluster_strength": {"ate": -0.3},
                },
                "fork": {"identical": False, "round": 4, "channel": "private"},
            },
        }],
        "forks": [{"patch": "identity", "factor_id": "NOOP", "identical": True}],
    })
    assert "Δ PPD" in md_full
    assert "Δ potential" in md_full
    assert "First divergence" in md_full


def test_and_event_ids_are_promise_and_draft():
    from src.engine.run_log import RunLog

    log = RunLog(run_id="and", config={})
    log.events = [
        {"event_id": "E003", "round": 3},
        {"event_id": "E030", "round": 30},
        {"event_id": "E052", "round": 52},
    ]
    assert and_event_ids(log) == ["E003", "E052"]
    op = draft_beat_op(log)
    assert op is not None
    assert op.target_event == "E052"


def test_first_divergence_marks_private_channel():
    from src.engine.run_log import RunLog

    factual = RunLog(run_id="f", config={})
    twin = RunLog(run_id="t", config={})
    base = {
        "round": 4,
        "agent": "phd_a",
        "type": "comply",
        "public_position": {"statement_type": "neutral", "authorship_claim": "any"},
        "private_intent": {"goal": "lay_low", "authorship_claim": "any"},
    }
    factual.actions = [dict(base)]
    twin.actions = [{**base, "private_intent": {"goal": "claim_credit", "authorship_claim": "first"}}]
    fork = first_divergence(factual, twin)
    assert fork["identical"] is False
    assert fork["channel"] == "private"
    assert fork["round"] == 4
    assert "First divergence" in table_forks({
        "forks": [_fork_row(fork)],
    })


def _fork_row(fork: dict) -> dict:
    return {"patch": "skip", "factor_id": "EVENT_SKIP:E052", **fork}


def test_memory_irf_carries_split_and_fork():
    cfg = _cfg(max_rounds=8)
    factual = run_factual(cfg)
    from src.engine.causal.estimands import memory_irf

    estimates = memory_irf(cfg, factual, "protest_authorship", [3])
    assert estimates[0].extras["split"]["public_private_divergence_mean"]
    assert "identical" in estimates[0].extras["fork"]
    assert "Δ PPD" in table_memory_irf({"memory_irf": [estimates[0].__dict__]})


def test_paper_crn_pair_records_split_ates():
    from src.experiments.paper_contrasts import run_crn_pair

    row = run_crn_pair("A", "A1", "A2", seed=0, max_rounds=6)
    assert "public_private_divergence_mean" in row["ates"]
    assert "ate" in row["ates"]["protest_authorship"]


def test_lesion_observation_is_a_causal_op():
    op = lesion("observation")
    assert "observation" in op.factor_id()


def test_split_y_has_private_channel():
    log = run_factual(_cfg())
    y = split_y(log)
    assert "public_private_divergence_mean" in y
    assert "trust_pi_logged" in y
    assert "trust_pi_path_mean" in y


def test_planted_oracle_still_the_and_lie():
    factors = planted_factors()
    shapley = exact_shapley(planted_outcome, factors)
    knockout = contrastive_leave_one_out(factors, factors)
    assert shapley["promise"] == 0.5
    assert knockout["promise"] == 1.0
    assert sum(knockout.values()) == 2.0
