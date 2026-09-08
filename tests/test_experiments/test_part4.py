"""Part 4 experiment integration tests."""

from __future__ import annotations

from src.engine.intervention import load_interventions
from src.engine.simulation import SimConfig, run_simulation
from src.experiments.conditions import EXPERIMENT_MATRIX, build_sim_config, get_condition
from src.experiments.metrics import compute_run_metrics
from src.experiments.paper_contrasts import run_crn_pair, shuffle_vs_full_test
from src.experiments.report import generate_report, generate_report_from_log
from src.experiments.runner import run_single


class TestConditions:
    def test_matrix_size(self):
        total = sum(len(v) for v in EXPERIMENT_MATRIX.values())
        assert total == 19  # 5+4+3+3+4

    def test_build_config_tags(self):
        cond = get_condition("A", "A2")
        cfg = build_sim_config(cond, seed=7)
        assert cfg.experiment_id == "A"
        assert cfg.condition_id == "A2"
        assert len(cfg.interventions) == 2


class TestExperimentRuns:
    def test_run_a2_completes(self):
        result = run_single("A", seed=1, condition_id="A2", max_rounds=60)
        log = result["log"]
        assert len(log.round_records) == 60
        assert "protest_authorship" in log.outcomes
        assert log.config["event_cast"]["idea"] == "phd_a"
        assert log.config["story_beats"]["draft_round"] == 52

    def test_run_validity_no_memory(self):
        result = run_single("V", seed=2, condition_id="V1", max_rounds=20)
        log = result["log"]
        assert log.config["disable_memory"] is True
        assert all(
            rec.get("agent_deltas", {}).get("phd_a", {}).get("memory_written") is None
            for rec in log.round_records
        )

    def test_shuffle_memory_config(self):
        log = run_simulation(SimConfig(max_rounds=30, seed=3, shuffle_memory=True, interventions=[]))
        assert log.config["shuffle_memory"] is True
        assert len(log.round_records) == 30


class TestMetricsAndReport:
    def test_compute_run_metrics(self):
        result = run_single("A", seed=4, condition_id="A1", max_rounds=30)
        metrics = compute_run_metrics(result["log"])
        assert metrics["run_id"] == result["log"].run_id
        assert "timeline" in metrics
        assert "trust_fragmentation_curve" in metrics

    def test_report_has_thirteen_sections(self, tmp_path):
        result = run_single("A", seed=5, condition_id="A1", max_rounds=10)
        text = generate_report_from_log(result["log"])
        for i in range(1, 15):
            assert f"## {i}." in text
        path = generate_report(
            experiment_id="A",
            condition_id="A1",
            seed=5,
            output_dir=tmp_path,
            log=result["log"],
        )
        assert "LLM Scoring Influence" in text
        assert "Action Field Explanation" in text
        assert "Causal Decompiler MRI" in text
        assert path.exists()


class TestCRNContrasts:
    def test_honor_pair_records_split_y(self):
        row = run_crn_pair("A", "A1", "A2", seed=0, max_rounds=8)
        assert row["control_id"] == "A1"
        assert "public_private_divergence_mean" in row["ates"]
        assert "ate" in row["ates"]["protest_authorship"]

    def test_shuffle_vs_full_is_crn(self):
        gate = shuffle_vs_full_test([0], max_rounds=8)
        assert gate["n_seeds"] == 1
        assert "ate_mean" in gate

    def test_new_outcomes_present(self):
        result = run_single("A", seed=6, condition_id="A2", max_rounds=55)
        outcomes = result["log"].outcomes
        for key in ("authorship_escalation_score", "authorship_escalation_potential", "post_r52_compliance", "withdraw_threat_event"):
            assert key in outcomes


class TestInterventionsLoaded:
    def test_part4_interventions_exist(self):
        ids = {i.intervention_id for i in load_interventions()}
        for required in (
            "INT_SKIP_E003",
            "INT_SKIP_E031",
            "INT_SKIP_E035",
            "INT_FALSE_MEMORY_INSERT",
            "INT_ALUMNI_POSITIVE",
            "INT_DELAYED_MEMORY_INSERT",
            "INT_B4_REBUTTAL_REQUEST",
        ):
            assert required in ids

