"""200-world mechanism benchmark and baselines."""

from __future__ import annotations

from src.engine.causal.baselines import evaluate_baselines, gumbel_agreement
from src.engine.causal.mechanisms import evaluate_benchmark, parameterized_worlds, recover_cstar
from src.engine.causal.worlds import evaluate_planted_worlds


def test_six_planted_worlds_remain_unit_test():
    result = evaluate_planted_worlds()
    assert result["minimal_cause_recovery"] == 1.0


def test_parameterized_worlds_are_200():
    worlds = parameterized_worlds(50)
    assert len(worlds) == 200
    families = {w.family for w in worlds}
    assert families == {"and_synergy", "or_redundancy", "delayed_mediation", "suppressor"}


def test_benchmark_smoke_recovers_causes():
    result = evaluate_benchmark(n_per_family=50, seed=11)
    assert result["n"] == 200
    assert result["cause_set_f1"] >= 0.9
    assert result["interaction_sign_accuracy"] >= 0.75
    assert result["false_attribution_rate"] <= 0.15


def test_and_world_needs_the_pair():
    world = next(w for w in parameterized_worlds(2, seed=0) if w.family == "and_synergy")
    recovered = recover_cstar(world)
    assert set(recovered) == set(world.true_minimal)


def test_baselines_underperform_oracle():
    payload = evaluate_baselines(n_per_family=8, seed=11)
    assert payload["oracle"]["f1"] >= payload["baselines"]["recency"]["f1"]
    assert payload["oracle"]["f1"] >= payload["baselines"]["llm_direct"]["f1"]
    g = gumbel_agreement(n=20, seed=11)
    assert g["shared_agreement"] == 1.0
