"""Cognitive sampling tests."""

from __future__ import annotations

from src.engine import SimConfig, run_simulation


def test_cognitive_sampling_marks_only_top_k_agents_for_llm():
    log = run_simulation(SimConfig(
        max_rounds=3,
        seed=4,
        mvp=False,
        llm_provider="scripted",
        policy_mode="dual_engine",
        cognitive_sampling_top_k=1,
    ))

    assert log.config["cognitive_sampling_top_k"] == 1
    sampled = [a for a in log.actions if (a.get("cognitive_sampling") or {}).get("sampled")]
    unsampled = [a for a in log.actions if (a.get("cognitive_sampling") or {}).get("enabled") and not (a.get("cognitive_sampling") or {}).get("sampled")]
    assert sampled
    assert unsampled
    assert all(a.get("llm_raw", {}).get("source") == "scripted_unsampled_render" for a in unsampled)
