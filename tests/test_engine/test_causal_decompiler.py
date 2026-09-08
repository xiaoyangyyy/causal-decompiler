"""Causal Decompiler: event-keyed twins, planted SCM, and MRI report smoke."""

from __future__ import annotations

from src.engine.causal import CausalDecompiler, delete_memory, run_factual, run_twin, skip_event
from src.engine.causal.estimands import crn_aligned_draws, memory_irf
from src.engine.causal.noise import STREAM_ACTION_JITTER, keyed_uniform
from src.engine.causal.toy import contrastive_leave_one_out, exact_shapley, planted_factors, planted_outcome
from src.engine.causal.twin import identity_holds
from src.engine.simulation import SimConfig


def _short_cfg(**kwargs) -> SimConfig:
    defaults = dict(mvp=True, seed=11, max_rounds=8, interventions=[], llm_provider="scripted")
    defaults.update(kwargs)
    return SimConfig(**defaults)


def test_keyed_uniform_ignores_sibling_draw_count():
    a = keyed_uniform(0, 4, "event_jitter", name="deadline_shift")
    keyed_uniform(0, 4, "event_jitter", name="unrelated_type")
    b = keyed_uniform(0, 4, "event_jitter", name="deadline_shift")
    assert a == b


def test_planted_and_shapley_splits_while_contrastive_overcounts():
    factors = planted_factors()
    shapley = exact_shapley(planted_outcome, factors)
    knockout = contrastive_leave_one_out(factors, factors)
    assert shapley["promise"] == 0.5
    assert shapley["draft"] == 0.5
    assert shapley["decoy"] == 0.0
    assert abs(sum(shapley.values()) - 1.0) < 1e-12
    assert knockout["promise"] == 1.0
    assert knockout["draft"] == 1.0
    assert knockout["decoy"] == 0.0
    assert sum(knockout.values()) == 2.0


def test_identity_twin_matches_factual_run():
    cfg = _short_cfg()
    factual = run_factual(cfg)
    twin = run_twin(cfg, [], llm_trace=factual.llm_cache)
    assert identity_holds(factual, twin)
    assert factual.noise_log
    stats = twin.outcomes["llm_trace_stats"]
    assert stats["run_misses"] == 0
    assert stats["run_hits"] > 0


def test_decompiler_attaches_mri_to_log_and_report():
    decompiler = CausalDecompiler()
    report = decompiler.decompile(
        _short_cfg(max_rounds=6),
        memory_rounds=[3],
        blame_limit=1,
        extra_ops=[],
    )
    assert report.identity_twin_ok
    assert report.llm_replay["identity_run_misses"] == 0
    assert decompiler.last_log is not None
    assert decompiler.last_log.outcomes["causal_mri"]["identity_twin_ok"] is True
    from src.experiments.report import generate_report_from_log

    text = generate_report_from_log(decompiler.last_log)
    assert "## 14." in text
    assert "Identity twin: ok" in text


def test_skip_event_keeps_later_keyed_draws():
    cfg = _short_cfg(max_rounds=6)
    factual = run_factual(cfg)
    skipped = run_twin(cfg, [skip_event(2, factual.events[1]["event_id"])], llm_trace=factual.llm_cache)
    assert crn_aligned_draws(factual, skipped, from_round=3)
    later_fact = {(d["round"], d["agent_id"], d["name"]) for d in factual.noise_log if d["stream"] == STREAM_ACTION_JITTER and d["round"] >= 3}
    later_twin = {(d["round"], d["agent_id"], d["name"]) for d in skipped.noise_log if d["stream"] == STREAM_ACTION_JITTER and d["round"] >= 3}
    assert later_fact
    assert later_fact == later_twin


def test_memory_irf_api_returns_delete_times():
    cfg = _short_cfg(max_rounds=10)
    factual = run_factual(cfg)
    estimates = memory_irf(cfg, factual, "protest_authorship", [4])
    assert len(estimates) == 1
    assert estimates[0].factor_id.startswith("MEMORY_DELETE")


def test_decompiler_smoke_report():
    report = CausalDecompiler().decompile(
        _short_cfg(max_rounds=6),
        memory_rounds=[3],
        blame_event_ids=None,
        extra_ops=[delete_memory(3)],
    )
    assert report.identity_twin_ok
    assert "protest_authorship" in report.split_y
    assert report.memory_irf
    assert report.shapley_toy["promise"] == 0.5
    assert report.contrastive_toy_lie["promise"] == 1.0
    assert report.three_worlds.get("factor_id", "").startswith("EVENT_SKIP")
    assert report.forks
    assert report.forks[0]["patch"] == "identity"


def test_llm_trace_replays_failures_without_recalling_inner():
    from src.engine.causal.llm_trace import LLMTrace, TracingAdapter
    from src.engine.llm_adapter import LLMError

    class BoomAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, system: str, user: str) -> dict:
            self.calls += 1
            raise LLMError("boom")

    inner = BoomAdapter()
    adapter = TracingAdapter(inner, LLMTrace())
    try:
        adapter.complete_json("sys", "user")
    except LLMError:
        pass
    try:
        adapter.complete_json("sys", "user")
    except LLMError:
        pass
    assert inner.calls == 1
    assert adapter.trace.misses == 1
    assert adapter.trace.hits == 1
