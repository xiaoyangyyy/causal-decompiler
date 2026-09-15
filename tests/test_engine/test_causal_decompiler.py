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
    assert report.forks
    assert report.forks[0]["patch"] == "identity"
    assert report.layer_interventions.get("worlds")
    assert report.hierarchical_search.get("evaluated")
    assert report.bidirectional_search.get("deletion")
    assert report.channels.get("ppg") is not None


def test_scripted_eight_round_full_battery_has_layers_and_k3():
    from src.experiments.paper_tables import render_paper_markdown

    report = CausalDecompiler().decompile(_short_cfg(max_rounds=8))
    worlds = report.layer_interventions.get("worlds") or {}
    assert len(worlds) == 4
    assert "do_belief" not in worlds
    assert report.hierarchical_search.get("evaluated")
    assert report.bidirectional_search.get("deletion")
    assert report.paired_effect.get("k") == 3
    md = render_paper_markdown(report.to_dict())
    assert "Hierarchical search" in md
    assert "Individual paired causal effect" in md
    assert "Minimal Effect-Recovery Set" in md
    assert "Layer-specific interventions" in md
    assert "not run on long LLM MRI" not in md
    assert "| k | 3 |" in md


def test_paper_top_k_uses_five_for_llm_even_on_short_runs():
    from src.engine.causal.search import PAPER_TOP_K_LLM, PAPER_TOP_K_SCRIPTED, paper_top_k

    assert paper_top_k(SimConfig(max_rounds=8, llm_provider="scripted")) == PAPER_TOP_K_SCRIPTED
    assert paper_top_k(SimConfig(max_rounds=8, llm_provider="ollama")) == PAPER_TOP_K_LLM
    assert paper_top_k(SimConfig(max_rounds=60, llm_provider="deepseek")) == PAPER_TOP_K_LLM


def test_long_llm_auto_battery_runs_layers_without_cheap_twins(monkeypatch):
    from src.engine.causal import decompiler as decmod
    from src.engine.causal.dynamics import distributional_k
    from src.engine.simulation import SimConfig

    assert distributional_k(SimConfig(max_rounds=60, llm_provider="deepseek")) == 1
    monkeypatch.setattr(decmod, "distributional_k", lambda cfg: 1)
    called = {"layers": 0, "hier": 0, "bi": 0}
    real_layers = decmod.layer_interventions
    real_hier = decmod.run_hierarchical_search
    real_bi = decmod.run_bidirectional_search

    def spy_layers(*args, **kwargs):
        called["layers"] += 1
        return real_layers(*args, **kwargs)

    def spy_hier(*args, **kwargs):
        called["hier"] += 1
        assert kwargs.get("new_twin_budget") == 2
        return real_hier(*args, **kwargs)

    def spy_bi(*args, **kwargs):
        called["bi"] += 1
        return real_bi(*args, **kwargs)

    monkeypatch.setattr(decmod, "layer_interventions", spy_layers)
    monkeypatch.setattr(decmod, "run_hierarchical_search", spy_hier)
    monkeypatch.setattr(decmod, "run_bidirectional_search", spy_bi)
    report = CausalDecompiler().decompile(
        _short_cfg(max_rounds=6),
        memory_rounds=[3],
        include_toy_shapley=False,
        include_story_shapley=False,
        include_three_worlds=False,
    )
    assert called["layers"] == 1
    assert called["hier"] == 1
    assert called["bi"] == 1
    assert report.layer_interventions.get("worlds")
    assert report.paired_effect.get("k") == 1


def test_information_algebra_reuses_matching_factor_ids(monkeypatch):
    from src.engine.causal.algebra import do_presence
    from src.engine.causal import dynamics as dyn

    cfg = _short_cfg(max_rounds=6)
    factual = run_factual(cfg)
    rec = factual.events[-1]
    eid = str(rec["event_id"])
    rnd = int(rec["round"])
    fid = do_presence(rnd, eid).factor_id()
    calls: list[str] = []
    real_twin = dyn.run_twin

    def spy(base, ops, llm_trace=None):
        calls.append(ops[0].factor_id() if ops else "NOOP")
        return real_twin(base, ops, llm_trace=llm_trace)

    monkeypatch.setattr(dyn, "run_twin", spy)
    result = dyn.information_algebra(
        cfg,
        factual,
        "protest_authorship",
        event_id=eid,
        round_num=rnd,
        prior_effects={
            fid: {
                "ate": -0.42,
                "fork": {"identical": False, "round": rnd, "channel": "event"},
                "split": {"public_private_divergence_mean": 0.1},
            }
        },
    )
    assert result["worlds"]["do_presence"]["reused"] is True
    assert result["worlds"]["do_presence"]["ate"] == -0.42
    assert fid not in calls
    assert result["reused"] >= 1


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
