"""Factual record and patched twin replay.

A no-op twin must reproduce the factual run round-by-round. After a patch,
streams that were not rewritten keep the same keyed draws (CRN), so later
rounds do not inherit a shifted PRNG queue.
"""

from __future__ import annotations

from src.engine.causal.algebra import CausalOp, apply_ops
from src.engine.causal.llm_trace import LLMTrace, bind_llm_trace, reset_llm_trace
from src.engine.causal.noise import bind_noise_salts, reset_noise_salts
from src.engine.run_log import RunLog
from src.engine.simulation import SimConfig, run_simulation


def load_factual(path) -> RunLog:
    """Reload a persisted factual run, including NoiseLog and LLMTrace sidecar."""
    from pathlib import Path

    return RunLog.from_jsonl(Path(path))


def sim_config_from_log(log: RunLog, **overrides) -> SimConfig:
    """Rebuild a SimConfig so twins can replay a persisted factual run."""
    from src.engine.intervention import load_interventions

    c = dict(log.config or {})
    catalog = {i.intervention_id: i for i in load_interventions()}
    inter_ids = c.get("interventions") or []
    interventions = [catalog[i] for i in inter_ids if i in catalog]
    cfg = SimConfig(
        max_rounds=int(c.get("max_rounds") or 60),
        seed=int(c.get("seed") or 0),
        mvp=bool(c.get("mvp")),
        active_agents=c.get("active_agents"),
        offstage_agents=c.get("offstage_agents"),
        disable_memory=bool(c.get("disable_memory")),
        shuffle_memory=bool(c.get("shuffle_memory")),
        disable_state_events=bool(c.get("disable_state_events")),
        experiment_id=c.get("experiment_id"),
        condition_id=c.get("condition_id"),
        policy_mode=str(c.get("policy_mode") or "dual_engine"),
        enable_llm_action_scoring=bool(c.get("enable_llm_action_scoring", True)),
        cognitive_policy_lambda=c.get("cognitive_policy_lambda", 0.35),
        llm_action_score_mix=float(c.get("llm_action_score_mix") or 0.35),
        hierarchy_lesion=bool(c.get("hierarchy_lesion")),
        status_lesion=bool(c.get("status_lesion")),
        trust_lesion=bool(c.get("trust_lesion")),
        observation_lesion=bool(c.get("observation_lesion")),
        cognitive_sampling_top_k=c.get("cognitive_sampling_top_k"),
        cognitive_sampling_threshold=float(c.get("cognitive_sampling_threshold") or 0.0),
        llm_provider=c.get("llm_provider"),
        llm_model=c.get("llm_model"),
        llm_temperature=c.get("llm_temperature"),
        interventions=interventions,
        run_id=None,
    )
    if overrides:
        from dataclasses import replace

        cfg = replace(cfg, **overrides)
    return cfg


def run_factual(config: SimConfig) -> RunLog:
    return run_simulation(config)


def run_replay(config: SimConfig, *, llm_trace: LLMTrace | None = None) -> RunLog:
    """Run a fully specified config while replaying an existing LLM trace."""
    token = bind_llm_trace(llm_trace) if llm_trace is not None else None
    try:
        return run_simulation(config)
    finally:
        if token is not None:
            reset_llm_trace(token)


def run_twin(base: SimConfig, ops: list[CausalOp], *, llm_trace: LLMTrace | None = None) -> RunLog:
    cfg, salts = apply_ops(base, ops)
    salt_token = bind_noise_salts(salts)
    llm_token = bind_llm_trace(llm_trace) if llm_trace is not None else None
    try:
        log = run_simulation(cfg)
    finally:
        reset_noise_salts(salt_token)
        if llm_token is not None:
            reset_llm_trace(llm_token)
    log.config["causal_ops"] = [op.factor_id() for op in ops]
    return log


def identity_holds(factual: RunLog, twin: RunLog) -> bool:
    if [e.get("event_id") for e in factual.events] != [e.get("event_id") for e in twin.events]:
        return False
    if [(a.get("round"), a.get("agent"), a.get("type")) for a in factual.actions] != [
        (a.get("round"), a.get("agent"), a.get("type")) for a in twin.actions
    ]:
        return False
    return True
