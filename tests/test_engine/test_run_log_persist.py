"""Persist NoiseLog + LLMTrace so twins can replay without a live API."""

from __future__ import annotations

from src.engine.causal.llm_trace import LLMTrace
from src.engine.causal.twin import load_factual
from src.engine.run_log import RunLog, llm_trace_sidecar


def test_jsonl_roundtrip_restores_events_noise_and_llm_trace(tmp_path):
    log = RunLog(run_id="persist1", config={"seed": 1, "condition_id": "A1"})
    log.events = [{"event_id": "E001", "round": 1, "type": "deadline", "source": "pi"}]
    log.actions = [{"round": 1, "agent": "phd_a", "type": "comply", "intensity": 0.4}]
    log.round_records = [{"round": 1, "event_id": "E001", "metrics": {"trust_phd_a_pi": 0.62, "public_private_divergence": 0.51}, "agent_deltas": {}}]
    log.noise_log = [{"round": 1, "stream": "event_jitter", "agent_id": None, "name": "u", "value": 0.13}]
    log.interventions_applied = [{"round": 3, "intervention_id": "INT_AUTH_EXPLICIT", "variant": "explicit"}]
    trace = LLMTrace()
    trace.by_key["abc123"] = {"ok": True, "action": "comply"}
    trace.errors["def456"] = "parse failed"
    trace.hits = 2
    trace.misses = 1
    log.llm_cache = trace
    log.outcomes["protest_authorship"] = 0.02

    path = tmp_path / "run_persist1.jsonl"
    log.write_jsonl(path)
    assert llm_trace_sidecar(path).exists()

    loaded = RunLog.from_jsonl(path)
    assert load_factual(path).llm_cache.by_key["abc123"]["ok"] is True
    assert loaded.events[0]["event_id"] == "E001"
    assert loaded.events[0]["type"] == "deadline"
    assert loaded.actions[0]["type"] == "comply"
    assert loaded.noise_log[0]["value"] == 0.13
    assert loaded.interventions_applied[0]["intervention_id"] == "INT_AUTH_EXPLICIT"
    assert loaded.llm_cache is not None
    assert loaded.llm_cache.by_key["abc123"]["action"] == "comply"
    assert loaded.llm_cache.errors["def456"] == "parse failed"
    assert loaded.outcomes["trust_pi_logged"] == 0.62
    assert loaded.outcomes["trust_pi_final"] == 0.62
    assert loaded.outcomes["public_private_divergence_mean"] == 0.51
