"""Tests for memory reconsolidation."""

from __future__ import annotations

from src.cognition.memory import RecallResult, _stable_unit, reconsolidate_memories
from src.world.loader import load_events, load_world


def test_reconsolidation_rewrites_old_memory_continuously():
    world = load_world()
    agent = world.agents["phd_a"]
    agent.memory.append({
        "memory_id": "M001",
        "owner": "phd_a",
        "round": 3,
        "event_ref": "E003",
        "content_type": "promise_fulfilled",
        "target": "pi",
        "valence": 0.65,
        "strength": 0.50,
        "strength_0": 0.50,
        "decay": 0.03,
        "rehearsal_count": 0.2,
        "evidence_quality": 0.45,
        "interpretation": "PI seemed to imply I would lead the paper.",
        "behavioral_hooks": ["ask_for_authorship"],
        "was_recalled": [],
        "objective_fact_ref": "E003.objective_fact",
    })
    event = next(e for e in load_events() if e.type == "authorship_ambiguity")
    recall = RecallResult({"M001": 1.0}, ["M001"], 0.65, 0.50, {})

    audit = reconsolidate_memories(agent, event, recall, event.round)

    assert audit["updated"]
    assert agent.memory[0]["valence"] < 0.65
    assert agent.memory[0]["strength"] > 0.50
    assert agent.memory[0]["reconsolidation_history"][0]["event_ref"] == event.event_id


def test_recall_embedding_is_stable_across_hash_seed(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("builtin hash() is not process-stable")

    monkeypatch.setattr("builtins.hash", boom)
    from src.cognition.memory import _memory_vector, _stable_unit

    assert _stable_unit("promise_broken") == _stable_unit("promise_broken")
    assert 0.0 <= _stable_unit("phd_a") < 1.0
    vec = _memory_vector(
        {
            "round": 3,
            "strength": 0.5,
            "valence": 0.1,
            "content_type": "promise_fulfilled",
            "target": "pi",
        },
        10,
    )
    assert len(vec) == 5
