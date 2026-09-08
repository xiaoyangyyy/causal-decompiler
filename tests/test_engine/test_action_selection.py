"""Tests for continuous action candidate generation."""

from __future__ import annotations

from src.cognition.memory import RecallResult
from src.engine.action_selection import generate_action_candidates, sample_action_candidate_legacy
from src.engine.event_agent import EventAgent
from src.world.actions import get_allowed_actions
from src.world.loader import load_world


def test_candidates_have_probabilities_and_motives():
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent().generate(52, world)
    allowed = [a.value for a in get_allowed_actions(agent.id)]

    candidates = generate_action_candidates(agent, event, world, None, allowed, seed=42)

    assert candidates
    assert abs(sum(c.probability for c in candidates) - 1.0) < 1e-6
    assert all(c.motives for c in candidates)
    assert all(c.field_decomposition for c in candidates)
    assert all("motive_contributions" in c.field_decomposition for c in candidates)
    assert all(0.0 <= c.intensity <= 1.0 for c in candidates)


def test_authorship_pressure_lifts_authorship_actions():
    world = load_world()
    agent = world.agents["phd_a"]
    agent.beliefs.pi_fairness = 0.20
    agent.beliefs.my_first_author_probability = 0.15
    agent.emotion.resentment = 0.80
    event = EventAgent().generate(52, world)
    allowed = [a.value for a in get_allowed_actions(agent.id)]

    candidates = generate_action_candidates(agent, event, world, None, allowed, seed=7)
    top_types = {c.type for c in candidates[:5]}

    assert top_types & {"ask_for_authorship", "challenge_claim", "confront", "document_contribution"}


def test_recall_pressure_lifts_ask_for_authorship():
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent().generate(52, world)
    allowed = [a.value for a in get_allowed_actions(agent.id)]
    cold = generate_action_candidates(agent, event, world, None, allowed, seed=7)
    hot = generate_action_candidates(
        agent,
        event,
        world,
        RecallResult(
            attention_weights={"M1": 1.0},
            recalled_ids=["M1"],
            recall_field_valence=-0.85,
            recall_field_strength=0.95,
            audit={},
        ),
        allowed,
        seed=7,
    )

    def tendency(cands, atype: str) -> float:
        return next(c.tendency for c in cands if c.type == atype)

    assert tendency(hot, "ask_for_authorship") > tendency(cold, "ask_for_authorship")


def test_legacy_sampling_is_seed_stable():
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent().generate(40, world)
    allowed = [a.value for a in get_allowed_actions(agent.id)]
    candidates = generate_action_candidates(agent, event, world, None, allowed, seed=3)

    a = sample_action_candidate_legacy(candidates, seed=3, round_num=40, agent_id="phd_a")
    b = sample_action_candidate_legacy(candidates, seed=3, round_num=40, agent_id="phd_a")

    assert a.type == b.type


def test_idea_draft_candidates_exclude_work_actions():
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent().generate(8, world).model_copy(update={
        "type": "authorship_draft",
        "round": 52,
        "event_id": "E052",
    })
    allowed = [a.value for a in get_allowed_actions(agent.id)]
    assert "write_section" in allowed
    candidates = generate_action_candidates(agent, event, world, None, allowed, seed=7)
    types = {c.type for c in candidates}
    assert "write_section" not in types
    assert "run_experiment" not in types
    assert types & {"ask_for_authorship", "privately_lobby_pi", "confront", "document_contribution"}


def test_credit_action_cannot_stay_team_support():
    from src.engine.role_policy import _align_public_to_action, _scripted_render_action

    public = _align_public_to_action(
        "ask_for_authorship",
        {"statement_type": "team_support", "authorship_claim": "any_authorship"},
    )
    assert public["statement_type"] == "self_advocacy"
    assert public["authorship_claim"] == "first_author"

    doc = _align_public_to_action("document_contribution", {"statement_type": "team_support"})
    assert doc["statement_type"] == "self_advocacy"

    lobby = _align_public_to_action("privately_lobby_pi", {"statement_type": "team_support"})
    assert lobby["statement_type"] == "team_support"

    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent().generate(52, world)
    rendered = _scripted_render_action(
        agent, event, {"type": "ask_for_authorship", "target": "pi", "intensity": 0.8},
    )
    assert rendered["public_position"]["statement_type"] == "self_advocacy"


def test_idea_draft_softmax_concentrates_on_top_credit_action():
    from src.engine.role_policy import _focus_draft_payloads

    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent().generate(8, world).model_copy(update={
        "type": "authorship_draft",
        "round": 52,
        "event_id": "E052",
    })
    payloads = [
        {"type": "ask_for_authorship", "fused_tendency": 0.77, "probability": 0.33},
        {"type": "privately_lobby_pi", "fused_tendency": 0.70, "probability": 0.33},
        {"type": "document_contribution", "fused_tendency": 0.63, "probability": 0.33},
        {"type": "write_section", "fused_tendency": 0.40, "probability": 0.01},
    ]
    focused = _focus_draft_payloads(event, agent, world, payloads)
    types = {p["type"] for p in focused}
    assert "write_section" not in types
    ask_p = next(p["probability"] for p in focused if p["type"] == "ask_for_authorship")
    doc_p = next(p["probability"] for p in focused if p["type"] == "document_contribution")
    assert ask_p > 0.50
    assert ask_p > doc_p + 0.25
