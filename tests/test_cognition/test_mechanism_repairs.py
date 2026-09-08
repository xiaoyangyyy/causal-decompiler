"""Tests for mechanism repairs and new social-pressure machinery."""

from __future__ import annotations

from src.cognition.authorship import DIM_WEIGHTS
from src.cognition.memory import write_memory
from src.cognition.pipeline import process_event_phase
from src.cognition.pressure_fields import compute_pressure_fields
from src.cognition.relationship import update_relationships
from src.cognition.reputation import current_reputation, update_reputation_from_action
from src.cognition.social_potential import compute_social_potential
from src.engine.event_agent import EventAgent
from src.world.loader import load_events, load_world
from src.world.organization import agent_contribution_share, can_directly_observe, resolve_event_cast


def _edge(world, src, tgt):
    return next(e for e in world.relationships if e.source == src and e.target == tgt)


def test_contribution_share_reads_dimension_ledger():
    world = load_world()
    world.project.contribution_ledger = {
        "idea": {"phd_a": 0.80, "phd_b": 0.20},
        "experiments": {"phd_a": 0.70, "phd_b": 0.30},
        "writing": {"phd_a": 0.60, "phd_b": 0.40},
        "data": {"phd_a": 0.50, "phd_b": 0.50},
        "code": {"phd_a": 0.50, "phd_b": 0.50},
        "rebuttal": {"phd_a": 0.50, "phd_b": 0.50},
        "funding": {"phd_a": 0.50, "phd_b": 0.50},
        "supervision": {"phd_a": 0.50, "phd_b": 0.50},
    }
    share = agent_contribution_share(world, "phd_a")
    field = compute_social_potential(world, world.agents["phd_a"], EventAgent().generate(8, world))

    assert share > 0.55
    assert field.evidence["contribution_share"] == round(share, 4)


def test_dim_weights_are_a_probability_simplex():
    assert abs(sum(DIM_WEIGHTS.values()) - 1.0) < 1e-9


def test_team_event_is_visible_to_internal_non_targets(llm_adapter):
    world = load_world()
    event = next(e for e in load_events() if e.visibility == "team")
    engineer = world.agents["engineer_e"]

    assert can_directly_observe(engineer, event) is True
    mem = write_memory(engineer, event, event.round, llm_adapter=llm_adapter)
    assert mem is not None


def test_bilateral_event_hides_from_non_participants(llm_adapter):
    world = load_world()
    event = next(e for e in load_events() if e.visibility == "bilateral")
    event.source = "phd_b"
    event.targets = ["pi"]
    engineer = world.agents["engineer_e"]

    assert can_directly_observe(engineer, event) is False
    assert write_memory(engineer, event, event.round, llm_adapter=llm_adapter) is None


def test_support_action_creates_reverse_obligation():
    world = load_world()
    event = load_events()[0]
    before = _edge(world, "phd_b", "phd_a").obligation

    update_relationships(
        world,
        event,
        recalls={},
        actions=[{"agent": "phd_a", "type": "support_teammate", "target": "phd_b", "intensity": 0.9}],
    )

    after = _edge(world, "phd_b", "phd_a").obligation
    assert after > before


def test_named_pressure_fields_are_recorded():
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent().generate(52, world)
    fields = compute_pressure_fields(world, agent, event)

    assert set(fields) == {
        "AuthorshipPressureField",
        "TrustCollapseField",
        "AuthorityComplianceField",
        "IntegrityRiskField",
    }
    assert all(0.0 <= item["value"] <= 1.0 for item in fields.values())


def test_reputation_moves_after_public_actions():
    world = load_world()
    agent = world.agents["phd_a"]
    before = current_reputation(agent)
    update_reputation_from_action(
        world,
        {"agent": "phd_a", "type": "hide_negative_result", "target": "project", "intensity": 0.9},
    )
    assert current_reputation(agent) < before


def test_state_events_use_existing_agents():
    world = load_world()
    event = EventAgent(seed=3).generate(8, world)
    cast = resolve_event_cast(world)

    assert event is not None
    assert event.source in world.agents
    assert all(target in world.agents or target == "project" for target in event.targets)
    assert cast.idea in world.agents
    assert cast.experimenter in world.agents
    assert cast.pi in world.agents


def test_rumor_can_write_second_hand_memory(llm_adapter):
    world = load_world()
    event = next(e for e in load_events() if e.visibility == "bilateral")
    event.source = "phd_b"
    event.targets = ["pi"]
    for edge in world.relationships:
        if {edge.source, edge.target} == {"phd_b", "phd_a"} or {edge.source, edge.target} == {"pi", "phd_a"}:
            edge.communication_frequency = 0.90
            edge.information_access = 0.90

    result = process_event_phase(world, event, llm_adapter=llm_adapter)
    rumor = result.agent_deltas.get("phd_a", {}).get("rumor_memory")
    assert rumor is not None
    assert rumor["interpretation"].startswith("Heard second-hand:")
    assert result.agent_deltas["phd_a"]["observation_channel"] == "rumor"
    assert any(m.get("channel") == "rumor" for m in world.agents["phd_a"].memory)


def test_bilateral_event_does_not_move_blind_agent_beliefs():
    world = load_world()
    events = {e.event_id: e for e in load_events()}
    event = events["E030"]
    eng_before = world.agents["engineer_e"].beliefs.pi_fairness
    phd_before = world.agents["phd_a"].beliefs.pi_fairness
    result = process_event_phase(world, event)
    assert result.agent_deltas["engineer_e"]["observation_channel"] == "none"
    assert result.agent_deltas["phd_a"]["observation_channel"] == "direct"
    assert world.agents["engineer_e"].beliefs.pi_fairness == eng_before
    assert world.agents["phd_a"].beliefs.pi_fairness < phd_before


def test_team_event_updates_internal_witness_affect():
    world = load_world()
    events = {e.event_id: e for e in load_events()}
    event = events["E014"]
    result = process_event_phase(world, event)
    assert result.agent_deltas["engineer_e"]["observation_channel"] == "direct"
    assert result.metrics["observation_direct_share"] > 0.4
    assert result.agent_deltas["engineer_e"]["emotion"] is not None


def test_trust_has_recovery_floor_not_absorbing_zero():
    world = load_world()
    event = load_events()[0]
    for edge in world.relationships:
        if edge.source == "phd_a" and edge.target == "pi":
            edge.trust = 0.10
            edge.resentment = 1.0
    for _ in range(80):
        update_relationships(world, event, recalls={})
        assert _edge(world, "phd_a", "pi").trust >= 0.06
    assert _edge(world, "phd_a", "pi").trust > 0.0


def test_neutral_rounds_do_not_pin_trust_to_floor():
    world = load_world()
    event = next(e for e in load_events() if e.type == "deadline_shift")
    start = _edge(world, "phd_a", "pi").trust
    for _ in range(40):
        update_relationships(world, event, recalls={})
    after = _edge(world, "phd_a", "pi").trust
    assert after > 0.20
    assert after > start - 0.08
    for edge in world.relationships:
        if edge.source in {"phd_a", "phd_b", "pi"} and edge.target in {"phd_a", "phd_b", "pi"}:
            assert edge.trust > 0.06


def test_authorship_draft_lowers_idea_pi_trust():
    world = load_world()
    draft = next(e for e in load_events() if e.event_id == "E052")
    before = _edge(world, "phd_a", "pi").trust
    update_relationships(world, draft, recalls={})
    assert _edge(world, "phd_a", "pi").trust < before - 0.02


def test_honored_draft_does_not_cut_trust_like_broken_draft():
    world_broken = load_world()
    world_honored = load_world()
    draft = next(e for e in load_events() if e.event_id == "E052")
    honored = draft.model_copy(update={"framing": "positive", "payload": {**draft.payload, "draft_severity": "honored"}})
    start = _edge(world_broken, "phd_a", "pi").trust
    update_relationships(world_broken, draft, recalls={})
    update_relationships(world_honored, honored, recalls={})
    assert _edge(world_broken, "phd_a", "pi").trust < start
    assert _edge(world_honored, "phd_a", "pi").trust > _edge(world_broken, "phd_a", "pi").trust
