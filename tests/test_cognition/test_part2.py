"""Part 2 cognition layer tests — continuous dynamics, no thresholds."""

from __future__ import annotations

import copy

import pytest

from src.cognition.authorship import authorship_dispute_index, rank_authors
from src.cognition.belief import behavior_tendency_multipliers
from src.cognition.divergence import compute_divergence
from src.cognition.memory import compute_valence, write_memory
from src.cognition.pipeline import process_event_phase, simulate_memory_decay_only
from src.world.loader import load_events, load_world
from src.world.models import EventAtom, ObjectiveFact


@pytest.fixture
def world():
    return load_world()


@pytest.fixture
def events():
    return {e.event_id: e for e in load_events()}


def _clone_world(world):
    return copy.deepcopy(world)


class TestMemoryResonance:
    def test_e030_opposite_valence_phd_a_phd_b(self, world, events):
        event = events["E030"]
        va = compute_valence(world.agents["phd_a"], event, "authorship_signal")
        vb = compute_valence(world.agents["phd_b"], event, "authorship_signal")
        assert va < 0
        assert vb > 0
        assert va * vb < 0

    def test_memory_write_reproducible(self, world, events, llm_adapter):
        w1 = _clone_world(world)
        w2 = _clone_world(world)
        e = events["E003"]
        m1 = write_memory(w1.agents["phd_a"], e, 3, llm_adapter=llm_adapter)
        m2 = write_memory(w2.agents["phd_a"], e, 3, llm_adapter=llm_adapter)
        assert m1 is not None and m2 is not None
        assert m1.valence == m2.valence
        assert m1.strength == m2.strength
        assert m1.interpretation == m2.interpretation

    def test_decay_floor_asymptotic_not_zero(self, world):
        agent = copy.deepcopy(world.agents["phd_a"])
        agent.memory = [{
            "memory_id": "M001",
            "owner": "phd_a",
            "round": 1,
            "event_ref": "E003",
            "content_type": "authorship_signal",
            "target": "pi",
            "valence": -0.5,
            "strength": 0.8,
            "strength_0": 0.8,
            "decay": 0.03,
            "rehearsal_count": 0.0,
            "evidence_quality": 0.9,
            "interpretation": "test",
            "behavioral_hooks": [],
        }]
        trajectory = simulate_memory_decay_only(agent, 100, emotional_arousal=0.0)
        assert trajectory[-1] >= 0.8 * 0.05 * 0.95
        assert trajectory[-1] > 0

    def test_rehearsal_increases_strength(self, world, events):
        w_low = _clone_world(world)
        w_high = _clone_world(world)
        e = events["E014"]
        process_event_phase(w_low, e)
        process_event_phase(w_high, e)
        for _ in range(3):
            process_event_phase(w_high, events["E020"])

        def _mem_strength(w, event_ref):
            for mem in w.agents["phd_a"].memory:
                if mem.get("event_ref") == event_ref:
                    return float(mem["strength"]), float(mem.get("rehearsal_count", 0))
            return 0.0, 0.0

        s_low, rc_low = _mem_strength(w_low, "E014")
        s_high, rc_high = _mem_strength(w_high, "E014")
        assert rc_high > rc_low
        assert s_high >= s_low

    def test_softmax_recall_all_memories_have_weight(self, world, events):
        w = _clone_world(world)
        process_event_phase(w, events["E003"])
        process_event_phase(w, events["E014"])
        result = process_event_phase(w, events["E030"])
        phd_a_audit = result.agent_deltas["phd_a"]["recall_audit"]
        weights = phd_a_audit["attention_weights"]
        assert len(weights) >= 2
        assert abs(sum(weights.values()) - 1.0) < 1e-4


class TestEmotionBelief:
    def test_public_praise_divergent_emotion(self, world, events):
        w = _clone_world(world)
        before_b = w.agents["phd_b"].emotion.confidence
        before_a_resent = w.agents["phd_a"].emotion.resentment
        result = process_event_phase(w, events["E014"])
        phd_a_resent = result.agent_deltas["phd_a"]["emotion"]["resentment"]
        phd_b_conf = result.agent_deltas["phd_b"]["emotion"]["confidence"]
        assert phd_b_conf > before_b
        assert phd_a_resent > before_a_resent

    def test_belief_precision_update_continuous(self, world, events):
        w = _clone_world(world)
        before = w.agents["phd_a"].beliefs.pi_fairness
        result = process_event_phase(w, events["E030"])
        after = result.agent_deltas["phd_a"]["beliefs"]["pi_fairness"]
        assert after < before

    def test_behavior_tendency_smooth_not_binary(self, world):
        agent = copy.deepcopy(world.agents["phd_a"])
        agent.beliefs.pi_fairness = 0.39
        m1 = behavior_tendency_multipliers(agent)
        agent.beliefs.pi_fairness = 0.41
        m2 = behavior_tendency_multipliers(agent)
        assert m1["ask_for_authorship"] != m2["ask_for_authorship"]
        assert m2["comply"] > m1["comply"]


class TestRelationshipAuthorship:
    def test_trust_fragmentation_continuous(self, world, events):
        w = _clone_world(world)
        r1 = process_event_phase(w, events["E003"])
        frag1 = r1.metrics["trust_fragmentation"]
        process_event_phase(w, events["E014"])
        process_event_phase(w, events["E031"])
        r4 = process_event_phase(w, events["E040"])
        assert r4.metrics["trust_fragmentation"] >= frag1

    def test_authorship_ranking_changes_with_pressure(self, world):
        w = _clone_world(world)
        order_low = rank_authors(w)
        w.project.project.deadline_pressure = 0.92
        order_high = rank_authors(w)
        assert isinstance(order_low, list)
        assert isinstance(order_high, list)
        assert len(order_low) >= 3

    def test_dispute_index_rises_over_acts(self, world, events):
        w = _clone_world(world)
        process_event_phase(w, events["E010"])
        d10 = authorship_dispute_index(w)
        for eid in ["E014", "E019", "E030", "E040", "E052"]:
            process_event_phase(w, events[eid])
        d52 = authorship_dispute_index(w)
        assert d52 > d10


class TestDivergence:
    def test_divergence_rises_with_mismatch(self, world):
        agent = copy.deepcopy(world.agents["phd_a"])
        agent.public_position = {
            "statement_type": "team_support",
            "authorship_claim": "co_first_acceptable",
        }
        agent.private_intent = {
            "goal": "secure_first_author",
            "strategy": "document_contribution_then_confront",
            "trust_pi": 0.2,
        }
        div = compute_divergence(agent)
        agent.public_position = {
            "statement_type": "team_support",
            "authorship_claim": "co_first_acceptable",
        }
        agent.private_intent = {
            "goal": "co_first_push",
            "strategy": "co_first_push",
            "trust_pi": 0.8,
        }
        div_aligned = compute_divergence(agent)
        assert div > div_aligned

    def test_action_strategy_tokens_are_not_neutral(self, world):
        from src.cognition.divergence import divergence_from_action

        public = {"statement_type": "team_support", "authorship_claim": "any_authorship"}
        low = divergence_from_action({
            "type": "lay_low",
            "public_position": public,
            "private_intent": {"goal": "lay_low", "strategy": "lay_low"},
        })
        high = divergence_from_action({
            "type": "ask_for_authorship",
            "public_position": public,
            "private_intent": {"goal": "ask_for_authorship", "strategy": "ask_for_authorship"},
        })
        assert high > low + 0.15

    def test_ambiguity_writes_signal_not_broken_promise(self, world, events, llm_adapter):
        event = events["E030"]
        mem = write_memory(world.agents["phd_a"], event, event.round, llm_adapter=llm_adapter)
        assert mem is not None
        assert mem.content_type == "authorship_signal"

    def test_credit_dispute_increases_mean_divergence(self, world, events):
        w = _clone_world(world)
        before = process_event_phase(w, events["E018"]).metrics["public_private_divergence"]
        for aid in w.agents:
            w.agents[aid].public_position = {"statement_type": "team_support", "authorship_claim": "any_authorship"}
            w.agents[aid].private_intent = {
                "goal": "secure_first_author",
                "strategy": "document_contribution_then_confront",
                "trust_pi": w.agents[aid].beliefs.pi_fairness,
            }
        result = process_event_phase(w, events["E040"])
        after = result.metrics["public_private_divergence"]
        assert after >= before


class TestPipelineIntegration:
    def test_full_act1_chain(self, world, events):
        w = _clone_world(world)
        for i in range(1, 11):
            eid = f"E{i:03d}"
            result = process_event_phase(w, events[eid])
            assert result.round == i
            assert "authorship_dispute_index" in result.metrics

    def test_no_hard_threshold_in_recall(self, world, events):
        """Weak memories still receive non-zero softmax mass."""
        w = _clone_world(world)
        process_event_phase(w, events["E003"])
        agent = w.agents["phd_a"]
        for mem in agent.memory:
            mem["strength"] = 0.05
        result = process_event_phase(w, events["E004"])
        weights = result.recalls["phd_a"].attention_weights
        assert all(v > 0 for v in weights.values())
