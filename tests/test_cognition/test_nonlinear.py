"""Tests for nonlinear dynamics primitives."""

from __future__ import annotations

import copy

from src.cognition.belief import apply_action_belief_feedback
from src.cognition.dynamics import (
    action_escalation_impulse,
    authorship_memory_cluster,
    combine_escalation_score,
    escalation_potential_from_state,
    nonlinear_belief_target,
)
from src.engine.simulation import SimConfig, run_simulation
from src.world.loader import load_world


class TestNonlinearDynamics:
    def test_betrayal_shock_amplified_by_promise_cluster(self):
        prior = 0.6
        low = nonlinear_belief_target(prior, -0.35, 0.8, cluster_amp=0.1)
        high = nonlinear_belief_target(prior, -0.35, 0.8, cluster_amp=0.9)
        assert low > high

    def test_betrayal_with_cluster_bites_harder_than_promise_bump(self):
        prior = 0.5
        promise = nonlinear_belief_target(prior, 0.25, 0.9, positive_boost=1.0)
        betray = nonlinear_belief_target(prior, -0.25, 0.9, cluster_amp=0.8)
        assert promise > prior
        assert betray < prior
        assert (prior - betray) > (promise - prior)

    def test_positive_shocks_not_amplified_by_cluster(self):
        prior = 0.4
        low = nonlinear_belief_target(prior, 0.30, 0.9, cluster_amp=0.0)
        high = nonlinear_belief_target(prior, 0.30, 0.9, cluster_amp=0.9)
        assert abs(high - low) < 0.02

    def test_escalation_potential_saturates(self):
        low = escalation_potential_from_state(
            {"pi_fairness": 0.7}, {"resentment": 0.2, "anger": 0.1},
            promise_broken=0.1, promise_cluster=0.2,
        )
        high = escalation_potential_from_state(
            {"pi_fairness": 0.1}, {"resentment": 0.9, "anger": 0.8},
            promise_broken=0.9, promise_cluster=1.2,
        )
        assert high > low
        assert high <= 1.0
        assert combine_escalation_score(high, 0.0) < high
        assert combine_escalation_score(high, 0.0) > 0.0

    def test_document_contribution_carries_soft_impulse(self):
        assert action_escalation_impulse("document_contribution", 0.8) > 0.05
        assert action_escalation_impulse("ask_for_authorship", 0.8) > action_escalation_impulse(
            "document_contribution", 0.8,
        )
        assert action_escalation_impulse("write_section", 0.8) == 0.0

    def test_cluster_gate_moves_in_observed_range(self):
        shared = dict(
            beliefs={"pi_fairness": 0.05},
            emotion={"resentment": 0.70, "anger": 0.60},
            promise_broken=0.83,
        )
        low = escalation_potential_from_state(shared["beliefs"], shared["emotion"], promise_broken=0.83, promise_cluster=1.0)
        high = escalation_potential_from_state(shared["beliefs"], shared["emotion"], promise_broken=0.83, promise_cluster=6.4)
        assert high > low + 0.02

    def test_promise_and_draft_gate(self):
        beliefs = {"pi_fairness": 0.05}
        emotion = {"resentment": 0.70, "anger": 0.60}
        draft_only = escalation_potential_from_state(
            beliefs, emotion, promise_broken=0.9, promise_cluster=4.0, promise_anchor=0.0,
        )
        promise_and_draft = escalation_potential_from_state(
            beliefs, emotion, promise_broken=0.9, promise_cluster=4.0, promise_anchor=0.85,
        )
        assert promise_and_draft > draft_only + 0.08

    def test_action_feedback_moves_beliefs(self):
        world = load_world()
        agent = copy.deepcopy(world.agents["phd_a"])
        before = agent.beliefs.pi_fairness
        apply_action_belief_feedback(agent, "confront", 0.9)
        assert agent.beliefs.pi_fairness < before

    def test_a2_honor_draft_lowers_escalation_vs_a1(self):
        from src.engine.intervention import load_interventions

        inters = {i.intervention_id: i for i in load_interventions()}
        a1 = run_simulation(SimConfig(max_rounds=52, seed=7, interventions=[inters["INT_AUTH_BASELINE"]]))
        a2 = run_simulation(SimConfig(
            max_rounds=52, seed=7,
            interventions=[inters["INT_AUTH_EXPLICIT"], inters["INT_E052_HONOR"]],
        ))
        assert a2.outcomes["authorship_escalation_potential"] <= a1.outcomes["authorship_escalation_potential"] + 0.05

    def test_memory_cluster_is_saturating(self):
        agent = load_world().agents["phd_a"]
        agent.memory = [
            {"round": 3, "content_type": "authorship_signal", "strength": 0.9},
            {"round": 20, "content_type": "promise_broken", "strength": 0.9},
        ]
        cluster = authorship_memory_cluster(agent, round_min=1, round_max=40, current_round=40)
        assert cluster < 1.8
