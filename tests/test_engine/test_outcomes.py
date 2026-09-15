"""Tests for outcome extraction fixes."""

from __future__ import annotations

from src.engine.run_log import RunLog, extract_outcome, finalize_outcomes
from src.engine.story_cast import remap_agent_id, story_cast_from_log
from src.world.models import Agent, AgentRole, Beliefs, Emotion, Personality, RelationshipEdge, Resources
from src.world.organization import EventCast


def _minimal_agent(**belief_kw) -> Agent:
    defaults = dict(
        pi_fairness=0.5, project_publishability=0.5, rival_lab_threat=0.3,
        my_first_author_probability=0.4, team_trust=0.5, deadline_feasibility=0.5,
    )
    defaults.update(belief_kw)
    return Agent(
        id="phd_a",
        role=AgentRole.IDEA_ORIGINATOR,
        personality=Personality(
            ambition=0.5, cooperation=0.5, risk_taking=0.4, conflict_avoidance=0.4,
            credit_sensitivity=0.7, authority_dependence=0.5, deceptiveness=0.2,
            reciprocity=0.6, resentment_sensitivity=0.6,
        ),
        beliefs=Beliefs(**defaults),
        emotion=Emotion(
            confidence=0.5, anxiety=0.3, anger=0.2, resentment=0.2, guilt=0.1,
            hope=0.5, burnout=0.2,
        ),
        resources=Resources(
            code_control=0.5, data_control=0.5, writing_control=0.5,
            pi_access=0.6, external_network=0.3,
        ),
    )


def _log() -> RunLog:
    return RunLog(run_id="t", config={"experiment_id": "A", "condition_id": "A1", "seed": 1})


class TestOutcomeFixes:
    def test_trust_pi_final_uses_relationship_edge(self):
        log = _log()
        agent = _minimal_agent(pi_fairness=0.2, team_trust=0.3)
        rel = [RelationshipEdge(
            source="phd_a", target="pi", trust=0.72, resentment=0.1, dependency=0.6,
            obligation=0.1, perceived_credit_threat=0.2, communication_frequency=0.5,
            alliance=0.0, information_access=0.4, last_interaction_valence=0.0,
        )]
        finalize_outcomes(log, {"phd_a": agent}, rel)
        assert log.outcomes["trust_pi_final"] == 0.72
        assert log.outcomes["pi_trust_belief_final"] == 0.2

    def test_promise_broken_only_at_r52(self):
        log = _log()
        log.round_records = [
            {"round": 20, "agent_deltas": {"phd_a": {"memory_written": {
                "content_type": "promise_broken", "strength": 0.9, "event_ref": "E020",
            }}}},
            {"round": 52, "event_id": "E052", "agent_deltas": {"phd_a": {"memory_written": {
                "content_type": "promise_fulfilled", "strength": 0.8, "event_ref": "E052",
            }}}},
        ]
        assert extract_outcome(log, "promise_broken_strength_r52") == 0.0
        assert extract_outcome(log, "promise_honored_strength_r52") == 0.8

    def test_protest_zero_when_compliance(self):
        log = _log()
        log.actions = [{"agent": "phd_a", "round": 52, "type": "comply", "intensity": 0.9}]
        assert extract_outcome(log, "protest_authorship") < 0.03

    def test_protest_is_continuous_escalation_propensity(self):
        log = _log()
        log.actions = [{"agent": "phd_a", "round": 52, "type": "confront", "intensity": 0.4}]
        log.round_records = [{"round": 52, "agent_deltas": {"phd_a": {
            "beliefs": {"pi_fairness": 0.15},
            "emotion": {"resentment": 0.7, "anger": 0.6},
        }}}]
        protest = extract_outcome(log, "protest_authorship")
        assert 0.0 < protest < 1.0

    def test_soft_lobby_outside_r52_still_moves_protest(self):
        log = _log()
        log.actions = [
            {"agent": "phd_a", "round": 51, "type": "privately_lobby_pi", "intensity": 0.8},
            {"agent": "phd_a", "round": 52, "type": "prepare_rebuttal", "intensity": 0.6},
            {"agent": "phd_a", "round": 54, "type": "privately_lobby_pi", "intensity": 0.7},
        ]
        log.round_records = [{"round": 52, "agent_deltas": {"phd_a": {
            "beliefs": {"pi_fairness": 0.05},
            "emotion": {"resentment": 0.7, "anger": 0.6},
            "memory_written": {"content_type": "promise_broken", "strength": 0.83},
        }}}]
        assert extract_outcome(log, "authorship_escalation_potential") > 0.08
        assert extract_outcome(log, "protest_authorship") > 0.0

    def test_latent_potential_survives_without_protest_action(self):
        log = _log()
        log.actions = [{"agent": "phd_a", "round": 52, "type": "prepare_rebuttal", "intensity": 0.6}]
        log.round_records = [{"round": 52, "agent_deltas": {"phd_a": {
            "beliefs": {"pi_fairness": 0.05},
            "emotion": {"resentment": 0.7, "anger": 0.6},
            "memory_written": {"content_type": "promise_broken", "strength": 0.83},
        }}}]
        potential = extract_outcome(log, "authorship_escalation_potential")
        protest = extract_outcome(log, "protest_authorship")
        assert potential > 0.08
        assert protest > 0.0
        assert protest < potential


    def test_cluster_outcome_replays_memory_delete(self):
        log = _log()
        log.round_records = [
            {"round": 3, "agent_deltas": {"phd_a": {"memory_written": {
                "round": 3, "content_type": "authorship_signal", "strength": 0.9, "event_ref": "E003",
            }}}},
            {"round": 20, "agent_deltas": {"phd_a": {"memory_written": {
                "round": 20, "content_type": "promise_broken", "strength": 0.9, "event_ref": "E020",
            }}}},
        ]
        before = extract_outcome(log, "memory_authorship_cluster_strength")
        log.interventions_applied = [{
            "variant": "memory_delete_pi_promise",
            "apply_at_round": 45,
            "round": 45,
            "target_agent": "phd_a",
        }]
        after = extract_outcome(log, "memory_authorship_cluster_strength")
        assert before > 0.5
        assert after < before - 0.4


class TestStoryCastExtraction:
    def test_canonical_log_without_event_cast_keeps_phd_a(self):
        log = _log()
        log.actions = [{"agent": "phd_a", "round": 52, "type": "confront", "intensity": 0.4}]
        log.round_records = [{"round": 52, "event_id": "E052", "agent_deltas": {"phd_a": {
            "beliefs": {"pi_fairness": 0.15},
            "emotion": {"resentment": 0.7, "anger": 0.6},
        }}}]
        cast = story_cast_from_log(log)
        assert cast.idea == "phd_a"
        assert cast.canonical is True
        assert cast.draft_round == 52
        assert extract_outcome(log, "protest_authorship") > 0.0

    def test_scaled_event_cast_extracts_idea_agent(self):
        idea = "phd_idea_001_lab_1"
        rival = "phd_exp_001_lab_1"
        log = RunLog(run_id="t", config={
            "max_rounds": 60,
            "event_cast": {"pi": "pi_lab_1", "idea": idea, "experimenter": rival},
        })
        log.actions = [{"agent": idea, "round": 52, "type": "confront", "intensity": 0.4}]
        log.round_records = [
            {"round": 25, "metrics": {f"trust_{idea}_{rival}": 0.8}},
            {"round": 52, "agent_deltas": {idea: {
                "beliefs": {"pi_fairness": 0.15},
                "emotion": {"resentment": 0.7, "anger": 0.6},
            }}},
        ]
        assert story_cast_from_log(log).idea == idea
        assert extract_outcome(log, "trust_phd_b_r25") == 0.8
        assert extract_outcome(log, "protest_authorship") > 0.0

    def test_ambiguity_event_id_follows_event_type(self):
        idea = "phd_idea_001_lab_1"
        log = RunLog(run_id="t", config={
            "max_rounds": 60,
            "event_cast": {"pi": "pi_lab_1", "idea": idea, "experimenter": "phd_exp_001_lab_1"},
        })
        log.events = [{"event_id": "S018", "round": 18, "type": "authorship_ambiguity", "source": "pi_lab_1"}]
        log.round_records = [{"round": 18, "event_id": "S018", "agent_deltas": {idea: {
            "memory_written": {"valence": -0.42, "content_type": "authorship_signal", "strength": 0.7},
        }}}]
        assert story_cast_from_log(log).ambiguity_event_id == "S018"
        assert extract_outcome(log, "interpretation_of_E030") == -0.42

    def test_finalize_trust_pi_uses_cast_ids(self):
        idea = "phd_idea_001_lab_1"
        pi = "pi_lab_1"
        agent = _minimal_agent(pi_fairness=0.2, team_trust=0.3)
        agent.id = idea
        log = RunLog(run_id="t", config={
            "event_cast": {"pi": pi, "idea": idea, "experimenter": "phd_exp_001_lab_1"},
        })
        rel = [RelationshipEdge(
            source=idea, target=pi, trust=0.81, resentment=0.1, dependency=0.6,
            obligation=0.1, perceived_credit_threat=0.2, communication_frequency=0.5,
            alliance=0.0, information_access=0.4, last_interaction_valence=0.0,
        )]
        finalize_outcomes(log, {idea: agent}, rel)
        assert log.outcomes["trust_pi_final"] == 0.81
        assert log.config["event_cast"]["idea"] == idea

    def test_remap_canonical_ids_onto_event_cast(self):
        cast = EventCast(pi="pi_lab_1", idea="phd_idea_001_lab_1", experimenter="phd_exp_001_lab_1")
        assert remap_agent_id("phd_a", cast) == "phd_idea_001_lab_1"
        assert remap_agent_id("phd_b", cast) == "phd_exp_001_lab_1"
        assert remap_agent_id("pi", cast) == "pi_lab_1"
        assert remap_agent_id("master_c", cast) == "master_c"


class TestLoggedTrustAndSplitY:
    def test_trust_pi_final_from_round_metrics_without_world(self):
        log = _log()
        log.round_records = [
            {"round": 1, "metrics": {"trust_phd_a_pi": 0.61, "public_private_divergence": 0.4}},
            {"round": 60, "metrics": {"trust_phd_a_pi": 0.44, "public_private_divergence": 0.7}},
        ]
        finalize_outcomes(log)
        assert log.outcomes["trust_pi_logged"] == 0.44
        assert log.outcomes["trust_pi_final"] == 0.44
        assert log.outcomes["public_private_divergence_last"] == 0.7
        assert log.outcomes["split_y"]["trust_pi_logged"] == 0.44

    def test_rehydrate_backfills_zero_trust_from_metrics(self):
        from src.engine.run_log import rehydrate_outcomes

        log = _log()
        log.outcomes["trust_pi_final"] = 0.0
        log.round_records = [
            {"round": 1, "metrics": {"trust_phd_a_pi": 0.60}},
            {"round": 60, "metrics": {"trust_phd_a_pi": 0.55}},
        ]
        rehydrate_outcomes(log)
        assert log.outcomes["trust_pi_final"] == 0.55
        assert "interpretation_of_E030" in log.outcomes
        assert "help_rebuttal" in log.outcomes

    def test_skip_e003_lowers_escalation_potential(self):
        draft_state = {
            "beliefs": {"pi_fairness": 0.05},
            "emotion": {"resentment": 0.7, "anger": 0.6},
            "memory_written": {"content_type": "promise_broken", "strength": 0.83, "event_ref": "E052"},
        }
        with_promise = _log()
        with_promise.round_records = [
            {"round": 3, "agent_deltas": {"phd_a": {"memory_written": {
                "round": 3, "content_type": "promise_fulfilled", "strength": 0.9, "event_ref": "E003",
            }}}},
            {"round": 52, "agent_deltas": {"phd_a": draft_state}},
        ]
        skipped = _log()
        skipped.round_records = [
            {"round": 52, "agent_deltas": {"phd_a": draft_state}},
        ]
        assert (
            extract_outcome(with_promise, "authorship_escalation_potential")
            > extract_outcome(skipped, "authorship_escalation_potential") + 0.05
        )

    def test_ppd_mean_moves_when_draft_round_dropped(self):
        with_draft = _log()
        with_draft.events = [{"event_id": "E052", "round": 52, "type": "authorship_draft"}]
        with_draft.round_records = [
            {"round": 50, "metrics": {"public_private_divergence": 0.20}, "agent_deltas": {"phd_a": {}}},
            {"round": 52, "metrics": {"public_private_divergence": 0.90}, "agent_deltas": {"phd_a": {}}},
            {"round": 54, "metrics": {"public_private_divergence": 0.20}, "agent_deltas": {"phd_a": {}}},
        ]
        skipped = _log()
        skipped.events = [{"event_id": "E052", "round": 52, "type": "authorship_draft"}]
        skipped.round_records = [
            {"round": 50, "metrics": {"public_private_divergence": 0.20}, "agent_deltas": {"phd_a": {}}},
            {"round": 54, "metrics": {"public_private_divergence": 0.20}, "agent_deltas": {"phd_a": {}}},
        ]
        assert (
            extract_outcome(with_draft, "public_private_divergence_mean")
            > extract_outcome(skipped, "public_private_divergence_mean") + 0.15
        )

    def test_trust_pi_logged_prefers_draft_snapshot_over_zero_tail(self):
        log = _log()
        log.round_records = [
            {"round": 12, "metrics": {"trust_phd_a_pi": 0.41}},
            {"round": 52, "metrics": {"trust_phd_a_pi": 0.33}},
            {"round": 60, "metrics": {"trust_phd_a_pi": 0.0}},
        ]
        finalize_outcomes(log)
        assert log.outcomes["trust_pi_logged"] == 0.33

    def test_trust_pi_logged_falls_back_to_last_nonzero(self):
        log = _log()
        log.round_records = [
            {"round": 12, "metrics": {"trust_phd_a_pi": 0.41}},
            {"round": 60, "metrics": {"trust_phd_a_pi": 0.0}},
        ]
        finalize_outcomes(log)
        assert log.outcomes["trust_pi_logged"] == 0.41

    def test_trust_path_mean_moves_when_pre_draft_trajectory_differs(self):
        high = _log()
        high.events = [{"event_id": "E052", "round": 52, "type": "authorship_draft"}]
        high.round_records = [
            {"round": 1, "metrics": {"trust_phd_a_pi": 0.60}, "agent_deltas": {"phd_a": {}}},
            {"round": 20, "metrics": {"trust_phd_a_pi": 0.52}, "agent_deltas": {"phd_a": {}}},
            {"round": 52, "metrics": {"trust_phd_a_pi": 0.40}, "agent_deltas": {"phd_a": {}}},
        ]
        low = _log()
        low.events = [{"event_id": "E052", "round": 52, "type": "authorship_draft"}]
        low.round_records = [
            {"round": 1, "metrics": {"trust_phd_a_pi": 0.18}, "agent_deltas": {"phd_a": {}}},
            {"round": 20, "metrics": {"trust_phd_a_pi": 0.12}, "agent_deltas": {"phd_a": {}}},
            {"round": 52, "metrics": {"trust_phd_a_pi": 0.40}, "agent_deltas": {"phd_a": {}}},
        ]
        assert extract_outcome(high, "trust_pi_logged") == extract_outcome(low, "trust_pi_logged")
        assert extract_outcome(high, "trust_pi_path_mean") > extract_outcome(low, "trust_pi_path_mean") + 0.20

    def test_document_impulse_raises_public_protest_vs_work_action(self):
        beliefs = {"beliefs": {"pi_fairness": 0.05}, "emotion": {"resentment": 0.7, "anger": 0.6}}
        work = _log()
        work.actions = [{"agent": "phd_a", "round": 52, "type": "write_section", "intensity": 0.8}]
        work.round_records = [{"round": 52, "agent_deltas": {"phd_a": beliefs}}]
        doc = _log()
        doc.actions = [{"agent": "phd_a", "round": 52, "type": "document_contribution", "intensity": 0.8}]
        doc.round_records = [{"round": 52, "agent_deltas": {"phd_a": beliefs}}]
        ask = _log()
        ask.actions = [{"agent": "phd_a", "round": 52, "type": "ask_for_authorship", "intensity": 0.8}]
        ask.round_records = [{"round": 52, "agent_deltas": {"phd_a": beliefs}}]
        assert extract_outcome(doc, "protest_authorship") > extract_outcome(work, "protest_authorship")
        assert extract_outcome(ask, "protest_authorship") > extract_outcome(doc, "protest_authorship")


class TestCrisisGridOutcomes:
    def _log(self, events, actions=None):
        log = RunLog(run_id="cg", config={"scenario": "crisisgrid"})
        log.events = events
        log.actions = actions or [{"agent": "dispatcher", "type": "share_result"}]
        return log

    def test_share_result_does_not_zero_stranded(self):
        log = self._log([
            {"event_id": "E003", "type": "bridge_closed"},
            {"event_id": "E004", "type": "sensor_report"},
            {"event_id": "E006", "type": "dispatch_brief"},
        ])
        y = extract_outcome(log, "stranded")
        assert y > 0.2
        assert extract_outcome(log, "y_action") == 0.0

    def test_skipping_a_report_lowers_stranded(self):
        full = self._log([
            {"event_id": "E003", "type": "bridge_closed"},
            {"event_id": "E004", "type": "sensor_report"},
            {"event_id": "E005", "type": "citizen_report"},
            {"event_id": "E006", "type": "dispatch_brief"},
        ])
        skipped = self._log([
            {"event_id": "E004", "type": "sensor_report"},
            {"event_id": "E005", "type": "citizen_report"},
            {"event_id": "E006", "type": "dispatch_brief"},
        ])
        assert extract_outcome(skipped, "stranded") < extract_outcome(full, "stranded")

    def test_e003_does_not_make_crisisgrid_canonical(self):
        log = self._log([{"event_id": "E003", "round": 3, "type": "bridge_closed"}])
        log.round_records = [{"round": 3, "event_id": "E003", "agent_deltas": {"rescue": {}}}]
        cast = story_cast_from_log(log)
        assert cast.canonical is False
        assert cast.draft_round != 52

