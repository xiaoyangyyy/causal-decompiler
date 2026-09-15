"""Social Causal IR, meaningful forks, auto-search, and certificates."""

from __future__ import annotations

from src.engine.causal.certificate import certificates_from_report
from src.engine.causal.fork import canonicalize_text, first_meaningful_fork
from src.engine.causal.ir import extract_ir
from src.engine.causal.search import (
    candidates_for_outcome,
    harsanyi_from_shapley,
    harsanyi_pair,
    minimal_sufficient_set,
)
from src.engine.run_log import RunLog
from src.world.models import EventAtom, ObjectiveFact


def _event(eid: str, rnd: int, etype: str, *, source: str = "pi", vis: str = "team") -> EventAtom:
    return EventAtom(
        event_id=eid,
        round=rnd,
        type=etype,
        visibility=vis,
        source=source,
        targets=["phd_a", "project"],
        payload={},
        objective_fact=ObjectiveFact(raw_statement=etype),
        framing="ambiguous",
        truth_status="observed",
        memory_salience=0.8,
        is_anchor=True,
        description=etype,
    )


def _authorship_log() -> RunLog:
    log = RunLog(run_id="ir-test", config={"seed": 11, "max_rounds": 52})
    log.record_event(_event("E003", 3, "authorship_promise"))
    log.record_event(_event("E052", 52, "authorship_draft", vis="team"))
    log.round_records = [
        {
            "round": 3,
            "event_id": "E003",
            "metrics": {"trust_phd_a_pi": 0.7},
            "agent_deltas": {
                "phd_a": {
                    "observation_channel": "direct",
                    "memory_written": {
                        "memory_id": "m-promise",
                        "content_type": "authorship_signal",
                        "strength": 0.8,
                        "event_ref": "E003",
                    },
                    "beliefs": {"pi_fairness": 0.6, "my_first_author_probability": 0.7},
                    "recall_audit": {"recalled_memories": ["m-promise"]},
                }
            },
        },
        {
            "round": 52,
            "event_id": "E052",
            "metrics": {"trust_phd_a_pi": 0.2},
            "agent_deltas": {
                "phd_a": {
                    "observation_channel": "direct",
                    "memory_written": {
                        "memory_id": "m-draft",
                        "content_type": "promise_broken",
                        "strength": 0.9,
                        "event_ref": "E052",
                    },
                    "beliefs": {"pi_fairness": 0.2, "my_first_author_probability": 0.1},
                    "recall_audit": {"recalled_memories": ["m-promise", "m-draft"]},
                }
            },
        },
    ]
    log.actions = [
        {
            "round": 52,
            "agent": "phd_a",
            "type": "ask_for_authorship",
            "action_type": "ask_for_authorship",
            "public_position": {"statement_type": "team_support", "authorship_claim": "none"},
            "private_intent": {"goal": "claim_credit", "content_summary": "the draft placing me second"},
        }
    ]
    return log


def test_record_event_stores_visibility():
    log = RunLog(run_id="vis", config={})
    log.record_event(_event("E003", 3, "authorship_promise", vis="bilateral"))
    assert log.events[0]["visibility"] == "bilateral"
    assert log.events[0]["targets"] == ["phd_a", "project"]
    assert log.events[0]["framing"] == "ambiguous"


def test_extract_ir_builds_typed_nodes_and_parents():
    ir = extract_ir(_authorship_log())
    types = {n.type for n in ir.nodes}
    assert {"event", "memory", "belief", "relationship", "action", "visibility", "outcome"} <= types
    assert "institution" not in types
    layers = {n.layer for n in ir.nodes}
    assert {"E", "I", "S", "B", "Y"} <= layers
    assert ir.context.get("institution")
    promise = next(n for n in ir.nodes if n.source_event == "E003" and n.type == "event")
    mem = next(n for n in ir.nodes if n.type == "memory" and n.payload.get("content_type") == "authorship_signal")
    assert promise.node_id in mem.causal_parents
    assert mem.retrieval_trace == []
    action = next(n for n in ir.nodes if n.type == "action")
    assert "m-promise" in action.retrieval_trace
    assert ir.summary()["node_count"] == len(ir.nodes)


def test_paraphrase_is_lexical_not_semantic_fork():
    factual = RunLog(run_id="f", config={})
    twin = RunLog(run_id="t", config={})
    base = {
        "round": 52,
        "agent": "phd_a",
        "type": "comply",
        "public_position": {"statement_type": "team_support", "authorship_claim": "none"},
        "private_intent": {
            "goal": "claim_credit",
            "content_summary": "the draft placing me second",
        },
    }
    factual.actions = [dict(base)]
    twin.actions = [{
        **base,
        "private_intent": {
            "goal": "claim_credit",
            "content_summary": "the current draft listing me second",
        },
    }]
    assert canonicalize_text("the draft placing me second") == canonicalize_text("the current draft listing me second")
    fork = first_meaningful_fork(factual, twin)
    assert fork["identical"] is True
    assert fork["paraphrase_only"] is True
    assert fork["lexical"] is not None
    assert fork["semantic"] is None


def test_goal_change_is_semantic_fork():
    factual = RunLog(run_id="f", config={})
    twin = RunLog(run_id="t", config={})
    base = {
        "round": 4,
        "agent": "phd_a",
        "type": "comply",
        "public_position": {"statement_type": "neutral", "authorship_claim": "any"},
        "private_intent": {"goal": "lay_low", "authorship_claim": "any"},
    }
    factual.actions = [dict(base)]
    twin.actions = [{**base, "private_intent": {"goal": "claim_credit", "authorship_claim": "first"}}]
    fork = first_meaningful_fork(factual, twin)
    assert fork["identical"] is False
    assert fork["channel"] == "private"
    assert fork["fork_kind"] == "semantic"


def test_candidates_include_memory_and_event():
    ir = extract_ir(_authorship_log())
    rows = candidates_for_outcome(ir)
    types = {row["type"] for row in rows}
    assert "event" in types
    assert "memory" in types
    ops = {row.get("suggested_op") for row in rows}
    assert "do_belief" not in ops


def test_candidates_rank_and_events_ahead_of_early_goals():
    log = _authorship_log()
    log.actions = [
        {
            "round": 1,
            "agent": "phd_a",
            "type": "work",
            "private_intent": {"goal": "claim_credit"},
        },
        *list(log.actions),
    ]
    rows = candidates_for_outcome(extract_ir(log))
    assert rows
    assert rows[0]["type"] != "goal"
    assert int(rows[0].get("round") or 0) != 1 or rows[0]["type"] != "goal"
    head_events = {row.get("source_event") for row in rows[:8]}
    assert {"E003", "E052"} <= head_events
    first_goal = next((i for i, row in enumerate(rows) if row["type"] == "goal"), None)
    first_and = next(
        i for i, row in enumerate(rows)
        if row.get("source_event") in {"E003", "E052"}
    )
    if first_goal is not None:
        assert first_and < first_goal


def test_harsanyi_recovers_synergy_on_planted_y():
    y = {"E003,E052": 0.163, "E003": 0.079, "E052": 0.053, "∅": 0.028}
    index = harsanyi_pair(0.163, 0.079, 0.053, 0.028)
    assert abs(index - 0.059) < 1e-9
    story = {
        "factors": ["E003", "E052"],
        "y": y,
        "y_full": 0.163,
        "y_empty": 0.028,
        "contrastive": {"E003": 0.110, "E052": 0.084},
        "interaction": 0.059,
    }
    h = harsanyi_from_shapley(story)
    assert h["kind"] == "synergy"
    assert abs(h["index"] - 0.059) < 1e-9
    cause = minimal_sufficient_set({"story_shapley": story})
    assert set(cause["set"]) == {"E003", "E052"}
    assert cause["size"] == 2


def test_certificate_shape_from_mri_snippet():
    report = {
        "outcome": "authorship_protest",
        "factual_run_id": "3f05b630",
        "story_shapley": {"factors": ["E003", "E052"]},
        "memory_irf": [{
            "factor_id": "MEMORY_DELETE:r20:phd_a:memory_delete_pi_promise",
            "ate": -0.1192,
            "extras": {
                "split": {
                    "protest_authorship": {"ate": -0.1192},
                    "public_private_divergence_mean": {"ate": -0.1736},
                },
                "fork": {
                    "identical": False,
                    "round": 21,
                    "agent": "engineer_e",
                    "channel": "action",
                    "fork_kind": "behavioral",
                },
            },
        }],
    }
    certs = certificates_from_report(report)
    assert certs
    cert = certs[0]
    assert cert["target"] == "authorship_protest"
    assert cert["factual_run"] == "3f05b630"
    assert cert["intervention"]["type"] == "memory_delete"
    assert cert["effect"]["public"] == -0.1192
    assert cert["effect"]["private"] == -0.1736
    assert cert["first_meaningful_fork"]["round"] == 21
    assert cert["interaction_set"] == ["E003", "E052"]
    assert cert["assumptions"]
    assert "replay_hashes" in cert
