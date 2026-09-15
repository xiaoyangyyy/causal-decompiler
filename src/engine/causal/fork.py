"""Layered forks: lexical vs semantic vs behavioral vs outcome.

Earliest string mismatch is not the paper object. Report earliest
meaningful fork (intent / action / outcome), and keep the lexical hit
in extras so paraphrase cannot look like a mechanism change.
"""

from __future__ import annotations

import re
from typing import Any

from src.engine.run_log import RunLog, extract_outcome

_WORD = re.compile(r"[a-z0-9]+")

_PHRASES = (
    ("placing me second", "second_author"),
    ("listing me second", "second_author"),
    ("draft placing me", "second_author"),
    ("draft listing me", "second_author"),
    ("secure first authorship", "seek_first_author"),
    ("first authorship", "seek_first_author"),
    ("first-author", "seek_first_author"),
    ("first author", "seek_first_author"),
    ("team support", "team_support"),
    ("self advocacy", "self_advocacy"),
    ("self-advocacy", "self_advocacy"),
    ("lay low", "lay_low"),
    ("claim credit", "claim_credit"),
)

_STOP = frozenset({
    "the", "a", "an", "on", "of", "to", "for", "and", "in", "with",
    "despite", "current", "paper", "that", "me", "my",
})


def canonicalize_text(value: Any) -> str:
    text = str(value or "").lower().replace("_", " ")
    for src, dst in _PHRASES:
        text = text.replace(src, f" {dst} ")
    tokens = [tok for tok in _WORD.findall(text) if tok not in _STOP]
    return " ".join(tokens)


def _stance_fields(payload: Any) -> tuple[str, str, str, str]:
    if not isinstance(payload, dict):
        raw = str(payload)
        return (raw, "", "", "")
    return (
        str(payload.get("statement_type") or ""),
        str(payload.get("authorship_claim") or payload.get("strategy") or ""),
        str(payload.get("goal") or ""),
        str(payload.get("content_summary") or ""),
    )


def stance_key(payload: Any) -> tuple[str, ...]:
    return _stance_fields(payload)


def semantic_stance_key(payload: Any) -> tuple[str, ...]:
    return tuple(canonicalize_text(part) for part in _stance_fields(payload))


def _action_index(log: RunLog) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for act in log.actions:
        key = (int(act.get("round") or 0), str(act.get("agent") or ""))
        out[key] = act
    return out


def _hit(kind: str, rnd: int, agent: str, channel: str, factual: Any, twin: Any) -> dict[str, Any]:
    return {
        "identical": False,
        "fork_kind": kind,
        "round": rnd,
        "agent": agent,
        "channel": channel,
        "factual": factual,
        "twin": twin,
    }


def classify_forks(factual: RunLog, twin: RunLog, *, outcome: str = "protest_authorship") -> dict[str, Any]:
    """Walk the paired transcript and label the first hit of each layer."""
    fact = _action_index(factual)
    other = _action_index(twin)
    lexical = semantic = behavioral = None
    for key in sorted(set(fact) | set(other)):
        rnd, agent = key
        fa = fact.get(key)
        ta = other.get(key)
        if fa is None or ta is None:
            hit = _hit(
                "behavioral",
                rnd,
                agent,
                "presence",
                None if fa is None else str(fa.get("type")),
                None if ta is None else str(ta.get("type")),
            )
            behavioral = behavioral or hit
            lexical = lexical or hit
            semantic = semantic or hit
            break
        if str(fa.get("type")) != str(ta.get("type")):
            hit = _hit("behavioral", rnd, agent, "action", str(fa.get("type")), str(ta.get("type")))
            behavioral = behavioral or hit
            lexical = lexical or hit
            semantic = semantic or hit
            continue
        pub_l = stance_key(fa.get("public_position")) != stance_key(ta.get("public_position"))
        pub_s = semantic_stance_key(fa.get("public_position")) != semantic_stance_key(ta.get("public_position"))
        if pub_l and lexical is None:
            lexical = _hit(
                "lexical", rnd, agent, "public",
                stance_key(fa.get("public_position")),
                stance_key(ta.get("public_position")),
            )
        if pub_s and semantic is None:
            semantic = _hit(
                "semantic", rnd, agent, "public",
                semantic_stance_key(fa.get("public_position")),
                semantic_stance_key(ta.get("public_position")),
            )
        priv_l = stance_key(fa.get("private_intent")) != stance_key(ta.get("private_intent"))
        priv_s = semantic_stance_key(fa.get("private_intent")) != semantic_stance_key(ta.get("private_intent"))
        if priv_l and lexical is None:
            lexical = _hit(
                "lexical", rnd, agent, "private",
                stance_key(fa.get("private_intent")),
                stance_key(ta.get("private_intent")),
            )
        if priv_s and semantic is None:
            semantic = _hit(
                "semantic", rnd, agent, "private",
                semantic_stance_key(fa.get("private_intent")),
                semantic_stance_key(ta.get("private_intent")),
            )
        if behavioral is not None and semantic is not None and lexical is not None:
            break

    y0 = extract_outcome(factual, outcome)
    y1 = extract_outcome(twin, outcome)
    outcome_hit = None
    if abs(y1 - y0) >= 0.02:
        outcome_hit = {
            "identical": False,
            "fork_kind": "outcome",
            "round": None,
            "agent": None,
            "channel": "outcome",
            "factual": y0,
            "twin": y1,
        }

    meaningful = behavioral or semantic or outcome_hit
    transcript_identical = behavioral is None and semantic is None
    reported_identical = transcript_identical and outcome_hit is None
    return {
        "identical": bool(reported_identical),
        "lexical": lexical,
        "semantic": semantic,
        "behavioral": behavioral,
        "outcome": outcome_hit,
        "earliest_meaningful": None if transcript_identical and outcome_hit is None else meaningful,
        "paraphrase_only": bool(lexical is not None and transcript_identical),
    }


def flatten_meaningful_fork(layers: dict[str, Any]) -> dict[str, Any]:
    """Paper-facing fork row: meaningful hit on top, layered extras underneath."""
    meaningful = layers.get("earliest_meaningful")
    if layers.get("identical") or meaningful is None:
        base: dict[str, Any] = {
            "identical": True,
            "round": None,
            "agent": None,
            "channel": None,
            "factual": None,
            "twin": None,
            "fork_kind": None,
        }
    else:
        base = {
            "identical": False,
            "round": meaningful.get("round"),
            "agent": meaningful.get("agent"),
            "channel": meaningful.get("channel"),
            "factual": meaningful.get("factual"),
            "twin": meaningful.get("twin"),
            "fork_kind": meaningful.get("fork_kind"),
        }
    for key in ("lexical", "semantic", "behavioral", "outcome", "earliest_meaningful", "paraphrase_only"):
        base[key] = layers.get(key)
    return base


def first_meaningful_fork(factual: RunLog, twin: RunLog, *, outcome: str = "protest_authorship") -> dict[str, Any]:
    return flatten_meaningful_fork(classify_forks(factual, twin, outcome=outcome))
