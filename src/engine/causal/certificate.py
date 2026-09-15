"""Machine-checkable causal certificates from an MRI report.

Each evaluated intervention becomes a JSON object a debugger can store:
target, factual run, do(), split effects, first meaningful fork, interaction
set, and the identification assumptions that were actually used.
"""

from __future__ import annotations

from typing import Any

from src.engine.causal.search import harsanyi_from_shapley, minimal_sufficient_set

DEFAULT_ASSUMPTIONS = (
    "CRN identity twin on frozen U",
    "LLM prompt replay until the first prompt mismatch",
    "Memory IRF is an interventional analogue, not a natural indirect effect",
    "Minimal cause set is searched over evaluated ops, not a full do-algebra",
    "Reported fork is earliest meaningful (semantic/behavioral), not string mismatch",
)


def _fork_payload(fork: dict[str, Any] | None) -> dict[str, Any] | None:
    if not fork or fork.get("identical"):
        return None
    meaningful = fork.get("earliest_meaningful") or fork
    if not meaningful or meaningful.get("identical"):
        return None
    return {
        "round": meaningful.get("round"),
        "agent": meaningful.get("agent"),
        "channel": meaningful.get("channel"),
        "fork_kind": meaningful.get("fork_kind") or fork.get("fork_kind"),
        "paraphrase_only": bool(fork.get("paraphrase_only")),
    }


def _split_effects(item: dict[str, Any]) -> dict[str, float]:
    split = (item.get("extras") or {}).get("split") or {}
    out: dict[str, float] = {}
    for key in ("protest_authorship", "authorship_escalation_potential", "public_private_divergence_mean"):
        ate = (split.get(key) or {}).get("ate")
        if ate is None and key == "protest_authorship":
            ate = item.get("ate")
        if ate is not None:
            label = {"protest_authorship": "public", "authorship_escalation_potential": "potential", "public_private_divergence_mean": "private"}.get(key, key)
            out[label] = float(ate)
    if "public" not in out and item.get("ate") is not None:
        out["public"] = float(item.get("ate") or 0.0)
    return out


def _parse_intervention(factor_id: str, kind: str) -> dict[str, Any]:
    bits = [b for b in str(factor_id or "").split(":") if b]
    round_num = None
    factor = factor_id
    for bit in bits:
        if bit.startswith("r") and bit[1:].isdigit():
            round_num = int(bit[1:])
        elif bit not in {"EVENT_SKIP", "MEMORY_DELETE", "EVENT_OVERRIDE", "OBSERVE_LOCK", "MECHANISM_LESION", "POLICY_LAMBDA"}:
            factor = bit
    op_type = {
        "skip": "event_skip",
        "memory_irf": "memory_delete",
        "three_worlds": "event_skip",
        "identity": "noop",
    }.get(kind, kind)
    if "MEMORY" in str(factor_id):
        op_type = "memory_delete"
    elif "EVENT_SKIP" in str(factor_id) or kind == "skip":
        op_type = "event_skip"
    return {"type": op_type, "factor": factor, "round": round_num, "factor_id": factor_id}


def _cert(
    *,
    target: str,
    factual_run: str,
    intervention: dict[str, Any],
    effect: dict[str, float],
    fork: dict[str, Any] | None,
    interaction_set: list[str],
    replay_hashes: list[str],
) -> dict[str, Any]:
    return {
        "target": target,
        "factual_run": factual_run,
        "intervention": intervention,
        "effect": effect,
        "first_meaningful_fork": _fork_payload(fork),
        "interaction_set": interaction_set,
        "uncertainty": {},
        "assumptions": list(DEFAULT_ASSUMPTIONS),
        "replay_hashes": replay_hashes,
    }


def certificates_from_report(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One certificate per evaluated MRI intervention (plus the joint AND set)."""
    report = report or {}
    target = str(report.get("outcome") or "authorship_protest")
    run_id = str(report.get("factual_run_id") or "")
    story = report.get("story_shapley") or {}
    interaction_set = list(story.get("factors") or [])
    hashes = [h for h in (run_id,) if h]
    certs: list[dict[str, Any]] = []

    for item in report.get("memory_irf") or []:
        certs.append(_cert(
            target=target,
            factual_run=run_id,
            intervention=_parse_intervention(str(item.get("factor_id") or ""), "memory_irf"),
            effect=_split_effects(item),
            fork=(item.get("extras") or {}).get("fork"),
            interaction_set=interaction_set,
            replay_hashes=hashes,
        ))
    for item in report.get("contrastive") or []:
        certs.append(_cert(
            target=target,
            factual_run=run_id,
            intervention=_parse_intervention(str(item.get("factor_id") or ""), "skip"),
            effect=_split_effects(item) or {"public": float(item.get("ate") or 0.0)},
            fork=(item.get("extras") or {}).get("fork"),
            interaction_set=interaction_set,
            replay_hashes=hashes,
        ))
    for item in report.get("total_effects") or []:
        certs.append(_cert(
            target=target,
            factual_run=run_id,
            intervention=_parse_intervention(str(item.get("factor_id") or ""), str(item.get("name") or "op")),
            effect=_split_effects(item) or {"public": float(item.get("ate") or 0.0)},
            fork=(item.get("extras") or {}).get("fork"),
            interaction_set=interaction_set,
            replay_hashes=hashes,
        ))

    worlds = report.get("three_worlds") or {}
    if worlds:
        certs.append(_cert(
            target=target,
            factual_run=run_id,
            intervention=_parse_intervention(str(worlds.get("factor_id") or ""), "three_worlds"),
            effect={
                "public": float(worlds.get("ate_total") or 0.0),
                "gated": float(worlds.get("gated_channel") or 0.0),
            },
            fork=worlds.get("fork_w1"),
            interaction_set=interaction_set,
            replay_hashes=hashes + [str(rid) for rid in (worlds.get("run_ids") or {}).values() if rid],
        ))

    min_set = report.get("minimal_cause") or minimal_sufficient_set(report)
    interaction = report.get("interaction") or harsanyi_from_shapley(story)
    if min_set.get("set"):
        certs.append({
            "target": target,
            "factual_run": run_id,
            "intervention": {
                "type": "minimal_sufficient_set",
                "factor": ",".join(str(x) for x in min_set.get("set") or []),
                "round": None,
            },
            "effect": {"total": float(min_set.get("total_effect") or 0.0)},
            "first_meaningful_fork": None,
            "interaction_set": list(min_set.get("set") or interaction_set),
            "uncertainty": {},
            "assumptions": list(DEFAULT_ASSUMPTIONS),
            "replay_hashes": hashes,
            "minimal_cause": min_set,
            "harsanyi": interaction,
        })
    return certs
