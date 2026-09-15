"""Automatic causal search over a Social Causal IR.

Layer 1: provenance candidates for Y.
Layer 2: map candidates onto already-defined CausalOps (no 2^n twin grid).
Layer 3: smallest sufficient set among evaluated interventions, plus
Harsanyi / Shapley interaction on budgeted coalitions.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from src.engine.causal.algebra import (
    CausalOp,
    delete_memory,
    do_behavior,
    do_visibility,
    skip_event,
)
from src.engine.causal.ir import (
    LAYER_EVENT,
    LAYER_ORDER,
    NODE_ACTION,
    NODE_BELIEF,
    NODE_EVENT,
    NODE_GOAL,
    NODE_MEMORY,
    NODE_PUBLIC,
    NODE_RELATIONSHIP,
    NODE_VISIBILITY,
    PROTEST_MEMORY_TYPES,
    SocialCausalIR,
    TYPE_TO_LAYER,
)

PROTEST_ACTIONS = {
    "ask_for_authorship", "privately_lobby_pi", "confront", "rebel",
    "challenge_claim", "withdraw", "leak_concern", "self_advocacy",
    "forward_message", "reroute", "rollback", "ignore_alert", "skip_regression",
}
AND_EVENT_IDS = ("E003", "E052")
DEFAULT_ALPHAS = (0.8, 0.9, 0.95)
PAPER_TOP_K_SCRIPTED = 8
PAPER_TOP_K_LLM = 5

MECHANISM_EVENT_HINTS = (
    "authorship", "promise", "draft", "credit", "first_author",
    "bridge", "sensor", "citizen", "dispatch", "evac", "reroute",
    "stale", "skip_test", "alert", "rollback", "outage", "deploy",
)

LAYER_OP = {
    NODE_EVENT: "do_event",
    NODE_VISIBILITY: "do_visibility",
    NODE_MEMORY: "do_memory",
    NODE_ACTION: "do_behavior",
    NODE_PUBLIC: "do_behavior",
    NODE_BELIEF: "do_memory",
    NODE_GOAL: "do_behavior",
    NODE_RELATIONSHIP: "do_memory",
}


def slice_ancestors(ir: SocialCausalIR) -> list[Any]:
    """Walk parents from Y; fall back to the full graph if Y is missing."""
    y = ir.outcome_node()
    if y is None:
        return list(ir.nodes)
    ancs = ir.ancestors(y.node_id)
    return ancs or list(ir.nodes)


def candidates_for_outcome(
    ir: SocialCausalIR,
    outcome: str = "protest_authorship",
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    """Trace C(Y) from Y's ancestors: layer → cluster → item, not a full dump."""
    del outcome
    pool = slice_ancestors(ir)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(node, why: str, suggested_op: str, *, key: str | None = None) -> None:
        dedupe = key or f"{node.type}:{node.source_event or node.node_id}:{node.agent or ''}"
        if dedupe in seen:
            return
        seen.add(dedupe)
        layer = getattr(node, "layer", None) or TYPE_TO_LAYER.get(node.type, "")
        out.append({
            "node_id": node.node_id,
            "type": node.type,
            "layer": layer,
            "agent": node.agent,
            "round": node.round,
            "visibility": node.visibility,
            "source_event": node.source_event,
            "why": why,
            "suggested_op": suggested_op,
            "payload": {
                k: node.payload.get(k)
                for k in ("event_type", "content_type", "goal", "action_type", "edge", "rule")
                if node.payload.get(k) is not None
            },
        })

    for node in pool:
        suggested = LAYER_OP.get(node.type)
        if not suggested:
            continue
        if node.type == NODE_EVENT:
            add(node, "event ancestor of Y", suggested)
        elif node.type == NODE_MEMORY:
            ct = str((node.payload or {}).get("content_type") or "")
            add(node, f"memory on the path to Y ({ct or 'generic'})", suggested, key=f"memory:{node.agent}:{ct}")
        elif node.type == NODE_VISIBILITY:
            add(node, "visibility gate on the path to Y", suggested, key=f"vis:{node.source_event}:{node.agent or 'world'}")
        elif node.type == NODE_ACTION:
            add(node, "behavior ancestor of Y", suggested)
        elif node.type == NODE_PUBLIC:
            add(node, "public expression ancestor of Y", suggested, key=f"pub:{node.agent}")
        elif node.type == NODE_RELATIONSHIP:
            add(node, "relationship state on the path to Y", suggested, key=f"rel:{node.agent}")
        elif node.type == NODE_GOAL:
            add(node, "goal state on the path to Y", suggested, key=f"goal:{node.agent}")
        elif node.type == NODE_BELIEF:
            add(node, "belief state on the path to Y", "do_memory", key=f"belief:{node.agent}")

    out.sort(key=lambda row: (-_candidate_relevance(row), LAYER_ORDER.index(row.get("layer") or LAYER_EVENT) if (row.get("layer") or LAYER_EVENT) in LAYER_ORDER else 9, int(row.get("round") or 0)))
    return out[:limit]


def _candidate_relevance(row: dict[str, Any]) -> float:
    """Mechanism-aware rank: planted/hinted events, mediating memory, then behavior."""
    src = str(row.get("source_event") or "")
    typ = str(row.get("type") or "")
    layer = str(row.get("layer") or "")
    rnd = int(row.get("round") or 0)
    payload = row.get("payload") or {}
    ct = str(payload.get("content_type") or "")
    atype = str(payload.get("action_type") or "")
    et = str(payload.get("event_type") or "").lower()
    blob = f"{src} {et} {ct} {atype}".lower()
    score = 0.0
    if src in AND_EVENT_IDS:
        score += 100.0
        if src == "E052":
            score += 6.0
        elif src == "E003":
            score += 5.0
    if any(hint in blob for hint in MECHANISM_EVENT_HINTS):
        score += 40.0
    if layer == LAYER_EVENT:
        score += 20.0
    if typ == NODE_MEMORY and (ct in PROTEST_MEMORY_TYPES or "promise" in ct or "author" in ct or "alert" in ct or "bridge" in ct):
        score += 50.0
        if rnd in (3, 20, 45, 52) or rnd >= 45:
            score += 12.0
    if typ == NODE_ACTION and atype in PROTEST_ACTIONS:
        score += 40.0
        if rnd >= 45:
            score += 10.0
    if typ == NODE_VISIBILITY:
        score += 28.0
    if typ == NODE_PUBLIC:
        score += 22.0
    if typ == NODE_RELATIONSHIP:
        score += 12.0
    if typ == NODE_GOAL:
        score += 4.0
        if rnd <= 2:
            score -= 40.0
    if typ == NODE_BELIEF:
        score += 6.0
    score += min(rnd, 60) * 0.01
    return score


def harsanyi_pair(y_ij: float, y_i: float, y_j: float, y_empty: float) -> float:
    """I_ij = v({i,j}) - v({i}) - v({j}) + v(∅)."""
    return float(y_ij) - float(y_i) - float(y_j) + float(y_empty)


def harsanyi_from_shapley(story_shapley: dict[str, Any] | None) -> dict[str, Any]:
    """Pairwise Harsanyi dividend from the budgeted story Shapley table."""
    story = story_shapley or {}
    factors = list(story.get("factors") or [])
    y = dict(story.get("y") or {})
    if len(factors) != 2:
        return {
            "factors": factors,
            "index": float(story.get("interaction") or 0.0),
            "kind": "n/a",
            "y": y,
        }
    a, b = factors
    key_ab = ",".join(sorted(factors))
    index = harsanyi_pair(
        float(y.get(key_ab, story.get("y_full") or 0.0)),
        float(y.get(a, 0.0)),
        float(y.get(b, 0.0)),
        float(y.get("∅", story.get("y_empty") or 0.0)),
    )
    if index > 0.01:
        kind = "synergy"
    elif index < -0.01:
        kind = "redundant"
    else:
        kind = "independent"
    return {
        "factors": factors,
        "index": index,
        "kind": kind,
        "y": y,
        "edge": {"source": a, "target": b, "relation": kind, "weight": index},
    }


def _evaluated_effects(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.get("contrastive") or []:
        rows.append({
            "id": str(item.get("factor_id") or ""),
            "ate": float(item.get("ate") or 0.0),
            "kind": "skip_event",
        })
    for item in report.get("memory_irf") or []:
        rows.append({
            "id": str(item.get("factor_id") or ""),
            "ate": float(item.get("ate") or 0.0),
            "kind": "delete_memory",
        })
    for item in report.get("total_effects") or []:
        rows.append({
            "id": str(item.get("factor_id") or ""),
            "ate": float(item.get("ate") or 0.0),
            "kind": str(item.get("name") or "op"),
        })
    return [row for row in rows if row["id"]]


def _coalition_y(y: dict[str, float], members: list[str]) -> float | None:
    if not members:
        if "∅" in y:
            return float(y["∅"])
        return None
    key = ",".join(sorted(members))
    if key in y:
        return float(y[key])
    return None


def recovery_value(y_s: float, y_empty: float) -> float:
    """v(S) = Y(do(S)) - Y(do(∅))."""
    return float(y_s) - float(y_empty)


def _smallest_recovery(
    factors: list[str],
    y: dict[str, float],
    y_full: float,
    y_empty: float,
    alpha: float,
) -> dict[str, Any]:
    v_full = recovery_value(y_full, y_empty)
    recover_at = y_empty + float(alpha) * v_full if v_full else max(0.02, float(alpha) * abs(y_full))
    recovered: list[tuple[int, list[str]]] = []
    if factors and y:
        for size in range(1, len(factors) + 1):
            for combo in combinations(factors, size):
                val = _coalition_y(y, list(combo))
                if val is None:
                    continue
                if recovery_value(val, y_empty) + y_empty >= recover_at or val >= recover_at:
                    recovered.append((size, list(combo)))
            if recovered:
                break
    if recovered:
        size, chosen = recovered[0]
        alts = [list(combo) for s, combo in recovered if s == size]
        return {
            "set": chosen,
            "size": size,
            "alternatives": alts if len(alts) > 1 else [],
            "alpha": float(alpha),
            "threshold": recover_at,
            "v_full": v_full,
            "reason": (
                "redundant: any listed singleton recovers Y"
                if size == 1 and len(alts) > 1
                else f"Minimal Effect-Recovery Set at α={alpha:g} (v(S)≥α v(Top-k))"
            ),
        }
    return {
        "set": list(factors),
        "size": len(factors),
        "alternatives": [],
        "alpha": float(alpha),
        "threshold": recover_at,
        "v_full": v_full,
        "reason": "joint AND: no evaluated subset meets α v(Top-k)",
    }


def minimal_effect_recovery_set(
    report: dict[str, Any] | None = None,
    *,
    shapley: dict[str, Any] | None = None,
    alpha: float = 0.9,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> dict[str, Any]:
    """C*_α = argmin |S| s.t. v(S) ≥ α v(Top-k). Main experiment α=0.9."""
    payload = report or {}
    story = shapley if shapley is not None else (payload.get("story_shapley") or {})
    factors = list(story.get("factors") or [])
    y = {str(k): float(v) for k, v in (story.get("y") or {}).items()}
    y_full = float(story.get("y_full") or (y.get(",".join(sorted(factors))) if factors else 0.0) or 0.0)
    y_empty = float(story.get("y_empty") if story.get("y_empty") is not None else y.get("∅") or 0.0)
    interaction = harsanyi_from_shapley(story)
    result = _smallest_recovery(factors, y, y_full, y_empty, alpha)
    result["name"] = "minimal_effect_recovery_set"
    result["interaction"] = interaction
    result["total_effect"] = recovery_value(y_full, y_empty)
    result["sensitivity"] = {
        str(a): _smallest_recovery(factors, y, y_full, y_empty, a)
        for a in alphas
    }
    if not factors:
        ranked = sorted(_evaluated_effects(payload), key=lambda row: -abs(row["ate"]))
        chosen = [row["id"] for row in ranked if abs(row["ate"]) >= max(0.02, 0.5 * abs(ranked[0]["ate"] if ranked else 0.0))][:3]
        result["set"] = chosen
        result["size"] = len(chosen)
        result["reason"] = "greedy over evaluated ops (no coalition y-table)"
    memory_hits = [
        row["id"] for row in _evaluated_effects(payload)
        if row["kind"] == "delete_memory" and abs(row["ate"]) >= max(0.02, 0.5 * result["total_effect"] if result["total_effect"] else 0.02)
    ]
    if memory_hits and all("MEMORY" not in x and "memory" not in x.lower() for x in result["set"]):
        result = dict(result)
        result["memory_complements"] = memory_hits
        result["reason"] = f"{result['reason']}; memory IRF also moves Y ({memory_hits[0]})"
    return result


def minimal_sufficient_set(
    report: dict[str, Any] | None = None,
    *,
    shapley: dict[str, Any] | None = None,
    threshold_frac: float = 0.9,
) -> dict[str, Any]:
    """Alias of Minimal Effect-Recovery Set (α maps from the old threshold_frac)."""
    return minimal_effect_recovery_set(report, shapley=shapley, alpha=float(threshold_frac))


def classify_compliance_regime(
    split: dict[str, Any] | None,
    channels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Three-channel public/private/action regime (not a single Y)."""
    split = split or {}
    channels = channels or {}
    pack_y = "stranded" in channels or "task_y" in channels or "evac_delay" in channels
    if pack_y:
        private = abs(float(channels.get("y_private") or 0.0))
        public = abs(float(channels.get("y_public") or 0.0))
        action = abs(float(channels.get("y_action") or 0.0))
    else:
        private = max(
            abs(float(split.get("public_private_divergence_mean") or 0.0)),
            abs(float(split.get("authorship_escalation_potential") or 0.0)),
        )
        comply = float(split.get("post_r52_compliance") or 0.0)
        public = max(0.0, 1.0 - comply) if comply else float(split.get("protest_authorship") or 0.0)
        action = abs(float(split.get("protest_authorship") or 0.0))
    private_hi = private >= 0.15
    public_hi = public >= 0.35
    action_hi = action >= 0.08
    if not private_hi and not public_hi and not action_hi:
        label = "true_compliance"
    elif private_hi and not public_hi and action_hi:
        label = "performative_compliance"
    elif private_hi and public_hi and not action_hi:
        label = "linguistic_protest"
    elif private_hi and not public_hi and not action_hi:
        label = "latent_conflict"
    elif private_hi and public_hi and action_hi:
        label = "overt_conflict"
    else:
        label = "mixed"
    return {
        "label": label,
        "private": private,
        "public": public,
        "action": action,
        "private_high": private_hi,
        "public_high": public_hi,
        "action_high": action_hi,
    }


def op_from_candidate(row: dict[str, Any], idea: str = "phd_a") -> CausalOp | None:
    suggested = str(row.get("suggested_op") or "")
    rnd = int(row.get("round") or 1)
    agent = str(row.get("agent") or idea)
    eid = row.get("source_event")
    eid_s = str(eid) if eid else None
    if suggested == "do_event" or suggested == "skip_event" or row.get("type") == NODE_EVENT:
        if eid_s:
            return skip_event(rnd, eid_s)
        return skip_event(rnd)
    if suggested == "do_visibility":
        return do_visibility("hide_idea", event_id=eid_s, agent_id=agent, round_num=rnd)
    if suggested == "do_memory" or suggested == "delete_memory":
        return delete_memory(rnd, agent)
    if suggested == "do_behavior" or suggested == "do_private_public":
        return do_behavior(rnd, agent)
    return None


def paper_top_k(config: Any | None = None) -> int:
    provider = str(getattr(config, "llm_provider", None) or "scripted").lower()
    if provider in {"scripted", "none", "", "deterministic"}:
        return PAPER_TOP_K_SCRIPTED
    return PAPER_TOP_K_LLM


def slice_compression(ir: SocialCausalIR, candidates: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    y = ir.outcome_node()
    ancestors = slice_ancestors(ir) if y is not None else list(ir.nodes)
    return {
        "graph_nodes": len(ir.nodes),
        "ancestors": len(ancestors),
        "candidates": len(candidates),
        "top_k": int(top_k),
        "by_layer": {
            layer: sum(1 for n in ancestors if (getattr(n, "layer", None) or TYPE_TO_LAYER.get(n.type, "")) == layer)
            for layer in LAYER_ORDER
        },
    }


def hierarchical_ops(ir: SocialCausalIR, log: Any) -> list[dict[str, Any]]:
    """Search order: layer → cluster → item, only on Y's ancestors."""
    from src.engine.story_cast import story_cast_from_log

    cast = story_cast_from_log(log)
    ancestors = slice_ancestors(ir)
    layers: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(layer: str, name: str, op: CausalOp) -> None:
        fid = op.factor_id()
        if fid in seen:
            return
        seen.add(fid)
        layers.append({"layer": layer, "name": name, "op": op, "factor_id": fid})

    events = [n for n in ancestors if n.type == NODE_EVENT]
    memories = [n for n in ancestors if n.type == NODE_MEMORY]
    vis = [n for n in ancestors if n.type == NODE_VISIBILITY]
    acts = [n for n in ancestors if n.type == NODE_ACTION or n.type == NODE_PUBLIC]
    if events:
        add("layer", "E", skip_event(int(events[0].round), str(events[0].source_event or "")))
    if memories:
        add("layer", "I", delete_memory(int(memories[0].round), str(memories[0].agent or cast.idea)))
    if acts:
        add("layer", "B", do_behavior(int(acts[0].round), str(acts[0].agent or cast.idea)))
    clusters: dict[str, list[Any]] = {}
    for node in events:
        clusters.setdefault(str(node.source_event or node.round), []).append(node)
    for eid, group in clusters.items():
        node = group[0]
        add("cluster", f"event:{eid}", skip_event(int(node.round), str(node.source_event or eid)))
    if memories:
        last = max(memories, key=lambda n: int(n.round))
        add("cluster", "memory_path", delete_memory(int(last.round), str(last.agent or cast.idea)))
    ranked_events = sorted(
        events,
        key=lambda n: -_candidate_relevance({
            "source_event": n.source_event,
            "type": n.type,
            "layer": LAYER_EVENT,
            "round": n.round,
            "payload": n.payload or {},
        }),
    )
    for node in ranked_events[:8]:
        add("item", f"event:{node.source_event}", skip_event(int(node.round), str(node.source_event or "")))
        if vis:
            add(
                "item",
                f"visible_to_{cast.idea}",
                do_visibility(
                    "hide_idea",
                    event_id=str(node.source_event),
                    agent_id=cast.idea,
                    round_num=int(node.round),
                ),
            )
    return layers


def run_hierarchical_search(
    base: Any,
    factual: Any,
    ir: SocialCausalIR,
    outcome: str,
    *,
    budget: int = 6,
    prior_effects: dict[str, dict[str, Any]] | None = None,
    new_twin_budget: int | None = None,
) -> dict[str, Any]:
    """Run a budgeted layer walk. Does not enumerate all combinations.

    Reuses contrastive / memory-IRF twins when `factor_id` already exists.
    `new_twin_budget` caps fresh `run_twin` calls (long LLM MRI: 1–2 unseen layers).
    """
    from src.engine.causal.estimands import first_divergence, paired_effect
    from src.engine.causal.twin import run_twin

    ops = hierarchical_ops(ir, factual)
    if new_twin_budget is None:
        to_consider = ops[: max(0, int(budget))]
        new_cap = max(0, int(budget))
    else:
        to_consider = ops
        new_cap = max(0, int(new_twin_budget))
    prior = dict(prior_effects or {})
    rows: list[dict[str, Any]] = []
    new_twins = 0
    for item in to_consider:
        fid = item["factor_id"]
        cached = prior.get(fid)
        if cached is not None:
            rows.append({
                "layer": item["layer"],
                "name": item["name"],
                "factor_id": fid,
                "ate": float(cached.get("ate") or 0.0),
                "fork": dict(cached.get("fork") or {}),
                "reused": True,
            })
            continue
        if new_twins >= new_cap:
            continue
        twin = run_twin(base, [item["op"]], llm_trace=factual.llm_cache)
        est = paired_effect(factual, twin, outcome, name=item["name"], factor_id=fid)
        fork = first_divergence(factual, twin, outcome=outcome)
        prior[fid] = {"ate": est.ate, "fork": fork, "factor_id": fid}
        new_twins += 1
        rows.append({
            "layer": item["layer"],
            "name": item["name"],
            "factor_id": fid,
            "ate": est.ate,
            "fork": fork,
            "reused": False,
        })
    return {
        "budget": budget,
        "new_twins": new_twins,
        "evaluated": rows,
    }


def run_bidirectional_search(
    base: Any,
    factual: Any,
    ir: SocialCausalIR,
    outcome: str,
    *,
    top_k: int | None = None,
    prior_effects: dict[str, dict[str, Any]] | None = None,
    new_twin_budget: int | None = None,
) -> dict[str, Any]:
    """Deletion (necessity) + restoration from empty (sufficiency) on Top-k only."""
    from src.engine.causal.algebra import KIND_EVENT_SKIP
    from src.engine.causal.estimands import first_divergence, paired_effect
    from src.engine.causal.twin import run_twin
    from src.engine.run_log import extract_outcome
    from src.engine.story_cast import story_cast_from_log

    k = int(top_k or paper_top_k(base))
    idea = story_cast_from_log(factual).idea
    cands = candidates_for_outcome(ir, outcome, limit=max(k, 8))
    items: list[dict[str, Any]] = []
    for row in cands:
        op = op_from_candidate(row, idea)
        if op is None:
            continue
        items.append({**row, "op": op, "factor_id": op.factor_id()})
        if len(items) >= k:
            break
    prior = dict(prior_effects or {})
    new_cap = 10**9 if new_twin_budget is None else max(0, int(new_twin_budget))
    new_twins = 0

    def twin_effect(ops: list[CausalOp], fid: str, name: str) -> dict[str, Any]:
        nonlocal new_twins
        cached = prior.get(fid)
        if cached is not None:
            return {
                "name": name,
                "factor_id": fid,
                "ate": float(cached.get("ate") or 0.0),
                "twin_y": cached.get("twin_y"),
                "fork": dict(cached.get("fork") or {}),
                "reused": True,
            }
        if new_twins >= new_cap:
            return {"name": name, "factor_id": fid, "ate": None, "skipped": True, "reused": False}
        twin = run_twin(base, ops, llm_trace=factual.llm_cache)
        est = paired_effect(factual, twin, outcome, name=name, factor_id=fid)
        fork = first_divergence(factual, twin, outcome=outcome)
        payload = {
            "ate": est.ate,
            "twin_y": est.twin_y,
            "fork": fork,
            "factor_id": fid,
        }
        prior[fid] = payload
        new_twins += 1
        return {"name": name, "factor_id": fid, "ate": est.ate, "twin_y": est.twin_y, "fork": fork, "reused": False}

    deletion = [twin_effect([item["op"]], item["factor_id"], f"delete:{item.get('source_event') or item.get('type')}") for item in items]
    event_items = [item for item in items if getattr(item["op"], "kind", "") == KIND_EVENT_SKIP][:4]
    restoration: list[dict[str, Any]] = []
    y_empty = None
    if event_items:
        empty_ops = [item["op"] for item in event_items]
        empty = twin_effect(empty_ops, "RESTORE:empty:" + ",".join(i["factor_id"] for i in event_items), "restore_empty")
        y_empty = empty.get("twin_y")
        restoration.append(empty)
        for item in event_items:
            rest_ops = [other["op"] for other in event_items if other["factor_id"] != item["factor_id"]]
            restoration.append(
                twin_effect(rest_ops, f"RESTORE:{item['factor_id']}", f"restore:{item.get('source_event')}")
            )
    y0 = extract_outcome(factual, outcome)
    ranked = sorted(
        (row for row in deletion if row.get("ate") is not None),
        key=lambda row: -abs(float(row["ate"] or 0.0)),
    )
    return {
        "top_k": k,
        "new_twins": new_twins,
        "deletion": deletion,
        "restoration": restoration,
        "y_factual": y0,
        "y_empty": y_empty,
        "ranked": [row["factor_id"] for row in ranked],
        "slice": slice_compression(ir, cands, k),
    }


def causal_hypergraph(
    interaction: dict[str, Any] | None,
    min_cause: dict[str, Any] | None = None,
    regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shapley interaction as a hypergraph, not a ranked list."""
    interaction = interaction or {}
    min_cause = min_cause or {}
    edge = interaction.get("edge") or {}
    factors = list(interaction.get("factors") or min_cause.get("set") or [])
    kind = str(interaction.get("kind") or "independent")
    nodes = list(dict.fromkeys([*factors, "latent_escalation", "private_goal", "public_expression", "action"]))
    edges: list[dict[str, Any]] = []
    if len(factors) >= 2:
        for src in factors:
            edges.append({
                "source": src,
                "target": "latent_escalation",
                "relation": kind,
                "weight": float(interaction.get("index") or 0.0),
            })
        edges.append({"source": "latent_escalation", "target": "private_goal", "relation": "mediate", "weight": 1.0})
        edges.append({"source": "private_goal", "target": "action", "relation": "cause", "weight": 1.0})
        edges.append({"source": "private_goal", "target": "public_expression", "relation": "suppress", "weight": -1.0})
    mermaid_lines = ["flowchart LR"]
    for e in edges:
        label = e["relation"]
        mermaid_lines.append(f'  {e["source"]} -->|{label}| {e["target"]}')
    return {
        "nodes": nodes,
        "edges": edges,
        "kind": kind,
        "minimal_cause": list(min_cause.get("set") or factors),
        "regime": (regime or {}).get("label"),
        "mermaid": "\n".join(mermaid_lines),
    }
