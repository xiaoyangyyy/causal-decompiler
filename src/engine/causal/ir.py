"""Social Causal IR: first-class nodes for do() on social variables.

A RunLog is still a transcript. This module turns it into a provenance graph
so later search can intervene on memory / event / visibility / relationship
nodes instead of deleting an unnamed string.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.engine.causal.noise import STREAM_ACTION_GUMBEL, noise_key
from src.engine.run_log import RunLog
from src.engine.story_cast import story_cast_from_log

MECHANISM_VERSION = "social_ir_v2"

NODE_EVENT = "event"
NODE_MEMORY = "memory"
NODE_BELIEF = "belief"
NODE_RELATIONSHIP = "relationship"
NODE_GOAL = "goal"
NODE_ACTION = "action"
NODE_PUBLIC = "public_expression"
NODE_VISIBILITY = "visibility"
NODE_INSTITUTION = "institution"
NODE_OUTCOME = "outcome"

LAYER_EVENT = "E"
LAYER_INFORMATION = "I"
LAYER_STATE = "S"
LAYER_BEHAVIOR = "B"
LAYER_OUTCOME = "Y"

TYPE_TO_LAYER = {
    NODE_EVENT: LAYER_EVENT,
    NODE_VISIBILITY: LAYER_INFORMATION,
    NODE_MEMORY: LAYER_INFORMATION,
    NODE_BELIEF: LAYER_STATE,
    NODE_RELATIONSHIP: LAYER_STATE,
    NODE_GOAL: LAYER_STATE,
    NODE_ACTION: LAYER_BEHAVIOR,
    NODE_PUBLIC: LAYER_BEHAVIOR,
    NODE_OUTCOME: LAYER_OUTCOME,
}

LAYER_ORDER = (LAYER_EVENT, LAYER_INFORMATION, LAYER_STATE, LAYER_BEHAVIOR, LAYER_OUTCOME)

PROTEST_MEMORY_TYPES = frozenset({
    "authorship_signal", "promise_fulfilled", "promise_broken", "credit_claim",
})


@dataclass
class IRNode:
    node_id: str
    type: str
    agent: str | None
    round: int
    visibility: str | None = None
    causal_parents: list[str] = field(default_factory=list)
    mechanism_version: str = MECHANISM_VERSION
    noise_key: str | None = None
    source_event: str | None = None
    retrieval_trace: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    layer: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["layer"] = self.layer or TYPE_TO_LAYER.get(self.type, "")
        return data


@dataclass
class SocialCausalIR:
    run_id: str
    mechanism_version: str
    nodes: list[IRNode]
    edges: list[tuple[str, str]]
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mechanism_version": self.mechanism_version,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "by_type": _count_types(self.nodes),
            "by_layer": _count_layers(self.nodes),
            "context": dict(self.context),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [{"parent": a, "child": b} for a, b in self.edges],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mechanism_version": self.mechanism_version,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "by_type": _count_types(self.nodes),
            "by_layer": _count_layers(self.nodes),
            "context": dict(self.context),
        }

    def outcome_node(self) -> IRNode | None:
        for node in reversed(self.nodes):
            if node.type == NODE_OUTCOME or node.layer == LAYER_OUTCOME:
                return node
        return None

    def get(self, node_id: str) -> IRNode | None:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def ancestors(self, node_id: str) -> list[IRNode]:
        parents = {child: [] for _, child in self.edges}
        for src, dst in self.edges:
            parents.setdefault(dst, []).append(src)
        seen: set[str] = set()
        stack = list(parents.get(node_id, []))
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(parents.get(nid, []))
        by_id = {n.node_id: n for n in self.nodes}
        return [by_id[i] for i in seen if i in by_id]


def _count_types(nodes: list[IRNode]) -> dict[str, int]:
    out: dict[str, int] = {}
    for node in nodes:
        out[node.type] = out.get(node.type, 0) + 1
    return out


def _count_layers(nodes: list[IRNode]) -> dict[str, int]:
    out: dict[str, int] = {layer: 0 for layer in LAYER_ORDER}
    for node in nodes:
        layer = node.layer or TYPE_TO_LAYER.get(node.type, "")
        if layer:
            out[layer] = out.get(layer, 0) + 1
    return out


def assign_layer(node: IRNode) -> IRNode:
    node.layer = TYPE_TO_LAYER.get(node.type, node.layer or "")
    return node


def _nid(*parts: Any) -> str:
    return ":".join(str(p) for p in parts if p is not None and str(p) != "")


def _noise_hex(seed: int, round_num: int, stream: str, agent_id: str | None, name: str) -> str:
    return format(noise_key(seed, round_num, stream, agent_id, name), "x")


def extract_ir(log: RunLog) -> SocialCausalIR:
    """Rebuild a Social Causal IR from a factual (or twin) transcript."""
    cast = story_cast_from_log(log)
    idea = cast.idea
    seed = int((log.config or {}).get("seed") or 0)
    nodes: list[IRNode] = []
    edges: list[tuple[str, str]] = []
    events_by_round: dict[int, dict[str, Any]] = {}

    for ev in log.events:
        rnd = int(ev.get("round") or 0)
        eid = str(ev.get("event_id") or f"R{rnd}")
        events_by_round[rnd] = ev
        vis = ev.get("visibility")
        event_id = _nid(NODE_EVENT, eid)
        nodes.append(IRNode(
            node_id=event_id,
            type=NODE_EVENT,
            agent=str(ev.get("source") or "") or None,
            round=rnd,
            visibility=str(vis) if vis else None,
            source_event=eid,
            payload={
                "event_type": ev.get("event_type") or ev.get("type"),
                "source": ev.get("source"),
                "targets": ev.get("targets") or [],
                "framing": ev.get("framing"),
            },
        ))
        vis_id = _nid(NODE_VISIBILITY, eid)
        nodes.append(IRNode(
            node_id=vis_id,
            type=NODE_VISIBILITY,
            agent=None,
            round=rnd,
            visibility=str(vis) if vis else None,
            causal_parents=[event_id],
            source_event=eid,
            payload={"channel": vis or ev.get("type")},
        ))
        edges.append((event_id, vis_id))

    institution_context = {
        "rule": "pi_decides_author_order",
        "agent": cast.pi,
        "round": int(cast.draft_round),
        "scenario": str((log.config or {}).get("scenario") or "labwars"),
    }

    mem_ids: dict[tuple[str, str], str] = {}
    belief_ids: dict[tuple[int, str], str] = {}

    for rec in log.round_records:
        rnd = int(rec.get("round") or 0)
        eid = str(rec.get("event_id") or (events_by_round.get(rnd) or {}).get("event_id") or f"R{rnd}")
        event_id = _nid(NODE_EVENT, eid)
        vis_id = _nid(NODE_VISIBILITY, eid)
        metrics = rec.get("metrics") or {}
        trust_key = f"trust_{idea}_{cast.pi}"
        if trust_key in metrics:
            rel_id = _nid(NODE_RELATIONSHIP, idea, cast.pi, rnd)
            parents = [p for p in (event_id, vis_id) if any(n.node_id == p for n in nodes)]
            nodes.append(IRNode(
                node_id=rel_id,
                type=NODE_RELATIONSHIP,
                agent=idea,
                round=rnd,
                visibility="private",
                causal_parents=parents,
                source_event=eid,
                payload={"edge": f"{idea}->{cast.pi}", "trust": float(metrics[trust_key])},
            ))
            for parent in parents:
                edges.append((parent, rel_id))
        for agent_id, delta in (rec.get("agent_deltas") or {}).items():
            if not isinstance(delta, dict):
                continue
            channel = delta.get("observation_channel")
            if channel:
                obs_vis = _nid(NODE_VISIBILITY, eid, agent_id)
                if not any(n.node_id == obs_vis for n in nodes):
                    parents = [p for p in (event_id, vis_id) if any(n.node_id == p for n in nodes)]
                    nodes.append(IRNode(
                        node_id=obs_vis,
                        type=NODE_VISIBILITY,
                        agent=str(agent_id),
                        round=rnd,
                        visibility=str(channel),
                        causal_parents=parents,
                        source_event=eid,
                        payload={"observation_channel": channel},
                    ))
                    for parent in parents:
                        edges.append((parent, obs_vis))
            mem = delta.get("memory_written")
            if isinstance(mem, dict) and mem.get("content_type"):
                mid = str(mem.get("memory_id") or _nid("mem", agent_id, rnd, mem.get("content_type")))
                node_id = _nid(NODE_MEMORY, mid)
                mem_ids[(str(agent_id), mid)] = node_id
                parents = [p for p in (event_id, vis_id) if any(n.node_id == p for n in nodes)]
                nodes.append(IRNode(
                    node_id=node_id,
                    type=NODE_MEMORY,
                    agent=str(agent_id),
                    round=rnd,
                    visibility=str(channel or "direct"),
                    causal_parents=parents,
                    source_event=str(mem.get("event_ref") or eid),
                    payload={
                        "content_type": mem.get("content_type"),
                        "strength": mem.get("strength"),
                        "event_ref": mem.get("event_ref"),
                    },
                ))
                for parent in parents:
                    edges.append((parent, node_id))
            beliefs = delta.get("beliefs")
            if isinstance(beliefs, dict) and agent_id == idea:
                bid = _nid(NODE_BELIEF, agent_id, rnd)
                belief_ids[(rnd, str(agent_id))] = bid
                parents = [p for p in (event_id,) if any(n.node_id == p for n in nodes)]
                mem_parent = next(
                    (n.node_id for n in reversed(nodes) if n.type == NODE_MEMORY and n.agent == agent_id and n.round == rnd),
                    None,
                )
                if mem_parent:
                    parents.append(mem_parent)
                nodes.append(IRNode(
                    node_id=bid,
                    type=NODE_BELIEF,
                    agent=str(agent_id),
                    round=rnd,
                    visibility="private",
                    causal_parents=parents,
                    source_event=eid,
                    payload={
                        "pi_fairness": beliefs.get("pi_fairness"),
                        "my_first_author_probability": beliefs.get("my_first_author_probability"),
                    },
                ))
                for parent in parents:
                    edges.append((parent, bid))

    actions_by_round: dict[int, list[dict[str, Any]]] = {}
    for act in log.actions:
        actions_by_round.setdefault(int(act.get("round") or 0), []).append(act)

    for rec in log.round_records:
        rnd = int(rec.get("round") or 0)
        eid = str(rec.get("event_id") or (events_by_round.get(rnd) or {}).get("event_id") or f"R{rnd}")
        event_id = _nid(NODE_EVENT, eid)
        deltas = rec.get("agent_deltas") or {}
        for act in actions_by_round.get(rnd, []):
            agent_id = str(act.get("agent") or "")
            atype = str(act.get("action_type") or act.get("type") or "")
            action_id = _nid(NODE_ACTION, agent_id, rnd, atype)
            recall = (deltas.get(agent_id) or {}).get("recall_audit") or {}
            recalled = [str(x) for x in (recall.get("recalled_memories") or [])]
            if recalled:
                mem_parents = [mem_ids[key] for key in mem_ids if key[0] == agent_id and key[1] in recalled]
            else:
                mem_parents = [
                    n.node_id for n in nodes
                    if n.type == NODE_MEMORY and n.agent == agent_id and n.round == rnd
                ]
            parents = [p for p in (event_id,) if any(n.node_id == p for n in nodes)]
            parents.extend(mem_parents[:6])
            belief_parent = belief_ids.get((rnd, agent_id))
            if belief_parent:
                parents.append(belief_parent)
            nodes.append(IRNode(
                node_id=action_id,
                type=NODE_ACTION,
                agent=agent_id,
                round=rnd,
                visibility="public",
                causal_parents=parents,
                mechanism_version=MECHANISM_VERSION,
                noise_key=_noise_hex(seed, rnd, STREAM_ACTION_GUMBEL, agent_id, atype or "a"),
                source_event=eid,
                retrieval_trace=recalled[:12],
                payload={"action_type": atype, "target": act.get("target"), "intensity": act.get("intensity")},
            ))
            for parent in parents:
                edges.append((parent, action_id))
            public = act.get("public_position") or {}
            pub_id = _nid(NODE_PUBLIC, agent_id, rnd)
            nodes.append(IRNode(
                node_id=pub_id,
                type=NODE_PUBLIC,
                agent=agent_id,
                round=rnd,
                visibility="public",
                causal_parents=[action_id],
                source_event=eid,
                payload={
                    "statement_type": public.get("statement_type"),
                    "authorship_claim": public.get("authorship_claim"),
                },
            ))
            edges.append((action_id, pub_id))
            private = act.get("private_intent") or {}
            if private:
                goal_id = _nid(NODE_GOAL, agent_id, rnd)
                nodes.append(IRNode(
                    node_id=goal_id,
                    type=NODE_GOAL,
                    agent=agent_id,
                    round=rnd,
                    visibility="private",
                    causal_parents=[action_id],
                    source_event=eid,
                    payload={
                        "goal": private.get("goal"),
                        "strategy": private.get("strategy"),
                    },
                ))
                edges.append((action_id, goal_id))

    for node in nodes:
        assign_layer(node)

    y_id = _nid(NODE_OUTCOME, str((log.config or {}).get("primary_outcome") or "Y"))
    behavior_parents = [
        n.node_id for n in nodes
        if n.layer == LAYER_BEHAVIOR
    ][-12:]
    y_node = assign_layer(IRNode(
        node_id=y_id,
        type=NODE_OUTCOME,
        agent=idea,
        round=int(log.round_records[-1].get("round") or 0) if log.round_records else 0,
        visibility="outcome",
        causal_parents=behavior_parents,
        payload={"outcome": str((log.config or {}).get("primary_outcome") or "Y")},
    ))
    nodes.append(y_node)
    for parent in behavior_parents:
        edges.append((parent, y_id))

    return SocialCausalIR(
        run_id=log.run_id,
        mechanism_version=MECHANISM_VERSION,
        nodes=nodes,
        edges=edges,
        context={"institution": institution_context},
    )
