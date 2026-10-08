"""Replay a frozen external log. This module does not import lab dynamics.

Interventions use the same four layer names as the decompiler
(do_event, do_visibility, do_memory, do_behavior). Effects come from the
rules stored in the log.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_trace(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _state(trace: dict[str, Any], intervention: dict[str, str] | None) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    events = set(trace["events"])
    visible = {event: set(agents) for event, agents in trace["visibility"].items()}
    memory = {agent: set(items) for agent, items in trace["memory"].items()}
    actions = dict(trace["actions"])
    if not intervention:
        return events, visible, memory, actions
    layer = intervention["layer"]
    target = intervention.get("target", "")
    agent = intervention.get("agent", "")
    if layer == "do_event":
        events.discard(target)
    elif layer == "do_visibility":
        visible.setdefault(target, set()).discard(agent)
    elif layer == "do_memory":
        memory.setdefault(agent, set()).discard(target)
    elif layer == "do_behavior":
        actions[agent] = intervention.get("to", "")
    else:
        raise ValueError(f"unknown layer {layer}")
    return events, visible, memory, actions


def _fires(rule: dict[str, Any], events: set[str], visible: dict[str, set[str]], memory: dict[str, set[str]], actions: dict[str, str]) -> bool:
    if "missing_event" in rule:
        return rule["missing_event"] not in events
    if "hidden_from" in rule:
        spec = rule["hidden_from"]
        return spec["agent"] not in visible.get(spec["event"], set())
    if "forgotten" in rule:
        spec = rule["forgotten"]
        return spec["event"] not in memory.get(spec["agent"], set())
    if "action_is" in rule:
        spec = rule["action_is"]
        return actions.get(spec["agent"]) == spec["value"]
    return False


def replay(trace: dict[str, Any], intervention: dict[str, str] | None = None) -> dict[str, Any]:
    events, visible, memory, actions = _state(trace, intervention)
    for rule in trace["program"]["rules"]:
        if _fires(rule, events, visible, memory, actions):
            return {"y": float(rule["y"]), "matched": True}
    return {"y": float(trace["program"]["else_y"]), "matched": False}


def external_claim(path: Path | str) -> str:
    """A log inside this repo is an interface check. A third-party claim needs an outside file."""
    resolved = Path(path).resolve()
    root = Path(__file__).resolve().parents[3]
    if resolved == root or root in resolved.parents:
        return "interface_check"
    return "third_party"


def external_mri(path: Path | str) -> dict[str, Any]:
    """Identity replay plus one intervention on each of the four layers."""
    trace = load_trace(path)
    factual = replay(trace)
    layers = [
        ("do_event", {"layer": "do_event", "target": "e1"}),
        ("do_visibility", {"layer": "do_visibility", "target": "e1", "agent": "a"}),
        ("do_memory", {"layer": "do_memory", "target": "e1", "agent": "a"}),
        ("do_behavior", {"layer": "do_behavior", "agent": "b", "to": "idle"}),
    ]
    effects = []
    for name, intervention in layers:
        twin = replay(trace, intervention)
        effects.append({
            "layer": name,
            "y": twin["y"],
            "ate": twin["y"] - factual["y"],
        })
    necessary = [item["layer"] for item in effects if item["ate"] != 0.0]
    return {
        "system": trace.get("system"),
        "identity_y": factual["y"],
        "identity_ok": factual["y"] == float(trace["y"]) and not factual["matched"],
        "effects": effects,
        "necessary_layers": necessary,
        "minimal_set_size": len(necessary),
        "claim": external_claim(path),
    }
