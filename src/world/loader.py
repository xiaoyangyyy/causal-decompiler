"""LabWars world loader — load and validate configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .models import (
    Agent,
    EventAtom,
    ProjectMetrics,
    ProjectState,
    RelationshipEdge,
    WorldState,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CONFIG_DIR = PROJECT_ROOT / "config"
SCENARIOS_DIR = CONFIG_DIR / "scenarios"


def scenario_config_dir(scenario: str | None = None) -> Path:
    name = str(scenario or "labwars")
    pack = SCENARIOS_DIR / name
    if (pack / "world.yaml").exists():
        return pack
    return CONFIG_DIR


def load_world_config(scenario: str | None = None) -> dict[str, Any]:
    cfg = _load_yaml(scenario_config_dir(scenario) / "world.yaml")
    cfg.setdefault("scenario", scenario or "labwars")
    return cfg


def load_agents(scenario: str | None = None) -> list[Agent]:
    data = _load_yaml(scenario_config_dir(scenario) / "agents" / "profiles.yaml")
    agents: list[Agent] = []
    for raw in data["agents"]:
        validate_against_schema(raw, "agent.schema.json")
        agents.append(Agent.model_validate(raw))
    return agents


def load_events(scenario: str | None = None) -> list[EventAtom]:
    data = _load_yaml(scenario_config_dir(scenario) / "events" / "anchors.yaml")
    events: list[EventAtom] = []
    for raw in data["events"]:
        validate_against_schema(raw, "event.schema.json")
        events.append(EventAtom.model_validate(raw))
    return events


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json_schema(name: str) -> dict[str, Any]:
    with (SCHEMAS_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


def _schema_validator(name: str) -> Draft202012Validator:
    schema = _load_json_schema(name)
    return Draft202012Validator(schema)


def validate_against_schema(instance: dict[str, Any], schema_name: str) -> None:
    validator = _schema_validator(schema_name)
    validator.validate(instance)


def _default_edge(source: str, target: str) -> RelationshipEdge:
    """Generate baseline relationship edge between two internal agents."""
    # PI-centric: students depend on PI; phd_a/phd_b have mutual credit threat
    base_trust = 0.55
    dependency = 0.30
    credit_threat = 0.25
    resentment = 0.10

    if source == "pi" or target == "pi":
        dependency = 0.65 if source != "pi" else 0.20
        base_trust = 0.60
    if {source, target} == {"phd_a", "phd_b"}:
        credit_threat = 0.55
        base_trust = 0.42
        resentment = 0.20
    if "master_c" in (source, target):
        dependency = 0.55
        credit_threat = 0.15
    if "engineer_e" in (source, target):
        base_trust = 0.65
        credit_threat = 0.10
    if "visiting_f" in (source, target):
        base_trust = 0.35
        dependency = 0.15

    return RelationshipEdge(
        source=source,
        target=target,
        trust=round(base_trust, 2),
        resentment=round(resentment, 2),
        dependency=round(dependency, 2),
        obligation=0.15,
        perceived_credit_threat=round(credit_threat, 2),
        communication_frequency=0.45,
        alliance=0.0,
        information_access=0.40,
        last_interaction_valence=0.0,
    )


def build_initial_relationships(internal_agents: list[str]) -> list[RelationshipEdge]:
    edges: list[RelationshipEdge] = []
    for i, src in enumerate(internal_agents):
        for tgt in internal_agents:
            if src != tgt:
                edges.append(_default_edge(src, tgt))
    return edges


def load_initial_project(world_cfg: dict[str, Any]) -> ProjectState:
    raw = {
        "project": world_cfg["initial_project"],
        "contribution_ledger": world_cfg["initial_contribution_ledger"],
        "author_order_draft": [],
        "submission_status": "in_progress",
        "current_round": 0,
        "target_conference": world_cfg["world"]["target_conference"],
    }
    validate_against_schema(raw, "project.schema.json")
    return ProjectState(
        project=ProjectMetrics.model_validate(raw["project"]),
        contribution_ledger=raw["contribution_ledger"],
        author_order_draft=raw["author_order_draft"],
        submission_status=raw["submission_status"],
        current_round=raw["current_round"],
        target_conference=raw["target_conference"],
    )


def load_world(scenario: str | None = None) -> WorldState:
    world_cfg = load_world_config(scenario)
    agents = load_agents(scenario)
    internal = world_cfg["internal_agents"]
    relationships = build_initial_relationships(internal)
    project = load_initial_project(world_cfg)

    return WorldState(
        agents={a.id: a for a in agents},
        relationships=relationships,
        project=project,
        world_config=world_cfg,
    )


def validate_events_schedule(events: list[EventAtom], scenario: str | None = None) -> list[str]:
    """Return list of validation errors (empty if valid)."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    prev_round = 0

    for event in sorted(events, key=lambda e: e.round):
        if event.event_id in seen_ids:
            errors.append(f"Duplicate event_id: {event.event_id}")
        seen_ids.add(event.event_id)

        if event.round < prev_round:
            errors.append(f"Round not monotonic at {event.event_id}: {event.round} < {prev_round}")
        prev_round = event.round

    world_cfg = load_world_config(scenario)
    mandatory = set(world_cfg["mandatory_anchor_events"])
    anchor_ids = {e.event_id for e in events if e.is_anchor}
    missing = mandatory - anchor_ids
    if missing:
        errors.append(f"Missing mandatory anchors: {sorted(missing)}")

    expected = int(world_cfg.get("world", {}).get("total_rounds") or 60)
    if len(events) != expected:
        errors.append(f"Expected {expected} events, got {len(events)}")

    return errors


def validate_relationship_coverage(
    relationships: list[RelationshipEdge], internal_agents: list[str]
) -> list[str]:
    errors: list[str] = []
    expected = {(s, t) for s in internal_agents for t in internal_agents if s != t}
    actual = {(e.source, e.target) for e in relationships}
    missing = expected - actual
    if missing:
        errors.append(f"Missing relationship edges: {len(missing)} pairs")
    return errors
