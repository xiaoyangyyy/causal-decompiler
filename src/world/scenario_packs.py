"""Write CrisisGrid and ReleaseOps scenario packs under config/scenarios/."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.world.loader import CONFIG_DIR

PERSONALITY = {
    "ambition": 0.55,
    "cooperation": 0.60,
    "risk_taking": 0.45,
    "conflict_avoidance": 0.40,
    "credit_sensitivity": 0.35,
    "authority_dependence": 0.40,
    "deceptiveness": 0.20,
    "reciprocity": 0.55,
    "resentment_sensitivity": 0.30,
}
BELIEFS = {
    "pi_fairness": 0.55,
    "project_publishability": 0.50,
    "rival_lab_threat": 0.25,
    "my_first_author_probability": 0.20,
    "team_trust": 0.55,
    "deadline_feasibility": 0.50,
    "my_contribution_recognized": 0.50,
    "others_are_free_riding": 0.30,
    "academic_integrity_risk": 0.20,
}
EMOTION = {
    "confidence": 0.55,
    "anxiety": 0.40,
    "anger": 0.15,
    "resentment": 0.15,
    "guilt": 0.10,
    "hope": 0.50,
    "burnout": 0.25,
}
RESOURCES = {
    "code_control": 0.40,
    "data_control": 0.40,
    "writing_control": 0.30,
    "pi_access": 0.40,
    "external_network": 0.35,
}
PROJECT = {
    "idea_clarity": 0.40,
    "experimental_strength": 0.30,
    "code_stability": 0.40,
    "writing_quality": 0.20,
    "novelty_risk": 0.30,
    "baseline_coverage": 0.30,
    "deadline_pressure": 0.55,
    "rival_threat": 0.20,
    "funding_pressure": 0.25,
    "authorship_conflict": 0.10,
    "team_morale": 0.65,
    "integrity_risk": 0.15,
}
LEDGER = {
    "idea": {"dispatcher": 0.5, "sensor_north": 0.5} if False else {"pi": 1.0},
}


def _agent(agent_id: str, role: str, name: str, goals: list[str], extra: dict | None = None) -> dict:
    return {
        "id": agent_id,
        "role": role,
        "display_name": name,
        "goals": goals,
        "personality": dict(PERSONALITY),
        "extra_traits": extra or {},
        "beliefs": dict(BELIEFS),
        "emotion": dict(EMOTION),
        "resources": dict(RESOURCES),
    }


def _event(eid: str, rnd: int, etype: str, source: str, targets: list[str], *, vis: str = "team", anchor: bool = False, desc: str = "") -> dict:
    return {
        "event_id": eid,
        "round": rnd,
        "type": etype,
        "visibility": vis,
        "source": source,
        "targets": targets,
        "payload": {},
        "objective_fact": {"raw_statement": desc or etype, "verifiable_claims": [etype]},
        "framing": "neutral",
        "truth_status": "verified",
        "memory_salience": 0.7,
        "is_anchor": anchor,
        "description": desc or etype,
    }


def crisisgrid_pack() -> dict:
    agents = [
        _agent("dispatcher", "pi", "Grid dispatcher", ["minimize_evac_time"], {"scenario_role": "dispatcher", "grid_cell": "hub"}),
        _agent("sensor_north", "engineer", "North sensor", ["report_hazards"], {"scenario_role": "sensor", "grid_cell": "N"}),
        _agent("sensor_south", "engineer", "South sensor", ["report_hazards"], {"scenario_role": "sensor", "grid_cell": "S"}),
        _agent("hospital", "collaborator", "Hospital", ["absorb_casualties"], {"scenario_role": "hospital", "grid_cell": "E"}),
        _agent("traffic", "experimenter", "Traffic control", ["keep_corridors_open"], {"scenario_role": "traffic", "grid_cell": "W"}),
        _agent("rescue", "idea_originator", "Rescue lead", ["reroute_around_bridge"], {"scenario_role": "rescue", "grid_cell": "C"}),
    ]
    internal = [a["id"] for a in agents]
    events = []
    types_cycle = [
        ("hospital_capacity", "hospital"),
        ("traffic_jam", "traffic"),
        ("team_meeting", "dispatcher"),
        ("sensor_report", "sensor_north"),
        ("citizen_report", "sensor_south"),
        ("dispatch_brief", "dispatcher"),
    ]
    for rnd in range(1, 25):
        etype, src = types_cycle[(rnd - 1) % len(types_cycle)]
        eid = f"E{rnd:03d}"
        events.append(_event(eid, rnd, etype, src, ["dispatcher", "rescue"], desc=etype, anchor=True))
    events[2] = _event("E003", 3, "bridge_closed", "traffic", ["dispatcher", "rescue"], vis="team", anchor=True, desc="bridge closed")
    events[7] = _event("E008", 8, "sensor_report", "sensor_north", ["dispatcher"], vis="bilateral", anchor=True, desc="sensor sees closure")
    events[8] = _event("E009", 9, "citizen_report", "sensor_south", ["dispatcher"], vis="public", anchor=True, desc="citizen reports jam")
    events[11] = _event("E012", 12, "dispatch_brief", "dispatcher", ["rescue", "hospital"], vis="team", anchor=True, desc="dispatcher belief update")
    events[17] = _event("E018", 18, "evac_order", "dispatcher", ["rescue", "traffic"], vis="team", anchor=True, desc="evac with or without reroute")
    world = {
        "world": {"days_per_round": 1, "total_rounds": 24, "total_days": 24, "target_conference": "CrisisGrid"},
        "internal_agents": internal,
        "external_agents": [],
        "mandatory_anchor_events": ["E003", "E008", "E009", "E012", "E018"],
        "initial_project": PROJECT,
        "initial_contribution_ledger": {
            "idea": {"dispatcher": 0.4, "rescue": 0.6},
            "experiments": {"sensor_north": 0.5, "sensor_south": 0.5},
            "writing": {"dispatcher": 1.0},
            "data": {"hospital": 0.5, "traffic": 0.5},
            "supervision": {"dispatcher": 1.0},
        },
        "scenario": "crisisgrid",
        "primary_outcome": "stranded",
        "topology": "spatial_comm",
    }
    return {"world": world, "agents": {"agents": agents}, "events": {"events": events}}


def releaseops_pack() -> dict:
    agents = [
        _agent("product", "pi", "Product owner", ["ship_on_time"], {"scenario_role": "product", "stage": "spec"}),
        _agent("developer", "engineer", "Developer", ["land_change"], {"scenario_role": "developer", "stage": "code"}),
        _agent("code_reviewer", "collaborator", "Code reviewer", ["gate_quality"], {"scenario_role": "reviewer", "stage": "review"}),
        _agent("tester", "experimenter", "Tester", ["catch_regressions"], {"scenario_role": "tester", "stage": "test"}),
        _agent("deployer", "postdoc", "Deployer", ["release_safely"], {"scenario_role": "deployer", "stage": "deploy"}),
        _agent("monitor", "idea_originator", "Monitor", ["page_on_outage"], {"scenario_role": "monitor", "stage": "observe"}),
    ]
    internal = [a["id"] for a in agents]
    events = []
    cycle = [
        ("team_meeting", "product"),
        ("stale_config", "developer"),
        ("test_skipped", "tester"),
        ("alert", "monitor"),
        ("rollback", "deployer"),
        ("deploy_failure", "deployer"),
    ]
    for rnd in range(1, 17):
        etype, src = cycle[(rnd - 1) % len(cycle)]
        events.append(_event(f"E{rnd:03d}", rnd, etype, src, ["product", "deployer"], desc=etype, anchor=True))
    events[2] = _event("E003", 3, "stale_config", "developer", ["tester", "deployer"], vis="team", anchor=True, desc="stale config merged")
    events[5] = _event("E006", 6, "test_skipped", "tester", ["deployer"], vis="bilateral", anchor=True, desc="regression skipped")
    events[8] = _event("E009", 9, "deploy_failure", "deployer", ["monitor", "product"], vis="team", anchor=True, desc="AND fault fires")
    events[10] = _event("E011", 11, "alert", "monitor", ["deployer"], vis="team", anchor=True, desc="alert can suppress outage")
    events[12] = _event("E013", 13, "rollback", "deployer", ["product"], vis="team", anchor=True, desc="rollback suppressor")
    world = {
        "world": {"days_per_round": 1, "total_rounds": 16, "total_days": 16, "target_conference": "ReleaseOps"},
        "internal_agents": internal,
        "external_agents": [],
        "mandatory_anchor_events": ["E003", "E006", "E009", "E011", "E013"],
        "initial_project": PROJECT,
        "initial_contribution_ledger": {
            "idea": {"product": 0.7, "developer": 0.3},
            "experiments": {"tester": 1.0},
            "writing": {"product": 1.0},
            "data": {"monitor": 1.0},
            "supervision": {"product": 0.6, "code_reviewer": 0.4},
        },
        "scenario": "releaseops",
        "primary_outcome": "task_y",
        "topology": "workflow_dag",
    }
    return {"world": world, "agents": {"agents": agents}, "events": {"events": events}}


def write_pack(name: str, pack: dict) -> Path:
    root = CONFIG_DIR / "scenarios" / name
    (root / "agents").mkdir(parents=True, exist_ok=True)
    (root / "events").mkdir(parents=True, exist_ok=True)
    (root / "world.yaml").write_text(yaml.safe_dump(pack["world"], sort_keys=False), encoding="utf-8")
    (root / "agents" / "profiles.yaml").write_text(yaml.safe_dump(pack["agents"], sort_keys=False), encoding="utf-8")
    (root / "events" / "anchors.yaml").write_text(yaml.safe_dump(pack["events"], sort_keys=False), encoding="utf-8")
    return root


def write_all() -> None:
    write_pack("crisisgrid", crisisgrid_pack())
    write_pack("releaseops", releaseops_pack())
    lab = CONFIG_DIR / "scenarios" / "labwars"
    lab.mkdir(parents=True, exist_ok=True)
    (lab / "README.md").write_text(
        "LabWars pack lives in `config/` (world.yaml, agents/, events/).\n"
        "`load_world('labwars')` falls back to that directory.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_all()
