"""Story-role mapping for experiment A–D outcome extraction.

The canonical 14-agent MRI story keeps phd_a / E030 / round 52. Scaled
populations reuse the same outcome names, resolving agents from EventCast
and story beats from event types plus max_rounds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.world.organization import EventCast

CANONICAL_EVENT_IDS = frozenset({
    "E003", "E020", "E025", "E030", "E031", "E038", "E040", "E047", "E052",
})

ROLE_ALIASES = {
    "phd_a": "idea",
    "phd_b": "experimenter",
    "pi": "pi",
    "postdoc_d": "postdoc",
    "engineer_e": "engineer",
    "lab_alumni": "alumni",
    "rival_lab_h": "rival",
}


@dataclass(frozen=True)
class StoryCast:
    """Resolved cast + story beats used by extract_outcome / metrics."""

    pi: str = "pi"
    idea: str = "phd_a"
    experimenter: str = "phd_b"
    postdoc: str | None = None
    engineer: str | None = None
    alumni: str | None = None
    rival: str | None = None
    canonical: bool = True
    draft_round: int = 52
    ambiguity_event_id: str = "E030"
    withdraw_round: int = 47
    withdraw_event_id: str = "E047"
    fairness_mid_round: int = 35
    trust_rounds: tuple[int, int, int] = (25, 44, 60)
    memory_cluster_min: int = 3
    memory_cluster_max: int = 40
    protest_end: int = 55
    compliance_end: int = 53
    help_rebuttal_start: int = 57
    document_start: int = 18
    document_end: int = 25
    demand_start: int = 47
    snapshot_rounds: tuple[int, ...] = (1, 20, 40, 60)

    @property
    def trust_early(self) -> int:
        return self.trust_rounds[0]

    @property
    def trust_mid(self) -> int:
        return self.trust_rounds[1]

    @property
    def trust_final(self) -> int:
        return self.trust_rounds[2]

    def event_cast_dict(self) -> dict[str, Any]:
        return {
            "pi": self.pi,
            "idea": self.idea,
            "experimenter": self.experimenter,
            "postdoc": self.postdoc,
            "engineer": self.engineer,
            "alumni": self.alumni,
            "rival": self.rival,
        }

    def beats_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "draft_round": self.draft_round,
            "ambiguity_event_id": self.ambiguity_event_id,
            "withdraw_round": self.withdraw_round,
            "withdraw_event_id": self.withdraw_event_id,
            "fairness_mid_round": self.fairness_mid_round,
            "trust_rounds": list(self.trust_rounds),
            "memory_cluster_min": self.memory_cluster_min,
            "memory_cluster_max": self.memory_cluster_max,
            "protest_end": self.protest_end,
            "compliance_end": self.compliance_end,
            "help_rebuttal_start": self.help_rebuttal_start,
            "document_start": self.document_start,
            "document_end": self.document_end,
            "demand_start": self.demand_start,
            "snapshot_rounds": list(self.snapshot_rounds),
        }


def remap_agent_id(agent_id: str | None, cast: EventCast | StoryCast | Mapping[str, Any] | None) -> str | None:
    """Map canonical MRI ids onto the live EventCast when present."""
    if not agent_id or cast is None:
        return agent_id
    attr = ROLE_ALIASES.get(agent_id)
    if not attr:
        return agent_id
    if isinstance(cast, Mapping):
        mapped = cast.get(attr)
    else:
        mapped = getattr(cast, attr, None)
    return mapped or agent_id


def event_cast_asdict(cast: EventCast) -> dict[str, Any]:
    return asdict(cast)


def story_cast_from_event_cast(cast: EventCast, *, canonical: bool = True, max_rounds: int = 60) -> StoryCast:
    beats = _canonical_beats() if canonical else _scaled_beats(max_rounds)
    return StoryCast(
        pi=cast.pi,
        idea=cast.idea,
        experimenter=cast.experimenter,
        postdoc=cast.postdoc,
        engineer=cast.engineer,
        alumni=cast.alumni,
        rival=cast.rival,
        **beats,
    )


def story_cast_from_log(log: Any, world_agents: Mapping[str, Any] | None = None) -> StoryCast:
    ids = _agent_ids_in_log(log, world_agents)
    cfg_cast = dict(log.config.get("event_cast") or {})
    idea = cfg_cast.get("idea") or ("phd_a" if "phd_a" in ids else _most_frequent_actor(log) or _first_id(ids, "phd_a"))
    pi = cfg_cast.get("pi") or ("pi" if "pi" in ids else _first_prefix(ids, "pi", "pi"))
    experimenter = cfg_cast.get("experimenter") or (
        "phd_b" if "phd_b" in ids else _first_other(ids, {idea, pi}, idea)
    )
    canonical = _is_canonical_story(log, ids)
    beats = _canonical_beats() if canonical else _beats_from_log(log)
    beats.update(_beats_from_config(log.config.get("story_beats") or {}))
    return StoryCast(
        pi=pi,
        idea=idea,
        experimenter=experimenter,
        postdoc=cfg_cast.get("postdoc"),
        engineer=cfg_cast.get("engineer"),
        alumni=cfg_cast.get("alumni"),
        rival=cfg_cast.get("rival"),
        **beats,
    )


def _canonical_beats() -> dict[str, Any]:
    return {
        "canonical": True,
        "draft_round": 52,
        "ambiguity_event_id": "E030",
        "withdraw_round": 47,
        "withdraw_event_id": "E047",
        "fairness_mid_round": 35,
        "trust_rounds": (25, 44, 60),
        "memory_cluster_min": 3,
        "memory_cluster_max": 40,
        "protest_end": 55,
        "compliance_end": 53,
        "help_rebuttal_start": 57,
        "document_start": 18,
        "document_end": 25,
        "demand_start": 47,
        "snapshot_rounds": (1, 20, 40, 60),
    }


def _scale_round(round_on_60: int, max_rounds: int) -> int:
    if max_rounds <= 0:
        return round_on_60
    return max(1, min(max_rounds, int(round(round_on_60 * max_rounds / 60))))


def _scaled_beats(max_rounds: int, *, ambiguity_id: str = "E030", withdraw_id: str = "E047") -> dict[str, Any]:
    draft = _scale_round(52, max_rounds)
    return {
        "canonical": False,
        "draft_round": draft,
        "ambiguity_event_id": ambiguity_id,
        "withdraw_round": _scale_round(47, max_rounds),
        "withdraw_event_id": withdraw_id,
        "fairness_mid_round": _scale_round(35, max_rounds),
        "trust_rounds": (
            _scale_round(25, max_rounds),
            _scale_round(44, max_rounds),
            max_rounds,
        ),
        "memory_cluster_min": min(3, max_rounds),
        "memory_cluster_max": _scale_round(40, max_rounds),
        "protest_end": min(max_rounds, draft + 3),
        "compliance_end": min(max_rounds, draft + 1),
        "help_rebuttal_start": _scale_round(57, max_rounds),
        "document_start": _scale_round(18, max_rounds),
        "document_end": _scale_round(25, max_rounds),
        "demand_start": _scale_round(47, max_rounds),
        "snapshot_rounds": (
            1,
            max(1, max_rounds // 3),
            max(1, (2 * max_rounds) // 3),
            max_rounds,
        ),
    }


def _beats_from_log(log: Any) -> dict[str, Any]:
    max_rounds = _max_round(log)
    amb_id, _amb_r = _first_event_of_type(log, "authorship_ambiguity")
    _draft_id, draft_r = _last_event_of_type(log, "authorship_draft")
    withdraw_id, withdraw_r = _first_event_of_type(log, "threat_withdraw")
    beats = _scaled_beats(
        max_rounds,
        ambiguity_id=amb_id or "E030",
        withdraw_id=withdraw_id or "E047",
    )
    if draft_r:
        beats["draft_round"] = draft_r
        beats["protest_end"] = min(max_rounds, draft_r + 3)
        beats["compliance_end"] = min(max_rounds, draft_r + 1)
    if withdraw_r:
        beats["withdraw_round"] = withdraw_r
        beats["demand_start"] = withdraw_r
    return beats


def _beats_from_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    allowed = set(_canonical_beats())
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in allowed:
            continue
        if key in {"trust_rounds", "snapshot_rounds"} and isinstance(value, list):
            out[key] = tuple(int(v) for v in value)
        else:
            out[key] = value
    return out


def _is_canonical_story(log: Any, ids: set[str]) -> bool:
    scenario = str((getattr(log, "config", None) or {}).get("scenario") or "labwars")
    if scenario in {"crisisgrid", "releaseops"}:
        return False
    explicit = (log.config.get("story_beats") or {}).get("canonical")
    if explicit is False:
        return False
    if explicit is True:
        return True
    for rec in log.round_records:
        if rec.get("event_id") in CANONICAL_EVENT_IDS:
            return True
    for ev in log.events:
        if ev.get("event_id") in CANONICAL_EVENT_IDS:
            return True
    if "phd_a" in ids and not (log.config.get("event_cast") or {}).get("idea"):
        return True
    if "phd_a" in ids and (log.config.get("event_cast") or {}).get("idea") in {None, "phd_a"}:
        return True
    return False


def _max_round(log: Any) -> int:
    configured = int(log.config.get("max_rounds") or 0)
    observed = 0
    for rec in log.round_records:
        observed = max(observed, int(rec.get("round") or 0))
    for act in log.actions:
        observed = max(observed, int(act.get("round") or 0))
    for ev in log.events:
        observed = max(observed, int(ev.get("round") or 0))
    return configured or observed or 60


def _agent_ids_in_log(log: Any, world_agents: Mapping[str, Any] | None = None) -> set[str]:
    ids = set(world_agents or {})
    for rec in log.round_records:
        ids.update((rec.get("agent_deltas") or {}).keys())
    for act in log.actions:
        if act.get("agent"):
            ids.add(act["agent"])
    cfg = log.config.get("event_cast") or {}
    for key in ("pi", "idea", "experimenter", "postdoc", "engineer", "alumni", "rival"):
        if cfg.get(key):
            ids.add(cfg[key])
    return {str(i) for i in ids if i and i != "project"}


def _most_frequent_actor(log: Any) -> str | None:
    counts: dict[str, int] = {}
    for act in log.actions:
        aid = act.get("agent")
        if aid:
            counts[str(aid)] = counts.get(str(aid), 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: (counts[k], k))


def _first_id(ids: set[str], default: str) -> str:
    return next(iter(sorted(ids)), default)


def _first_prefix(ids: set[str], prefix: str, default: str) -> str:
    for aid in sorted(ids):
        if aid.startswith(prefix):
            return aid
    return default


def _first_other(ids: set[str], excluded: set[str], default: str) -> str:
    for aid in sorted(ids):
        if aid not in excluded:
            return aid
    return default


def _first_event_of_type(log: Any, event_type: str) -> tuple[str | None, int | None]:
    for ev in log.events:
        if ev.get("type") == event_type:
            return ev.get("event_id"), int(ev.get("round") or 0)
    return None, None


def _last_event_of_type(log: Any, event_type: str) -> tuple[str | None, int | None]:
    found_id: str | None = None
    found_round: int | None = None
    for ev in log.events:
        if ev.get("type") == event_type:
            found_id = ev.get("event_id")
            found_round = int(ev.get("round") or 0)
    return found_id, found_round
