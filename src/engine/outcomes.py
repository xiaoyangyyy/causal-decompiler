"""Scenario-registered outcomes: three-channel Y, PPG/PCI, and task Y."""

from __future__ import annotations

from typing import Any

CHANNEL_KEYS = ("y_private", "y_public", "y_action", "ppg", "pci")


def scenario_of(log: Any) -> str:
    return str((getattr(log, "config", None) or {}).get("scenario") or "labwars")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def ppg_pci(private: float, public: float, action: float) -> dict[str, float]:
    private = float(private)
    public = float(public)
    action = float(action)
    return {
        "y_private": private,
        "y_public": public,
        "y_action": action,
        "ppg": private - public,
        "pci": private - (public + action) / 2.0,
    }


CRISISGRID_REPORT_TYPES = frozenset({"bridge_closed", "sensor_report", "citizen_report"})
CRISISGRID_FORWARD_TYPES = frozenset({"dispatch_brief", "evac_order"})
CRISISGRID_FORWARD_ACTIONS = frozenset({"share_result"})
CRISISGRID_REROUTE_ACTIONS = frozenset({"reroute"})
RELEASEOPS_DELAY_ACTIONS = frozenset({"defer_test"})
RELEASEOPS_REPAIR_ACTIONS = frozenset({"restore_release"})
RELEASEOPS_STALE_TYPES = frozenset({"stale_config"})
RELEASEOPS_SKIP_TYPES = frozenset({"test_skipped"})
RELEASEOPS_ALERT_TYPES = frozenset({"alert"})
RELEASEOPS_ROLLBACK_TYPES = frozenset({"rollback"})


def _horizon(log: Any, n_events: int) -> int:
    """Declared length of the run. The denominator is rounds, not the pack's event census."""
    config = getattr(log, "config", None) or {}
    declared = int(config.get("max_rounds") or 0)
    if declared > 0:
        return max(1, declared)
    recorded = len(getattr(log, "round_records", None) or [])
    if recorded > 0:
        return max(1, recorded)
    return max(1, n_events)


def _rate(count: float, horizon: int) -> float:
    return _clamp(float(count) / max(1, int(horizon)))


def _intensity_sum(log: Any, types: frozenset[str]) -> float:
    total = 0.0
    for action in getattr(log, "actions", None) or []:
        name = str(action.get("type") or action.get("action_type") or "")
        if name not in types:
            continue
        if action.get("intensity") is None:
            total += 1.0
        else:
            total += float(action["intensity"])
    return total


def _count_types(log: Any, types: frozenset[str]) -> int:
    return sum(1 for ev in (getattr(log, "events", None) or []) if str(ev.get("type") or "") in types)


def crisisgrid_channels(log: Any) -> dict[str, float]:
    """Primary outcome is stranded: uncovered reports per round.

    Forward events are dispatch and evacuation orders. The only forward action is
    `share_result`. The only reroute action is `reroute`. A memory covers a report only
    when its event_ref names that report and the write is not the automatic same-round
    observation. One cover cancels one report.
    """
    events = getattr(log, "events", None) or []
    horizon = _horizon(log, len(events))
    reports = _rate(_count_types(log, CRISISGRID_REPORT_TYPES), horizon)
    forward = _rate(
        _count_types(log, CRISISGRID_FORWARD_TYPES) + _intensity_sum(log, CRISISGRID_FORWARD_ACTIONS),
        horizon,
    )
    reroute = _rate(_intensity_sum(log, CRISISGRID_REROUTE_ACTIONS), horizon)
    remembered_n = _relevant_memory_count(log, CRISISGRID_REPORT_TYPES)
    report_n = _count_types(log, CRISISGRID_REPORT_TYPES)
    covered = min(
        report_n,
        _count_types(log, CRISISGRID_FORWARD_TYPES)
        + _intensity_sum(log, CRISISGRID_FORWARD_ACTIONS)
        + _intensity_sum(log, CRISISGRID_REROUTE_ACTIONS)
        + remembered_n,
    )
    stranded = _rate(report_n - covered, horizon)
    evac_delay = stranded
    util = _rate(covered, max(1, report_n))
    out = ppg_pci(stranded, forward, reroute)
    out.update({
        "evac_delay": evac_delay,
        "stranded": stranded,
        "resource_util": util,
        "observed": reports,
        "remembered": _rate(remembered_n, horizon),
    })
    return out


def releaseops_channels(log: Any) -> dict[str, float]:
    """Primary outcome is task_y. Delay raises the skip term. Restore raises rollback.

    Those two actions are not added into one action mass. A memory raises the alert
    term only when event_ref points at a stale-config event. The stale event stays
    in the fault count. Rates use the declared horizon.
    """
    events = getattr(log, "events", None) or []
    horizon = _horizon(log, len(events))
    stale = _rate(_count_types(log, RELEASEOPS_STALE_TYPES), horizon)
    skipped = _rate(
        _count_types(log, RELEASEOPS_SKIP_TYPES) + _intensity_sum(log, RELEASEOPS_DELAY_ACTIONS),
        horizon,
    )
    alert = _rate(
        _count_types(log, RELEASEOPS_ALERT_TYPES) + _relevant_memory_count(log, RELEASEOPS_STALE_TYPES),
        horizon,
    )
    rollback = _rate(
        _count_types(log, RELEASEOPS_ROLLBACK_TYPES) + _intensity_sum(log, RELEASEOPS_REPAIR_ACTIONS),
        horizon,
    )
    # min is the graded AND: the indicator product is 0 or 1, and the product of
    # two per-round rates squares a small number. There is no rollback-time floor.
    failure = _clamp(min(stale, skipped))
    stale_n = _count_types(log, RELEASEOPS_STALE_TYPES)
    if stale_n <= 0:
        suppression = 0.0
    else:
        alert_n = _count_types(log, RELEASEOPS_ALERT_TYPES) + _relevant_memory_count(log, RELEASEOPS_STALE_TYPES)
        rollback_n = _count_types(log, RELEASEOPS_ROLLBACK_TYPES) + _intensity_sum(log, RELEASEOPS_REPAIR_ACTIONS)
        suppression = min(1.0, alert_n / stale_n) * min(1.0, rollback_n / stale_n)
    outage = _clamp(failure * (1.0 - suppression))
    rollback_time = _clamp(failure * (1.0 - rollback))
    task = outage
    return {
        "y_private": task,
        "y_public": outage,
        "y_action": _rate(_intensity_sum(log, RELEASEOPS_REPAIR_ACTIONS), horizon),
        "ppg": 0.0,
        "pci": 0.0,
        "deploy_failure": failure,
        "rollback_time": rollback_time,
        "outage": outage,
        "task_y": task,
    }


def labwars_channels(log: Any, *, private: float, public: float, action: float) -> dict[str, float]:
    return ppg_pci(private, public, action)


def _memory_records(log: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for rec in getattr(log, "round_records", None) or []:
        deltas = rec.get("agent_deltas") or {}
        for delta in deltas.values():
            mem = (delta or {}).get("memory_written")
            if isinstance(mem, dict) and mem:
                found.append(mem)
    return found


def _remembered_ids(log: Any) -> set[str]:
    return {str(mem.get("event_ref")) for mem in _memory_records(log) if mem.get("event_ref")}


def _relevant_memory_count(log: Any, types: frozenset[str]) -> int:
    """Memories that name a qualifying event and are not the automatic observation write.

    The cognition step stores one memory on the event's own round. That write
    restates the event. A later round, a rehearsal, or a record with no round
    is what the outcome can use.
    """
    by_id = {
        str(event.get("event_id")): event
        for event in (getattr(log, "events", None) or [])
        if event.get("event_id") and str(event.get("type") or "") in types
    }
    seen: set[str] = set()
    count = 0
    for mem in _memory_records(log):
        ref = str(mem.get("event_ref") or "")
        event = by_id.get(ref)
        if event is None or ref in seen:
            continue
        mem_round = mem.get("round")
        event_round = event.get("round")
        rehearsed = float(mem.get("rehearsal_count") or 0.0) > 0.0
        automatic = (
            mem_round is not None
            and event_round is not None
            and int(mem_round) == int(event_round)
            and not rehearsed
        )
        if automatic:
            continue
        seen.add(ref)
        count += 1
    return count


def three_channel_y(log: Any, *, private: float | None = None, public: float | None = None, action: float | None = None) -> dict[str, float]:
    scenario = scenario_of(log)
    if scenario == "crisisgrid":
        return crisisgrid_channels(log)
    if scenario == "releaseops":
        return releaseops_channels(log)
    priv = 0.0 if private is None else private
    pub = 0.0 if public is None else public
    act = 0.0 if action is None else action
    return labwars_channels(log, private=priv, public=pub, action=act)
