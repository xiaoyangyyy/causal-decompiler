"""LabWars event system — types, registry, and payload helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventTypeSpec:
    type_id: str
    description: str
    default_visibility: str = "team"
    default_memory_salience: float = 0.5
    typical_content_types: tuple[str, ...] = ()


EVENT_TYPE_REGISTRY: dict[str, EventTypeSpec] = {
    "idea_claim": EventTypeSpec(
        "idea_claim", "idea 归属主张",
        typical_content_types=("credit_claim",),
    ),
    "experiment_success": EventTypeSpec(
        "experiment_success", "实验成功",
        default_memory_salience=0.55,
    ),
    "experiment_failure": EventTypeSpec(
        "experiment_failure", "实验失败",
        default_memory_salience=0.60,
    ),
    "baseline_failure": EventTypeSpec(
        "baseline_failure", "baseline 跑不过",
        default_memory_salience=0.65,
    ),
    "authorship_promise": EventTypeSpec(
        "authorship_promise", "署名承诺",
        default_visibility="bilateral",
        default_memory_salience=0.85,
        typical_content_types=("authorship_signal", "promise_fulfilled"),
    ),
    "authorship_ambiguity": EventTypeSpec(
        "authorship_ambiguity", "署名模糊化",
        default_visibility="bilateral",
        default_memory_salience=0.75,
        typical_content_types=("authorship_signal", "promise_broken"),
    ),
    "deadline_shift": EventTypeSpec(
        "deadline_shift", "截止日期变动",
        typical_content_types=("authority_signal",),
    ),
    "funding_pressure": EventTypeSpec(
        "funding_pressure", "资助方压力",
        typical_content_types=("authority_signal",),
    ),
    "rival_preprint": EventTypeSpec(
        "rival_preprint", "竞争组预印本",
        default_visibility="public",
        default_memory_salience=0.80,
        typical_content_types=("rival_threat",),
    ),
    "reviewer_feedback": EventTypeSpec(
        "reviewer_feedback", "审稿意见",
        default_memory_salience=0.55,
    ),
    "negative_result_hidden": EventTypeSpec(
        "negative_result_hidden", "隐瞒负面结果",
        default_visibility="private",
        default_memory_salience=0.70,
        typical_content_types=("integrity_signal",),
    ),
    "credit_dispute": EventTypeSpec(
        "credit_dispute", "贡献争议",
        default_memory_salience=0.75,
        typical_content_types=("credit_claim",),
    ),
    "private_lobbying": EventTypeSpec(
        "private_lobbying", "私下游说",
        default_visibility="bilateral",
        default_memory_salience=0.60,
        typical_content_types=("betrayal_signal", "authorship_signal"),
    ),
    "public_praise": EventTypeSpec(
        "public_praise", "公开表扬",
        default_memory_salience=0.70,
        typical_content_types=("credit_claim",),
    ),
    "public_blame": EventTypeSpec(
        "public_blame", "公开批评",
        default_memory_salience=0.65,
    ),
    "resource_reallocation": EventTypeSpec(
        "resource_reallocation", "资源重分配",
    ),
    "team_meeting": EventTypeSpec(
        "team_meeting", "组会",
        default_memory_salience=0.55,
    ),
    "submission_decision": EventTypeSpec(
        "submission_decision", "投稿决策",
        default_memory_salience=0.60,
    ),
    "authorship_draft": EventTypeSpec(
        "authorship_draft", "作者排序草案",
        default_memory_salience=0.90,
        typical_content_types=("authorship_signal", "promise_broken"),
    ),
    "threat_withdraw": EventTypeSpec(
        "threat_withdraw", "威胁退出",
        default_visibility="bilateral",
        default_memory_salience=0.85,
    ),
    "narrative_change": EventTypeSpec(
        "narrative_change", "叙事变更",
        default_memory_salience=0.70,
        typical_content_types=("betrayal_signal", "credit_claim"),
    ),
    "integrity_dispute": EventTypeSpec(
        "integrity_dispute", "学术诚信争议",
        default_memory_salience=0.72,
        typical_content_types=("integrity_signal",),
    ),
    "external_history": EventTypeSpec(
        "external_history", "外部历史信息",
        default_visibility="bilateral",
        default_memory_salience=0.75,
        typical_content_types=("historical_pattern",),
    ),
    "bridge_closed": EventTypeSpec("bridge_closed", "桥关闭", default_memory_salience=0.9, typical_content_types=("authority_signal",)),
    "sensor_report": EventTypeSpec("sensor_report", "传感器报告", default_memory_salience=0.8),
    "citizen_report": EventTypeSpec("citizen_report", "市民报告", default_visibility="public", default_memory_salience=0.7),
    "dispatch_brief": EventTypeSpec("dispatch_brief", "调度简报", default_memory_salience=0.75),
    "hospital_capacity": EventTypeSpec("hospital_capacity", "医院容量", default_memory_salience=0.55),
    "traffic_jam": EventTypeSpec("traffic_jam", "交通拥堵", default_memory_salience=0.6),
    "evac_order": EventTypeSpec("evac_order", "撤离命令", default_memory_salience=0.85),
    "stale_config": EventTypeSpec("stale_config", "过期配置", default_memory_salience=0.8, typical_content_types=("integrity_signal",)),
    "test_skipped": EventTypeSpec("test_skipped", "跳过测试", default_memory_salience=0.75),
    "alert": EventTypeSpec("alert", "生产告警", default_memory_salience=0.85),
    "rollback": EventTypeSpec("rollback", "回滚", default_memory_salience=0.8),
    "deploy_failure": EventTypeSpec("deploy_failure", "部署失败", default_memory_salience=0.9),
    "outage": EventTypeSpec("outage", "中断", default_visibility="public", default_memory_salience=0.9),
}


CONFLICT_TYPES: dict[int, dict[str, Any]] = {
    1: {"name": "Credit conflict", "variables": ["contribution_ledger", "credit_threat"]},
    2: {"name": "Resource conflict", "variables": ["resources", "dependency"]},
    3: {"name": "Narrative conflict", "variables": ["idea_clarity", "writing_control"]},
    4: {"name": "Integrity conflict", "variables": ["integrity_risk", "truth_status"]},
    5: {"name": "Career conflict", "variables": ["graduation_pressure", "job_market"]},
    6: {"name": "Authority conflict", "variables": ["authority_dependence", "pi_access"]},
    7: {"name": "Memory conflict", "variables": ["memory_interpretation_divergence"]},
}


def all_event_types() -> list[str]:
    return list(EVENT_TYPE_REGISTRY.keys())


def get_event_spec(event_type: str) -> EventTypeSpec:
    if event_type not in EVENT_TYPE_REGISTRY:
        raise KeyError(f"Unknown event type: {event_type}")
    return EVENT_TYPE_REGISTRY[event_type]


def apply_payload_to_project(project: dict[str, float], payload: dict[str, Any]) -> dict[str, float]:
    """Apply *_delta keys from event payload to project metrics."""
    updated = dict(project)
    for key, value in payload.items():
        if key.endswith("_delta") and isinstance(value, (int, float)):
            field = key[: -len("_delta")]
            if field in updated:
                updated[field] = max(0.0, min(1.0, updated[field] + float(value)))
    return updated
