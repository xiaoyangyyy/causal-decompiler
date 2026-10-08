"""LabWars action space — enums, categories, and project effects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


ActionCategory = Literal[
    "research",
    "political",
    "information",
    "emotional",
    "external",
    "communication",
]


class ActionType(str, Enum):
    # Research
    RUN_EXPERIMENT = "run_experiment"
    IMPROVE_BASELINE = "improve_baseline"
    WRITE_SECTION = "write_section"
    DEBUG_CODE = "debug_code"
    ANALYZE_FAILURE = "analyze_failure"
    PREPARE_REBUTTAL = "prepare_rebuttal"
    OPEN_SOURCE_CODE = "open_source_code"
    WITHHOLD_CODE = "withhold_code"
    # Political
    ASK_FOR_AUTHORSHIP = "ask_for_authorship"
    PRIVATELY_LOBBY_PI = "privately_lobby_pi"
    SUPPORT_TEAMMATE = "support_teammate"
    UNDERMINE_TEAMMATE = "undermine_teammate"
    FORM_ALLIANCE = "form_alliance"
    LEAK_CONCERN = "leak_concern"
    REQUEST_MEDIATION = "request_mediation"
    DELAY_RESPONSE = "delay_response"
    # Information
    SHARE_RESULT = "share_result"
    HIDE_NEGATIVE_RESULT = "hide_negative_result"
    SELECTIVELY_REPORT = "selectively_report"
    CHALLENGE_CLAIM = "challenge_claim"
    DOCUMENT_CONTRIBUTION = "document_contribution"
    CITE_PRIOR_MEMORY = "cite_prior_memory"
    # Emotional
    CONFRONT = "confront"
    WITHDRAW = "withdraw"
    APOLOGIZE = "apologize"
    BLAME = "blame"
    SEEK_VALIDATION = "seek_validation"
    COMPLY = "comply"
    REBEL = "rebel"
    # External
    CONTACT_COLLABORATOR = "contact_collaborator"
    CHECK_RIVAL_ARXIV = "check_rival_arxiv"
    SUBMIT_WORKSHOP_VERSION = "submit_workshop_version"
    TALK_TO_ALUMNI = "talk_to_alumni"
    NOTIFY_PROGRAM_OFFICER = "notify_program_officer"
    # Scenario packs. Kept out of the LabWars allowed set.
    REROUTE = "reroute"
    RESTORE_RELEASE = "restore_release"
    DEFER_TEST = "defer_test"


ACTION_CATEGORIES: dict[ActionType, ActionCategory] = {
    ActionType.RUN_EXPERIMENT: "research",
    ActionType.IMPROVE_BASELINE: "research",
    ActionType.WRITE_SECTION: "research",
    ActionType.DEBUG_CODE: "research",
    ActionType.ANALYZE_FAILURE: "research",
    ActionType.PREPARE_REBUTTAL: "research",
    ActionType.OPEN_SOURCE_CODE: "research",
    ActionType.WITHHOLD_CODE: "research",
    ActionType.ASK_FOR_AUTHORSHIP: "political",
    ActionType.PRIVATELY_LOBBY_PI: "political",
    ActionType.SUPPORT_TEAMMATE: "political",
    ActionType.UNDERMINE_TEAMMATE: "political",
    ActionType.FORM_ALLIANCE: "political",
    ActionType.LEAK_CONCERN: "political",
    ActionType.REQUEST_MEDIATION: "political",
    ActionType.DELAY_RESPONSE: "political",
    ActionType.SHARE_RESULT: "information",
    ActionType.HIDE_NEGATIVE_RESULT: "information",
    ActionType.SELECTIVELY_REPORT: "information",
    ActionType.CHALLENGE_CLAIM: "information",
    ActionType.DOCUMENT_CONTRIBUTION: "information",
    ActionType.CITE_PRIOR_MEMORY: "information",
    ActionType.CONFRONT: "emotional",
    ActionType.WITHDRAW: "emotional",
    ActionType.APOLOGIZE: "emotional",
    ActionType.BLAME: "emotional",
    ActionType.SEEK_VALIDATION: "emotional",
    ActionType.COMPLY: "emotional",
    ActionType.REBEL: "emotional",
    ActionType.CONTACT_COLLABORATOR: "external",
    ActionType.CHECK_RIVAL_ARXIV: "external",
    ActionType.SUBMIT_WORKSHOP_VERSION: "external",
    ActionType.TALK_TO_ALUMNI: "external",
    ActionType.NOTIFY_PROGRAM_OFFICER: "external",
    ActionType.REROUTE: "external",
    ActionType.RESTORE_RELEASE: "research",
    ActionType.DEFER_TEST: "research",
}


COOPERATION_ACTIONS: set[ActionType] = {
    ActionType.SUPPORT_TEAMMATE,
    ActionType.SHARE_RESULT,
    ActionType.OPEN_SOURCE_CODE,
    ActionType.APOLOGIZE,
    ActionType.COMPLY,
    ActionType.REQUEST_MEDIATION,
    ActionType.DOCUMENT_CONTRIBUTION,
}

CONFLICT_ACTIONS: set[ActionType] = {
    ActionType.UNDERMINE_TEAMMATE,
    ActionType.CONFRONT,
    ActionType.BLAME,
    ActionType.REBEL,
    ActionType.CHALLENGE_CLAIM,
    ActionType.ASK_FOR_AUTHORSHIP,
    ActionType.WITHHOLD_CODE,
    ActionType.HIDE_NEGATIVE_RESULT,
}


@dataclass(frozen=True)
class ProjectEffect:
    field: str
    delta: float


@dataclass(frozen=True)
class ActionSpec:
    action: ActionType
    category: ActionCategory
    project_effects: tuple[ProjectEffect, ...] = ()
    cooperation_level: float = 0.5
    conflict_level: float = 0.0
    authorship_claim_intensity: float = 0.0
    notes: str = ""


ACTION_REGISTRY: dict[ActionType, ActionSpec] = {
    ActionType.RUN_EXPERIMENT: ActionSpec(
        ActionType.RUN_EXPERIMENT, "research",
        (ProjectEffect("experimental_strength", 0.08),),
        cooperation_level=0.7,
    ),
    ActionType.IMPROVE_BASELINE: ActionSpec(
        ActionType.IMPROVE_BASELINE, "research",
        (ProjectEffect("baseline_coverage", 0.08),),
        cooperation_level=0.65,
    ),
    ActionType.WRITE_SECTION: ActionSpec(
        ActionType.WRITE_SECTION, "research",
        (ProjectEffect("writing_quality", 0.08),),
        cooperation_level=0.6,
    ),
    ActionType.DEBUG_CODE: ActionSpec(
        ActionType.DEBUG_CODE, "research",
        (ProjectEffect("code_stability", 0.10),),
        cooperation_level=0.75,
    ),
    ActionType.ANALYZE_FAILURE: ActionSpec(
        ActionType.ANALYZE_FAILURE, "research",
        (ProjectEffect("novelty_risk", -0.05),),
        cooperation_level=0.6,
    ),
    ActionType.PREPARE_REBUTTAL: ActionSpec(
        ActionType.PREPARE_REBUTTAL, "research",
        (ProjectEffect("writing_quality", 0.06),),
        cooperation_level=0.55,
    ),
    ActionType.OPEN_SOURCE_CODE: ActionSpec(
        ActionType.OPEN_SOURCE_CODE, "research",
        (ProjectEffect("code_stability", 0.05),),
        cooperation_level=0.85,
    ),
    ActionType.WITHHOLD_CODE: ActionSpec(
        ActionType.WITHHOLD_CODE, "research",
        (ProjectEffect("code_stability", -0.05),),
        cooperation_level=0.2, conflict_level=0.7,
    ),
    ActionType.ASK_FOR_AUTHORSHIP: ActionSpec(
        ActionType.ASK_FOR_AUTHORSHIP, "political",
        (ProjectEffect("authorship_conflict", 0.12),),
        conflict_level=0.6, authorship_claim_intensity=0.8,
    ),
    ActionType.PRIVATELY_LOBBY_PI: ActionSpec(
        ActionType.PRIVATELY_LOBBY_PI, "political",
        conflict_level=0.4, authorship_claim_intensity=0.5,
    ),
    ActionType.SUPPORT_TEAMMATE: ActionSpec(
        ActionType.SUPPORT_TEAMMATE, "political",
        cooperation_level=0.9, conflict_level=0.05,
    ),
    ActionType.UNDERMINE_TEAMMATE: ActionSpec(
        ActionType.UNDERMINE_TEAMMATE, "political",
        (ProjectEffect("team_morale", -0.05),),
        cooperation_level=0.1, conflict_level=0.85,
    ),
    ActionType.FORM_ALLIANCE: ActionSpec(
        ActionType.FORM_ALLIANCE, "political",
        cooperation_level=0.75, conflict_level=0.3,
    ),
    ActionType.LEAK_CONCERN: ActionSpec(
        ActionType.LEAK_CONCERN, "political",
        (ProjectEffect("authorship_conflict", 0.08),),
        conflict_level=0.5,
    ),
    ActionType.REQUEST_MEDIATION: ActionSpec(
        ActionType.REQUEST_MEDIATION, "political",
        (ProjectEffect("authorship_conflict", -0.05),),
        cooperation_level=0.7, conflict_level=0.3,
    ),
    ActionType.DELAY_RESPONSE: ActionSpec(
        ActionType.DELAY_RESPONSE, "political",
        (ProjectEffect("deadline_pressure", 0.08),),
        conflict_level=0.4,
    ),
    ActionType.SHARE_RESULT: ActionSpec(
        ActionType.SHARE_RESULT, "information",
        cooperation_level=0.85,
    ),
    ActionType.HIDE_NEGATIVE_RESULT: ActionSpec(
        ActionType.HIDE_NEGATIVE_RESULT, "information",
        (ProjectEffect("integrity_risk", 0.15),),
        cooperation_level=0.15, conflict_level=0.5,
    ),
    ActionType.SELECTIVELY_REPORT: ActionSpec(
        ActionType.SELECTIVELY_REPORT, "information",
        conflict_level=0.45,
    ),
    ActionType.CHALLENGE_CLAIM: ActionSpec(
        ActionType.CHALLENGE_CLAIM, "information",
        (ProjectEffect("authorship_conflict", 0.10),),
        conflict_level=0.7, authorship_claim_intensity=0.6,
    ),
    ActionType.DOCUMENT_CONTRIBUTION: ActionSpec(
        ActionType.DOCUMENT_CONTRIBUTION, "information",
        cooperation_level=0.65, authorship_claim_intensity=0.5,
    ),
    ActionType.CITE_PRIOR_MEMORY: ActionSpec(
        ActionType.CITE_PRIOR_MEMORY, "information",
        cooperation_level=0.5, authorship_claim_intensity=0.4,
    ),
    ActionType.CONFRONT: ActionSpec(
        ActionType.CONFRONT, "emotional",
        (ProjectEffect("team_morale", -0.08),),
        conflict_level=0.8,
    ),
    ActionType.WITHDRAW: ActionSpec(
        ActionType.WITHDRAW, "emotional",
        (ProjectEffect("team_morale", -0.10),),
        cooperation_level=0.05, conflict_level=0.6,
    ),
    ActionType.APOLOGIZE: ActionSpec(
        ActionType.APOLOGIZE, "emotional",
        cooperation_level=0.8, conflict_level=0.1,
    ),
    ActionType.BLAME: ActionSpec(
        ActionType.BLAME, "emotional",
        (ProjectEffect("team_morale", -0.06),),
        conflict_level=0.75,
    ),
    ActionType.SEEK_VALIDATION: ActionSpec(
        ActionType.SEEK_VALIDATION, "emotional",
        cooperation_level=0.55,
    ),
    ActionType.COMPLY: ActionSpec(
        ActionType.COMPLY, "emotional",
        cooperation_level=0.7, conflict_level=0.15,
    ),
    ActionType.REBEL: ActionSpec(
        ActionType.REBEL, "emotional",
        (ProjectEffect("authorship_conflict", 0.08),),
        cooperation_level=0.1, conflict_level=0.85,
    ),
    ActionType.CONTACT_COLLABORATOR: ActionSpec(
        ActionType.CONTACT_COLLABORATOR, "external",
        cooperation_level=0.5,
    ),
    ActionType.CHECK_RIVAL_ARXIV: ActionSpec(
        ActionType.CHECK_RIVAL_ARXIV, "external",
        cooperation_level=0.6,
    ),
    ActionType.SUBMIT_WORKSHOP_VERSION: ActionSpec(
        ActionType.SUBMIT_WORKSHOP_VERSION, "external",
        (ProjectEffect("rival_threat", 0.05),),
        conflict_level=0.4,
    ),
    ActionType.TALK_TO_ALUMNI: ActionSpec(
        ActionType.TALK_TO_ALUMNI, "external",
        cooperation_level=0.45,
    ),
    ActionType.NOTIFY_PROGRAM_OFFICER: ActionSpec(
        ActionType.NOTIFY_PROGRAM_OFFICER, "external",
        (ProjectEffect("funding_pressure", 0.05),),
    ),
    ActionType.REROUTE: ActionSpec(ActionType.REROUTE, "external", cooperation_level=0.7),
    ActionType.RESTORE_RELEASE: ActionSpec(ActionType.RESTORE_RELEASE, "research", cooperation_level=0.8),
    ActionType.DEFER_TEST: ActionSpec(ActionType.DEFER_TEST, "research", cooperation_level=0.3),
}

PACK_ONLY_ACTIONS = frozenset({
    ActionType.REROUTE,
    ActionType.RESTORE_RELEASE,
    ActionType.DEFER_TEST,
})


def pack_action_names(scenario: str) -> list[str]:
    """Actions a pack agent can take. LabWars verbs stay out of this list."""
    if scenario == "crisisgrid":
        return [ActionType.SHARE_RESULT.value, ActionType.REROUTE.value]
    if scenario == "releaseops":
        return [
            ActionType.RESTORE_RELEASE.value,
            ActionType.DEFER_TEST.value,
            ActionType.ANALYZE_FAILURE.value,
            ActionType.DEBUG_CODE.value,
            ActionType.SHARE_RESULT.value,
        ]
    return []


# Per-agent action restrictions (agent_id -> blocked actions)
AGENT_ACTION_RESTRICTIONS: dict[str, set[ActionType]] = {
    "engineer_e": {ActionType.ASK_FOR_AUTHORSHIP},
    "visiting_f": {ActionType.PRIVATELY_LOBBY_PI},
    "reviewer_1": set(ActionType),
    "reviewer_2": set(ActionType),
    "reviewer_3": set(ActionType),
    "rival_lab_h": {
        a for a in ActionType if ACTION_CATEGORIES[a] != "external"
    } - {ActionType.CHECK_RIVAL_ARXIV, ActionType.SUBMIT_WORKSHOP_VERSION},
}


def all_action_types() -> list[str]:
    return [a.value for a in ActionType]


def get_allowed_actions(agent_id: str, burnout: float = 0.0) -> list[ActionType]:
    """Return legally available actions; burnout affects probability, not availability."""
    blocked = set(AGENT_ACTION_RESTRICTIONS.get(agent_id, set())) | set(PACK_ONLY_ACTIONS)
    return [a for a in ActionType if a not in blocked]


def apply_project_effects(project_metrics: dict[str, float], action: ActionType) -> dict[str, float]:
    spec = ACTION_REGISTRY[action]
    updated = dict(project_metrics)
    for effect in spec.project_effects:
        current = updated.get(effect.field, 0.0)
        updated[effect.field] = max(0.0, min(1.0, current + effect.delta))
    return updated
