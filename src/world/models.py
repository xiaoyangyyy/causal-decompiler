"""LabWars world layer — data models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class AgentRole(str, Enum):
    PI = "pi"
    IDEA_ORIGINATOR = "idea_originator"
    EXPERIMENTER = "experimenter"
    MASTER_STUDENT = "master_student"
    POSTDOC = "postdoc"
    ENGINEER = "engineer"
    VISITING_STUDENT = "visiting_student"
    COLLABORATOR = "collaborator"
    RIVAL_LAB = "rival_lab"
    REVIEWER = "reviewer"
    PROGRAM_OFFICER = "program_officer"
    ALUMNI = "alumni"
    OPERATOR = "operator"


class Personality(BaseModel):
    ambition: float
    cooperation: float
    risk_taking: float
    conflict_avoidance: float
    credit_sensitivity: float
    authority_dependence: float
    deceptiveness: float
    reciprocity: float
    resentment_sensitivity: float

    @field_validator("*", mode="before")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return clamp(float(v))


class Beliefs(BaseModel):
    pi_fairness: float
    project_publishability: float
    rival_lab_threat: float
    my_first_author_probability: float
    team_trust: float
    deadline_feasibility: float
    my_contribution_recognized: float = 0.5
    others_are_free_riding: float = 0.0
    academic_integrity_risk: float = 0.0

    @field_validator("*", mode="before")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return clamp(float(v))


class Emotion(BaseModel):
    confidence: float
    anxiety: float
    anger: float
    resentment: float
    guilt: float
    hope: float
    burnout: float
    envy: float = 0.0
    fear: float = 0.0
    loyalty: float = 0.5

    @field_validator("*", mode="before")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return clamp(float(v))


class Resources(BaseModel):
    code_control: float
    data_control: float
    writing_control: float
    pi_access: float
    external_network: float

    @field_validator("*", mode="before")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return clamp(float(v))


class Agent(BaseModel):
    id: str
    role: AgentRole
    display_name: str = ""
    goals: list[str] = Field(default_factory=list)
    personality: Personality
    extra_traits: dict[str, Any] = Field(default_factory=dict)
    beliefs: Beliefs
    emotion: Emotion
    resources: Resources
    memory: list[dict[str, Any]] = Field(default_factory=list)
    memory_recall_log: list[dict[str, Any]] = Field(default_factory=list)
    public_position: dict[str, Any] = Field(default_factory=dict)
    private_intent: dict[str, Any] = Field(default_factory=dict)
    action_history: list[dict[str, Any]] = Field(default_factory=list)


class ObjectiveFact(BaseModel):
    raw_statement: str | None = None
    verifiable_claims: list[str] = Field(default_factory=list)


class EventAtom(BaseModel):
    event_id: str
    round: int
    type: str
    visibility: str
    source: str
    targets: list[str]
    payload: dict[str, Any] = Field(default_factory=dict)
    objective_fact: ObjectiveFact
    framing: str
    truth_status: str
    memory_salience: float
    is_anchor: bool
    description: str = ""
    act: int | None = None

    @field_validator("memory_salience", mode="before")
    @classmethod
    def _clamp_salience(cls, v: float) -> float:
        return clamp(float(v))


class RelationshipEdge(BaseModel):
    source: str
    target: str
    trust: float
    resentment: float
    dependency: float
    obligation: float
    perceived_credit_threat: float
    communication_frequency: float
    alliance: float
    information_access: float
    last_interaction_valence: float

    @field_validator(
        "trust",
        "resentment",
        "dependency",
        "obligation",
        "perceived_credit_threat",
        "communication_frequency",
        "alliance",
        "information_access",
        mode="before",
    )
    @classmethod
    def _clamp_edge(cls, v: float) -> float:
        return clamp(float(v))

    @field_validator("last_interaction_valence", mode="before")
    @classmethod
    def _clamp_valence(cls, v: float) -> float:
        return clamp(float(v), -1.0, 1.0)


class ProjectMetrics(BaseModel):
    idea_clarity: float = 0.0
    experimental_strength: float = 0.0
    code_stability: float = 0.0
    writing_quality: float = 0.0
    novelty_risk: float = 0.0
    baseline_coverage: float = 0.0
    deadline_pressure: float = 0.0
    rival_threat: float = 0.0
    funding_pressure: float = 0.0
    authorship_conflict: float = 0.0
    team_morale: float = 0.0
    integrity_risk: float = 0.0
    hazard: float = 0.0
    outage_mass: float = 0.0
    skip_mass: float = 0.0

    @field_validator("*", mode="before")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return clamp(float(v))


class ProjectState(BaseModel):
    project: ProjectMetrics
    contribution_ledger: dict[str, dict[str, float]] = Field(default_factory=dict)
    author_order_draft: list[str] = Field(default_factory=list)
    submission_status: str = "in_progress"
    current_round: int = 0
    target_conference: str = "NeurIPS"


class WorldState(BaseModel):
    agents: dict[str, Agent]
    relationships: list[RelationshipEdge]
    project: ProjectState
    world_config: dict[str, Any] = Field(default_factory=dict)
