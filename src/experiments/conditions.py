"""Experiment condition definitions — A/B/C/D + validity controls."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.engine.intervention import Intervention, load_interventions
from src.engine.simulation import SimConfig

_INTERVENTIONS = {i.intervention_id: i for i in load_interventions()}


@dataclass
class ExperimentCondition:
    experiment_id: str
    condition_id: str
    label: str
    intervention_ids: list[str] = field(default_factory=list)
    disable_memory: bool = False
    shuffle_memory: bool = False
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    seed_offset: int = 0
    primary_outcomes: list[str] = field(default_factory=list)


def _i(*ids: str) -> list[Intervention]:
    return [_INTERVENTIONS[iid] for iid in ids if iid in _INTERVENTIONS]


EXPERIMENT_A: dict[str, ExperimentCondition] = {
    "A1": ExperimentCondition(
        "A", "A1", "baseline", ["INT_AUTH_BASELINE"],
        primary_outcomes=[
            "trust_pi_final", "pi_fairness_r52",
            "authorship_escalation_score", "authorship_escalation_potential",
            "memory_authorship_cluster_strength",
        ],
    ),
    "A2": ExperimentCondition(
        "A", "A2", "explicit_promise", ["INT_AUTH_EXPLICIT", "INT_E052_HONOR"],
        primary_outcomes=[
            "trust_pi_final", "pi_fairness_r52",
            "authorship_escalation_score", "post_r52_compliance",
            "memory_authorship_cluster_strength",
        ],
    ),
    "A3": ExperimentCondition(
        "A", "A3", "ambiguous_promise", ["INT_AUTH_AMBIGUOUS"],
        primary_outcomes=[
            "trust_pi_final", "pi_fairness_r52",
            "authorship_escalation_score", "authorship_escalation_potential",
            "memory_authorship_cluster_strength",
        ],
    ),
    "A4": ExperimentCondition(
        "A", "A4", "no_promise", ["INT_SKIP_E003"],
        primary_outcomes=[
            "trust_pi_final", "pi_fairness_r52",
            "authorship_escalation_score", "authorship_escalation_potential",
            "memory_authorship_cluster_strength",
        ],
    ),
    "A5": ExperimentCondition(
        "A", "A5", "explicit_plus_delete", ["INT_AUTH_EXPLICIT", "INT_MEMORY_DELETE"],
        primary_outcomes=[
            "trust_pi_final", "authorship_escalation_score",
            "memory_authorship_cluster_strength", "post_r52_compliance",
        ],
    ),
}

EXPERIMENT_B: dict[str, ExperimentCondition] = {
    "B1": ExperimentCondition("B", "B1", "baseline", [], primary_outcomes=["help_rebuttal", "demand_authorship_exchange", "passive_cooperation"]),
    "B2": ExperimentCondition("B", "B2", "strengthen_betrayal", ["INT_MEMORY_STRENGTHEN_BETRAYAL"], primary_outcomes=["help_rebuttal", "demand_authorship_exchange"]),
    "B3": ExperimentCondition("B", "B3", "skip_E031", ["INT_SKIP_E031"], primary_outcomes=["help_rebuttal", "demand_authorship_exchange"]),
    "B4": ExperimentCondition("B", "B4", "rebuttal_request", ["INT_B4_REBUTTAL_REQUEST"], primary_outcomes=["help_rebuttal", "demand_authorship_exchange"]),
}

EXPERIMENT_C: dict[str, ExperimentCondition] = {
    "C1": ExperimentCondition("C", "C1", "false_memory", ["INT_FALSE_MEMORY_INSERT", "INT_MEMORY_CORRECT"], primary_outcomes=["trust_phd_b_r25", "trust_phd_b_r44", "trust_phd_b_r60", "trust_recovery_rate"]),
    "C2": ExperimentCondition("C", "C2", "false_no_correct", ["INT_FALSE_MEMORY_INSERT"], primary_outcomes=["trust_phd_b_r60", "trust_recovery_rate"]),
    "C3": ExperimentCondition("C", "C3", "baseline", [], primary_outcomes=["trust_phd_b_r60"]),
}

EXPERIMENT_D: dict[str, ExperimentCondition] = {
    "D1": ExperimentCondition("D", "D1", "baseline", [], primary_outcomes=["pi_fairness_r35", "pi_fairness_r52", "protest_authorship", "interpretation_of_E030"]),
    "D2": ExperimentCondition("D", "D2", "skip_E035", ["INT_SKIP_E035"], primary_outcomes=["pi_fairness_r35", "protest_authorship", "interpretation_of_E030"]),
    "D3": ExperimentCondition("D", "D3", "positive_alumni", ["INT_ALUMNI_POSITIVE"], primary_outcomes=["pi_fairness_r35", "protest_authorship", "interpretation_of_E030"]),
}

EXPERIMENT_VALIDITY: dict[str, ExperimentCondition] = {
    "V1": ExperimentCondition("V", "V1", "no_memory", [], disable_memory=True, primary_outcomes=["protest_authorship", "memory_authorship_cluster_strength"]),
    "V2": ExperimentCondition("V", "V2", "shuffled_memory", [], shuffle_memory=True, primary_outcomes=["protest_authorship", "memory_authorship_cluster_strength"]),
    "V3": ExperimentCondition("V", "V3", "delayed_insert", ["INT_DELAYED_MEMORY_INSERT"], primary_outcomes=["protest_authorship", "memory_authorship_cluster_strength"]),
    "V6": ExperimentCondition("V", "V6", "full_memory_baseline", ["INT_AUTH_EXPLICIT"], primary_outcomes=["protest_authorship", "memory_authorship_cluster_strength"]),
}

EXPERIMENT_MATRIX: dict[str, dict[str, ExperimentCondition]] = {
    "A": EXPERIMENT_A,
    "B": EXPERIMENT_B,
    "C": EXPERIMENT_C,
    "D": EXPERIMENT_D,
    "V": EXPERIMENT_VALIDITY,
}


def get_condition(experiment_id: str, condition_id: str | None = None) -> ExperimentCondition:
    exp = experiment_id.upper()
    table = EXPERIMENT_MATRIX.get(exp)
    if not table:
        raise ValueError(f"Unknown experiment: {experiment_id}")
    cid = condition_id or next(iter(table))
    if cid not in table:
        raise ValueError(f"Unknown condition {cid} for experiment {exp}")
    return table[cid]


def list_conditions(experiment_id: str) -> list[str]:
    return list(EXPERIMENT_MATRIX[experiment_id.upper()].keys())


def build_sim_config(
    condition: ExperimentCondition,
    seed: int,
    *,
    max_rounds: int = 60,
    output_dir: str | None = None,
) -> SimConfig:
    from pathlib import Path

    run_id = f"{condition.experiment_id}{condition.condition_id}_seed{seed}"
    provider = condition.llm_provider or os.environ.get("LABWARS_LLM_PROVIDER") or None
    top_k_raw = os.environ.get("LABWARS_COGNITIVE_TOP_K")
    top_k = int(top_k_raw) if top_k_raw else None
    return SimConfig(
        max_rounds=max_rounds,
        seed=seed + condition.seed_offset,
        interventions=_i(*condition.intervention_ids),
        disable_memory=condition.disable_memory,
        shuffle_memory=condition.shuffle_memory,
        llm_provider=provider,
        llm_model=condition.llm_model,
        llm_temperature=condition.llm_temperature,
        experiment_id=condition.experiment_id,
        condition_id=condition.condition_id,
        run_id=run_id,
        output_dir=Path(output_dir) if output_dir else None,
        cognitive_sampling_top_k=top_k,
    )


EXPERIMENT_PRIMARY_OUTCOME = {
    "A": "authorship_escalation_score",
    "B": "help_rebuttal",
    "C": "trust_phd_b_r60",
    "D": "interpretation_of_E030",
    "V": "protest_authorship",
}


def primary_outcome_for(experiment_id: str) -> str:
    return EXPERIMENT_PRIMARY_OUTCOME[experiment_id.upper()]


def report_outcomes_for(experiment_id: str) -> list[str]:
    from src.engine.run_log import SPLIT_Y_KEYS

    exp = experiment_id.upper()
    keys: list[str] = []
    table = EXPERIMENT_MATRIX[exp]
    for cond in table.values():
        for key in cond.primary_outcomes:
            if key not in keys:
                keys.append(key)
    for key in SPLIT_Y_KEYS:
        if key not in keys:
            keys.append(key)
    return keys
