"""CRN-paired A/B/C/D/V contrasts for the paper — not independent seeds.

Each pair shares the control run's LLM trace so treatment is a patched twin,
not a new draw from the model.
"""

from __future__ import annotations

from typing import Any

from src.engine.causal.estimands import PAPER_SPLIT_KEYS
from src.engine.causal.twin import run_replay
from src.engine.run_log import extract_outcome
from src.experiments.conditions import build_sim_config, get_condition, primary_outcome_for, report_outcomes_for

PAPER_CONTRASTS: dict[str, list[tuple[str, str, str]]] = {
    "A": [
        ("A1", "A2", "honor_vs_baseline"),
        ("A1", "A5", "delete_vs_baseline"),
        ("A2", "A5", "delete_vs_honor"),
        ("A1", "A4", "skip_promise_vs_baseline"),
    ],
    "B": [
        ("B1", "B2", "strengthen_betrayal"),
        ("B1", "B3", "skip_E031"),
        ("B1", "B4", "rebuttal_request"),
    ],
    "C": [
        ("C3", "C2", "false_memory_vs_baseline"),
        ("C2", "C1", "correction_vs_false"),
        ("C3", "C1", "corrected_vs_baseline"),
    ],
    "D": [
        ("D1", "D2", "skip_alumni"),
        ("D1", "D3", "positive_alumni"),
    ],
    "V": [
        ("V6", "V1", "no_memory"),
        ("V6", "V2", "shuffled_memory"),
        ("V6", "V3", "delayed_insert"),
    ],
}


def _pair_outcomes(experiment_id: str) -> list[str]:
    keys = [primary_outcome_for(experiment_id)]
    for key in (*PAPER_SPLIT_KEYS, *report_outcomes_for(experiment_id)):
        if key not in keys:
            keys.append(key)
    return keys


def run_crn_pair(
    experiment_id: str,
    control_id: str,
    treatment_id: str,
    seed: int,
    *,
    max_rounds: int = 60,
    outcomes: list[str] | None = None,
) -> dict[str, Any]:
    control = get_condition(experiment_id, control_id)
    treatment = get_condition(experiment_id, treatment_id)
    ctrl_cfg = build_sim_config(control, seed, max_rounds=max_rounds)
    treat_cfg = build_sim_config(treatment, seed, max_rounds=max_rounds)
    ctrl_log = run_replay(ctrl_cfg)
    treat_log = run_replay(treat_cfg, llm_trace=ctrl_log.llm_cache)
    keys = outcomes or _pair_outcomes(experiment_id)
    ates = {
        key: {
            "control": extract_outcome(ctrl_log, key),
            "treatment": extract_outcome(treat_log, key),
            "ate": extract_outcome(treat_log, key) - extract_outcome(ctrl_log, key),
        }
        for key in keys
    }
    replay = treat_log.outcomes.get("llm_trace_stats") or {}
    return {
        "experiment_id": experiment_id.upper(),
        "control_id": control_id,
        "treatment_id": treatment_id,
        "seed": seed,
        "control_run_id": ctrl_log.run_id,
        "treatment_run_id": treat_log.run_id,
        "ates": ates,
        "llm_replay": {"hits": replay.get("run_hits", 0), "misses": replay.get("run_misses", 0)},
    }


def flatten_pair(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    label = f"{row['control_id']}→{row['treatment_id']}"
    for outcome, stats in (row.get("ates") or {}).items():
        out.append({
            "label": label,
            "experiment_id": row.get("experiment_id"),
            "outcome": outcome,
            "ate": stats.get("ate", 0.0),
            "control_mean": stats.get("control", 0.0),
            "treatment_mean": stats.get("treatment", 0.0),
            "n": 1,
            "seed": row.get("seed"),
        })
    return out


def run_experiment_contrasts(
    experiment_id: str,
    *,
    seeds: list[int] | None = None,
    max_rounds: int = 60,
    pairs: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    exp = experiment_id.upper()
    seed_values = seeds or [0]
    planned = pairs or PAPER_CONTRASTS[exp]
    rows: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []
    for seed in seed_values:
        for control_id, treatment_id, _name in planned:
            row = run_crn_pair(exp, control_id, treatment_id, seed, max_rounds=max_rounds)
            row["name"] = _name
            rows.append(row)
            flat.extend(flatten_pair(row))
    return {
        "experiment_id": exp,
        "n_seeds": len(seed_values),
        "max_rounds": max_rounds,
        "pairs": [{"control": a, "treatment": b, "name": n} for a, b, n in planned],
        "runs": rows,
        "table_rows": flat,
    }


def run_paper_contrasts(
    experiments: list[str] | None = None,
    *,
    seeds: list[int] | None = None,
    max_rounds: int = 60,
) -> dict[str, Any]:
    experiments = [e.upper() for e in (experiments or list(PAPER_CONTRASTS))]
    by_exp = {
        exp: run_experiment_contrasts(exp, seeds=seeds, max_rounds=max_rounds)
        for exp in experiments
    }
    table_rows: list[dict[str, Any]] = []
    for block in by_exp.values():
        table_rows.extend(block.get("table_rows") or [])
    return {"experiments": by_exp, "table_rows": table_rows}


def shuffle_vs_full_test(
    seeds: list[int],
    outcome: str = "protest_authorship",
    *,
    max_rounds: int = 60,
) -> dict[str, Any]:
    """CRN validity gate: V6 (full memory) vs V2 (shuffled) on shared LLM traces."""
    ates: list[float] = []
    for seed in seeds:
        row = run_crn_pair("V", "V6", "V2", seed, max_rounds=max_rounds)
        ates.append(float(row["ates"][outcome]["ate"]))
    mean_ate = sum(ates) / len(ates) if ates else 0.0
    return {
        "outcome": outcome,
        "n_seeds": len(seeds),
        "ate_mean": mean_ate,
        "ates": ates,
        "passes_nonzero_shift": any(abs(v) > 1e-9 for v in ates),
    }
