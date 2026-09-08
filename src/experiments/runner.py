"""Single A/B/C/D/V condition run. The paper MRI is `run_paper_protocol`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.engine.simulation import run_simulation
from src.experiments.conditions import build_sim_config, get_condition
from src.experiments.metrics import compute_run_metrics


def run_single(
    experiment_id: str,
    seed: int,
    condition_id: str | None = None,
    *,
    max_rounds: int = 60,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    condition = get_condition(experiment_id, condition_id)
    cfg = build_sim_config(condition, seed, max_rounds=max_rounds, output_dir=str(output_dir) if output_dir else None)
    log = run_simulation(cfg)
    metrics = compute_run_metrics(log)
    return {"log": log, "metrics": metrics, "condition": condition}
