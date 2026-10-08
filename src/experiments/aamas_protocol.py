"""Pre-registered AAMAS grid.

Primary endpoints, fixed before the run, are the three outcome channels
y_private, y_public, and y_action. Contrasts are the differences of
seed-means. They are not passed through a cutoff.

The cross-model contrast uses two providers. If the second provider cannot
be contacted, that contrast is recorded as unavailable and is not replaced
by another channel after the fact.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.engine.run_log import three_channel_y
from src.engine.simulation import SimConfig, run_simulation
from src.world.loader import PROJECT_ROOT

CHANNELS = ("y_private", "y_public", "y_action")
# Anchor fraction scales every anchor's salience. It does not delete events.
# ReleaseOps in papers/results/aamas_preregistered.json was run for 16 rounds
# with index thinning. Do not overwrite that file.
SCENARIO_ROUNDS = {
    "labwars": 60,
    "crisisgrid": 24,
    "releaseops": 18,
}
DEFAULT_SEEDS = (0, 1, 2)


def cell_config(
    scenario: str,
    seed: int,
    provider: str,
    *,
    anchor_keep_fraction: float = 1.0,
    rounds: int | None = None,
    model: str | None = None,
    policy_mode: str = "dual_engine",
    cognitive_sampling_top_k: int | None = 1,
    causal_do: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> SimConfig:
    """Build one cell.

    The recorded grid uses dual_engine and top_k=1. Pass policy_mode=\"llm_native\"
    and cognitive_sampling_top_k=None to let every participant propose actions.
    """
    full = scenario == "labwars"
    return SimConfig(
        max_rounds=int(rounds if rounds is not None else SCENARIO_ROUNDS[scenario]),
        seed=seed,
        mvp=not full,
        llm_provider=provider,
        llm_model=model,
        scenario=scenario,
        anchor_keep_fraction=anchor_keep_fraction,
        policy_mode=policy_mode,
        cognitive_sampling_top_k=cognitive_sampling_top_k,
        causal_do=dict(causal_do or {}),
        output_dir=output_dir,
    )


def run_cell(config: SimConfig) -> dict[str, Any]:
    log = run_simulation(config)
    channels = three_channel_y(log)
    return {
        "scenario": config.scenario,
        "seed": config.seed,
        "provider": config.llm_provider or "scripted",
        "anchor_keep_fraction": config.anchor_keep_fraction,
        "rounds": config.max_rounds,
        "policy_mode": config.policy_mode,
        "cognitive_sampling_top_k": config.cognitive_sampling_top_k,
        "causal_do": dict(config.causal_do or {}),
        "run_id": log.run_id,
        "channels": {key: float(channels.get(key, 0.0)) for key in CHANNELS},
    }


def _means(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals = {key: 0.0 for key in CHANNELS}
    for row in rows:
        for key in CHANNELS:
            totals[key] += float(row["channels"][key])
    n = len(rows) or 1
    return {key: totals[key] / n for key in CHANNELS}


def contrast_diffs(left: dict[str, float], right: dict[str, float]) -> dict[str, Any]:
    return {"diffs": {key: float(right[key]) - float(left[key]) for key in CHANNELS}}


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report endpoint differences. Does not choose a new endpoint or a cutoff."""
    by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_anchor: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_provider[str(row["provider"])].append(row)
        by_anchor[float(row["anchor_keep_fraction"])].append(row)
    providers = sorted(by_provider)
    model_contrast: dict[str, Any]
    if len(providers) < 2:
        model_contrast = {
            "status": "second_model_unavailable",
            "providers": providers,
        }
    else:
        first, second = providers[0], providers[1]
        model_contrast = {
            "status": "compared",
            "providers": [first, second],
            **contrast_diffs(_means(by_provider[first]), _means(by_provider[second])),
        }
    anchor_contrast: dict[str, Any]
    if 1.0 in by_anchor and 0.5 in by_anchor:
        anchor_contrast = {
            "status": "compared",
            **contrast_diffs(_means(by_anchor[1.0]), _means(by_anchor[0.5])),
        }
    else:
        anchor_contrast = {"status": "not_run"}
    claim = "cross_model_not_identified" if model_contrast["status"] == "second_model_unavailable" else "differences_reported"
    return {
        "claim": claim,
        "model_contrast": model_contrast,
        "anchor_contrast": anchor_contrast,
    }


def run_grid(
    *,
    scenarios: tuple[str, ...] = ("labwars", "crisisgrid", "releaseops"),
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    providers: tuple[str, ...] = ("scripted",),
    anchor_fractions: tuple[float, ...] = (1.0, 0.5),
    rounds: int | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for scenario in scenarios:
        for provider in providers:
            for fraction in anchor_fractions:
                for seed in seeds:
                    try:
                        rows.append(run_cell(cell_config(
                            scenario,
                            seed,
                            provider,
                            anchor_keep_fraction=fraction,
                            rounds=rounds,
                            model=model,
                        )))
                    except Exception as exc:
                        errors.append({
                            "scenario": scenario,
                            "provider": provider,
                            "seed": str(seed),
                            "anchor_keep_fraction": str(fraction),
                            "error": f"{type(exc).__name__}: {exc}",
                        })
    return {"rows": rows, "errors": errors, "decision": decide(rows)}


def write_grid(payload: dict[str, Any], path: Path | None = None) -> Path:
    recorded = PROJECT_ROOT / "papers" / "results" / "aamas_preregistered.json"
    if path is None:
        raise FileExistsError(
            "The completed grid is papers/results/aamas_preregistered.json "
            "(ReleaseOps was 16 rounds under the presence score). "
            "Pass an explicit path for a new run."
        )
    if path.resolve() == recorded.resolve():
        raise FileExistsError(f"Refusing to overwrite the completed grid at {recorded}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def probe_provider(provider: str, model: str | None = None) -> str | None:
    """One short round. Returns an error string when the provider is unusable."""
    try:
        run_cell(cell_config("releaseops", 0, provider, rounds=1, model=model))
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None
