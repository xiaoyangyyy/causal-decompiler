"""CLI for the Causal Decompiler v2 protocol.

paper      — MRI of one frozen trajectory
benchmark  — 200 mechanism worlds (CI default)
matrix     — 3 scenarios × models × seeds (explicit; not CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.engine.causal.baselines import evaluate_baselines
from src.engine.causal.mechanisms import evaluate_benchmark
from src.engine.simulation import SimConfig
from src.experiments.conditions import list_conditions
from src.experiments.paper_protocol import run_paper_protocol
from src.experiments.paper_tables import render_paper_markdown
from src.experiments.report import generate_report
from src.experiments.runner import run_single
from src.world.loader import PROJECT_ROOT

PAPER_DEFAULT_TOP_K = 1
SCENARIO_OUTCOMES = {
    "labwars": "protest_authorship",
    "crisisgrid": "stranded",
    "releaseops": "task_y",
}
SCENARIO_ROUNDS = {
    "labwars": 8,
    "crisisgrid": 8,
    "releaseops": 8,
}


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _paper_cfg(args: argparse.Namespace) -> SimConfig:
    top_k = args.sampled_top_k if args.sampled_top_k is not None else PAPER_DEFAULT_TOP_K
    scenario = str(getattr(args, "scenario", None) or "labwars")
    return SimConfig(
        max_rounds=args.rounds,
        seed=args.seed,
        mvp=not args.full_cast and scenario == "labwars",
        interventions=[],
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        policy_mode=args.policy_mode,
        cognitive_sampling_top_k=top_k,
        output_dir=PROJECT_ROOT / "output" / "runs",
        scenario=scenario,
    )


def cmd_run(args: argparse.Namespace) -> None:
    result = run_single(args.experiment, args.seed, args.condition, max_rounds=args.rounds)
    log = result["log"]
    print(f"run_id={log.run_id} rounds={len(log.round_records)}")
    for key in result["condition"].primary_outcomes:
        print(f"  {key}={log.outcomes.get(key)}")


def cmd_report(args: argparse.Namespace) -> None:
    path = generate_report(
        run_id=args.run_id,
        experiment_id=args.experiment,
        condition_id=args.condition,
        seed=args.seed,
        output_dir=args.output,
    )
    print(path)


def cmd_paper(args: argparse.Namespace) -> None:
    if args.include_lambda:
        print(
            "warning: --include-lambda rewrites prompts; LLM cache misses are expected.",
            file=sys.stderr,
        )
    factual_path = args.from_jsonl or None
    cfg = None if factual_path else _paper_cfg(args)
    contrasts = [part.strip() for part in (args.contrasts or "").split(",") if part.strip()]
    seeds = _parse_int_list(args.contrast_seeds) if args.contrast_seeds else [args.seed]
    memory_rounds = _parse_int_list(args.memory_rounds) if args.memory_rounds else None
    scenario = str(getattr(args, "scenario", None) or "labwars")
    outcome = args.outcome or SCENARIO_OUTCOMES.get(scenario, "protest_authorship")
    result = run_paper_protocol(
        cfg,
        from_jsonl=factual_path,
        outcome=outcome,
        memory_rounds=memory_rounds,
        auto_battery=not args.lite,
        include_lambda=args.include_lambda,
        contrasts=contrasts or None,
        contrast_seeds=seeds,
        write_output=True,
        output_dir=args.output,
    )
    print(result.summary)
    if result.markdown_path:
        print(result.markdown_path)


def cmd_benchmark(args: argparse.Namespace) -> None:
    bench = evaluate_benchmark(n_per_family=args.n_per_family, seed=args.seed)
    base = evaluate_baselines(n_per_family=min(8, args.n_per_family), seed=args.seed)
    payload = {**bench, "baselines": base}
    print(
        f"n={bench['n']} F1={bench['cause_set_f1']:.3f} "
        f"sign={bench['interaction_sign_accuracy']:.3f} "
        f"false_attr={bench['false_attribution_rate']:.3f}"
    )
    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "benchmark_200.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        md = render_paper_markdown({"benchmark": bench, "baselines": base, "findings": [
            f"200-world F1={bench['cause_set_f1']:.3f}",
        ]})
        (out / "benchmark_200.md").write_text(md, encoding="utf-8")
        print(path)


def matrix_model_for(provider: str, explicit: str | None) -> str | None:
    """Ollama defaults to llama3.2; DeepSeek keeps config/llm.deepseek.yaml."""
    if explicit:
        return explicit
    if provider == "ollama":
        return "llama3.2"
    return None


def _matrix_cell_key(row: dict) -> tuple[str, str, int]:
    return (str(row.get("scenario")), str(row.get("provider")), int(row.get("seed")))


def _write_matrix(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": rows}, indent=2), encoding="utf-8")


def cmd_matrix(args: argparse.Namespace) -> None:
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    seeds = _parse_int_list(args.seeds) if args.seeds else [args.seed]
    top_k = args.sampled_top_k if args.sampled_top_k is not None else PAPER_DEFAULT_TOP_K
    out = Path(args.output) if args.output else PROJECT_ROOT / "output" / "reports"
    path = out / "matrix.json"
    rows: list[dict] = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = list(payload.get("cells") or [])
    done = {_matrix_cell_key(row) for row in rows}
    planned = [(provider, scenario, int(seed)) for provider in providers for scenario in scenarios for seed in seeds]
    total = len(planned)
    for idx, (provider, scenario, seed) in enumerate(planned, start=1):
        key = (scenario, provider, seed)
        if key in done:
            print(f"[{idx}/{total}] skip {scenario} {provider} seed={seed}")
            continue
        cfg = SimConfig(
            max_rounds=int(args.rounds or SCENARIO_ROUNDS.get(scenario, 8)),
            seed=int(seed),
            mvp=False,
            llm_provider=provider,
            llm_model=matrix_model_for(provider, args.llm_model),
            cognitive_sampling_top_k=top_k,
            scenario=scenario,
            output_dir=PROJECT_ROOT / "output" / "runs",
        )
        outcome = SCENARIO_OUTCOMES.get(scenario, "protest_authorship")
        try:
            result = run_paper_protocol(
                cfg,
                outcome=outcome,
                auto_battery=not args.lite,
                write_output=True,
                output_dir=out,
            )
            row = {
                "scenario": scenario,
                "provider": provider,
                "seed": seed,
                "identity": result.report.identity_twin_ok,
                "y": result.report.factual_y,
                "cstar": list((result.report.minimal_cause or {}).get("set") or []),
                "run_id": result.report.factual_run_id,
            }
            print(
                f"[{idx}/{total}] {scenario} {provider} seed={seed} "
                f"identity={result.report.identity_twin_ok} Y={result.report.factual_y:.4f}"
            )
        except Exception as exc:
            row = {
                "scenario": scenario,
                "provider": provider,
                "seed": seed,
                "identity": False,
                "y": None,
                "cstar": [],
                "run_id": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[{idx}/{total}] {scenario} {provider} seed={seed} FAILED {row['error']}")
        rows.append(row)
        done.add(key)
        _write_matrix(path, rows)
    print(path)


def _add_paper_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", "-s", type=int, default=11)
    parser.add_argument("--full-cast", action="store_true", help="14-agent story instead of MVP")
    parser.add_argument("--llm-provider", default="scripted")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--policy-mode", default="dual_engine", choices=["social_physics", "dual_engine", "llm_native"])
    parser.add_argument("--sampled-top-k", type=int, default=None, help="LLM-scored agents per round (default 1)")
    parser.add_argument("--memory-rounds", default="", help="Comma-separated delete times for the memory IRF")
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--scenario", default="labwars", choices=["labwars", "crisisgrid", "releaseops"])
    parser.add_argument("--from-jsonl", default=None, help="Replay MRI from a persisted factual jsonl")
    parser.add_argument("--contrasts", default="", help="Comma-separated experiments, e.g. A or A,C")
    parser.add_argument("--contrast-seeds", default="", help="Comma-separated seeds for CRN contrast table")
    parser.add_argument("--include-lambda", action="store_true", help="Optional field-vs-LLM lesion (cache misses)")
    parser.add_argument("--lite", action="store_true", help="Identity + split-Y only")
    parser.add_argument("--output", "-o", default=None)
    parser.set_defaults(func=cmd_paper)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labwars-experiments",
        description="Causal Decompiler v2: MRI, 200-world benchmark, scenario matrix",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run one A/B/C/D/V condition (debug, not the paper MRI)")
    p_run.add_argument("--experiment", "-e", required=True, help="A|B|C|D|V")
    p_run.add_argument("--condition", "-c", default=None, help="Condition id e.g. A2")
    p_run.add_argument("--seed", "-s", type=int, default=42)
    p_run.add_argument("--rounds", type=int, default=60)
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="Write a trajectory decompilation report")
    p_report.add_argument("--run-id", default=None)
    p_report.add_argument("--experiment", "-e", default="A")
    p_report.add_argument("--condition", "-c", default="A1")
    p_report.add_argument("--seed", "-s", type=int, default=42)
    p_report.add_argument("--output", "-o", default=None)
    p_report.set_defaults(func=cmd_report)

    p_paper = sub.add_parser("paper", help="Paper protocol: MRI of one frozen trajectory")
    _add_paper_args(p_paper)

    p_alias = sub.add_parser("decompile", help="Alias of paper")
    _add_paper_args(p_alias)

    p_bench = sub.add_parser("benchmark", help="200 parameterized mechanism worlds")
    p_bench.add_argument("--n-per-family", type=int, default=50)
    p_bench.add_argument("--seed", "-s", type=int, default=11)
    p_bench.add_argument("--output", "-o", default=None)
    p_bench.set_defaults(func=cmd_benchmark)

    p_matrix = sub.add_parser("matrix", help="Scenario × model × seed grid (not CI)")
    p_matrix.add_argument("--scenarios", default="labwars,crisisgrid,releaseops")
    p_matrix.add_argument("--providers", default="scripted")
    p_matrix.add_argument("--seeds", default="11")
    p_matrix.add_argument("--seed", "-s", type=int, default=11)
    p_matrix.add_argument("--rounds", type=int, default=8)
    p_matrix.add_argument("--llm-model", default=None)
    p_matrix.add_argument("--sampled-top-k", type=int, default=None, help="LLM-scored agents per round (default 1)")
    p_matrix.add_argument("--lite", action="store_true")
    p_matrix.add_argument("--output", "-o", default=None)
    p_matrix.set_defaults(func=cmd_matrix)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "condition", None) is None and hasattr(args, "experiment"):
        try:
            args.condition = list_conditions(args.experiment)[0]
        except (ValueError, KeyError):
            pass
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
