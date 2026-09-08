"""CLI for the Causal Decompiler paper protocol.

The conference object is an MRI of one frozen 14-agent / 60-round trajectory.
Scale matrices, independent-seed ATE grids, and egalitarian/sampling sweeps
are not part of this surface.

Commands:
  paper   — MRI tables plus optional CRN contrasts (the public entry)
  run     — one A/B/C/D/V condition (debug, not the paper MRI)
  report  — trajectory markdown from a condition or existing log
"""

from __future__ import annotations

import argparse
import sys

from src.engine.simulation import SimConfig
from src.experiments.conditions import list_conditions
from src.experiments.paper_protocol import run_paper_protocol
from src.experiments.report import generate_report
from src.experiments.runner import run_single
from src.world.loader import PROJECT_ROOT

PAPER_DEFAULT_TOP_K = 1


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _paper_cfg(args: argparse.Namespace) -> SimConfig:
    top_k = args.sampled_top_k if args.sampled_top_k is not None else PAPER_DEFAULT_TOP_K
    return SimConfig(
        max_rounds=args.rounds,
        seed=args.seed,
        mvp=not args.full_cast,
        interventions=[],
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        policy_mode=args.policy_mode,
        cognitive_sampling_top_k=top_k,
        output_dir=PROJECT_ROOT / "output" / "runs",
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
    result = run_paper_protocol(
        cfg,
        from_jsonl=factual_path,
        outcome=args.outcome,
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


def _add_paper_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", "-s", type=int, default=11)
    parser.add_argument("--full-cast", action="store_true", help="14-agent story instead of MVP")
    parser.add_argument("--llm-provider", default="scripted")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--policy-mode", default="dual_engine", choices=["social_physics", "dual_engine", "llm_native"])
    parser.add_argument("--sampled-top-k", type=int, default=None, help="LLM-scored agents per round (default 1)")
    parser.add_argument("--memory-rounds", default="", help="Comma-separated delete times for the memory IRF")
    parser.add_argument("--outcome", default="protest_authorship")
    parser.add_argument("--from-jsonl", default=None, help="Replay MRI from a persisted factual jsonl")
    parser.add_argument("--contrasts", default="", help="Comma-separated experiments, e.g. A or A,C")
    parser.add_argument("--contrast-seeds", default="", help="Comma-separated seeds for CRN contrast table")
    parser.add_argument("--include-lambda", action="store_true", help="Optional field-vs-LLM lesion (cache misses)")
    parser.add_argument("--lite", action="store_true", help="Identity + split-Y + toy Shapley only")
    parser.add_argument("--output", "-o", default=None)
    parser.set_defaults(func=cmd_paper)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labwars-experiments",
        description="LabWars Causal Decompiler: MRI of one frozen lab trajectory",
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

    p_paper = sub.add_parser("paper", help="Paper protocol: MRI tables plus optional CRN contrasts")
    _add_paper_args(p_paper)

    p_alias = sub.add_parser("decompile", help="Alias of paper (MRI only; pass --contrasts for CRN pairs)")
    _add_paper_args(p_alias)

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
