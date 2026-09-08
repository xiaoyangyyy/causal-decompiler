"""Agent MRI decompilation report generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.engine.run_log import RunLog
from src.experiments.metrics import compute_run_metrics
from src.experiments.runner import run_single
from src.world.loader import PROJECT_ROOT

TEMPLATE_PATH = PROJECT_ROOT / "config" / "report_template.md"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "output" / "reports"


def _format_timeline(timeline: list[dict[str, Any]]) -> str:
    if not timeline:
        return "_No salient memory nodes recorded._"
    lines = []
    for node in timeline:
        lines.append(
            f"- R{node['round']} | {node['agent']} | {node['event_ref']} | "
            f"strength={node.get('strength', 0):.2f} | valence={node.get('valence', 0):+.2f} | "
            f"{node.get('interpretation', '')}"
        )
    return "\n".join(lines)


def _format_trust_snapshots(snaps: dict[int, dict[str, float]]) -> str:
    if not snaps:
        return "_Trust snapshots unavailable._"
    lines = []
    for rnd in sorted(snaps):
        edges = ", ".join(f"{k.split('_', 1)[1]}={v:.3f}" for k, v in snaps[rnd].items())
        lines.append(f"- R{rnd}: {edges}")
    return "\n".join(lines)


def _format_curve(curve: list[dict[str, float]], key: str) -> str:
    if not curve:
        return "_No data._"
    sample = curve[:: max(1, len(curve) // 10)]
    return "\n".join(f"- R{p['round']}: {key}={p.get(key, 0):.3f}" for p in sample)



def _format_causal_path(path: dict[str, Any]) -> str:
    if not path or not path.get("nodes"):
        return "_No path-level causal chain extracted._"
    lines = [f"Finding: {path.get('finding', '')}".strip()]
    for node in path.get("nodes", []):
        metrics = []
        if "strength" in node:
            metrics.append(f"strength={float(node.get('strength', 0)):.2f}")
        if "valence" in node:
            metrics.append(f"valence={float(node.get('valence', 0)):+.2f}")
        if "intensity" in node:
            metrics.append(f"intensity={float(node.get('intensity', 0)):.2f}")
        suffix = f" ({', '.join(metrics)})" if metrics else ""
        lines.append(
            f"- R{node.get('round')} -> {node.get('kind')}:{node.get('label')} "
            f"[{node.get('event_id') or 'action'}]{suffix} - {node.get('detail', '')}"
        )
    phases = path.get("trajectory_phases", {})
    if phases.get("phases"):
        lines.append("Trajectory phases:")
        for phase in phases.get("phases", []):
            lines.append(f"- {phase.get('name')} | {phase.get('round_range')} | {phase.get('signal')}")
        lines.append(f"Phase transition peak: R{phases.get('phase_transition_round')} pressure={float(phases.get('phase_pressure_peak', 0)):.3f}")
    outcome = path.get("outcome_summary", {})
    lines.append(
        "Outcome: "
        f"protest={float(outcome.get('protest_authorship', 0)):.3f}, "
        f"escalation={float(outcome.get('authorship_escalation_score', 0)):.3f}, "
        f"memory_cluster={float(outcome.get('memory_authorship_cluster_strength', 0)):.3f}, "
        f"promise_broken_R52={float(outcome.get('promise_broken_strength_r52', 0)):.3f}."
    )
    lines.append(f"Counterfactual: {path.get('counterfactual_hint', '')}")
    return "\n".join(lines)


def _format_rank_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return "none"
    return ", ".join(f"{item.get('type')}={float(item.get('score', 0)):.3f}" for item in items)


def _format_llm_scoring_influence(influence: dict[str, Any]) -> str:
    if not influence or not influence.get("scored_action_count"):
        return "_No LLM-scored action candidates recorded._"
    lines = [
        f"- Scored actions: {influence.get('scored_action_count', 0)}",
        f"- Mean LLM Override Pressure: {float(influence.get('mean_override_pressure', 0)):.3f}",
        f"- Max LLM Override Pressure: {float(influence.get('max_override_pressure', 0)):.3f}",
        f"- Mean selected-action rank lift: {float(influence.get('mean_selected_rank_lift', 0)):.3f}",
    ]
    for item in influence.get("examples", [])[:5]:
        ranks = item.get("selected_ranks", {})
        lines.append(
            f"- R{item.get('round')} {item.get('agent')} selected `{item.get('selected_action')}` "
            f"ranks(field={ranks.get('field')}, llm={ranks.get('llm')}, fused={ranks.get('fused')}), "
            f"pressure={float(item.get('override_pressure', 0)):.3f}"
        )
        lines.append(f"  field top3: {_format_rank_items(item.get('field_top3', []))}")
        lines.append(f"  llm top3: {_format_rank_items(item.get('llm_top3', []))}")
        lines.append(f"  fused top3: {_format_rank_items(item.get('fused_top3', []))}")
    return "\n".join(lines)


def _format_force_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return "none"
    return ", ".join(f"{item.get('force')}={float(item.get('contribution', 0)):+.3f}" for item in items)


def _format_action_field_explanations(explanations: list[dict[str, Any]]) -> str:
    if not explanations:
        return "_No action-field decompositions recorded._"
    lines = []
    for item in explanations[:8]:
        mem = item.get("memory_trigger") or {}
        mem_text = mem.get("content_type") or "none"
        if mem.get("memory_id"):
            mem_text = f"{mem.get('content_type')}:{mem.get('memory_id')} attention={float(mem.get('attention', 0)):.3f}"
        lines.append(
            f"- R{item.get('round')} {item.get('agent')} `{item.get('action')}` -> {item.get('target')} "
            f"field={float(item.get('field_tendency', 0)):.3f}, fused={float(item.get('fused_tendency', 0)):.3f}, p={float(item.get('probability', 0)):.3f}"
        )
        lines.append(f"  positive: {_format_force_items(item.get('positive_forces', []))}")
        lines.append(f"  negative: {_format_force_items(item.get('negative_forces', []))}")
        lines.append(f"  memory_trigger: {mem_text}")
    return "\n".join(lines)


def _format_llm_footprint(footprint: dict[str, Any]) -> str:
    if not footprint or not footprint.get("scored_action_count"):
        return "_No LLM influence footprint recorded._"
    lines = [
        f"- Scored actions: {footprint.get('scored_action_count', 0)}",
        f"- Mean field->LLM pressure: {float(footprint.get('mean_field_to_llm_pressure', 0)):.3f}",
        f"- Max field->LLM pressure: {float(footprint.get('max_field_to_llm_pressure', 0)):.3f}",
        f"- Mean selected LLM rank lift: {float(footprint.get('mean_selected_llm_rank_lift', 0)):.3f}",
    ]
    for item in footprint.get("examples", [])[:5]:
        lines.append(
            f"- R{item.get('round')} {item.get('agent')} `{item.get('selected_action')}` "
            f"field_rank={item.get('field_rank')} llm_rank={item.get('llm_rank')} fused_rank={item.get('fused_rank')} "
            f"pressure={float(item.get('field_to_llm_pressure', 0)):.3f}; private_strategy={item.get('private_strategy')}"
        )
        lines.append(f"  field top3: {_format_rank_items(item.get('field_top3', []))}")
        lines.append(f"  llm top3: {_format_rank_items(item.get('llm_top3', []))}")
    return "\n".join(lines)

def _format_causal_mri(report: dict[str, Any] | None) -> str:
    if not report:
        return "_Causal decompiler not attached. Run `python -m src.experiments paper`._"
    lines = [
        f"- Identity twin: {'ok' if report.get('identity_twin_ok') else 'FAILED'}",
        f"- Outcome {report.get('outcome')}: factual Y={float(report.get('factual_y', 0)):.4f}",
    ]
    split = report.get("split_y") or {}
    if split:
        lines.append("- Split Y: " + ", ".join(f"{k}={float(v):.3f}" for k, v in split.items()))
    for finding in (report.get("findings") or [])[:6]:
        lines.append(f"- Finding: {finding}")
    replay = report.get("llm_replay") or {}
    lines.append(
        f"- LLM replay: identity hits={replay.get('identity_run_hits', 0)} "
        f"misses={replay.get('identity_run_misses', 0)}"
    )
    for item in (report.get("memory_irf") or [])[:6]:
        split = (item.get("extras") or {}).get("split") or {}
        ppd = (split.get("public_private_divergence_mean") or {}).get("ate")
        extra = f", ΔPPD={float(ppd):+.4f}" if ppd is not None else ""
        trust = (split.get("trust_pi_path_mean") or {}).get("ate")
        if trust is not None:
            extra += f", Δtrust_path={float(trust):+.4f}"
        lines.append(
            f"- IRF {item.get('factor_id')}: Δprotest={float(item.get('ate', 0)):+.4f}{extra}"
        )
    for item in (report.get("forks") or []):
        if item.get("patch") == "identity" or item.get("identical"):
            continue
        lines.append(
            f"- Fork {item.get('factor_id')}: R{item.get('round')} {item.get('channel')} "
            f"({item.get('agent')})"
        )
        break
    for item in (report.get("contrastive") or [])[:6]:
        lines.append(
            f"- Skip {item.get('factor_id')}: ATE={float(item.get('ate', 0)):+.4f}"
        )
    locus = report.get("point_of_commitment")
    if locus:
        lines.append(f"- Point of commitment: {locus.get('factor_id')} ATE={float(locus.get('ate', 0)):+.4f}")
    story = report.get("story_shapley") or {}
    if story.get("shapley"):
        lines.append(
            "- Story Shapley: "
            + ", ".join(f"{k}={float(v):.3f}" for k, v in story["shapley"].items())
            + f" (AND-lie={story.get('and_lie')})"
        )
    worlds = report.get("three_worlds") or {}
    if worlds:
        lines.append(
            f"- Three-worlds: total={float(worlds.get('ate_total', 0)):+.4f} "
            f"gated={float(worlds.get('gated_channel', 0)):+.4f} "
            f"hypocrisy={float(worlds.get('hypocrisy_index', 0)):+.4f}"
        )
    shapley = report.get("shapley_toy") or {}
    if shapley:
        lines.append("- Planted Shapley: " + ", ".join(f"{k}={v:.2f}" for k, v in shapley.items()))
    for note in (report.get("notes") or [])[:4]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def generate_report_from_log(log: RunLog, metrics: dict[str, Any] | None = None) -> str:
    metrics = metrics or compute_run_metrics(log)
    outcomes = metrics["outcomes"]
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    split = outcomes.get("split_y") if isinstance(outcomes.get("split_y"), dict) else {}
    if not split:
        split = {k: outcomes.get(k, 0) for k in (
            "protest_authorship", "authorship_escalation_potential", "public_private_divergence_mean", "post_r52_compliance",
            "trust_pi_final", "trust_pi_logged", "trust_pi_path_mean", "pi_fairness_r52",
            "promise_broken_strength_r52", "promise_honored_strength_r52",
            "memory_authorship_cluster_strength", "authorship_dispute_index",
        )}
    latent = (
        f"- Split Y: protest={float(split.get('protest_authorship', 0)):.3f}, "
        f"potential={float(split.get('authorship_escalation_potential', outcomes.get('authorship_escalation_potential', 0))):.3f}, "
        f"PPD(mean)={float(split.get('public_private_divergence_mean', 0)):.3f}, "
        f"PPD(last)={float(split.get('public_private_divergence_last', outcomes.get('public_private_divergence_last', 0))):.3f}, "
        f"R52 comply={float(split.get('post_r52_compliance', 0)):.3f}\n"
        f"- Trust: relationship={float(split.get('trust_pi_final', outcomes.get('trust_pi_final', 0))):.3f}, "
        f"logged={float(split.get('trust_pi_logged', outcomes.get('trust_pi_logged', 0))):.3f}, "
        f"path_mean={float(split.get('trust_pi_path_mean', outcomes.get('trust_pi_path_mean', 0))):.3f}, "
        f"pi_fairness_r52={float(split.get('pi_fairness_r52', outcomes.get('pi_fairness_r52', 0))):.3f}\n"
        f"- Memory cluster strength: {float(outcomes.get('memory_authorship_cluster_strength', 0)):.3f}\n"
        f"- Authority compliance: {float(outcomes.get('authority_compliance', 0)):.3f}"
    )

    memory_causal = (
        f"- Authorship memory cluster (R3鈥揜40): {outcomes.get('memory_authorship_cluster_strength', 0):.3f}\n"
        f"- Promise broken strength @R52: {outcomes.get('promise_broken_strength_r52', 0):.3f}\n"
        f"- Memory IRF (not |ΔM/ΔY| mediation) is in the Causal Decompiler MRI section"
    )

    interventions = log.interventions_applied
    inter_lines = "\n".join(
        f"- R{i.get('round')}: {i.get('intervention_id')} ({i.get('variant')})"
        for i in interventions
    ) or "_No interventions applied._"

    div_peaks = metrics.get("divergence_peaks", [])
    div_text = "\n".join(
        f"- R{p['round']} divergence={p['divergence']:.3f} ({p.get('event_id')})" for p in div_peaks[:8]
    ) or "_No divergence ranking available._"

    failure = f"- Critic violations: {metrics.get('critic_count', 0)}\n"
    if log.critic_violations:
        failure += "\n".join(
            f"  - R{v.get('round')} {v.get('agent')}: {v.get('rule_id', v.get('message', 'violation'))}"
            for v in log.critic_violations[:5]
        )

    probes = log.outcomes.get("probe_suggestions") or []
    probe_text = "\n".join(
        f"- R{p.get('round')}: {p.get('variant')} - {p.get('reason')}" for p in probes
    ) or "_No probe suggestions._"

    replacements = {
        "{{run_id}}": log.run_id,
        "{{experiment_id}}": str(metrics.get("experiment_id") or log.config.get("experiment_id", "NA")),
        "{{condition_id}}": str(metrics.get("condition_id") or log.config.get("condition_id", "NA")),
        "{{seed}}": str(metrics.get("seed") or log.config.get("seed", "NA")),
        "{{timeline_section}}": _format_timeline(metrics.get("timeline", [])),
        "{{latent_section}}": latent,
        "{{trust_section}}": _format_trust_snapshots(metrics.get("trust_snapshots", {})),
        "{{authorship_section}}": _format_curve(metrics.get("authorship_dispute_curve", []), "authorship_dispute_index"),
        "{{memory_causal_section}}": memory_causal,
        "{{intervention_section}}": inter_lines,
        "{{divergence_section}}": div_text,
        "{{failure_section}}": failure,
        "{{probe_section}}": probe_text,
        "{{causal_path_section}}": _format_causal_path(metrics.get("path_level_causal_chain", {})),
        "{{llm_scoring_section}}": _format_llm_scoring_influence(metrics.get("llm_scoring_influence", {})),
        "{{action_field_explanation_section}}": _format_action_field_explanations(metrics.get("action_field_explanations", [])),
        "{{llm_footprint_section}}": _format_llm_footprint(metrics.get("llm_influence_footprint", {})),
        "{{causal_mri_section}}": _format_causal_mri(log.outcomes.get("causal_mri")),
    }
    text = template
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def generate_report(
    run_id: str | None = None,
    *,
    experiment_id: str = "A",
    condition_id: str = "A1",
    seed: int = 42,
    output_dir: Path | str | None = None,
    log: RunLog | None = None,
) -> Path:
    out_dir = Path(output_dir) if output_dir else DEFAULT_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if log is None:
        result = run_single(experiment_id, seed, condition_id)
        log = result["log"]
        metrics = result["metrics"]
    else:
        metrics = compute_run_metrics(log)

    rid = run_id or log.run_id
    report_text = generate_report_from_log(log, metrics)
    path = out_dir / f"report_{rid}.md"
    path.write_text(report_text, encoding="utf-8")

    meta = {"run_id": rid, "metrics": metrics}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
