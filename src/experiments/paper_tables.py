"""Paper tables from a Causal MRI report and CRN contrasts.

These are the figures a reviewer should be able to regenerate from JSON.
"""

from __future__ import annotations

from typing import Any


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def table_split_y(report: dict[str, Any]) -> str:
    split = report.get("split_y") or {}
    lines = [
        "### Table. Split-Y (public vs private)",
        "",
        "| Estimand | Value |",
        "|---|---:|",
    ]
    for key, val in split.items():
        lines.append(f"| `{key}` | {_fmt(val, 4)} |")
    return "\n".join(lines)


def table_memory_irf(report: dict[str, Any]) -> str:
    rows = report.get("memory_irf") or []
    lines = [
        "### Table. Memory IRF (public vs private over delete-time)",
        "",
        "| Delete at | Δ protest | Δ potential | Δ PPD | Δ trust_path | Δ cluster | Fork |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    if not rows:
        lines.append("| _none_ | | | | | | |")
        return "\n".join(lines)
    for item in rows:
        split = (item.get("extras") or {}).get("split") or {}
        fork = (item.get("extras") or {}).get("fork") or {}
        fork_cell = "identical" if fork.get("identical") else f"R{fork.get('round')} {fork.get('channel')}"
        lines.append(
            f"| `{item.get('factor_id')}` | "
            f"{_fmt((split.get('protest_authorship') or {}).get('ate', item.get('ate')), 4)} | "
            f"{_fmt((split.get('authorship_escalation_potential') or {}).get('ate'), 4)} | "
            f"{_fmt((split.get('public_private_divergence_mean') or {}).get('ate'), 4)} | "
            f"{_fmt((split.get('trust_pi_path_mean') or {}).get('ate'), 4)} | "
            f"{_fmt((split.get('memory_authorship_cluster_strength') or {}).get('ate'), 4)} | "
            f"{fork_cell} |"
        )
    return "\n".join(lines)


def table_forks(report: dict[str, Any]) -> str:
    rows = report.get("forks") or []
    lines = [
        "### Table. First divergence from the factual transcript",
        "",
        "| Patch | Factor | Round | Agent | Channel | Factual | Twin |",
        "|---|---|---:|---|---|---|---|",
    ]
    if not rows:
        lines.append("| _none_ | | | | | | |")
        return "\n".join(lines)
    for item in rows:
        if item.get("identical"):
            lines.append(
                f"| `{item.get('patch')}` | `{item.get('factor_id')}` | | | identical | | |"
            )
            continue
        lines.append(
            f"| `{item.get('patch')}` | `{item.get('factor_id')}` | {item.get('round')} | "
            f"`{item.get('agent')}` | {item.get('channel')} | `{item.get('factual')}` | `{item.get('twin')}` |"
        )
    return "\n".join(lines)


def table_shapley(report: dict[str, Any]) -> str:
    toy = report.get("shapley_toy") or {}
    lie = report.get("contrastive_toy_lie") or {}
    story = report.get("story_shapley") or {}
    lines = [
        "### Table. Contrastive skip lies on AND causes; Shapley does not",
        "",
        "| Factor | Planted Shapley | Planted knockout | Story Shapley | Story knockout |",
        "|---|---:|---:|---:|---:|",
    ]
    keys = list(dict.fromkeys([*toy.keys(), *lie.keys(), *(story.get("shapley") or {}).keys(), *(story.get("contrastive") or {}).keys()]))
    if not keys:
        lines.append("| _none_ | | | | |")
        return "\n".join(lines)
    story_phi = story.get("shapley") or {}
    story_ko = story.get("contrastive") or {}
    for key in keys:
        lines.append(
            f"| `{key}` | {_fmt(toy.get(key), 3)} | {_fmt(lie.get(key), 3)} | "
            f"{_fmt(story_phi.get(key), 4)} | {_fmt(story_ko.get(key), 4)} |"
        )
    if story:
        lines += [
            "",
            f"Story total effect `{_fmt(story.get('total_effect'), 4)}`; "
            f"contrastive sum `{_fmt(story.get('contrastive_sum'), 4)}`; "
            f"interaction `{_fmt(story.get('interaction'), 4)}`; "
            f"AND-lie={story.get('and_lie')}.",
        ]
    return "\n".join(lines)


def table_three_worlds(report: dict[str, Any]) -> str:
    worlds = report.get("three_worlds") or {}
    lines = [
        "### Table. Three worlds (factual / do / do+omniscient)",
        "",
        "| World | Y | PPD | R52 comply | cluster |",
        "|---|---:|---:|---:|---:|",
    ]
    if not worlds:
        lines.append("| _not run_ | | | | |")
        return "\n".join(lines)
    split = worlds.get("split") or {}
    y = worlds.get("y") or {}
    for world in ("w0", "w1", "w2"):
        s = split.get(world) or {}
        lines.append(
            f"| {world} | {_fmt(y.get(world), 4)} | {_fmt(s.get('public_private_divergence_mean'), 4)} | "
            f"{_fmt(s.get('post_r52_compliance'), 4)} | {_fmt(s.get('memory_authorship_cluster_strength'), 4)} |"
        )
    lines += [
        "",
        f"ATE total={_fmt(worlds.get('ate_total'), 4)}; "
        f"omniscient={_fmt(worlds.get('ate_omniscient'), 4)}; "
        f"gated channel={_fmt(worlds.get('gated_channel'), 4)}; "
        f"hypocrisy index={_fmt(worlds.get('hypocrisy_index'), 4)} "
        f"on `{worlds.get('factor_id', '')}`.",
    ]
    return "\n".join(lines)


def table_identity(report: dict[str, Any]) -> str:
    replay = report.get("llm_replay") or {}
    lines = [
        "### Table. Identity twin (CRN + LLM replay)",
        "",
        "| Check | Value |",
        "|---|---|",
        f"| identity_twin_ok | {report.get('identity_twin_ok')} |",
        f"| factual Y | {_fmt(report.get('factual_y'), 4)} |",
        f"| replay hits | {replay.get('identity_run_hits', replay.get('run_hits', ''))} |",
        f"| replay misses | {replay.get('identity_run_misses', replay.get('run_misses', ''))} |",
        f"| run | `{report.get('factual_run_id', '')}` |",
    ]
    return "\n".join(lines)


def table_contrasts(contrasts: list[dict[str, Any]]) -> str:
    lines = [
        "### Table. CRN-paired condition contrasts (split-Y)",
        "",
        "| Pair | Outcome | ATE | Control | Treatment | N |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if not contrasts:
        lines.append("| _none_ | | | | | |")
        return "\n".join(lines)
    for row in contrasts:
        lines.append(
            f"| {row.get('label')} | `{row.get('outcome')}` | {_fmt(row.get('ate'), 4)} | "
            f"{_fmt(row.get('control_mean'), 4)} | {_fmt(row.get('treatment_mean'), 4)} | {row.get('n', 1)} |"
        )
    return "\n".join(lines)


def latex_split_y(report: dict[str, Any]) -> str:
    split = report.get("split_y") or {}
    rows = "\n".join(
        "%s & %s \\\\" % (str(key).replace("_", r"\_"), _fmt(val, 4))
        for key, val in split.items()
    )
    return (
        "\\begin{tabular}{lr}\n\\toprule\nEstimand & Value \\\\\n\\midrule\n"
        + rows
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def render_paper_markdown(
    report: dict[str, Any],
    *,
    contrasts: list[dict[str, Any]] | None = None,
) -> str:
    findings = report.get("findings") or []
    finding_block = "\n".join(f"- {line}" for line in findings) or "_No auto-findings._"
    sections = [
        "# Causal Decompiler — paper tables",
        "",
        "## Findings",
        "",
        finding_block,
        "",
        table_identity(report),
        "",
        table_split_y(report),
        "",
        table_memory_irf(report),
        "",
        table_forks(report),
        "",
        table_shapley(report),
        "",
        table_three_worlds(report),
        "",
        table_contrasts(contrasts or []),
    ]
    return "\n".join(sections) + "\n"
