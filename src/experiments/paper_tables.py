"""Paper tables from a Causal MRI report and CRN contrasts.

These are the figures a reviewer should be able to regenerate from JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
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
        "### Table. First meaningful fork",
        "",
        "| Patch | Factor | Kind | Round | Agent | Channel | Factual | Twin |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    if not rows:
        lines.append("| _none_ | | | | | | | |")
        return "\n".join(lines)
    for item in rows:
        if item.get("identical"):
            note = "paraphrase only" if item.get("paraphrase_only") else "identical"
            lines.append(
                f"| `{item.get('patch')}` | `{item.get('factor_id')}` | {note} | | | | | |"
            )
            continue
        lines.append(
            f"| `{item.get('patch')}` | `{item.get('factor_id')}` | {item.get('fork_kind') or ''} | "
            f"{item.get('round')} | `{item.get('agent')}` | {item.get('channel')} | "
            f"`{item.get('factual')}` | `{item.get('twin')}` |"
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


def table_social_ir(report: dict[str, Any]) -> str:
    ir = report.get("social_ir") or {}
    by_layer = ir.get("by_layer") or {}
    lines = [
        "### Table. Five-layer Social Causal IR",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| run | `{ir.get('run_id', report.get('factual_run_id', ''))}` |",
        f"| mechanism | `{ir.get('mechanism_version', '')}` |",
        f"| nodes | {ir.get('node_count', '')} |",
        f"| edges | {ir.get('edge_count', '')} |",
        f"| E | {by_layer.get('E', '')} |",
        f"| I | {by_layer.get('I', '')} |",
        f"| S | {by_layer.get('S', '')} |",
        f"| B | {by_layer.get('B', '')} |",
        f"| Y | {by_layer.get('Y', '')} |",
    ]
    regime = report.get("compliance_regime") or {}
    if regime.get("label"):
        lines.append(f"| regime | `{regime.get('label')}` |")
    return "\n".join(lines)


def table_candidates(report: dict[str, Any]) -> str:
    rows = report.get("candidates") or []
    lines = [
        "### Table. Provenance candidates C(Y)",
        "",
        "| Type | Round | Event | Agent | Why | Suggested do() |",
        "|---|---:|---|---|---|---|",
    ]
    if not rows:
        lines.append("| _none_ | | | | | |")
        return "\n".join(lines)
    for item in rows[:16]:
        lines.append(
            f"| `{item.get('type')}` | {item.get('round')} | `{item.get('source_event') or ''}` | "
            f"`{item.get('agent') or ''}` | {item.get('why')} | `{item.get('suggested_op')}` |"
        )
    return "\n".join(lines)


def table_minimal_cause(report: dict[str, Any]) -> str:
    cause = report.get("minimal_cause") or {}
    interaction = report.get("interaction") or cause.get("interaction") or {}
    lines = [
        "### Table. Minimal Effect-Recovery Set C*_α",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| C*_α | `{', '.join(str(x) for x in (cause.get('set') or []))}` |",
        f"| α | {cause.get('alpha', 0.9)} |",
        f"| size | {cause.get('size', '')} |",
        f"| reason | {cause.get('reason', '')} |",
        f"| Harsanyi I | {_fmt(interaction.get('index'), 4)} ({interaction.get('kind', '')}) |",
        f"| factors | `{', '.join(str(x) for x in (interaction.get('factors') or []))}` |",
    ]
    sensitivity = report.get("cstar_sensitivity") or cause.get("sensitivity") or {}
    if sensitivity:
        lines += ["", "α sensitivity:"]
        for alpha, payload in sensitivity.items():
            payload = payload or {}
            lines.append(
                f"- α={alpha}: `{', '.join(str(x) for x in (payload.get('set') or []))}` "
                f"(size {payload.get('size', '')})"
            )
    return "\n".join(lines)


def table_certificates(report: dict[str, Any]) -> str:
    rows = report.get("certificates") or []
    lines = [
        "### Table. Causal certificates",
        "",
        "| Target | do() | Δ public | Δ private | Meaningful fork | Interaction |",
        "|---|---|---:|---:|---|---|",
    ]
    if not rows:
        lines.append("| _none_ | | | | | |")
        return "\n".join(lines)
    for item in rows[:12]:
        op = item.get("intervention") or {}
        effect = item.get("effect") or {}
        fork = item.get("first_meaningful_fork") or {}
        fork_cell = ""
        if fork:
            fork_cell = f"R{fork.get('round')} {fork.get('agent')} {fork.get('channel')}"
        lines.append(
            f"| `{item.get('target')}` | `{op.get('type')}:{op.get('factor')}` | "
            f"{_fmt(effect.get('public'), 4)} | {_fmt(effect.get('private'), 4)} | "
            f"{fork_cell} | `{', '.join(str(x) for x in (item.get('interaction_set') or []))}` |"
        )
    return "\n".join(lines)


def table_mirf_surface(report: dict[str, Any]) -> str:
    surface = report.get("memory_irf_surface") or {}
    observe = surface.get("observe") or []
    delete = surface.get("delete") or []
    matrix = surface.get("matrix") or {}
    post = surface.get("post_auc") or {}
    lines = [
        "### Table. Memory IRF surface MIRF(r, t) and PostAUC",
        "",
    ]
    if not delete or not observe:
        lines.append("_Surface not computed._")
        return "\n".join(lines)
    header = "| delete \\ t | " + " | ".join(f"t={t}" for t in observe) + " | PostAUC |"
    sep = "|" + "|".join(["---"] + ["---:"] * (len(observe) + 1)) + "|"
    lines += [header, sep]
    for r in delete:
        row = matrix.get(str(r)) or {}
        cells = " | ".join(_fmt(row.get(str(t)), 3) for t in observe)
        lines.append(f"| r={r} | {cells} | {_fmt(post.get(str(r)), 3)} |")
    return "\n".join(lines)


def table_slice(report: dict[str, Any]) -> str:
    stats = report.get("slice_stats") or {}
    lines = [
        "### Table. Ancestor slice compression",
        "",
        "| Stage | Count |",
        "|---|---:|",
        f"| graph nodes | {stats.get('graph_nodes', '')} |",
        f"| Y ancestors | {stats.get('ancestors', '')} |",
        f"| candidates | {stats.get('candidates', '')} |",
        f"| Top-k | {stats.get('top_k', '')} |",
    ]
    return "\n".join(lines)


def table_bidirectional(report: dict[str, Any]) -> str:
    payload = report.get("bidirectional_search") or {}
    lines = [
        "### Table. Bidirectional search (deletion + restoration)",
        "",
        "| Direction | Factor | ATE | Reused |",
        "|---|---|---:|---|",
    ]
    rows = list(payload.get("deletion") or []) + list(payload.get("restoration") or [])
    if not rows:
        lines.append("| _none_ | | | |")
        return "\n".join(lines)
    for item in rows[:16]:
        name = str(item.get("name") or "")
        direction = "restore" if name.startswith("restore") else "delete"
        reused = "yes" if item.get("reused") else "no"
        lines.append(
            f"| {direction} | `{item.get('factor_id')}` | {_fmt(item.get('ate'), 4)} | {reused} |"
        )
    extra = []
    if payload.get("top_k") is not None:
        extra.append(f"Top-k={payload.get('top_k')}")
    if payload.get("new_twins") is not None:
        extra.append(f"new twins={payload.get('new_twins')}")
    if extra:
        lines += ["", "; ".join(extra) + "."]
    return "\n".join(lines)


def table_layer_interventions(report: dict[str, Any]) -> str:
    info = report.get("layer_interventions") or {}
    worlds = info.get("worlds") or {}
    lines = [
        "### Table. Layer-specific interventions",
        "",
        "| do() | ΔY | fork |",
        "|---|---:|---|",
    ]
    if not worlds:
        lines.append("| _none_ | | |")
        return "\n".join(lines)
    for name, payload in worlds.items():
        fork = (payload or {}).get("fork") or {}
        fork_s = "identical" if fork.get("identical") else f"R{fork.get('round')} {fork.get('channel')}"
        lines.append(f"| `{name}` | {_fmt((payload or {}).get('ate'), 4)} | {fork_s} |")
    return "\n".join(lines)


def table_channels(report: dict[str, Any]) -> str:
    ch = report.get("channels") or {}
    split = report.get("split_y") or {}
    lines = [
        "### Table. Three-channel Y / PPG / PCI",
        "",
        "| Channel | Value |",
        "|---|---:|",
        f"| Y private | {_fmt(ch.get('y_private', split.get('y_private')), 4)} |",
        f"| Y public | {_fmt(ch.get('y_public', split.get('y_public')), 4)} |",
        f"| Y action | {_fmt(ch.get('y_action', split.get('y_action')), 4)} |",
        f"| PPG | {_fmt(ch.get('ppg', split.get('ppg')), 4)} |",
        f"| PCI | {_fmt(ch.get('pci', split.get('pci')), 4)} |",
    ]
    for key in ("evac_delay", "stranded", "resource_util", "deploy_failure", "rollback_time", "outage", "task_y"):
        if key in ch:
            lines.append(f"| `{key}` | {_fmt(ch.get(key), 4)} |")
    return "\n".join(lines)


def table_benchmark(report: dict[str, Any]) -> str:
    bench = report.get("benchmark") or {}
    if not bench:
        return ""
    lines = [
        "### Table. 200 mechanism worlds",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n | {bench.get('n', '')} |",
        f"| cause-set F1 | {_fmt(bench.get('cause_set_f1'), 3)} |",
        f"| interaction-sign accuracy | {_fmt(bench.get('interaction_sign_accuracy'), 3)} |",
        f"| intervention budget | {_fmt(bench.get('intervention_budget_mean'), 1)} |",
        f"| false attribution | {_fmt(bench.get('false_attribution_rate'), 3)} |",
    ]
    return "\n".join(lines)


def table_baselines(report: dict[str, Any]) -> str:
    payload = report.get("baselines") or {}
    if not payload:
        return ""
    lines = [
        "### Table. Baselines and ablations",
        "",
        "| Method | F1 | False attr. |",
        "|---|---:|---:|",
    ]
    for group in ("baselines", "ablations"):
        for name, row in (payload.get(group) or {}).items():
            lines.append(f"| `{name}` | {_fmt(row.get('f1'), 3)} | {_fmt(row.get('false_attribution'), 3)} |")
    return "\n".join(lines)


def table_hierarchical_search(report: dict[str, Any]) -> str:
    payload = report.get("hierarchical_search") or {}
    rows = payload.get("evaluated") or []
    lines = [
        "### Table. Hierarchical search",
        "",
        "| Layer | Name | Factor | ATE | Reused | Fork |",
        "|---|---|---|---:|---|---|",
    ]
    if not rows:
        lines.append("| _none_ | | | | | |")
        return "\n".join(lines)
    for item in rows:
        fork = item.get("fork") or {}
        fork_s = "identical" if fork.get("identical") else f"R{fork.get('round')} {fork.get('channel')}"
        reused = "yes" if item.get("reused") else "no"
        lines.append(
            f"| `{item.get('layer')}` | {item.get('name')} | `{item.get('factor_id')}` | "
            f"{_fmt(item.get('ate'), 4)} | {reused} | {fork_s} |"
        )
    extra = []
    if payload.get("new_twins") is not None:
        extra.append(f"new twins={payload.get('new_twins')}")
    if extra:
        lines += ["", "; ".join(extra) + "."]
    return "\n".join(lines)


def table_paired_effect(report: dict[str, Any]) -> str:
    pe = report.get("paired_effect") or {}
    lines = [
        "### Table. Individual paired causal effect",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| k | {pe.get('k', '')} |",
        f"| factor | `{pe.get('factor_id') or ''}` |",
        f"| mean | {_fmt(pe.get('mean', pe.get('paired_mean_difference')), 4)} |",
        f"| notes | {pe.get('notes', '')} |",
    ]
    return "\n".join(lines)


def table_hypergraph(report: dict[str, Any]) -> str:
    graph = report.get("hypergraph") or {}
    mermaid = graph.get("mermaid") or ""
    lines = [
        "### Causal hypergraph (Harsanyi interactions)",
        "",
        f"Kind `{graph.get('kind', '')}`; C* `{', '.join(str(x) for x in (graph.get('minimal_cause') or []))}`; "
        f"regime `{graph.get('regime') or ''}`.",
        "",
    ]
    if mermaid:
        lines += ["```mermaid", mermaid, "```"]
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


def _cstar_cell(values: list[list[str]]) -> str:
    unique = sorted({tuple(item) for item in values})
    if not unique:
        return ""
    if len(unique) == 1:
        return "{" + ", ".join(unique[0]) + "}"
    return "; ".join("{" + ", ".join(item) + "}" for item in unique)


def table_matrix_cells(payload: dict[str, Any]) -> str:
    cells = list(payload.get("cells") or [])
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cells:
        key = (str(row.get("scenario") or ""), str(row.get("provider") or ""))
        groups.setdefault(key, []).append(row)
    lines = [
        "### Table. LLM protocol matrix (8 rounds, sampled-top-k=1)",
        "",
        "Protocol gate, not the 14-agent / 60-round social MRI.",
        "",
        "| Scenario | Provider | n | Identity | Mean Y | $C_{0.9}^*$ | Errors |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    order = (
        ("labwars", "ollama"),
        ("labwars", "deepseek"),
        ("crisisgrid", "ollama"),
        ("crisisgrid", "deepseek"),
        ("releaseops", "ollama"),
        ("releaseops", "deepseek"),
    )
    seen = set()
    for key in list(order) + [k for k in groups if k not in order]:
        rows = groups.get(key) or []
        if not rows or key in seen:
            continue
        seen.add(key)
        n = len(rows)
        ident = sum(1 for row in rows if row.get("identity") is True)
        ys = [float(row["y"]) for row in rows if row.get("y") is not None]
        mean_y = sum(ys) / len(ys) if ys else None
        errors = sum(1 for row in rows if row.get("error"))
        cstar = _cstar_cell([list(row.get("cstar") or []) for row in rows])
        scenario, provider = key
        provider_label = "llama3.2" if provider == "ollama" else provider
        lines.append(
            f"| {scenario} | {provider_label} | {n} | {ident}/{n} | "
            f"{_fmt(mean_y, 4) if mean_y is not None else ''} | `{cstar}` | {errors} |"
        )
    n_all = len(cells)
    ident_all = sum(1 for row in cells if row.get("identity") is True)
    err_all = sum(1 for row in cells if row.get("error"))
    lines += [
        "",
        f"Total identity {ident_all}/{n_all}; error cells {err_all}.",
    ]
    return "\n".join(lines)


def table_benchmark_families(bench: dict[str, Any]) -> str:
    families = bench.get("by_family") or {}
    lines = [
        "### Table. 200 worlds by mechanism family",
        "",
        "| Family | n | F1 | Sign acc. | False attr. |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in families.items():
        note = " (paper caveat)" if name == "and_synergy" and float(row.get("sign_accuracy") or 0) < 1 else ""
        lines.append(
            f"| `{name}`{note} | {row.get('n', '')} | {_fmt(row.get('f1'), 3)} | "
            f"{_fmt(row.get('sign_accuracy'), 3)} | {_fmt(row.get('false_attribution'), 3)} |"
        )
    return "\n".join(lines)


def render_aggregate_tables(
    *,
    benchmark: dict[str, Any],
    matrix: dict[str, Any],
    sixty_round: dict[str, Any] | None = None,
) -> str:
    bench = dict(benchmark)
    baselines = bench.pop("baselines", None) if "baselines" in bench and "cause_set_f1" in bench else None
    if baselines is None and "baselines" in benchmark:
        baselines = benchmark.get("baselines")
        bench = {k: v for k, v in benchmark.items() if k != "baselines"}
    findings = [
        f"200-world F1={_fmt(bench.get('cause_set_f1'), 3)}; "
        f"interaction-sign={_fmt(bench.get('interaction_sign_accuracy'), 3)}; "
        f"false attribution={_fmt(bench.get('false_attribution_rate'), 3)}.",
        "AND-family sign accuracy is 0.40; OR / delayed / suppressor are 1.00.",
        "60-cell LLM matrix is an 8-round protocol gate (identity + C*), not 14×60 social dynamics.",
    ]
    if sixty_round:
        run_id = sixty_round.get("factual_run_id") or ""
        findings.append(
            "Cached 60-round jsonl `3f05b630` failed identity (hits=129, misses=601); "
            f"fresh DeepSeek 14×60 is run `{run_id}`."
        )
    shell = {"benchmark": bench, "baselines": baselines or {}, "findings": findings}
    sections = [
        "# Causal Decompiler — aggregate paper tables",
        "## Findings",
        "\n".join(f"- {line}" for line in shell["findings"]),
        table_benchmark(shell),
        table_benchmark_families(bench),
        table_baselines(shell),
        table_matrix_cells(matrix),
    ]
    if sixty_round:
        sections += ["## 14-agent / 60-round MRI", _sixty_round_section(sixty_round)]
    return "\n\n".join(section for section in sections if section) + "\n"


def _sixty_round_section(report: dict[str, Any]) -> str:
    cause = report.get("minimal_cause") or {}
    channels = report.get("channels") or {}
    split = report.get("split_y") or {}
    replay = report.get("llm_replay") or {}
    cstar = ", ".join(str(x) for x in (cause.get("set") or []))
    y_private = channels.get("y_private", split.get("y_private"))
    y_public = channels.get("y_public", split.get("y_public"))
    lines = [
        table_identity(report),
        "",
        table_channels(report),
        "",
        table_minimal_cause(report),
        "",
        table_layer_interventions(report),
        "",
        table_memory_irf(report),
        "",
        "8-round LabWars matrix Y≈0.0115, $C^*$={E003}. "
        f"This 60-round run Y={_fmt(report.get('factual_y'), 4)}, $C^*$=`{{{cstar}}}`; "
        f"private={_fmt(y_private, 4)}, public={_fmt(y_public, 4)}, "
        f"PPG={_fmt(channels.get('ppg', split.get('ppg')), 4)}.",
        "",
        f"Replay hits={replay.get('identity_run_hits', replay.get('run_hits', ''))} "
        f"misses={replay.get('identity_run_misses', replay.get('run_misses', ''))}.",
    ]
    return "\n".join(lines)


def write_aggregate_tables(
    *,
    benchmark_path: str | Path,
    matrix_path: str | Path,
    output_path: str | Path,
    sixty_round: dict[str, Any] | None = None,
) -> Path:
    bench = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    text = render_aggregate_tables(benchmark=bench, matrix=matrix, sixty_round=sixty_round)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def render_paper_markdown(
    report: dict[str, Any],
    *,
    contrasts: list[dict[str, Any]] | None = None,
) -> str:
    findings = report.get("findings") or []
    finding_block = "\n".join(f"- {line}" for line in findings) or "_No auto-findings._"
    sections = [
        "# Causal Decompiler v2 — paper tables",
        "",
        "## Findings",
        "",
        finding_block,
        "",
        table_identity(report),
        "",
        table_social_ir(report),
        "",
        table_slice(report),
        "",
        table_channels(report),
        "",
        table_split_y(report),
        "",
        table_memory_irf(report),
        "",
        table_mirf_surface(report),
        "",
        table_forks(report),
        "",
        table_candidates(report),
        "",
        table_hierarchical_search(report),
        "",
        table_bidirectional(report),
        "",
        table_minimal_cause(report),
        "",
        table_layer_interventions(report),
        "",
        table_hypergraph(report),
        "",
        table_paired_effect(report),
        "",
        table_certificates(report),
        "",
        table_benchmark(report),
        "",
        table_baselines(report),
        "",
        table_contrasts(contrasts or []),
    ]
    return "\n".join(section for section in sections if section is not None).replace("\n\n\n", "\n\n") + "\n"
