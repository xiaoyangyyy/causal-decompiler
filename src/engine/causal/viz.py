"""Causal certificate HTML + mermaid for the debugger surface."""

from __future__ import annotations

from html import escape
from typing import Any


def mermaid_hypergraph(graph: dict[str, Any] | None) -> str:
    graph = graph or {}
    if graph.get("mermaid"):
        return str(graph["mermaid"])
    return "flowchart LR\n  empty[no interaction]"


def render_certificate_html(report: dict[str, Any]) -> str:
    """Standalone HTML a reviewer can open next to the markdown tables."""
    run = escape(str(report.get("factual_run_id") or ""))
    findings = report.get("findings") or []
    certs = report.get("certificates") or []
    graph = report.get("hypergraph") or {}
    worlds = (report.get("planted_worlds") or {}).get("worlds") or []
    surface = report.get("memory_irf_surface") or {}
    matrix = surface.get("matrix") or {}
    observe = surface.get("observe") or []
    delete = surface.get("delete") or []
    post_auc = surface.get("post_auc") or {}
    info = report.get("information_algebra") or {}
    info_worlds = info.get("worlds") or {}
    paired = report.get("paired_effect") or {}
    ci = paired.get("bootstrap_ci") or {}
    min_cause = report.get("minimal_cause") or {}
    mermaid = escape(mermaid_hypergraph(graph))

    finding_li = "".join(f"<li>{escape(str(x))}</li>" for x in findings[:8])
    cert_rows = []
    for item in certs[:16]:
        op = item.get("intervention") or {}
        effect = item.get("effect") or {}
        fork = item.get("first_meaningful_fork") or {}
        fork_s = ""
        if fork:
            fork_s = f"R{fork.get('round')} {fork.get('agent')} {fork.get('channel')}"
        cert_rows.append(
            "<tr>"
            f"<td><code>{escape(str(op.get('type')))}</code></td>"
            f"<td><code>{escape(str(op.get('factor')))}</code></td>"
            f"<td>{escape(str(effect.get('public', '')))}</td>"
            f"<td>{escape(str(effect.get('private', '')))}</td>"
            f"<td>{escape(fork_s)}</td>"
            "</tr>"
        )
    world_rows = []
    for row in worlds:
        world_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('world')))}</td>"
            f"<td>{escape(str(row.get('recovered')))}</td>"
            f"<td>{'yes' if row.get('minimal_cause_recovery') else 'no'}</td>"
            f"<td>{'yes' if row.get('interaction_recovery') else 'no'}</td>"
            f"<td>{escape(str(row.get('fork_localization')))}</td>"
            "</tr>"
        )
    irf_head = "".join(f"<th>t={t}</th>" for t in observe)
    irf_body = []
    for r in delete:
        cells = "".join(
            f"<td>{(matrix.get(str(r)) or {}).get(str(t), '')}</td>"
            for t in observe
        )
        irf_body.append(f"<tr><th>r={r}</th>{cells}<td>{post_auc.get(str(r), '')}</td></tr>")
    info_rows = []
    for name, payload in info_worlds.items():
        info_rows.append(
            f"<tr><td><code>{escape(name)}</code></td>"
            f"<td>{escape(str((payload or {}).get('ate')))}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Causal certificate {run}</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; color: #111; background: #fff; }}
    h1, h2 {{ font-weight: 600; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; font-size: 14px; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    code {{ font-family: Consolas, monospace; }}
    .meta {{ color: #555; }}
  </style>
</head>
<body>
  <h1>Causal Decompiler certificate</h1>
  <p class="meta">factual run <code>{run}</code> · C* = <code>{escape(str(min_cause.get('set')))}</code></p>
  <h2>Findings</h2>
  <ul>{finding_li or '<li>none</li>'}</ul>
  <h2>Causal hypergraph</h2>
  <pre>{mermaid}</pre>
  <h2>Certificates</h2>
  <table>
    <tr><th>do()</th><th>factor</th><th>Δ public</th><th>Δ private</th><th>meaningful fork</th></tr>
    {''.join(cert_rows) or '<tr><td colspan="5">none</td></tr>'}
  </table>
  <h2>Memory IRF surface (PostAUC-normalized)</h2>
  <table>
    <tr><th>delete \\ t</th>{irf_head}<th>PostAUC</th></tr>
    {''.join(irf_body) or '<tr><td colspan="2">none</td></tr>'}
  </table>
  <h2>Information algebra</h2>
  <table>
    <tr><th>intervention</th><th>ΔY</th></tr>
    {''.join(info_rows) or '<tr><td colspan="2">not run</td></tr>'}
  </table>
  <p class="meta">{escape(str(info.get('identity') or ''))}</p>
  <h2>Paired replicates</h2>
  <p>K={escape(str(paired.get('k', 1)))} mean={escape(str(paired.get('mean', '')))}
     CI=[{escape(str(ci.get('low', '')))}, {escape(str(ci.get('high', '')))}]
     sign={escape(str(paired.get('sign_consistency', '')))}
     fork_p={escape(str(paired.get('fork_probability', '')))}</p>
  <h2>Planted social worlds</h2>
  <table>
    <tr><th>world</th><th>recovered C*</th><th>min-set</th><th>interaction</th><th>fork</th></tr>
    {''.join(world_rows) or '<tr><td colspan="5">none</td></tr>'}
  </table>
</body>
</html>
"""
