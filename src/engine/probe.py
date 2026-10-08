"""Probe Agent — suggest interventions with continuous activation scores."""

from __future__ import annotations

import math
from typing import Any


def _gate(x: float, center: float, steepness: float = 6.0) -> float:
    z = max(-20.0, min(20.0, (x - center) * steepness))
    return 1.0 / (1.0 + math.exp(-z))


class ProbeAgent:
    def suggest(self, round_records: list[dict[str, Any]], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not round_records:
            return []

        latest = round_records[-1]
        metrics = latest.get("metrics", {})
        rnd = latest.get("round", 0)

        tf = float(metrics.get("trust_fragmentation", 0))
        adi = float(metrics.get("authorship_dispute_index", 0))
        ppd = float(metrics.get("public_private_divergence", 0))
        round_gate = _gate(float(rnd), center=30.0, steepness=0.25)

        suggestions = [
            {
                "round": rnd + 1,
                "intervention_type": "memory_intervention",
                "variant": "memory_correct_false_rumor",
                "activation": round(_gate(tf, center=0.55), 4),
                "reason": f"trust_fragmentation={tf:.3f}; continuous trust-recovery probe",
            },
            {
                "round": rnd + 1,
                "intervention_type": "authorship_framing",
                "variant": "merit_frame",
                "activation": round(_gate(adi, center=0.45) * round_gate, 4),
                "reason": f"authorship_dispute_index={adi:.3f}; continuous merit-framing probe",
            },
            {
                "round": rnd + 1,
                "intervention_type": "memory_intervention",
                "variant": "memory_strengthen_betrayal",
                "target_event": "E031",
                "activation": round(_gate(ppd, center=0.35), 4),
                "reason": f"public_private_divergence={ppd:.3f}; continuous betrayal-memory probe",
            },
        ]
        suggestions.sort(key=lambda p: p["activation"], reverse=True)
        return suggestions

    def suggest_from_mri(self, report: Any) -> list[dict[str, Any]]:
        """Turn a CausalMRIReport into the next interventions worth running."""
        suggestions: list[dict[str, Any]] = []
        irf = list(getattr(report, "memory_irf", None) or [])
        if irf:
            best = max(irf, key=lambda item: abs(float(item.get("ate", 0.0))))
            suggestions.append({
                "round": _round_from_factor(str(best.get("factor_id", ""))),
                "intervention_type": "memory_intervention",
                "variant": "memory_delete_pi_promise",
                "activation": round(min(1.0, abs(float(best.get("ate", 0.0)))), 4),
                "reason": (
                    f"causal decompiler IRF {best.get('factor_id')} "
                    f"ATE={float(best.get('ate', 0.0)):+.3f} on {getattr(report, 'outcome', 'Y')}"
                ),
                "source": "causal_decompiler",
            })
        contrastive = list(getattr(report, "contrastive", None) or [])
        if contrastive:
            best = max(contrastive, key=lambda item: abs(float(item.get("ate", 0.0))))
            suggestions.append({
                "round": _round_from_factor(str(best.get("factor_id", ""))),
                "intervention_type": "event_skip",
                "variant": "skip",
                "activation": round(min(1.0, abs(float(best.get("ate", 0.0)))), 4),
                "reason": (
                    f"contrastive skip {best.get('factor_id')} "
                    f"ATE={float(best.get('ate', 0.0)):+.3f}; "
                    f"commitment={getattr(report, 'point_of_commitment', None)}"
                ),
                "source": "causal_decompiler",
            })
        split = getattr(report, "split_y", None) or {}
        protest = float(split.get("protest_authorship", 0.0))
        private = float(split.get("public_private_divergence_mean", 0.0))
        if protest != 0.0 or private != 0.0:
            suggestions.append({
                "round": 0,
                "intervention_type": "observe_lock",
                "variant": "omniscient",
                "activation": round(min(1.0, max(abs(protest), abs(private))), 4),
                "reason": (
                    f"split-Y: protest={protest:.3f} vs public_private_divergence={private:.3f}"
                ),
                "source": "causal_decompiler",
            })
        suggestions.sort(key=lambda p: p["activation"], reverse=True)
        return suggestions


def _round_from_factor(factor_id: str) -> int:
    for part in factor_id.split(":"):
        if part.startswith("r") and part[1:].isdigit():
            return int(part[1:])
    return 0