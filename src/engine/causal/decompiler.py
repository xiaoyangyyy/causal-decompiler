"""Causal Decompiler: factual run → frozen U → patched IR → MRI report.

This is the user-facing pass. Estimators stay thin; the report is the
decompiled listing of what actually moved the outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.engine.causal.algebra import CausalOp
from src.engine.causal.estimands import (
    EffectEstimate,
    and_event_ids,
    contrastive_event_effects,
    default_memory_irf_rounds,
    draft_beat_op,
    first_divergence,
    lambda_lesion_effects,
    memory_irf,
    paired_effect,
    paired_split_effects,
    point_of_commitment,
    split_y,
    story_shapley,
    three_worlds,
)
from src.engine.causal.toy import contrastive_leave_one_out, exact_shapley, planted_factors, planted_outcome
from src.engine.causal.twin import identity_holds, run_factual, run_twin, sim_config_from_log
from src.engine.run_log import RunLog, extract_outcome
from src.engine.simulation import SimConfig


@dataclass
class CausalMRIReport:
    outcome: str
    factual_y: float
    split_y: dict[str, float]
    identity_twin_ok: bool
    total_effects: list[dict[str, Any]] = field(default_factory=list)
    split_effects: list[dict[str, Any]] = field(default_factory=list)
    memory_irf: list[dict[str, Any]] = field(default_factory=list)
    contrastive: list[dict[str, Any]] = field(default_factory=list)
    point_of_commitment: dict[str, Any] | None = None
    shapley_toy: dict[str, float] = field(default_factory=dict)
    contrastive_toy_lie: dict[str, float] = field(default_factory=dict)
    story_shapley: dict[str, Any] = field(default_factory=dict)
    three_worlds: dict[str, Any] = field(default_factory=dict)
    forks: list[dict[str, Any]] = field(default_factory=list)
    lambda_effects: list[dict[str, Any]] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    llm_replay: dict[str, Any] = field(default_factory=dict)
    factual_run_id: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        irf = self.memory_irf[0] if self.memory_irf else None
        skip = self.contrastive[0] if self.contrastive else None
        replay = self.llm_replay or {}
        lines = [
            f"run={self.factual_run_id} identity={self.identity_twin_ok} Y={self.factual_y:.4f}",
            (
                f"replay hits={replay.get('identity_run_hits', 0)} "
                f"misses={replay.get('identity_run_misses', 0)}"
            ),
        ]
        lines.extend(f"finding: {item}" for item in (self.findings or [])[:6])
        if irf:
            split = (irf.get("extras") or {}).get("split") or {}
            ppd = (split.get("public_private_divergence_mean") or {}).get("ate")
            pot = (split.get("authorship_escalation_potential") or {}).get("ate")
            extra = f" ΔPPD={float(ppd):+.4f}" if ppd is not None else ""
            if pot is not None:
                extra += f" Δpotential={float(pot):+.4f}"
            trust = (split.get("trust_pi_path_mean") or {}).get("ate")
            if trust is not None:
                extra += f" Δtrust_path={float(trust):+.4f}"
            lines.append(f"memory IRF {irf.get('factor_id')} ATE={float(irf.get('ate', 0.0)):+.4f}{extra}")
        fork = next(
            (item for item in self.forks if not item.get("identical") and item.get("patch") != "identity"),
            None,
        )
        if fork:
            lines.append(f"fork {fork.get('factor_id')} R{fork.get('round')} {fork.get('channel')}")
        if skip:
            lines.append(f"contrastive {skip.get('factor_id')} ATE={float(skip.get('ate', 0.0)):+.4f}")
        return "\n".join(lines)


def _dump_effects(effects: list[EffectEstimate]) -> list[dict[str, Any]]:
    return [asdict(e) for e in effects]


def _fork_entry(patch: str, factor_id: str, fork: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(fork or {})
    payload["patch"] = patch
    payload["factor_id"] = factor_id
    return payload


def _split_ate(item: dict[str, Any], key: str) -> float:
    split = item.get("extras", {}).get("split") or {}
    return float((split.get(key) or {}).get("ate") or 0.0)


def _findings(report: CausalMRIReport) -> list[str]:
    lines: list[str] = []
    if report.identity_twin_ok:
        lines.append("Identity twin reproduced the factual trajectory under frozen U and LLM replay.")
    else:
        lines.append("Identity twin failed — later ATEs are not CRN-identified.")
    split = report.split_y or {}
    protest = float(split.get("protest_authorship", report.factual_y) or 0.0)
    potential = float(split.get("authorship_escalation_potential", 0.0) or 0.0)
    ppd = float(split.get("public_private_divergence_mean", 0.0) or 0.0)
    comply = float(split.get("post_r52_compliance", 0.0) or 0.0)
    trust_logged = float(split.get("trust_pi_logged", 0.0) or 0.0)
    trust_path = float(split.get("trust_pi_path_mean", 0.0) or 0.0)
    if trust_logged <= 0.061 and trust_path <= 0.10:
        lines.append(
            f"Trust channel is pinned near the recovery floor "
            f"(logged={trust_logged:.3f}, path_mean={trust_path:.3f}); "
            "ATEs on trust_pi_logged are not identified."
        )
    elif trust_path > 0.12:
        lines.append(
            f"Trust path-mean ({trust_path:.3f}) is above the floor — "
            "do(M)/skip can move the private relationship channel."
        )
    if protest < 0.05 and potential > 0.08:
        lines.append(
            f"Public protest ({protest:.3f}) is action-gated; latent authorship potential "
            f"({potential:.3f}) is the live MRI channel."
        )
    if ppd > 0.15 and protest < 0.05:
        lines.append(
            f"Split-Y: private divergence ({ppd:.3f}) is large while public protest ({protest:.3f}) stays compressed — "
            "the hidden transcript is the estimand, not the binary revolt."
        )
    if comply > 0.5 and ppd > 0.2:
        lines.append("Public compliance coexists with private divergence (hypocrisy / dual transcript).")
    shapley = report.story_shapley or {}
    if shapley.get("and_lie"):
        lines.append(
            "Contrastive skip overcounts AND causes "
            f"(sum={float(shapley.get('contrastive_sum', 0)):.3f} vs total={float(shapley.get('total_effect', 0)):.3f}); "
            "Shapley splits the joint effect."
        )
    irf = report.memory_irf or []
    if len(irf) >= 2:
        ranked = sorted(
            irf,
            key=lambda item: max(
                abs(float(item.get("ate", 0.0))),
                abs(_split_ate(item, "authorship_escalation_potential")),
            ),
        )
        early = irf[0]
        late = irf[-1]
        if abs(float(late.get("ate", 0.0))) < abs(float(early.get("ate", 0.0))) * 0.5:
            lines.append(
                "Memory IRF: late deletion moves Y less than early deletion — the past has already committed."
            )
        best = ranked[-1]
        lines.append(
            f"Largest memory-IRF move: {best.get('factor_id')} ATE={float(best.get('ate', 0.0)):+.4f}."
        )
        for item in irf:
            d_pub = _split_ate(item, "protest_authorship")
            d_pot = _split_ate(item, "authorship_escalation_potential")
            d_ppd = _split_ate(item, "public_private_divergence_mean")
            d_comply = _split_ate(item, "post_r52_compliance")
            if abs(d_pot) > abs(d_pub) + 0.01:
                lines.append(
                    f"IRF {item.get('factor_id')}: latent potential Δ={d_pot:+.3f} moves while "
                    f"public protest Δ={d_pub:+.3f} stays compressed."
                )
                break
            if abs(d_ppd) > abs(d_pub) + 0.02 and abs(d_ppd) > abs(d_comply) + 0.02:
                lines.append(
                    f"IRF {item.get('factor_id')}: private PPD Δ={d_ppd:+.3f} moves more than "
                    f"public protest Δ={d_pub:+.3f} / R52 comply Δ={d_comply:+.3f}."
                )
                break
    for fork in report.forks or []:
        if fork.get("patch") == "identity":
            continue
        if fork.get("identical"):
            continue
        lines.append(
            f"Fork: {fork.get('factor_id')} first leaves the factual transcript at "
            f"R{fork.get('round')} ({fork.get('agent')}) on the {fork.get('channel')} channel."
        )
        break
    worlds = report.three_worlds or {}
    if worlds.get("hypocrisy_index") is not None and abs(float(worlds.get("hypocrisy_index") or 0.0)) > 0.02:
        lines.append(
            f"Three-worlds hypocrisy index={float(worlds['hypocrisy_index']):+.3f}: "
            "private divergence moved more than public compliance under the same do()."
        )
    toy = report.shapley_toy or {}
    lie = report.contrastive_toy_lie or {}
    if toy and lie:
        lines.append(
            "Planted AND oracle: Shapley "
            + ", ".join(f"{k}={v:.2f}" for k, v in toy.items())
            + "; contrastive knockout "
            + ", ".join(f"{k}={v:.0f}" for k, v in lie.items())
            + "."
        )
    return lines


class CausalDecompiler:
    def __init__(self) -> None:
        self.last_log = None

    def decompile(
        self,
        config: SimConfig,
        *,
        outcome: str = "protest_authorship",
        extra_ops: list[CausalOp] | None = None,
        memory_rounds: list[int] | None = None,
        blame_event_ids: list[str] | None = None,
        blame_limit: int | None = None,
        include_toy_shapley: bool = True,
        auto_battery: bool = True,
        include_story_shapley: bool | None = None,
        include_three_worlds: bool | None = None,
        include_lambda: bool = False,
        factual: RunLog | None = None,
    ) -> CausalMRIReport:
        if factual is None:
            factual = run_factual(config)
        return self._analyze(
            config,
            factual,
            outcome=outcome,
            extra_ops=extra_ops,
            memory_rounds=memory_rounds,
            blame_event_ids=blame_event_ids,
            blame_limit=blame_limit,
            include_toy_shapley=include_toy_shapley,
            auto_battery=auto_battery,
            include_story_shapley=include_story_shapley,
            include_three_worlds=include_three_worlds,
            include_lambda=include_lambda,
        )

    def decompile_log(
        self,
        factual: RunLog,
        config: SimConfig | None = None,
        **kwargs: Any,
    ) -> CausalMRIReport:
        cfg = config or sim_config_from_log(factual)
        return self.decompile(cfg, factual=factual, **kwargs)

    def _analyze(
        self,
        config: SimConfig,
        factual: RunLog,
        *,
        outcome: str,
        extra_ops: list[CausalOp] | None,
        memory_rounds: list[int] | None,
        blame_event_ids: list[str] | None,
        blame_limit: int | None,
        include_toy_shapley: bool,
        auto_battery: bool,
        include_story_shapley: bool | None,
        include_three_worlds: bool | None,
        include_lambda: bool,
    ) -> CausalMRIReport:
        twin0 = run_twin(config, [], llm_trace=factual.llm_cache)
        notes = [
            "Abduction is event-keyed NoiseLog, not a global PRNG queue.",
            "LLM outputs are record-replayed from the factual prompt cache.",
            "Memory IRF is an interventional analogue, not a natural indirect effect.",
            "Contrastive skip lies on AND causes (E003 × E052); Shapley on the planted SCM is the oracle.",
            "Split-Y is the paper estimand: public compliance and private divergence are not the same Y.",
        ]
        report = CausalMRIReport(
            outcome=outcome,
            factual_y=extract_outcome(factual, outcome),
            split_y=split_y(factual),
            identity_twin_ok=identity_holds(factual, twin0),
            notes=notes,
            forks=[_fork_entry("identity", "NOOP", first_divergence(factual, twin0))],
        )
        if not report.identity_twin_ok:
            report.notes.append("FAIL: no-op twin diverged from factual run.")
        twin_stats = twin0.outcomes.get("llm_trace_stats") or {}
        if twin_stats.get("run_misses", 0) == 0 and twin_stats.get("run_hits", 0) > 0:
            report.notes.append("Identity twin replayed every LLM prompt (zero cache misses).")

        for op in extra_ops or []:
            twin = run_twin(config, [op], llm_trace=factual.llm_cache)
            effect = paired_effect(factual, twin, outcome, name=op.kind, factor_id=op.factor_id())
            effect.extras["fork"] = first_divergence(factual, twin)
            report.total_effects.append(asdict(effect))
            report.split_effects.append({
                "factor_id": op.factor_id(),
                "split": {k: asdict(v) for k, v in paired_split_effects(factual, twin, name=op.kind, factor_id=op.factor_id()).items()},
            })
            report.forks.append(_fork_entry(op.kind, op.factor_id(), effect.extras["fork"]))

        if memory_rounds is None and auto_battery:
            memory_rounds = default_memory_irf_rounds(factual)
        if memory_rounds:
            report.memory_irf = _dump_effects(
                memory_irf(config, factual, outcome, memory_rounds)
            )
            for item in report.memory_irf:
                report.forks.append(_fork_entry("memory_irf", str(item.get("factor_id")), (item.get("extras") or {}).get("fork")))

        event_ids = blame_event_ids
        if event_ids is None:
            if auto_battery:
                event_ids = and_event_ids(factual)
            else:
                limit = 3 if blame_limit is None else max(0, int(blame_limit))
                event_ids = [e["event_id"] for e in factual.events[:limit]]
        if event_ids:
            contrastive = contrastive_event_effects(config, factual, outcome, event_ids)
            report.contrastive = _dump_effects(contrastive)
            locus = point_of_commitment(contrastive)
            report.point_of_commitment = asdict(locus) if locus else None
            for item in report.contrastive:
                report.forks.append(_fork_entry("skip", str(item.get("factor_id")), (item.get("extras") or {}).get("fork")))

        want_shapley = include_story_shapley if include_story_shapley is not None else auto_battery
        if want_shapley:
            report.story_shapley = story_shapley(config, factual, and_event_ids(factual), outcome)

        want_worlds = include_three_worlds if include_three_worlds is not None else auto_battery
        if want_worlds:
            op = draft_beat_op(factual)
            if op is not None:
                report.three_worlds = three_worlds(config, factual, op, outcome)
                report.forks.append(
                    _fork_entry("three_worlds", op.factor_id(), report.three_worlds.get("fork_w1"))
                )

        if include_lambda:
            report.lambda_effects = _dump_effects(lambda_lesion_effects(config, factual, outcome))

        if include_toy_shapley:
            factors = planted_factors()
            report.shapley_toy = exact_shapley(planted_outcome, factors)
            report.contrastive_toy_lie = contrastive_leave_one_out(factors, factors)
            report.notes.append(
                "Planted AND: factual knockout credits 1+1 (overcount); Shapley splits 0.5/0.5/0."
            )

        from src.engine.probe import ProbeAgent

        report.factual_run_id = factual.run_id
        report.llm_replay = {
            "identity_run_hits": twin_stats.get("run_hits", 0),
            "identity_run_misses": twin_stats.get("run_misses", 0),
            "cache": factual.outcomes.get("llm_trace_stats") or {},
        }
        report.findings = _findings(report)
        report.probes = ProbeAgent().suggest_from_mri(report)
        factual.outcomes["causal_mri"] = report.to_dict()
        heuristic = list(factual.outcomes.get("probe_suggestions") or [])
        factual.outcomes["probe_suggestions"] = heuristic + report.probes
        self.last_log = factual
        return report
