"""Dynamic estimands: MIRF surface, CPG, information algebra, paired K-CI."""

from __future__ import annotations

from typing import Any

from src.engine.causal.algebra import (
    CausalOp,
    do_belief,
    do_behavior,
    do_memory,
    do_presence,
    do_private_public,
    do_visibility,
    resample,
    skip_event,
)
from src.engine.causal.estimands import PAPER_SPLIT_KEYS, first_divergence, split_y
from src.engine.causal.fork import first_meaningful_fork
from src.engine.causal.noise import STREAM_ACTION_GUMBEL, STREAM_ACTION_SAMPLE
from src.engine.causal.twin import run_twin
from src.engine.run_log import PROTEST_ACTIONS, RunLog, extract_outcome
from src.engine.simulation import SimConfig
from src.engine.story_cast import story_cast_from_log

CHANNEL_PRIVATE = "private"
CHANNEL_PUBLIC = "public"
CHANNEL_ACTION = "action"


def cheap_twins(config: SimConfig) -> bool:
    provider = str(config.llm_provider or "scripted")
    return provider == "scripted" or int(config.max_rounds or 0) <= 12


def distributional_k(config: SimConfig) -> int:
    """K=3 only on scripted/short twins. Long LLM MRI stays k=1 CRN."""
    if cheap_twins(config) and int(config.max_rounds or 0) >= 8:
        return 3
    return 1


def index_prior_effects(*groups: Any) -> dict[str, dict[str, Any]]:
    """Map factor_id → {ate, fork, split} from already-run MRI twins."""
    index: dict[str, dict[str, Any]] = {}
    for rows in groups:
        if isinstance(rows, dict):
            rows = rows.get("evaluated") or rows.get("worlds") or []
            if isinstance(rows, dict):
                rows = [
                    {"factor_id": name, **(payload or {})}
                    for name, payload in rows.items()
                ]
        for item in rows or []:
            if not isinstance(item, dict):
                continue
            fid = str(item.get("factor_id") or "")
            if not fid:
                continue
            extras = item.get("extras") or {}
            fork = item.get("fork") or extras.get("fork") or {}
            split_raw = item.get("split") or extras.get("split") or {}
            split: dict[str, float] = {}
            for key, val in (split_raw or {}).items():
                if isinstance(val, dict) and "ate" in val:
                    split[str(key)] = float(val.get("ate") or 0.0)
                else:
                    try:
                        split[str(key)] = float(val)
                    except (TypeError, ValueError):
                        continue
            index[fid] = {
                "ate": float(item.get("ate") or 0.0),
                "fork": fork,
                "split": split,
                "cpg": extras.get("cpg") or item.get("cpg") or {},
                "factor_id": fid,
            }
    return index


def truncate_log(log: RunLog, t: int) -> RunLog:
    """Prefix the transcript through round t so Y_t is identified."""
    cut = RunLog(run_id=f"{log.run_id}:t{t}", config=dict(log.config or {}))
    cut.events = [e for e in log.events if int(e.get("round") or 0) <= t]
    cut.actions = [a for a in log.actions if int(a.get("round") or 0) <= t]
    cut.round_records = [r for r in log.round_records if int(r.get("round") or 0) <= t]
    cut.interventions_applied = [
        item for item in log.interventions_applied
        if int(item.get("round") or item.get("apply_at_round") or 0) <= t
    ]
    cut.noise_log = [d for d in log.noise_log if int(d.get("round") or 0) <= t]
    cut.outcomes = dict(log.outcomes or {})
    return cut


def observe_times(log: RunLog, start: int) -> list[int]:
    rounds = sorted({int(r.get("round") or 0) for r in log.round_records if int(r.get("round") or 0) >= start})
    if not rounds:
        return []
    t_max = rounds[-1]
    if t_max - start <= 12:
        return rounds
    grid = [rnd for rnd in rounds if (rnd - start) % 5 == 0]
    if t_max not in grid:
        grid.append(t_max)
    return grid


def outcome_at(log: RunLog, t: int, outcome: str) -> float:
    return float(extract_outcome(truncate_log(log, t), outcome) or 0.0)


def post_auc(curve: dict[int, float], delete_at: int) -> float:
    """Remaining-time normalized mean of ΔY_t for t >= r."""
    pts = [v for t, v in curve.items() if int(t) >= int(delete_at)]
    if not pts:
        return 0.0
    return float(sum(pts) / len(pts))


def mirf_curve(factual: RunLog, twin: RunLog, outcome: str, delete_at: int) -> dict[str, Any]:
    times = observe_times(factual, delete_at)
    curve = {t: outcome_at(twin, t, outcome) - outcome_at(factual, t, outcome) for t in times}
    return {
        "delete_at": delete_at,
        "curve": curve,
        "post_auc": post_auc(curve, delete_at),
        "terminal": curve.get(max(curve) if curve else delete_at, 0.0),
    }


def mirf_surface_from_twins(
    factual: RunLog,
    twins: list[tuple[int, RunLog]],
    outcome: str,
) -> dict[str, Any]:
    """MIRF_m(r, t) for already-run delete times. No new twins."""
    rows = [mirf_curve(factual, twin, outcome, r) for r, twin in twins]
    observe = sorted({t for row in rows for t in row["curve"]})
    matrix = {
        str(row["delete_at"]): {str(t): row["curve"].get(t) for t in observe}
        for row in rows
    }
    return {
        "observe": observe,
        "delete": [row["delete_at"] for row in rows],
        "matrix": matrix,
        "post_auc": {str(row["delete_at"]): row["post_auc"] for row in rows},
        "terminal": {str(row["delete_at"]): row["terminal"] for row in rows},
    }


def assemble_surface(memory_irf: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in memory_irf or []:
        payload = (item.get("extras") or {}).get("mirf") or {}
        if payload.get("curve") is not None:
            rows.append(payload)
    if not rows:
        return {}
    observe = sorted({int(t) for row in rows for t in (row.get("curve") or {})})
    return {
        "observe": observe,
        "delete": [row.get("delete_at") for row in rows],
        "matrix": {
            str(row.get("delete_at")): {str(t): (row.get("curve") or {}).get(t) for t in observe}
            for row in rows
        },
        "post_auc": {str(row.get("delete_at")): row.get("post_auc") for row in rows},
        "terminal": {str(row.get("delete_at")): row.get("terminal") for row in rows},
    }


def tri_channel_at(log: RunLog, t: int, idea: str) -> dict[str, float]:
    rec = next((r for r in log.round_records if int(r.get("round") or 0) == t), None)
    metrics = (rec or {}).get("metrics") or {}
    private = float(metrics.get("public_private_divergence_idea") or metrics.get("public_private_divergence") or 0.0)
    acts = [a for a in log.actions if int(a.get("round") or 0) == t and str(a.get("agent") or "") == idea]
    public = 0.0
    action = 0.0
    for act in acts:
        stmt = str(((act.get("public_position") or {}).get("statement_type") or "")).lower()
        if stmt in {"self_advocacy", "protest"}:
            public = 1.0
        if str(act.get("type") or act.get("action_type") or "") in PROTEST_ACTIONS:
            action = 1.0
    return {CHANNEL_PRIVATE: private, CHANNEL_PUBLIC: public, CHANNEL_ACTION: action}


def cpg_path(factual: RunLog, twin: RunLog) -> dict[str, Any]:
    """CPG(I, t) = ΔY_private − ΔY_public, plus the action channel."""
    idea = story_cast_from_log(factual).idea
    rounds = sorted({int(r.get("round") or 0) for r in factual.round_records})
    series: list[dict[str, Any]] = []
    for t in rounds:
        y0 = tri_channel_at(factual, t, idea)
        y1 = tri_channel_at(twin, t, idea)
        delta = {k: y1[k] - y0[k] for k in (CHANNEL_PRIVATE, CHANNEL_PUBLIC, CHANNEL_ACTION)}
        series.append({"round": t, **delta, "cpg": delta[CHANNEL_PRIVATE] - delta[CHANNEL_PUBLIC]})
    if not series:
        return {"series": [], "mean_cpg": 0.0, "regime": classify_cpg_regime(0, 0, 0)}
    mean_priv = sum(p[CHANNEL_PRIVATE] for p in series) / len(series)
    mean_pub = sum(p[CHANNEL_PUBLIC] for p in series) / len(series)
    mean_act = sum(p[CHANNEL_ACTION] for p in series) / len(series)
    mean_cpg = sum(p["cpg"] for p in series) / len(series)
    return {
        "series": series,
        "mean_cpg": mean_cpg,
        "delta": {"private": mean_priv, "public": mean_pub, "action": mean_act},
        "regime": classify_cpg_regime(mean_priv, mean_pub, mean_act),
    }


def classify_cpg_regime(private: float, public: float, action: float) -> str:
    priv_hi = abs(private) >= 0.05 or private >= 0.15
    pub_hi = public >= 0.35
    act_hi = action >= 0.08
    # For deltas, use sign-aware magnitude vs factual baseline levels elsewhere.
    if not priv_hi and not pub_hi and not act_hi:
        return "true_compliance"
    if priv_hi and not pub_hi and act_hi:
        return "performative_compliance"
    if priv_hi and pub_hi and not act_hi:
        return "linguistic_protest"
    if priv_hi and not pub_hi and not act_hi:
        return "latent_conflict"
    if priv_hi and pub_hi and act_hi:
        return "overt_conflict"
    return "mixed"


def cpg_from_split(split0: dict[str, float], split1: dict[str, float]) -> dict[str, Any]:
    private = float(split1.get("public_private_divergence_mean") or 0) - float(split0.get("public_private_divergence_mean") or 0)
    public = float(split1.get("protest_authorship") or 0) - float(split0.get("protest_authorship") or 0)
    action = public
    comply0 = float(split0.get("post_r52_compliance") or 0)
    comply1 = float(split1.get("post_r52_compliance") or 0)
    public_expr = (1.0 - comply1) - (1.0 - comply0)
    return {
        "private": private,
        "public": public_expr,
        "action": action,
        "cpg": private - public_expr,
        "regime": classify_cpg_regime(
            abs(private) + float(split0.get("public_private_divergence_mean") or 0),
            1.0 - comply0,
            float(split0.get("protest_authorship") or 0),
        ),
    }


def bootstrap_ci(samples: list[float], *, n_boot: int = 400, alpha: float = 0.05) -> dict[str, float]:
    if not samples:
        return {"low": 0.0, "high": 0.0, "mean": 0.0}
    mean = sum(samples) / len(samples)
    if len(samples) == 1:
        return {"low": mean, "high": mean, "mean": mean}
    rng_mod = 2_147_483_647
    boots: list[float] = []
    seed = 11
    for i in range(n_boot):
        acc = 0.0
        for j in range(len(samples)):
            seed = (seed * 1103515245 + 12345 + i * 17 + j) % rng_mod
            acc += samples[seed % len(samples)]
        boots.append(acc / len(samples))
    boots.sort()
    lo = boots[int(alpha / 2 * (n_boot - 1))]
    hi = boots[int((1 - alpha / 2) * (n_boot - 1))]
    return {"low": lo, "high": hi, "mean": mean}


def paired_replicates(
    base: SimConfig,
    factual: RunLog,
    ops: list[CausalOp],
    outcome: str,
    *,
    k: int = 1,
    factor_id: str | None = None,
) -> dict[str, Any]:
    """Individual paired causal effect. K=1 is deterministic CRN replay.

    Distributional mode resamples the action_sample stream so post-fork LLM
    or scripted draws are not mistaken for the intervention.
    """
    k = max(1, int(k))
    diffs: list[float] = []
    forks: list[dict[str, Any]] = []
    y0 = extract_outcome(factual, outcome)
    for i in range(k):
        extra = [] if i == 0 else [
            resample(STREAM_ACTION_GUMBEL, name="*", salt=i + 1),
            resample(STREAM_ACTION_SAMPLE, name="*", salt=i + 1),
        ]
        twin = run_twin(base, list(ops) + extra, llm_trace=factual.llm_cache)
        y1 = extract_outcome(twin, outcome)
        diffs.append(float(y1) - float(y0))
        forks.append(first_meaningful_fork(factual, twin, outcome=outcome))
    mean = sum(diffs) / len(diffs)
    signs = [1 if d > 1e-9 else (-1 if d < -1e-9 else 0) for d in diffs]
    mean_sign = 1 if mean > 1e-9 else (-1 if mean < -1e-9 else 0)
    sign_consistency = sum(1 for s in signs if s == mean_sign or mean_sign == 0) / len(signs)
    fork_prob = sum(1 for f in forks if not f.get("identical")) / len(forks)
    return {
        "name": "individual_paired_causal_effect",
        "factor_id": factor_id or (ops[0].factor_id() if ops else "NOOP"),
        "k": k,
        "mean": mean,
        "paired_mean_difference": mean,
        "samples": diffs,
        "bootstrap_ci": bootstrap_ci(diffs),
        "sign_consistency": sign_consistency,
        "fork_probability": fork_prob,
        "first_meaningful_fork": forks[0] if forks else None,
        "model": str(base.llm_provider or "scripted"),
        "seed": int(base.seed),
        "notes": "Not an ATE. One factual trajectory, K paired twins.",
    }


def information_algebra(
    base: SimConfig,
    factual: RunLog,
    outcome: str,
    *,
    event_id: str | None = None,
    round_num: int | None = None,
    agent_id: str | None = None,
    prior_effects: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Five do()s that split event / visibility / memory / belief / expression."""
    idea = agent_id or story_cast_from_log(factual).idea
    if round_num is None or event_id is None:
        rec = next((e for e in factual.events if str(e.get("event_id")) in {"E052", event_id or ""}), None)
        rec = rec or (factual.events[-1] if factual.events else None)
        if rec is None:
            return {}
        event_id = str(rec.get("event_id"))
        round_num = int(rec.get("round") or 1)
    ops = {
        "do_presence": do_presence(int(round_num), event_id),
        "do_visibility": do_visibility("hide_idea", event_id=event_id, agent_id=idea, round_num=int(round_num)),
        "do_memory": do_memory(int(round_num), idea),
        "do_belief": do_belief(int(round_num), idea, {"pi_fairness": 0.9, "my_first_author_probability": 0.8}),
        "do_private_public": do_private_public(int(round_num), idea),
    }
    worlds: dict[str, Any] = {}
    y0 = extract_outcome(factual, outcome)
    split0 = split_y(factual)
    prior = dict(prior_effects or {})
    reused = 0
    for name, op in ops.items():
        fid = op.factor_id()
        cached = prior.get(fid)
        if cached is not None:
            reused += 1
            worlds[name] = {
                "factor_id": fid,
                "ate": float(cached.get("ate") or 0.0),
                "split": dict(cached.get("split") or {}),
                "fork": dict(cached.get("fork") or {}),
                "cpg": dict(cached.get("cpg") or {}),
                "reused": True,
            }
            continue
        twin = run_twin(base, [op], llm_trace=factual.llm_cache)
        split1 = split_y(twin)
        worlds[name] = {
            "factor_id": fid,
            "ate": float(extract_outcome(twin, outcome) - y0),
            "split": {key: float(split1.get(key, 0) - split0.get(key, 0)) for key in PAPER_SPLIT_KEYS},
            "fork": first_divergence(factual, twin, outcome=outcome),
            "cpg": cpg_from_split(split0, split1),
            "reused": False,
        }
    return {
        "event_id": event_id,
        "round": round_num,
        "worlds": worlds,
        "reused": reused,
        "identity": (
            "event effect ≠ visibility effect ≠ memory effect ≠ "
            "belief effect ≠ expression-constraint effect"
        ),
    }


def layer_interventions(
    base: SimConfig,
    factual: RunLog,
    outcome: str,
    *,
    event_id: str | None = None,
    round_num: int | None = None,
    agent_id: str | None = None,
    prior_effects: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Paper layer-specific dos: event, visibility, memory, behavior. No do_belief."""
    idea = agent_id or story_cast_from_log(factual).idea
    if round_num is None or event_id is None:
        rec = next((e for e in factual.events if str(e.get("event_id")) in {"E052", event_id or ""}), None)
        rec = rec or (factual.events[-1] if factual.events else None)
        if rec is None:
            return {}
        event_id = str(rec.get("event_id"))
        round_num = int(rec.get("round") or 1)
    ops = {
        "do_event": skip_event(int(round_num), event_id),
        "do_visibility": do_visibility("hide_idea", event_id=event_id, agent_id=idea, round_num=int(round_num)),
        "do_memory": do_memory(int(round_num), idea),
        "do_behavior": do_behavior(int(round_num), idea),
    }
    worlds: dict[str, Any] = {}
    y0 = extract_outcome(factual, outcome)
    split0 = split_y(factual)
    prior = dict(prior_effects or {})
    reused = 0
    for name, op in ops.items():
        fid = op.factor_id()
        cached = prior.get(fid)
        if cached is not None:
            reused += 1
            worlds[name] = {
                "factor_id": fid,
                "ate": float(cached.get("ate") or 0.0),
                "split": dict(cached.get("split") or {}),
                "fork": dict(cached.get("fork") or {}),
                "cpg": dict(cached.get("cpg") or {}),
                "reused": True,
            }
            continue
        twin = run_twin(base, [op], llm_trace=factual.llm_cache)
        split1 = split_y(twin)
        worlds[name] = {
            "factor_id": fid,
            "ate": float(extract_outcome(twin, outcome) - y0),
            "split": {key: float(split1.get(key, 0) - split0.get(key, 0)) for key in PAPER_SPLIT_KEYS},
            "fork": first_divergence(factual, twin, outcome=outcome),
            "cpg": cpg_from_split(split0, split1),
            "reused": False,
        }
    return {
        "event_id": event_id,
        "round": round_num,
        "worlds": worlds,
        "reused": reused,
        "identity": "layer-specific interventions: do_event / do_visibility / do_memory / do_behavior",
    }
