"""Role Policy Agent - LLM candidate scoring plus constrained stance rendering."""

from __future__ import annotations

import math
import os
from typing import Any

from src.cognition.memory import RecallResult
from src.cognition.pressure_fields import compute_pressure_fields
from src.cognition.social_potential import SOCIAL_POTENTIAL_DIMENSIONS, compute_social_potential
from src.engine.action_selection import generate_action_candidates
from src.engine.causal.noise import STREAM_ACTION_SAMPLE, keyed_uniform
from src.engine.diversity import (
    avoid_actions,
    filter_allowed_actions,
)
from src.engine.llm_adapter import LLMAdapter, LLMError, QuotaExhaustedError
from src.engine.prompts import (
    ACTION_SCORING_SYSTEM,
    LLM_NATIVE_POLICY_SYSTEM,
    ROLE_POLICY_SYSTEM,
    build_action_scoring_prompt,
    build_llm_native_policy_prompt,
    build_role_policy_prompt,
)
from src.world.actions import ActionType, get_allowed_actions
from src.world.models import Agent, EventAtom, WorldState
from src.world.organization import default_private_intent, observation_gain, primary_authority, resolve_event_cast

from .event_agent import is_agent_active


def _pick_target(agent: Agent, world: WorldState, event: EventAtom, suggested: str | None) -> str:
    if suggested and (suggested in world.agents or suggested in {"project", "shared_doc"}):
        return suggested
    if event.source != agent.id and event.source in world.agents:
        return event.source
    internal = world.world_config.get("internal_agents", [])
    authority = primary_authority(world, agent)
    if authority and authority in internal and agent.id != authority:
        return authority
    for t in event.targets:
        if t in world.agents and t != agent.id:
            return t
    return "project"


def _normalize_action_response(
    raw: dict[str, Any],
    agent: Agent,
    event: EventAtom,
    world: WorldState,
    allowed: list[ActionType],
) -> dict[str, Any]:
    allowed_values = {a.value for a in allowed}
    primary = raw.get("primary_action") or raw
    atype = str(primary.get("type", ""))
    if atype not in allowed_values:
        atype = allowed[0].value

    intensity = float(primary.get("intensity", 0.5))
    intensity = max(0.0, min(1.0, intensity))
    target = _pick_target(agent, world, event, primary.get("target"))

    comm = raw.get("communication_action") or {}
    comm_type = str(comm.get("type", "share_result"))
    comm_target = _pick_target(agent, world, event, comm.get("target"))

    public = raw.get("public_position") or {"statement_type": "neutral", "authorship_claim": "any_authorship"}
    private = raw.get("private_intent") or default_private_intent(agent, atype)

    return {
        "agent": agent.id,
        "type": atype,
        "target": target,
        "intensity": round(intensity, 4),
        "communication_action": {
            "type": comm_type,
            "target": comm_target,
            "content_summary": str(comm.get("content_summary", f"{agent.id} re {event.type}")),
        },
        "public_position": _align_public_to_action(atype, public),
        "private_intent": private,
        "llm_raw": raw,
    }


def _softmax(values: list[float], temperature: float = 0.22) -> list[float]:
    if not values:
        return []
    temp = max(temperature, 1e-4)
    peak = max(values)
    exps = [math.exp((v - peak) / temp) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _sample_payload(payloads: list[dict[str, Any]], *, seed: int, round_num: int, agent_id: str) -> dict[str, Any]:
    if not payloads:
        raise ValueError("No action payloads to sample")
    needle = keyed_uniform(seed, round_num, STREAM_ACTION_SAMPLE, agent_id=agent_id, name="llm_fused")
    total = 0.0
    for payload in payloads:
        total += float(payload.get("probability", 0.0))
        if needle <= total:
            return payload
    return payloads[-1]


_DRAFT_CREDIT_ACTIONS = frozenset({
    "ask_for_authorship", "privately_lobby_pi", "confront", "challenge_claim",
    "document_contribution", "request_mediation", "cite_prior_memory",
    "leak_concern", "rebel", "withdraw",
})

# Public credit claims cannot be rendered as team_support (lobby may stay hypocritical).
_PUBLIC_CREDIT_ACTIONS = frozenset({
    "ask_for_authorship", "confront", "challenge_claim", "document_contribution",
    "cite_prior_memory", "rebel", "withdraw", "request_mediation",
})
_TEAM_SUPPORT_ACTIONS = frozenset({"comply", "support_teammate"})


def stance_prior_for_action(action_type: str) -> dict[str, str]:
    if action_type in _PUBLIC_CREDIT_ACTIONS:
        return {"statement_type": "self_advocacy", "authorship_claim": "first_author"}
    if action_type in _TEAM_SUPPORT_ACTIONS:
        return {"statement_type": "team_support", "authorship_claim": "any_authorship"}
    return {"statement_type": "neutral", "authorship_claim": "any_authorship"}


def _align_public_to_action(action_type: str, public: dict[str, Any] | None) -> dict[str, Any]:
    aligned = dict(public or {})
    stmt = str(aligned.get("statement_type") or "neutral")
    if action_type in _PUBLIC_CREDIT_ACTIONS and stmt in {"team_support", "neutral", ""}:
        aligned["statement_type"] = "self_advocacy"
        if not aligned.get("authorship_claim") or aligned.get("authorship_claim") == "any_authorship":
            aligned["authorship_claim"] = "first_author"
    elif action_type in _TEAM_SUPPORT_ACTIONS and stmt in {"self_advocacy", ""}:
        aligned["statement_type"] = "team_support"
    aligned.setdefault("statement_type", "neutral")
    aligned.setdefault("authorship_claim", "any_authorship")
    return aligned


def _focus_draft_payloads(event: EventAtom, agent: Agent, world: WorldState, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if event.type != "authorship_draft":
        return payloads
    if agent.id != (resolve_event_cast(world).idea or "phd_a"):
        return payloads
    credit = [p for p in payloads if str(p.get("type")) in _DRAFT_CREDIT_ACTIONS]
    pool = credit or payloads
    pool = sorted(
        pool,
        key=lambda item: float(item.get("fused_tendency", item.get("probability", item.get("tendency", 0.0)))),
        reverse=True,
    )[:3]
    key = "fused_tendency" if any("fused_tendency" in item for item in pool) else "probability"
    return _normalize_probability_payloads(pool, key=key, temperature=0.09)


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, score))




def _scripted_render_action(agent: Agent, event: EventAtom, selected_payload: dict[str, Any]) -> dict[str, Any]:
    """Render a selected field action without an LLM call for unsampled agents."""
    action_type = str(selected_payload.get("type", "document_contribution"))
    target = str(selected_payload.get("target") or "project")
    return {
        "agent": agent.id,
        "type": action_type,
        "target": target,
        "intensity": round(float(selected_payload.get("intensity", 0.5)), 4),
        "communication_action": {
            "type": "share_result",
            "target": target,
            "content_summary": f"{agent.id} follows field-selected {action_type} under {event.type}",
        },
        "public_position": _align_public_to_action(
            action_type, {"statement_type": "neutral", "authorship_claim": "any_authorship"},
        ),
        "private_intent": {
            "goal": default_private_intent(agent, action_type)["goal"],
            "strategy": action_type,
            "trust_pi": agent.beliefs.pi_fairness,
        },
        "llm_raw": {"source": "scripted_unsampled_render"},
    }


def _normalize_probability_payloads(
    payloads: list[dict[str, Any]],
    key: str = "fused_tendency",
    temperature: float = 0.22,
) -> list[dict[str, Any]]:
    if not payloads:
        return []
    probs = _softmax([float(item.get(key, 0.0)) for item in payloads], temperature=temperature)
    out = []
    for item, prob in zip(payloads, probs):
        copied = dict(item)
        copied["probability"] = round(prob, 5)
        out.append(copied)
    return out


def _cognitive_sampling_scores(
    world: WorldState,
    event: EventAtom,
    recalls: dict[str, RecallResult],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    top_k = config.get("cognitive_sampling_top_k")
    if top_k is None:
        return {}
    threshold = float(config.get("cognitive_sampling_threshold", 0.0) or 0.0)
    rows: list[dict[str, Any]] = []
    for agent_id, agent in world.agents.items():
        if not is_agent_active(agent_id, event.round, config):
            continue
        potential = compute_social_potential(world, agent, event, recalls.get(agent_id))
        dims = potential.dimensions
        score = max(0.0, min(1.0,
            0.34 * float(dims.get("uncertainty", 0.0))
            + 0.24 * float(dims.get("memory_pressure", 0.0))
            + 0.18 * float(dims.get("trust_deficit", 0.0))
            + 0.14 * float(dims.get("power_constraint", 0.0))
            + 0.10 * float(potential.total_pressure)
        ))
        rows.append({"agent": agent_id, "score": round(score, 4), "dimensions": dims})
    rows.sort(key=lambda item: (float(item["score"]), str(item["agent"])), reverse=True)
    selected = {item["agent"] for item in rows[: max(0, int(top_k))] if float(item["score"]) >= threshold}
    return {
        item["agent"]: {
            "enabled": True,
            "sampled": item["agent"] in selected,
            "score": item["score"],
            "top_k": int(top_k),
            "threshold": threshold,
            "rank": idx + 1,
            "dimensions": item["dimensions"],
        }
        for idx, item in enumerate(rows)
    }
class RolePolicyAgent:
    def __init__(self, llm: LLMAdapter, max_retries: int | None = None) -> None:
        self.llm = llm
        if max_retries is None:
            max_retries = int(os.environ.get("LABWARS_POLICY_RETRIES", "3"))
        self.max_retries = max_retries


    def _generate_llm_native_candidates(
        self,
        agent: Agent,
        event: EventAtom,
        world: WorldState,
        recall: RecallResult | None,
        allowed_str: list[str],
        avoid_actions: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Let the LLM propose the action candidate set for the llm_native contrast mode."""
        prompt = build_llm_native_policy_prompt(
            agent,
            event,
            world,
            recall,
            allowed_str,
            avoid_actions=avoid_actions,
        )
        raw = self.llm.complete_json(LLM_NATIVE_POLICY_SYSTEM, prompt)
        raw_candidates = raw.get("native_candidates", [])
        allowed = set(allowed_str)
        payloads: list[dict[str, Any]] = []
        seen: set[str] = set()
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                if not isinstance(item, dict):
                    continue
                action_type = str(item.get("type", ""))
                if action_type not in allowed or action_type in seen:
                    continue
                seen.add(action_type)
                plausibility = _coerce_score(item.get("plausibility", 0.5))
                intensity = _coerce_score(item.get("intensity", 0.5))
                cognitive_tendency = (plausibility - 0.5) * 2.0
                payloads.append({
                    "type": action_type,
                    "target": _pick_target(agent, world, event, item.get("target")),
                    "tendency": round(cognitive_tendency, 4),
                    "social_physics_tendency": 0.0,
                    "llm_score": round(plausibility, 4),
                    "llm_cognitive_tendency": round(cognitive_tendency, 4),
                    "fused_tendency": round(cognitive_tendency, 4),
                    "intensity": round(intensity, 4),
                    "motives": {
                        "llm_native_plausibility": round(plausibility, 4),
                    },
                    "field_decomposition": {
                        "baseline": 0.0,
                        "motive_contributions": {},
                        "event_affinity": 0.0,
                        "memory_trigger": {},
                        "raw_tendency": round(cognitive_tendency, 4),
                        "final_tendency": round(cognitive_tendency, 4),
                        "llm_native_public_reason": str(item.get("public_reason", ""))[:160],
                        "llm_native_private_reason": str(item.get("private_reason", ""))[:160],
                    },
                    "parameter_source": "llm_native_policy",
                    "cognitive_policy_lambda": 1.0,
                    "social_physics_weight": 0.0,
                    "scoring_source": "llm_native_generated",
                    "llm_score_reason": str(item.get("private_reason") or item.get("public_reason") or "")[:160],
                })
        if not payloads:
            fallback = allowed_str[0]
            payloads.append({
                "type": fallback,
                "target": _pick_target(agent, world, event, None),
                "tendency": 0.0,
                "social_physics_tendency": 0.0,
                "llm_score": 0.5,
                "llm_cognitive_tendency": 0.0,
                "fused_tendency": 0.0,
                "intensity": 0.5,
                "motives": {"llm_native_fallback": 1.0},
                "field_decomposition": {"baseline": 0.0, "motive_contributions": {}, "memory_trigger": {}},
                "parameter_source": "llm_native_fallback",
                "cognitive_policy_lambda": 1.0,
                "social_physics_weight": 0.0,
                "scoring_source": "llm_native_fallback",
                "llm_score_reason": "fallback after invalid native candidates",
            })
        return _normalize_probability_payloads(payloads, key="fused_tendency"), {
            "enabled": True,
            "source": "llm_native_generated",
            "policy_mode": "llm_native",
            "raw": raw,
        }

    def _score_candidates(
        self,
        agent: Agent,
        event: EventAtom,
        world: WorldState,
        recall: RecallResult | None,
        candidate_payload: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Ask the LLM for subjective candidate plausibility, then fuse with field scores."""
        if not config.get("enable_llm_action_scoring", True):
            return candidate_payload, {"enabled": False, "source": "field_only"}

        raw_lambda = config.get("cognitive_policy_lambda")
        if raw_lambda is None:
            raw_lambda = config.get("llm_action_score_mix", 0.35)
        cognitive_lambda = max(0.0, min(1.0, float(raw_lambda)))
        social_lambda = 1.0 - cognitive_lambda
        try:
            prompt = build_action_scoring_prompt(agent, event, world, recall, candidate_payload)
            raw = self.llm.complete_json(ACTION_SCORING_SYSTEM, prompt)
        except QuotaExhaustedError:
            raise
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            return candidate_payload, {"enabled": True, "source": "field_only_fallback", "error": str(exc)}

        raw_scores = raw.get("candidate_scores", [])
        score_map: dict[str, dict[str, Any]] = {}
        if isinstance(raw_scores, list):
            for item in raw_scores:
                if not isinstance(item, dict):
                    continue
                action_type = str(item.get("type", ""))
                if action_type:
                    score_map[action_type] = item

        fused: list[dict[str, Any]] = []
        for candidate in candidate_payload:
            item = dict(candidate)
            score_item = score_map.get(str(candidate.get("type", "")), {})
            llm_score = _coerce_score(score_item.get("plausibility", 0.5))
            field_tendency = float(candidate.get("tendency", 0.0))
            llm_cognitive_tendency = (llm_score - 0.5) * 2.0
            fused_tendency = social_lambda * field_tendency + cognitive_lambda * llm_cognitive_tendency
            item["field_probability"] = candidate.get("probability", 0.0)
            item["social_physics_tendency"] = round(field_tendency, 4)
            item["llm_score"] = round(llm_score, 4)
            item["llm_cognitive_tendency"] = round(llm_cognitive_tendency, 4)
            item["llm_score_reason"] = str(score_item.get("reason", ""))[:160]
            item["fused_tendency"] = round(fused_tendency, 4)
            item["cognitive_policy_lambda"] = round(cognitive_lambda, 4)
            item["social_physics_weight"] = round(social_lambda, 4)
            item["scoring_source"] = "dual_engine_fused"
            fused.append(item)

        probs = _softmax([float(item["fused_tendency"]) for item in fused], temperature=0.22)
        for item, prob in zip(fused, probs):
            item["probability"] = round(prob, 5)
        return fused, {
            "enabled": True,
            "source": "dual_engine_fused",
            "cognitive_policy_lambda": cognitive_lambda,
            "social_physics_weight": social_lambda,
            "raw": raw,
        }

    def decide(
        self,
        agent: Agent,
        event: EventAtom,
        world: WorldState,
        recall: RecallResult | None,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not is_agent_active(agent.id, event.round, config):
            return None

        allowed = get_allowed_actions(agent.id, burnout=agent.emotion.burnout)
        if not allowed:
            return None

        avoid = avoid_actions(agent)
        dynamic_avoid = list(avoid)
        allowed_str, applied_avoid = filter_allowed_actions(
            [a.value for a in allowed],
            dynamic_avoid,
        )

        seed = int(config.get("seed", 0) or 0)
        policy_mode = str(config.get("policy_mode", "dual_engine"))
        sampling_audit = config.get("cognitive_sampling", {}) or {}
        llm_sampled = bool(sampling_audit.get("sampled", True))
        local_config = dict(config)
        if not llm_sampled:
            local_config["enable_llm_action_scoring"] = False
        if config.get("observation_lesion"):
            obs_gain = 1.0
        else:
            channel = (recall.audit or {}).get("observation_channel") if recall else None
            obs_gain = observation_gain(str(channel or "direct"))
        candidates = generate_action_candidates(
            agent,
            event,
            world,
            recall,
            allowed_str,
            avoid_actions=dynamic_avoid,
            seed=seed,
            observation_gain_value=obs_gain,
        )
        base_payload = [c.to_dict() for c in candidates]
        if policy_mode == "llm_native" and llm_sampled:
            try:
                candidate_payload, scoring_audit = self._generate_llm_native_candidates(
                    agent,
                    event,
                    world,
                    recall,
                    allowed_str,
                    dynamic_avoid,
                )
            except QuotaExhaustedError:
                raise
            except (LLMError, KeyError, TypeError, ValueError) as exc:
                candidate_payload = base_payload
                scoring_audit = {"enabled": True, "source": "llm_native_fallback_to_field", "policy_mode": policy_mode, "error": str(exc)}
        elif policy_mode == "social_physics":
            candidate_payload = base_payload
            scoring_audit = {"enabled": False, "source": "field_only", "policy_mode": policy_mode}
        else:
            candidate_payload, scoring_audit = self._score_candidates(
                agent,
                event,
                world,
                recall,
                base_payload,
                local_config,
            )
            scoring_audit["policy_mode"] = policy_mode
        if not candidate_payload:
            candidate_payload = base_payload
        candidate_payload = _focus_draft_payloads(event, agent, world, candidate_payload)
        if not candidate_payload:
            candidate_payload = base_payload
        selected_payload = _sample_payload(
            candidate_payload,
            seed=seed,
            round_num=event.round,
            agent_id=agent.id,
        )
        social_potential = compute_social_potential(
            world, agent, event, recall, target=selected_payload.get("target")
        )
        selected_social_pressure = social_potential.pressure_for_action(str(selected_payload.get("type", "")))
        social_potential_ablation = {
            dim: round(
                social_potential.pressure_for_action(str(selected_payload.get("type", "")), lesions=[dim]),
                4,
            )
            for dim in SOCIAL_POTENTIAL_DIMENSIONS
        }
        pressure_fields = compute_pressure_fields(
            world, agent, event, recall, potential=social_potential
        )

        if not llm_sampled:
            act = _scripted_render_action(agent, event, selected_payload)
            private = act.setdefault("private_intent", {})
            private.setdefault("private_motives", selected_payload.get("motives", {}))
            act["action_candidates"] = candidate_payload
            act["selected_action"] = selected_payload
            act["private_motives"] = selected_payload.get("motives", {})
            act["social_potential"] = social_potential.to_dict()
            act["selected_social_pressure"] = round(selected_social_pressure, 4)
            act["selected_social_pressure_decomposition"] = social_potential.action_decomposition(str(selected_payload.get("type", "")))
            act["social_potential_ablation"] = social_potential_ablation
            act["pressure_fields"] = pressure_fields
            act["llm_action_scoring"] = {**scoring_audit, "cognitive_sampling": sampling_audit, "source": scoring_audit.get("source", "field_only_unsampled")}
            act["cognitive_sampling"] = sampling_audit
            return act

        last_error = ""
        retry_note = ""

        for attempt in range(self.max_retries + 1):
            user_prompt = build_role_policy_prompt(
                agent,
                event,
                world,
                recall,
                allowed_str,
                action_candidates=candidate_payload,
                sampled_action=selected_payload,
                avoid_actions=dynamic_avoid,
                retry_note=retry_note,
                validation_error=last_error,
                stance_prior=stance_prior_for_action(str(selected_payload.get("type", ""))),
            )
            try:
                raw = self.llm.complete_json(ROLE_POLICY_SYSTEM, user_prompt)
                raw["primary_action"] = {
                    "type": selected_payload["type"],
                    "target": selected_payload["target"],
                    "intensity": selected_payload["intensity"],
                }
                act = _normalize_action_response(raw, agent, event, world, allowed)
                private = act.setdefault("private_intent", {})
                private.setdefault("private_motives", selected_payload.get("motives", {}))
                act["action_candidates"] = candidate_payload
                act["selected_action"] = selected_payload
                act["private_motives"] = selected_payload.get("motives", {})
                act["social_potential"] = social_potential.to_dict()
                act["selected_social_pressure"] = round(selected_social_pressure, 4)
                act["selected_social_pressure_decomposition"] = social_potential.action_decomposition(
                    str(selected_payload.get("type", ""))
                )
                act["social_potential_ablation"] = social_potential_ablation
                act["pressure_fields"] = pressure_fields
                act["llm_action_scoring"] = {**scoring_audit, "cognitive_sampling": sampling_audit}
                act["cognitive_sampling"] = sampling_audit
                return act
            except QuotaExhaustedError:
                raise
            except (LLMError, KeyError, TypeError, ValueError) as exc:
                last_error = str(exc)
                retry_note = "Keep the sampled action fixed and repair only JSON/public/private fields."

        act = _scripted_render_action(agent, event, selected_payload)
        private = act.setdefault("private_intent", {})
        private.setdefault("private_motives", selected_payload.get("motives", {}))
        act["action_candidates"] = candidate_payload
        act["selected_action"] = selected_payload
        act["private_motives"] = selected_payload.get("motives", {})
        act["social_potential"] = social_potential.to_dict()
        act["selected_social_pressure"] = round(selected_social_pressure, 4)
        act["selected_social_pressure_decomposition"] = social_potential.action_decomposition(
            str(selected_payload.get("type", ""))
        )
        act["social_potential_ablation"] = social_potential_ablation
        act["pressure_fields"] = pressure_fields
        act["llm_action_scoring"] = {
            **scoring_audit,
            "cognitive_sampling": sampling_audit,
            "source": "scripted_render_fallback",
            "error": last_error[:240],
        }
        act["cognitive_sampling"] = sampling_audit
        return act

    def decide_all(
        self,
        world: WorldState,
        event: EventAtom,
        recalls: dict[str, RecallResult],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        sampling = _cognitive_sampling_scores(world, event, recalls, config)
        for agent_id, agent in world.agents.items():
            local_config = dict(config)
            if sampling:
                local_config["cognitive_sampling"] = sampling.get(agent_id, {"enabled": True, "sampled": False})
            act = self.decide(agent, event, world, recalls.get(agent_id), local_config)
            if act:
                actions.append(act)
        return actions
