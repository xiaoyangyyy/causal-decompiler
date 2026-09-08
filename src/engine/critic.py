"""Critic Agent 鈥?action consistency audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.world.actions import ActionType, get_allowed_actions
from src.world.models import Agent, WorldState


@dataclass
class Violation:
    code: str
    severity: str  # hard | soft
    message: str



ACTION_PUBLIC_COMPATIBILITY: dict[str, set[str]] = {
    "confront": {"self_advocacy", "challenge", "concern", "authorship_claim", "neutral"},
    "challenge_claim": {"self_advocacy", "challenge", "concern", "authorship_claim", "neutral"},
    "ask_for_authorship": {"self_advocacy", "authorship_claim", "concern", "neutral"},
    "privately_lobby_pi": {"self_advocacy", "neutral", "concern", "team_support"},
    "undermine_teammate": {"neutral", "concern", "team_support"},
    "support_teammate": {"team_support", "neutral"},
    "document_contribution": {"self_advocacy", "authorship_claim", "neutral", "concern"},
    "request_mediation": {"self_advocacy", "authorship_claim", "concern", "neutral"},
    "cite_prior_memory": {"self_advocacy", "authorship_claim", "neutral", "concern"},
    "withdraw": {"self_advocacy", "concern", "neutral"},
    "rebel": {"self_advocacy", "challenge", "concern"},
    "comply": {"team_support", "neutral"},
}

ESCALATING_ACTIONS = {"confront", "challenge_claim", "ask_for_authorship", "withdraw", "rebel", "blame"}
class CriticAgent:
    def check(self, action: dict[str, Any], agent: Agent, world: WorldState) -> list[Violation]:
        violations: list[Violation] = []
        atype = action.get("type")
        intensity = float(action.get("intensity", 0.5))

        allowed = {a.value for a in get_allowed_actions(agent.id, agent.emotion.burnout)}
        if atype not in allowed:
            violations.append(Violation("illegal_action", "hard", f"{atype} not in allowed set for {agent.id}"))

        if not (0.0 <= intensity <= 1.0):
            violations.append(Violation("intensity_range", "hard", f"intensity {intensity} out of [0,1]"))

        if agent.id == "engineer_e" and atype == "ask_for_authorship":
            violations.append(Violation("role_consistency", "hard", "engineer_e should not ask_for_authorship"))

        if atype == "cite_prior_memory" and not agent.memory:
            violations.append(Violation("memory_consistency", "hard", "cite_prior_memory with empty memory"))

        public = action.get("public_position", {})
        if public.get("statement_type") == "team_support" and action.get("type") == "undermine_teammate":
            conflict_score = 1.0 - agent.personality.deceptiveness
            violations.append(Violation(
                "public_private_conflict",
                "soft",
                f"team_support public with undermine action; conflict_score={conflict_score:.3f}",
            ))

        statement_type = str(public.get("statement_type", "neutral"))
        compatible = ACTION_PUBLIC_COMPATIBILITY.get(str(atype), {"neutral", statement_type})
        if statement_type not in compatible:
            violations.append(Violation(
                "llm_public_action_drift",
                "soft",
                f"public statement_type={statement_type} weakly mismatches sampled action={atype}",
            ))

        private = action.get("private_intent", {})
        strategy = str(private.get("strategy", ""))
        if str(atype) in ESCALATING_ACTIONS and strategy in {"comply", "support_teammate", "lay_low"}:
            violations.append(Violation(
                "llm_private_strategy_drift",
                "soft",
                f"private strategy={strategy} under-explains sampled action={atype}",
            ))

        selected = action.get("selected_action") or {}
        if selected and selected.get("type") and selected.get("type") != atype:
            violations.append(Violation(
                "sampled_action_overridden",
                "hard",
                f"action type {atype} differs from sampled {selected.get('type')}",
            ))

        return violations

    def fix_or_reject(
        self,
        action: dict[str, Any],
        agent: Agent,
        violations: list[Violation],
    ) -> tuple[dict[str, Any], list[Violation]]:
        hard = [v for v in violations if v.severity == "hard"]
        if not hard:
            return action, violations

        fallback_pool = ["comply", "document_contribution", "share_result", "write_section", "seek_validation"]
        allowed = {a.value for a in get_allowed_actions(agent.id, agent.emotion.burnout)}
        fallback = next((f for f in fallback_pool if f in allowed), "comply" if "comply" in allowed else list(allowed)[0])

        fixed = dict(action)
        fixed["type"] = fallback
        fixed["intensity"] = min(float(action.get("intensity", 0.5)), 0.6)
        fixed["_critic_fallback"] = True
        return fixed, violations


