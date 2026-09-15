"""Intervention algebra for the Causal Decompiler.

One CausalOp rewrites one node of the unrolled SCM. Structural ops change
inputs (skip event, lock observation). Mechanism ops change equations
(delete memory at t, lesion a channel, mix field vs LLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from src.engine.intervention import Intervention
from src.engine.simulation import SimConfig

KIND_EVENT_SKIP = "EVENT_SKIP"
KIND_EVENT_OVERRIDE = "EVENT_OVERRIDE"
KIND_MEMORY_DELETE = "MEMORY_DELETE"
KIND_OBSERVE_LOCK = "OBSERVE_LOCK"
KIND_POLICY_LAMBDA = "POLICY_LAMBDA"
KIND_MECHANISM_LESION = "MECHANISM_LESION"
KIND_RESAMPLE = "RESAMPLE"
KIND_DO_BELIEF = "DO_BELIEF"
KIND_DO_PUBLIC = "DO_PUBLIC"


@dataclass(frozen=True)
class CausalOp:
    kind: str
    round: int | None = None
    target_event: str | None = None
    target_agent: str | None = None
    variant: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def factor_id(self) -> str:
        bits = [self.kind]
        if self.round is not None:
            bits.append(f"r{self.round}")
        if self.target_event:
            bits.append(self.target_event)
        if self.target_agent:
            bits.append(self.target_agent)
        if self.variant:
            bits.append(self.variant)
        extra = self.payload.get("name") or self.payload.get("stream") or self.payload.get("channel")
        if extra:
            bits.append(str(extra))
        return ":".join(bits)


def skip_event(round_num: int, event_id: str | None = None) -> CausalOp:
    return CausalOp(kind=KIND_EVENT_SKIP, round=round_num, target_event=event_id)


def override_event(round_num: int, variant: str, event_id: str | None = None, override: dict[str, Any] | None = None) -> CausalOp:
    return CausalOp(
        kind=KIND_EVENT_OVERRIDE,
        round=round_num,
        target_event=event_id,
        variant=variant,
        payload={"override": dict(override or {})},
    )


def delete_memory(round_num: int, agent_id: str = "phd_a") -> CausalOp:
    return CausalOp(kind=KIND_MEMORY_DELETE, round=round_num, target_agent=agent_id, variant="memory_delete_pi_promise")


def observe_lock(variant: str = "omniscient") -> CausalOp:
    return CausalOp(kind=KIND_OBSERVE_LOCK, variant=variant)


def set_policy_lambda(value: float) -> CausalOp:
    return CausalOp(kind=KIND_POLICY_LAMBDA, payload={"lambda": float(value)})


def lesion(channel: str) -> CausalOp:
    return CausalOp(kind=KIND_MECHANISM_LESION, payload={"channel": channel})


def do_presence(round_num: int, event_id: str | None = None) -> CausalOp:
    """Change whether the event occurs; keep original visibility if it does."""
    return skip_event(round_num, event_id)


def do_event(round_num: int, event_id: str | None = None) -> CausalOp:
    """Paper name for do_presence / skip_event."""
    return skip_event(round_num, event_id)


def do_visibility(
    variant: str = "omniscient",
    *,
    event_id: str | None = None,
    agent_id: str | None = None,
    round_num: int | None = None,
) -> CausalOp:
    """Keep the event; change who can observe it."""
    return CausalOp(
        kind=KIND_OBSERVE_LOCK,
        variant=variant,
        round=round_num,
        target_event=event_id,
        target_agent=agent_id,
    )


def do_memory(round_num: int, agent_id: str = "phd_a") -> CausalOp:
    """Keep the event and its observers; change whether it is stored."""
    return delete_memory(round_num, agent_id)


def do_belief(round_num: int, agent_id: str, beliefs: dict[str, Any] | None = None) -> CausalOp:
    """Force private beliefs without changing the event or its visibility."""
    return CausalOp(
        kind=KIND_DO_BELIEF,
        round=round_num,
        target_agent=agent_id,
        payload={"beliefs": dict(beliefs or {"pi_fairness": 0.9})},
    )


def do_private_public(round_num: int, agent_id: str, statement_type: str = "team_support") -> CausalOp:
    """Force the public expression channel; leave private intent intact."""
    return CausalOp(
        kind=KIND_DO_PUBLIC,
        round=round_num,
        target_agent=agent_id,
        payload={"statement_type": statement_type},
    )


def do_behavior(round_num: int, agent_id: str, statement_type: str = "team_support") -> CausalOp:
    """Layer-B intervention: constrain expression/action while leaving S intact."""
    return do_private_public(round_num, agent_id, statement_type)


def resample(stream: str, *, round_num: int | None = None, agent_id: str | None = None, name: str = "u", salt: int = 1) -> CausalOp:
    return CausalOp(
        kind=KIND_RESAMPLE,
        round=round_num,
        target_agent=agent_id,
        payload={"stream": stream, "name": name, "salt": int(salt)},
    )


def _intervention_from_op(op: CausalOp) -> Intervention | None:
    if op.kind == KIND_EVENT_SKIP:
        if op.round is None:
            raise ValueError("EVENT_SKIP requires a round")
        return Intervention(
            intervention_id=op.factor_id(),
            type="event_skip",
            variant="skip",
            apply_at_round=op.round,
            target_event=op.target_event,
            skip_event=True,
        )
    if op.kind == KIND_EVENT_OVERRIDE:
        if op.round is None:
            raise ValueError("EVENT_OVERRIDE requires a round")
        return Intervention(
            intervention_id=op.factor_id(),
            type="authorship_framing",
            variant=op.variant or "custom",
            apply_at_round=op.round,
            target_event=op.target_event,
            override=dict(op.payload.get("override") or {}),
        )
    if op.kind == KIND_MEMORY_DELETE:
        if op.round is None:
            raise ValueError("MEMORY_DELETE requires a round")
        return Intervention(
            intervention_id=op.factor_id(),
            type="memory_intervention",
            variant=op.variant or "memory_delete_pi_promise",
            apply_at_round=op.round,
            target_agent=op.target_agent or "phd_a",
            target_event=op.target_event,
        )
    return None


def apply_ops(base: SimConfig, ops: list[CausalOp]) -> tuple[SimConfig, dict[str, int]]:
    """Translate CausalOps into a twin SimConfig plus CRN resample salts."""
    cfg = replace(base, interventions=list(base.interventions))
    salts: dict[str, int] = {}
    extra: list[Intervention] = []
    for op in ops:
        inter = _intervention_from_op(op)
        if inter is not None:
            extra.append(inter)
            continue
        if op.kind == KIND_OBSERVE_LOCK:
            if op.variant in {None, "omniscient", "gated"}:
                cfg = replace(cfg, observation_lesion=(op.variant != "gated"))
            else:
                do = dict(cfg.causal_do or {})
                do.update({
                    "hide_event_id": op.target_event,
                    "hide_from": op.target_agent,
                    "hide_from_round": op.round,
                })
                cfg = replace(cfg, causal_do=do)
        elif op.kind == KIND_DO_BELIEF:
            do = dict(cfg.causal_do or {})
            do.update({
                "force_belief": dict(op.payload.get("beliefs") or {}),
                "force_belief_agent": op.target_agent,
                "force_belief_from": op.round,
            })
            cfg = replace(cfg, causal_do=do)
        elif op.kind == KIND_DO_PUBLIC:
            do = dict(cfg.causal_do or {})
            do.update({
                "force_public": str(op.payload.get("statement_type") or "team_support"),
                "force_public_agent": op.target_agent,
                "force_public_from": op.round,
            })
            cfg = replace(cfg, causal_do=do)
        elif op.kind == KIND_POLICY_LAMBDA:
            cfg = replace(cfg, cognitive_policy_lambda=float(op.payload["lambda"]))
        elif op.kind == KIND_MECHANISM_LESION:
            channel = str(op.payload.get("channel") or "")
            if channel == "hierarchy":
                cfg = replace(cfg, hierarchy_lesion=True)
            elif channel == "status":
                cfg = replace(cfg, status_lesion=True)
            elif channel == "trust":
                cfg = replace(cfg, trust_lesion=True)
            elif channel == "observation":
                cfg = replace(cfg, observation_lesion=True)
            elif channel == "memory":
                cfg = replace(cfg, disable_memory=True)
            elif channel == "state_events":
                cfg = replace(cfg, disable_state_events=True)
            else:
                raise ValueError(f"Unknown lesion channel: {channel}")
        elif op.kind == KIND_RESAMPLE:
            stream = str(op.payload.get("stream") or "")
            name = str(op.payload.get("name") or "u")
            agent = op.target_agent or "-"
            salt = int(op.payload.get("salt") or 1)
            if name == "*" or op.payload.get("global"):
                salts[stream] = salt
                salts[f"{stream}|*|*"] = salt
            elif op.round is not None:
                salts[f"{op.round}|{stream}|{agent}|{name}"] = salt
            else:
                salts[f"{stream}|{agent}|{name}"] = salt
        else:
            raise ValueError(f"Unknown CausalOp kind: {op.kind}")
    if extra:
        cfg = replace(cfg, interventions=list(cfg.interventions) + extra)
    return cfg, salts
