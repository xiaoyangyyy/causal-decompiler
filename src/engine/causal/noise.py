"""Event-keyed exogenous noise for twin-world replay.

Each draw is a function of (seed, round, stream, agent, name), not of how
many times the global PRNG has been called. That keeps paired factual /
counterfactual runs aligned after skip_event or memory_delete changes
control flow (Starsim / event-keyed CRN).
"""

from __future__ import annotations

import hashlib
import math
import random
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Iterator


STREAM_EVENT_JITTER = "event_jitter"
STREAM_EVENT_SAMPLE = "event_sample"
STREAM_ACTION_JITTER = "action_jitter"
STREAM_ACTION_SAMPLE = "action_sample"
STREAM_ACTION_GUMBEL = "action_gumbel"
STREAM_LLM_SCORE = "llm_score"
STREAM_RUMOR = "rumor"
STREAM_CRITIC = "critic"
STREAM_SHUFFLE = "shuffle"


@dataclass
class NoiseDraw:
    round: int
    stream: str
    agent_id: str | None
    name: str
    value: float


@dataclass
class NoiseLog:
    draws: list[NoiseDraw] = field(default_factory=list)

    def to_list(self) -> list[dict[str, Any]]:
        return [
            {
                "round": d.round,
                "stream": d.stream,
                "agent_id": d.agent_id,
                "name": d.name,
                "value": d.value,
            }
            for d in self.draws
        ]


_RECORDER: ContextVar[NoiseLog | None] = ContextVar("labwars_noise_log", default=None)
_SALTS: ContextVar[dict[str, int]] = ContextVar("labwars_noise_salts", default={})


def bind_noise_log(log: NoiseLog | None) -> Token:
    return _RECORDER.set(log)


def reset_noise_log(token: Token) -> None:
    _RECORDER.reset(token)


def bind_noise_salts(salts: dict[str, int] | None) -> Token:
    return _SALTS.set(dict(salts or {}))


def reset_noise_salts(token: Token) -> None:
    _SALTS.reset(token)


def _salt(stream: str, agent_id: str | None, name: str, round_num: int) -> int:
    salts = _SALTS.get() or {}
    agent = agent_id or "-"
    for key in (
        f"{round_num}|{stream}|{agent}|{name}",
        f"{stream}|{agent}|{name}",
        f"{stream}|*|*",
        stream,
    ):
        if key in salts:
            return int(salts[key])
    return 0


def noise_key(seed: int, round_num: int, stream: str, agent_id: str | None = None, name: str = "u") -> int:
    material = (
        f"{int(seed)}|{int(round_num)}|{stream}|{agent_id or '-'}|{name}|"
        f"{_salt(stream, agent_id, name, round_num)}"
    )
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def keyed_uniform(
    seed: int,
    round_num: int,
    stream: str,
    agent_id: str | None = None,
    name: str = "u",
) -> float:
    """U(0,1) keyed by the modeled event, independent of sibling draw count."""
    value = random.Random(noise_key(seed, round_num, stream, agent_id, name)).random()
    recorder = _RECORDER.get()
    if recorder is not None:
        recorder.draws.append(
            NoiseDraw(round=round_num, stream=stream, agent_id=agent_id, name=name, value=value)
        )
    return value


def keyed_gumbel(
    seed: int,
    round_num: int,
    stream: str,
    agent_id: str | None = None,
    name: str = "u",
) -> float:
    """Standard Gumbel(0,1) keyed like CRN uniforms. Shared G_{i,t,a} across twins."""
    u = keyed_uniform(seed, round_num, stream, agent_id, name)
    u = min(1.0 - 1e-12, max(1e-12, u))
    return -math.log(-math.log(u))


def keyed_uniform_centered(
    seed: int,
    round_num: int,
    stream: str,
    *,
    agent_id: str | None = None,
    name: str = "u",
    amplitude: float,
) -> float:
    return (keyed_uniform(seed, round_num, stream, agent_id, name) * 2.0 - 1.0) * amplitude


def iter_stream(log: NoiseLog, stream: str) -> Iterator[NoiseDraw]:
    for draw in log.draws:
        if draw.stream == stream:
            yield draw
