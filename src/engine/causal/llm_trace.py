"""Record-replay for LLM JSON calls.

Physics noise is event-keyed. LLM text cannot be regenerated from a hash of
the seed, so the factual run records complete_json outputs and twins replay
them when the prompt is unchanged. Cache misses (a patch that rewrites the
prompt) fall through to the live adapter.
"""

from __future__ import annotations

import copy
import hashlib
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from src.engine.llm_adapter import LLMAdapter


def prompt_key(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n---\n{user}".encode("utf-8")).hexdigest()


@dataclass
class LLMTrace:
    by_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def stats(self) -> dict[str, int]:
        return {
            "unique_prompts": len(self.by_key),
            "cached_errors": len(self.errors),
            "hits": self.hits,
            "misses": self.misses,
        }

    def snapshot_hits_misses(self) -> tuple[int, int]:
        return self.hits, self.misses

    def copy(self, *, reset_counters: bool = True) -> "LLMTrace":
        return LLMTrace(
            by_key=copy.deepcopy(self.by_key),
            errors=dict(self.errors),
            hits=0 if reset_counters else self.hits,
            misses=0 if reset_counters else self.misses,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_key": copy.deepcopy(self.by_key),
            "errors": dict(self.errors),
            "hits": self.hits,
            "misses": self.misses,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LLMTrace":
        payload = data or {}
        return cls(
            by_key=copy.deepcopy(payload.get("by_key") or {}),
            errors=dict(payload.get("errors") or {}),
            hits=int(payload.get("hits") or 0),
            misses=int(payload.get("misses") or 0),
        )


_TRACE: ContextVar[LLMTrace | None] = ContextVar("labwars_llm_trace", default=None)


def bind_llm_trace(trace: LLMTrace | None) -> Token:
    return _TRACE.set(trace)


def reset_llm_trace(token: Token) -> None:
    _TRACE.reset(token)


def current_llm_trace() -> LLMTrace | None:
    return _TRACE.get()


class TracingAdapter(LLMAdapter):
    def __init__(self, inner: LLMAdapter, trace: LLMTrace) -> None:
        self.inner = inner
        self.trace = trace

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        from src.engine.llm_adapter import LLMError

        key = prompt_key(system, user)
        cached_error = self.trace.errors.get(key)
        if cached_error is not None:
            self.trace.hits += 1
            raise LLMError(cached_error)
        cached = self.trace.by_key.get(key)
        if cached is not None:
            self.trace.hits += 1
            return copy.deepcopy(cached)
        self.trace.misses += 1
        try:
            response = self.inner.complete_json(system, user)
        except Exception as exc:
            from src.engine.llm_adapter import QuotaExhaustedError

            if isinstance(exc, QuotaExhaustedError):
                raise
            self.trace.errors[key] = str(exc)
            raise
        self.trace.by_key[key] = copy.deepcopy(response)
        return copy.deepcopy(response)
