"""LLM adapter — OpenAI / Anthropic / Ollama backends."""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLM_CONFIG_PATH = PROJECT_ROOT / "config" / "llm.yaml"
DEEPSEEK_CONFIG_PATH = PROJECT_ROOT / "config" / "llm.deepseek.yaml"


def _load_dotenv() -> None:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


class LLMError(RuntimeError):
    pass


class QuotaExhaustedError(LLMError):
    """Provider billed out / insufficient balance. Do not retry."""


class LLMAdapter(ABC):
    @abstractmethod
    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        ...



class ScriptedAdapter(LLMAdapter):
    """Deterministic local adapter for non-LLM scale baselines."""

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        try:
            payload = json.loads(user)
        except json.JSONDecodeError:
            payload = {}
        lower_system = system.lower()
        if "memory interpretation" in lower_system:
            event_type = payload.get("event_type") or payload.get("current_event", {}).get("type", "event")
            return {"interpretation": f"scripted interpretation of {event_type}"}
        if "candidate_scores" in lower_system or "score candidate actions" in lower_system:
            return {
                "candidate_scores": [
                    {
                        "type": candidate.get("type", "document_contribution"),
                        "plausibility": 0.5,
                        "reason": "neutral scripted baseline",
                    }
                    for candidate in payload.get("action_candidates", [])
                ]
            }
        if "native_candidates" in lower_system or "propose candidate actions" in lower_system:
            allowed = payload.get("allowed_actions") or ["document_contribution"]
            return {
                "native_candidates": [
                    {
                        "type": action_type,
                        "target": "project",
                        "intensity": 0.5,
                        "plausibility": 0.5,
                        "public_reason": "scripted native baseline",
                        "private_reason": "scripted native baseline",
                    }
                    for action_type in allowed[:4]
                ]
            }
        sampled = payload.get("sampled_action") or {}
        action_type = sampled.get("type") or (payload.get("allowed_actions") or ["document_contribution"])[0]
        target = sampled.get("target") or "project"
        return {
            "primary_action": {
                "type": action_type,
                "target": target,
                "intensity": float(sampled.get("intensity", 0.5)),
            },
            "communication_action": {
                "type": "share_result",
                "target": target,
                "content_summary": "scripted constrained rendering",
            },
            "public_position": {
                "statement_type": "neutral",
                "authorship_claim": "any_authorship",
            },
            "private_intent": {
                "goal": "follow_selected_action",
                "strategy": action_type,
                "trust_pi": payload.get("state", {}).get("beliefs", {}).get("pi_fairness", 0.5),
            },
        }

def _parse_json_content(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    last_exc: Exception | None = None
    for blob in candidates:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
        if isinstance(data, dict):
            return data
        last_exc = LLMError("LLM JSON root must be an object")
    raise LLMError(f"LLM returned non-JSON: {text[:200]}") from last_exc


def _exception_message(exc: Exception) -> str:
    parts = [str(exc)]
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(str(status))
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body)[:300])
    return " ".join(parts)


def _is_quota_exhausted(err: str) -> bool:
    lowered = err.lower()
    markers = (
        "insufficient_quota",
        "insufficient quota",
        "insufficient_balance",
        "insufficient balance",
        "exceeded your current quota",
        "exceeded your quota",
        "quota exceeded",
        "payment required",
        "please recharge",
        "account overdrawn",
        "no more credits",
        "credit is not enough",
        "余额不足",
        "欠费",
        "充值",
        " 402",
        "402 ",
        "status_code=402",
        "error code: 402",
        "error code: 401",
        "authentication f",
        "invalid api key",
        "incorrect api key",
    )
    if any(m in lowered for m in markers):
        return True
    # Bare 401/402 status tokens from OpenAI client strings.
    if "401" in lowered and ("unauthorized" in lowered or "invalid" in lowered or "api key" in lowered):
        return True
    if "402" in lowered:
        return True
    return False


def _is_transient_api_error(err: str) -> bool:
    if _is_quota_exhausted(err):
        return False
    lowered = err.lower()
    markers = (
        "429", "502", "503", "504",
        "rate_limit", "concurrent_limit", "key_concurrent",
        "bad gateway", "gateway timeout", "service unavailable",
        "nginx", "timeout", "timed out", "connection reset",
    )
    return any(m in lowered for m in markers)


def _retry_backoff_sec(attempt: int, err: str) -> float:
    if "concurrent" in err.lower() or "503" in err:
        return min(90.0, 10.0 * (attempt + 1))
    if "502" in err or "504" in err or "bad gateway" in err.lower():
        return min(60.0, 6.0 * (attempt + 1))
    if "429" in err or "rate_limit" in err.lower():
        return min(30.0, 3.0 * (attempt + 1))
    if "timeout" in err.lower() or "timed out" in err.lower():
        return min(45.0, 5.0 * (attempt + 1))
    return min(30.0, 4.0 * (attempt + 1))


class OpenAIAdapter(LLMAdapter):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        request_delay_sec: float = 0.0,
        top_p: float | None = None,
        max_retries: int = 8,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise LLMError(f"{api_key_env} not set")
        kwargs: dict[str, Any] = {"api_key": key, "timeout": 180.0}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_delay_sec = request_delay_sec
        self.top_p = top_p
        self.max_retries = max_retries
        self.extra_body = extra_body
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        if self.request_delay_sec <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.request_delay_sec:
            time.sleep(self.request_delay_sec - elapsed)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        use_json_object = True
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "messages": [
                        {"role": "system", "content": system + ("" if use_json_object else "\nRespond with a JSON object only.")},
                        {"role": "user", "content": user},
                    ],
                }
                if use_json_object:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.top_p is not None:
                    kwargs["top_p"] = self.top_p
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                response = self.client.chat.completions.create(**kwargs)
                self._last_request_at = time.monotonic()
                message = response.choices[0].message
                content = message.content or ""
                if not str(content).strip():
                    for attr in ("reasoning_content", "refusal"):
                        alt = getattr(message, attr, None)
                        if alt:
                            content = alt
                            break
                data = _parse_json_content(content or "{}")
                if data:
                    return data
                if os.environ.get("LABWARS_FAST_JSON"):
                    return data
                use_json_object = False
                last_exc = LLMError("empty JSON object from model")
                continue
            except Exception as exc:
                self._last_request_at = time.monotonic()
                last_exc = exc
                err = _exception_message(exc)
                if _is_quota_exhausted(err):
                    raise QuotaExhaustedError(f"DeepSeek/OpenAI quota exhausted: {exc}") from exc
                if _is_transient_api_error(err) and attempt + 1 < self.max_retries:
                    time.sleep(_retry_backoff_sec(attempt, err))
                    continue
                raise LLMError(f"OpenAI-compatible API error: {exc}") from exc
        raise LLMError(f"OpenAI-compatible API error after retries: {last_exc}") from last_exc


class AnthropicAdapter(LLMAdapter):
    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system + "\nRespond with JSON only.",
            messages=[{"role": "user", "content": user}],
        )
        content = message.content[0].text if message.content else "{}"
        return _parse_json_content(content)


class OllamaAdapter(LLMAdapter):
    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434",
    ) -> None:
        import urllib.error
        import urllib.request

        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self._urllib = urllib.request
        self._urllib_error = urllib.error

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        payload = json.dumps({
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode("utf-8")
        req = self._urllib.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._urllib.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except self._urllib_error.URLError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        content = data.get("message", {}).get("content", "{}")
        return _parse_json_content(content)


def _extra_body_from_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    extra = dict(cfg.get("extra_body") or {})
    thinking = cfg.get("thinking")
    if thinking in {False, "disabled", "off", "none"}:
        extra["thinking"] = {"type": "disabled"}
    elif isinstance(thinking, dict):
        extra["thinking"] = thinking
    return extra or None


def load_llm_config(path: Path | None = None) -> dict[str, Any]:
    env_path = os.environ.get("LABWARS_LLM_CONFIG")
    p = path or (Path(env_path) if env_path else LLM_CONFIG_PATH)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 1024}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def get_adapter(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    top_p: float | None = None,
    config: dict[str, Any] | None = None,
) -> LLMAdapter:
    cfg = {**load_llm_config(), **(config or {})}
    prov = (provider or cfg.get("provider", "openai")).lower()
    if prov == "deepseek":
        ds = load_llm_config(DEEPSEEK_CONFIG_PATH)
        return get_adapter(
            provider="openai",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            top_p=top_p,
            config={**ds, **(config or {})},
        )
    mdl = model or os.environ.get("LABWARS_LLM_MODEL") or cfg.get("model", "gpt-4o-mini")
    temp = temperature if temperature is not None else float(cfg.get("temperature", 0.7))
    tokens = max_tokens or int(cfg.get("max_tokens", 1024))
    tp = top_p if top_p is not None else cfg.get("top_p")

    if prov == "openai":
        env_key = cfg.get("api_key_env", "OPENAI_API_KEY")
        url = base_url or cfg.get("base_url")
        delay = float(os.environ.get("LABWARS_REQUEST_DELAY_SEC", cfg.get("request_delay_sec", 0)))
        retries = int(os.environ.get("LABWARS_LLM_MAX_RETRIES", cfg.get("max_retries", 8)))
        return OpenAIAdapter(
            model=mdl,
            temperature=temp,
            max_tokens=tokens,
            api_key=api_key or os.environ.get(env_key),
            api_key_env=env_key,
            base_url=url,
            request_delay_sec=delay,
            top_p=float(tp) if tp is not None else None,
            max_retries=retries,
            extra_body=_extra_body_from_config(cfg),
        )
    if prov == "anthropic":
        env_key = cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        return AnthropicAdapter(model=mdl, temperature=temp, max_tokens=tokens, api_key=api_key or os.environ.get(env_key))
    if prov == "ollama":
        return OllamaAdapter(model=mdl, temperature=temp, base_url=base_url or cfg.get("base_url", "http://localhost:11434"))
    if prov in {"scripted", "deterministic", "none"}:
        return ScriptedAdapter()
    raise LLMError(f"Unknown LLM provider: {prov}")
