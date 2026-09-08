"""Live DeepSeek smoke for the Causal Decompiler.

Reads DEEPSEEK_API_KEY from the environment or gitignored .env.
Never prints the key.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.llm_adapter import OpenAIAdapter  # noqa: E402
from src.engine.simulation import SimConfig  # noqa: E402
from src.experiments.paper_protocol import run_paper_protocol  # noqa: E402

MODELS = ("deepseek-v4-flash", "deepseek-chat")


def _pick_adapter() -> OpenAIAdapter:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    last_error = None
    for model in MODELS:
        adapter = OpenAIAdapter(
            model=model,
            temperature=0.2,
            max_tokens=256,
            api_key=key,
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            request_delay_sec=0.4,
            max_retries=2,
            extra_body={"thinking": {"type": "disabled"}},
        )
        try:
            ping = adapter.complete_json(
                "Output JSON only.",
                '{"task":"ping","reply_schema":{"ok":true}}',
            )
            print(f"PING_OK model={model} payload={json.dumps(ping, ensure_ascii=False)[:200]}")
            return adapter
        except Exception as exc:
            last_error = exc
            print(f"PING_FAIL model={model} err={type(exc).__name__}: {str(exc)[:300]}")
    raise SystemExit(f"DeepSeek ping failed: {last_error}")


def main() -> None:
    adapter = _pick_adapter()
    result = run_paper_protocol(
        SimConfig(
            max_rounds=3,
            seed=7,
            mvp=True,
            interventions=[],
            llm_adapter=adapter,
            llm_provider="deepseek",
            policy_mode="dual_engine",
            cognitive_sampling_top_k=1,
        ),
        include_toy_shapley=True,
        auto_battery=False,
        write_output=True,
        output_dir=ROOT / "output" / "reports",
    )
    report = result.report
    print(result.summary)
    print(
        "LIVE_ASSERT "
        f"identity={report.identity_twin_ok} "
        f"replay_hits={report.llm_replay.get('identity_run_hits')} "
        f"replay_misses={report.llm_replay.get('identity_run_misses')}"
    )
    if not report.identity_twin_ok:
        raise SystemExit("identity twin failed under DeepSeek")
    if report.llm_replay.get("identity_run_misses", 1) != 0:
        raise SystemExit("identity twin made unexpected LLM cache misses")
    print("DEEPSEEK_CAUSAL_OK")


if __name__ == "__main__":
    main()
