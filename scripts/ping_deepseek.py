"""Ping DeepSeek with the paper config. Never prints the key."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.llm_adapter import get_adapter  # noqa: E402


def main() -> None:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    print(f"key_set={bool(key)} key_len={len(key)}", flush=True)
    adapter = get_adapter(provider="deepseek")
    payload = adapter.complete_json("Output JSON only.", '{"task":"ping","ok":true}')
    print(f"model={adapter.model}", flush=True)
    print(f"ping_ok={json.dumps(payload, ensure_ascii=False)[:300]}", flush=True)


if __name__ == "__main__":
    main()
