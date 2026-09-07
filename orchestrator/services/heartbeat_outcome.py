"""PRD-234 S3 — read an agent run's outcome honestly.

``AgentFactory.execute_with_prompt`` answers a dict; a refusal or failure is
``{"status": "error", "error": "…"}``. The heartbeat tick used to pull the
"result text" out of that dict with a fallback of ``str(exec_result)`` and file
it as a green ``llm_analysis`` finding — so a failing agent reported healthy
forever. This is the one place that decides what a run's dict means.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


def read_exec_outcome(exec_result: Any) -> Tuple[str, bool, str]:
    """``(text, is_error, error_detail)`` for a factory result."""
    if not isinstance(exec_result, dict):
        text = str(exec_result or "")[:500]
        return text, False, ""
    if str(exec_result.get("status") or "").lower() == "error":
        detail = exec_result.get("error") or exec_result.get("result") or exec_result.get("message") or "agent execution failed"
        return "", True, str(detail)[:500]
    text: Any = (
        exec_result.get("result")
        or exec_result.get("response")
        or exec_result.get("output")
        or exec_result.get("content")
        or ""
    )
    if isinstance(text, dict):
        text = text.get("result") or text.get("response") or str(text)
    return str(text or "")[:1000], False, ""


def tokens_of(exec_result: Any) -> int:
    if not isinstance(exec_result, dict):
        return 0
    try:
        return int(exec_result.get("tokens_used") or 0)
    except (TypeError, ValueError):
        return 0
