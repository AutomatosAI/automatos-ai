"""P229-RVW-2 — ``platform_ask_orchestrator`` bound to the server's clarification subject,
moved out of platform_executor.py (PRD-268 US-C19d, K3).

The executor binds every ``platform_ask_orchestrator`` call through it; its body is unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["bind_ask_orchestrator_context"]


def bind_ask_orchestrator_context(
    params: Dict[str, Any],
    caller_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Bind ``platform_ask_orchestrator`` to the SERVER-threaded clarification
    subject, never a tool parameter (P229-RVW-2).

    STRIP any client/LLM-supplied ``_run_id`` / ``_task_id`` / ``_field_id`` FIRST
    — mirroring the ``_agent_id`` hardening in ``exec_platform`` — THEN inject the
    server ``field_context`` values. A prompt-injected tool call therefore cannot
    smuggle a FOREIGN task id past the binding: with an empty field_context (the
    non-mission lanes — channels, webhooks, board, scheduled — where
    agent_factory defaults the mode to TASK_EXECUTION but threads no run) the keys
    stay ABSENT and the handler fails closed to proceed-with-assumption; with a
    real field_context only the server values survive. Rebuild-don't-mutate.
    """
    bound = {
        k: v for k, v in params.items()
        if k not in ("_run_id", "_task_id", "_field_id")
    }
    fctx = (caller_context or {}).get("field_context") or {}
    for src, dst in (("run_id", "_run_id"), ("task_id", "_task_id"), ("field_id", "_field_id")):
        val = fctx.get(src)
        if val is not None:
            bound[dst] = val
    return bound
