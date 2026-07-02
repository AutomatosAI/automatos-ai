"""Policy plane — errors-as-data envelope (PRD-174 §4.2).

Formalises the half-existing failure envelope (`tool_router.execute_and_format`
returned an ad-hoc `{success, llm_context, error_type, fatal_error}` shape) into
the typed `{code, message_for_model, remediation, retryable}` the *model* reads.
Every policy denial returns its reason as tool content so Auto can adapt
(escalate, ask, drop a field) instead of emitting "Unknown error".

Two directions:

- :func:`verdict_to_result` — turn a blocking :class:`Verdict` (deny/ask) into
  the executor result dict a tool call returns, carrying the structured error so
  the loop surfaces it to the LLM.
- :func:`ensure_error_envelope` — normalise any failing tool result so it always
  exposes a `policy_error` block, giving the model a stable shape to reason over.

Stdlib-only.
"""
from __future__ import annotations

from typing import Any, Dict

from modules.policy.types import Decision, PolicyError, Verdict


def verdict_to_result(verdict: Verdict, tool_name: str) -> Dict[str, Any]:
    """Render a blocking verdict as a tool-execution result dict.

    The result is shaped so both the tool loop and `tool_router` treat it as a
    non-success outcome, while `policy_error` + `llm_context` carry the
    model-readable reason. A denial is NOT an exception here — it is data.
    """
    err = verdict.error or PolicyError(
        code="policy_denied",
        message_for_model=verdict.reason or f"Action '{tool_name}' was blocked by policy.",
    )
    blocked = verdict.decision is Decision.DENY
    return {
        "success": False,
        "tool": tool_name,
        # Legacy-compatible keys the existing formatter/loop already read, so a
        # policy block flows through the same non-success path as any tool error
        # (the loop surfaces `error`/`llm_context` to the model).
        "error": err.message_for_model,
        "llm_context": err.message_for_model,
        "permission_denied": blocked,
        "requires_approval": verdict.decision is Decision.ASK,
        # The formalised errors-as-data payload (the new contract).
        "policy_error": err.to_dict(),
        "policy_decision": verdict.decision.value,
        "fatal_error": False,
        "error_type": err.code,
    }


def ensure_error_envelope(result: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee a failing result carries a `policy_error` block.

    Non-blocking success results pass through untouched. A failure that already
    has `policy_error` is left alone; otherwise we synthesise one from the
    result's own `error`/`error_type` so every denial the model sees has the
    same `{code, message_for_model, remediation, retryable}` shape.
    """
    if not isinstance(result, dict):
        return result
    if result.get("success"):
        return result
    if result.get("policy_error"):
        return result

    message = (
        result.get("error")
        or result.get("message")
        or result.get("llm_context")
        or "Tool call failed."
    )
    code = result.get("error_type") or (
        "rate_limited" if result.get("rate_limited")
        else "permission_denied" if result.get("permission_denied")
        else "requires_confirmation" if result.get("requires_confirmation")
        else "tool_error"
    )
    retryable = bool(result.get("rate_limited")) or code == "requires_confirmation"
    enriched = dict(result)
    enriched["policy_error"] = PolicyError(
        code=code,
        message_for_model=str(message),
        remediation=result.get("remediation"),
        retryable=retryable,
    ).to_dict()
    return enriched
