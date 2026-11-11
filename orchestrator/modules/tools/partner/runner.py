"""Partner tools: one call, from the model's arguments to the result (PRD-275).

``UnifiedToolExecutor.execute_tool`` routes by name inside a body far over the code-shape
limit, so a partner tool runs on its seam (``runs_partner_tools``, the innermost wrapper):

1. A name shaped ``<pack>_<tool>`` that is not one of the executor's own tools is looked up:
   the workspace has the pack installed for the calling agent, and the agent lists it
   (``resolve``). Any
   other name, and every name while ``PARTNER_TOOLS_ENABLED`` is off, takes the executor's
   own path untouched.
2. The executor's own gates judge the call first: a widget turn's scopes, a lean turn's
   grants, the policy plane.
3. The arguments are checked against the operation before the key is read.
4. The key is read through the credential store, the GET is sent (``client``), and the
   answer is cut to the pack's caps (``shape``).
5. One audit row is written for the call (``store.record_call``). A call whose row cannot be
   written gives no answer.

The executor's telemetry row is written for every call, as for any tool. The key, the
headers and the query are never logged, returned or recorded.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from modules.tools.partner import client
from modules.tools.partner.answer import refused, succeeded
from modules.tools.partner.offer import granted_packs, partner_tools_enabled
from modules.tools.partner.pack import Operation, Tool, split_tool_name
from modules.tools.partner.request import (
    PartnerRefusal, auth_headers, checked_arguments, checked_base_url, chosen_operation, request_url,
)
from modules.tools.partner.shape import UNEXPECTED_ANSWER, shaped
from modules.tools.partner.store import credential_fields, installed_packs, may_use, on_own_session, record_call

logger = logging.getLogger(__name__)

Execute = Callable[..., Awaitable[Dict[str, Any]]]

# Prefixes the executor routes itself; a pack cannot be named for one (pack.RESERVED_PACK_NAMES).
EXECUTOR_PREFIXES = ("platform_", "workspace_", "composio_")
OK, AFTER_OK = 200, 300
KEY_REFUSED_STATUSES = (401, 403)
NOT_FOUND_STATUS = 404
TOO_MANY_STATUS = 429

KEY_REFUSED = "partner_key_refused"
NOT_FOUND = "partner_not_found"
RATE_LIMITED = "partner_rate_limited"
HTTP_ERROR = "partner_http_error"
CREDENTIAL_FAILED = "partner_credential_failed"
AUDIT_FAILED = "partner_audit_failed"
FAILED = "partner_tool_failed"


@dataclass(frozen=True)
class Resolved:
    """A partner tool the calling agent may run, and the credential its pack uses."""

    pack_name: str
    tool: Tool
    name: str
    credential_id: Optional[int]


@dataclass(frozen=True)
class Outcome:
    """How a call that reached the key ended: its result, and what the audit row says."""

    result: Dict[str, Any]
    operation: Operation
    status_code: Optional[int] = None
    error_code: Optional[str] = None
    key_read: bool = True  # False: the credential was refused, so nothing was sent or is recorded


def might_be_partner_tool(executor: Any, tool_name: Any) -> bool:
    """``tool_name`` is shaped like a pack's tool and is not a name the executor routes itself."""
    from modules.tools.execution.refused_calls import is_a_tool
    from modules.tools.execution.unified_executor import TOOL_ALIASES

    if split_tool_name(tool_name) is None or tool_name.startswith(EXECUTOR_PREFIXES):
        return False
    return tool_name not in TOOL_ALIASES and not is_a_tool(executor, tool_name)


def resolve(db: Any, workspace_id: Any, agent_id: Any, tool_name: str) -> Optional[Resolved]:
    """The partner tool ``tool_name`` names for this agent: its workspace has the pack installed
    for this agent, and the agent's configuration lists it. None for anything else."""
    parts = split_tool_name(tool_name)
    if parts is None or parts[0] not in granted_packs(db, agent_id, workspace_id):
        return None
    for installed in installed_packs(db, workspace_id, [parts[0]]):
        if not may_use(installed, agent_id):
            return None
        for tool in installed.pack.tools:
            if tool.name == parts[1]:
                return Resolved(installed.pack.name, tool, tool_name, installed.credential_id)
    return None


def _http_refusal(resolved: Resolved, status_code: int) -> PartnerRefusal:
    if status_code in KEY_REFUSED_STATUSES:
        scope = f" The key may lack the scope '{resolved.tool.scope}'." if resolved.tool.scope else ""
        return PartnerRefusal(KEY_REFUSED, f"The partner refused the key (HTTP {status_code}).{scope}", status_code)
    if status_code == NOT_FOUND_STATUS:
        return PartnerRefusal(NOT_FOUND, "The partner has no such record (HTTP 404).", status_code)
    if status_code == TOO_MANY_STATUS:
        return PartnerRefusal(RATE_LIMITED, "The partner is limiting requests (HTTP 429). Try again later.",
                              status_code)
    return PartnerRefusal(HTTP_ERROR, f"The partner answered HTTP {status_code}.", status_code)


def _result_of(resolved: Resolved, operation: Operation, answer: client.Answer) -> Dict[str, Any]:
    """The result for the partner's answer; a refusal for anything but a 2xx with JSON in it."""
    if not OK <= answer.status_code < AFTER_OK:
        raise _http_refusal(resolved, answer.status_code)
    try:
        body = json.loads(answer.body.decode("utf-8"))
    except ValueError:
        raise PartnerRefusal(UNEXPECTED_ANSWER, "The partner's answer was not JSON.", answer.status_code) from None
    kept = shaped(body, operation.response)
    return succeeded(resolved.pack_name, resolved.name, operation, answer.status_code, kept)


async def _credential(resolved: Resolved, workspace_id: Any) -> Mapping[str, Any]:
    try:
        return await asyncio.to_thread(on_own_session, credential_fields, workspace_id, resolved.credential_id)
    except PartnerRefusal:
        raise
    except Exception:  # noqa: BLE001 — logged; the call fails clearly and nothing is sent
        logger.exception("[partner-tools] the credential of pack %s could not be read", resolved.pack_name)
        raise PartnerRefusal(CREDENTIAL_FAILED, "The pack's credential could not be read.") from None


async def _call(resolved: Resolved, operation: Operation, arguments: Mapping[str, Any], workspace_id: Any) -> Outcome:
    """Read the key, send the GET and shape the answer; every way it can end is an Outcome."""
    try:
        credential = await _credential(resolved, workspace_id)
    except PartnerRefusal as refusal:
        result = refused(resolved.pack_name, resolved.name, refusal, operation)
        return Outcome(result, operation, error_code=refusal.code, key_read=False)
    try:
        url = request_url(checked_base_url(credential.get("base_url")), operation, arguments)
        answer = await client.get(url, auth_headers(credential))
        return Outcome(_result_of(resolved, operation, answer), operation, answer.status_code)
    except PartnerRefusal as refusal:
        result = refused(resolved.pack_name, resolved.name, refusal, operation)
        return Outcome(result, operation, refusal.status_code, refusal.code)
    except Exception:  # noqa: BLE001 — logged; the call fails clearly and is still recorded
        logger.exception("[partner-tools] %s failed after its key was read", resolved.name)
        failure = PartnerRefusal(FAILED, "The partner tool failed.")
        return Outcome(refused(resolved.pack_name, resolved.name, failure, operation), operation, error_code=FAILED)


async def _recorded(resolved: Resolved, outcome: Outcome, started: float, agent_id: Any) -> Dict[str, Any]:
    """The call's result once its audit row is written; a refusal when the row cannot be."""
    duration_ms = int((time.monotonic() - started) * 1000)
    succeeded_call = bool(outcome.result.get("success"))
    if outcome.key_read and resolved.credential_id is not None:
        try:
            await asyncio.to_thread(
                on_own_session, record_call, resolved.credential_id, tool=resolved.name,
                operation=outcome.operation.name, status_code=outcome.status_code, duration_ms=duration_ms,
                success=succeeded_call, agent_id=agent_id, error_code=outcome.error_code)
        except Exception:  # noqa: BLE001 — logged; an unrecorded call's answer is withheld
            logger.exception("[partner-tools] the audit row for %s could not be written", resolved.name)
            unrecorded = PartnerRefusal(AUDIT_FAILED, "The call could not be recorded, so its answer is withheld.")
            return refused(resolved.pack_name, resolved.name, unrecorded, outcome.operation)
    logger.info("[partner-tools] %s %s: status=%s success=%s %dms", resolved.name, outcome.operation.name,
                outcome.status_code, succeeded_call, duration_ms)
    return outcome.result


async def run(resolved: Resolved, arguments: Mapping[str, Any], *, workspace_id: Any, agent_id: Any) -> Dict[str, Any]:
    """Run one call of a partner tool and return its result. Never raises a refusal."""
    started = time.monotonic()
    try:
        operation, own = chosen_operation(resolved.tool, arguments)
        checked = checked_arguments(operation, own)
    except PartnerRefusal as refusal:  # nothing was sent and the key was not read
        return refused(resolved.pack_name, resolved.name, refusal)
    outcome = await _call(resolved, operation, checked, workspace_id)
    return await _recorded(resolved, outcome, started, agent_id)


async def _resolved_for(executor: Any, call: Mapping[str, Any]) -> Optional[Resolved]:
    """The partner tool this call names, or None for a call that is the executor's own."""
    tool_name, workspace_id = call.get("tool_name"), call.get("workspace_id")
    if not partner_tools_enabled() or workspace_id is None or not might_be_partner_tool(executor, tool_name):
        return None
    try:
        return await asyncio.to_thread(on_own_session, resolve, workspace_id, call.get("agent_id"), tool_name)
    except Exception:  # noqa: BLE001 — logged; the executor answers as it would for any name it does not have
        logger.exception("[partner-tools] could not look up '%s'; it takes the executor's own path", tool_name)
        return None


async def _gated_run(executor: Any, resolved: Resolved, call: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the call behind the executor's own gates, and write its telemetry row and trace.
    A partner's answer is not kept as a tool-outcome memory: the partner's records stay the
    partner's, read again each time they are asked for."""
    from core.observability.tracer import fire_tool_trace
    from modules.tools.execution.telemetry import fire_telemetry

    tool_name, agent_id, workspace_id = resolved.name, call.get("agent_id"), call.get("workspace_id")
    parameters = call.get("parameters") if isinstance(call.get("parameters"), dict) else {}
    trace = call.get("trace_id") or "no-trace"
    started = time.monotonic()
    result: Dict[str, Any] = {"success": False, "error": "The partner tool did not run.", "tool": tool_name}
    try:
        blocked = executor._widget_gate(tool_name, parameters, trace) or executor._policy_gate_check(
            tool_name, parameters, agent_id=agent_id, workspace_id=workspace_id,
            caller_context=call.get("caller_context"), trace=trace)
        result = blocked if blocked is not None else await run(
            resolved, parameters, workspace_id=workspace_id, agent_id=agent_id)
        return result
    except Exception:  # noqa: BLE001 — logged; the model is told the tool failed, never why in internals
        logger.exception("[partner-tools] %s failed", tool_name)
        result = {"success": False, "error": "The partner tool failed.", "error_code": FAILED, "tool": tool_name}
        return result
    finally:
        duration_ms, ran = int((time.monotonic() - started) * 1000), bool(result.get("success"))
        fire_telemetry(tool_name=tool_name, parameters=parameters, agent_id=agent_id, workspace_id=workspace_id,
                       result=result, execution_time_ms=duration_ms, caller_context=call.get("caller_context"))
        fire_tool_trace(tool_name=tool_name, success=ran, duration_ms=duration_ms, workspace_id=workspace_id,
                        agent_id=agent_id, error=None if ran else result.get("error"))


def runs_partner_tools(execute: Execute) -> Execute:
    """Wrap ``UnifiedToolExecutor.execute_tool``: a call to one of the workspace's partner tools,
    by an agent that lists its pack, runs here; every other call runs as it did."""
    signature = inspect.signature(execute)

    @functools.wraps(execute)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        resolved = await _resolved_for(self, bound.arguments)
        if resolved is None:
            return await execute(self, *args, **kwargs)
        return await _gated_run(self, resolved, bound.arguments)

    return wrapped


__all__ = ["Outcome", "Resolved", "might_be_partner_tool", "resolve", "run", "runs_partner_tools"]
