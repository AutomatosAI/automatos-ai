from __future__ import annotations

import logging
import uuid
from typing import Optional
from contextvars import ContextVar, Token

# Import Phase 2 ContextVars from automatos_logging so both systems share
# the same variables. The LogRelayHandler reads these directly.
from core.monitoring.automatos_logging import (
    request_id_var,
    correlation_id_var,
    workspace_id_var,
    user_id_var,
    agent_id_var,
    workflow_id_var,
    run_id_var,
    tenant_id_var,
    http_method_var,
    http_path_var,
)

# Re-export for backwards compat (existing code imports from here)
__all__ = [
    "request_id_var",
    "correlation_id_var",
    "workspace_id_var",
    "user_id_var",
    "agent_id_var",
    "workflow_id_var",
    "run_id_var",
    "tenant_id_var",
    "http_method_var",
    "http_path_var",
    "ContextFilter",
    "install_request_context_logging",
    "set_request_id",
    "clear_request_id",
    "set_run_context",
    "set_request_context",
]


class ContextFilter(logging.Filter):
    """Logging filter that injects request/run context into records.

    Ensures attributes always exist on LogRecord so format strings
    like %(request_id)s never raise KeyError.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        record.request_id = request_id_var.get("")
        record.correlation_id = correlation_id_var.get("")
        record.workspace_id = workspace_id_var.get("")
        record.user_id = user_id_var.get("")
        record.agent_id = agent_id_var.get("")
        record.workflow_id = workflow_id_var.get("")
        record.run_id = run_id_var.get("")
        record.tenant_id = tenant_id_var.get("")
        record.http_method = http_method_var.get("")
        record.http_path = http_path_var.get("")
        return True


def install_request_context_logging(format_with_context: Optional[str] = None) -> None:
    """Attach the context filter to all handlers and optionally update formats."""
    context_filter = ContextFilter()
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.addFilter(context_filter)
        if format_with_context:
            handler.setFormatter(logging.Formatter(format_with_context))

    if not format_with_context:
        default_fmt = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[req=%(request_id)s ws=%(workspace_id)s agent=%(agent_id)s] - %(message)s"
        )
        for handler in logger.handlers:
            handler.setFormatter(logging.Formatter(default_fmt))


def set_request_id(request_id: Optional[str] = None) -> Token[str]:
    """Set request_id for current context; returns a token to reset later."""
    rid = request_id or uuid.uuid4().hex[:12]
    return request_id_var.set(rid)


def clear_request_id(token: Token[str]) -> None:
    request_id_var.reset(token)


def set_run_context(
    *,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """Set workflow/agent context for the current async context."""
    if run_id is not None:
        run_id_var.set(run_id)
    if agent_id is not None:
        agent_id_var.set(agent_id)
    if workflow_id is not None:
        workflow_id_var.set(workflow_id)
    if tenant_id is not None:
        tenant_id_var.set(tenant_id)


def set_request_context(
    *,
    request_id: str = "",
    workspace_id: str = "",
    user_id: str = "",
    method: str = "",
    path: str = "",
    correlation_id: str = "",
    agent_id: str = "",
) -> None:
    """Set full request context for structured logging.

    Call from FastAPI middleware to auto-enrich all logs
    within the request lifecycle. No need to pass extra={} at call sites.
    """
    if request_id:
        request_id_var.set(request_id)
    if workspace_id:
        workspace_id_var.set(workspace_id)
    if user_id:
        user_id_var.set(user_id)
    if method:
        http_method_var.set(method)
    if path:
        http_path_var.set(path)
    if correlation_id:
        correlation_id_var.set(correlation_id)
    if agent_id:
        agent_id_var.set(agent_id)
