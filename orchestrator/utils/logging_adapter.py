from __future__ import annotations

import logging
import uuid
from typing import Optional
from contextvars import ContextVar, Token


# Context variables used to enrich log records
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
run_id_var: ContextVar[str] = ContextVar("run_id", default="-")
agent_id_var: ContextVar[str] = ContextVar("agent_id", default="-")
workflow_id_var: ContextVar[str] = ContextVar("workflow_id", default="-")
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="-")


class ContextFilter(logging.Filter):
    """Logging filter that injects request/run context into records."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        # Ensure attributes always exist to avoid KeyError in formatters
        record.request_id = request_id_var.get()
        record.run_id = run_id_var.get()
        record.agent_id = agent_id_var.get()
        record.workflow_id = workflow_id_var.get()
        record.tenant_id = tenant_id_var.get()
        return True


def install_request_context_logging(format_with_context: Optional[str] = None) -> None:
    """Attach the context filter to all handlers and optionally update formats.

    If format_with_context is not provided, a reasonable default is used.
    """
    context_filter = ContextFilter()
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.addFilter(context_filter)
        if format_with_context:
            handler.setFormatter(logging.Formatter(format_with_context))

    if not format_with_context:
        default_fmt = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[req=%(request_id)s run=%(run_id)s agent=%(agent_id)s wf=%(workflow_id)s tenant=%(tenant_id)s] - %(message)s"
        )
        for handler in logger.handlers:
            handler.setFormatter(logging.Formatter(default_fmt))


def set_request_id(request_id: Optional[str] = None) -> Token[str]:
    """Set request_id for current context; returns a token to reset later."""
    rid = request_id or uuid.uuid4().hex[:12]
    return request_id_var.set(rid)


def clear_request_id(token: Token[str]) -> None:
    request_id_var.reset(token)


def set_run_context(*, run_id: Optional[str] = None, agent_id: Optional[str] = None, workflow_id: Optional[str] = None, tenant_id: Optional[str] = None) -> None:
    if run_id is not None:
        run_id_var.set(run_id)
    if agent_id is not None:
        agent_id_var.set(agent_id)
    if workflow_id is not None:
        workflow_id_var.set(workflow_id)
    if tenant_id is not None:
        tenant_id_var.set(tenant_id)


