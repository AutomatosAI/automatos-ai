"""The checks PUT /api/agents/{id}/model-config runs before it stores a model_config.

Moved out of api/agent_endpoints.py (19 Oct 2026, over the 800-line file limit) when
the save began refusing a body that names no model: the save replaces the agent's
whole model_config, so a blank model_id would wipe the agent's model past the
catalog and policy checks below. ``checked_model_config`` is the one entry point.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from core.models import Agent


def _refuse_a_deprecated_route(model: Any, model_id: str) -> None:
    """PRD-239 S5: a route the provider no longer offers is refused with the reason;
    saving it would only fail at the first chat turn."""
    if getattr(model, "status", "active") != "deprecated":
        return
    from core.llm import providers as _registry

    _spec = _registry.get_spec(model.serving_provider)
    _route = _spec.label if _spec else (model.serving_provider or "the provider")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"{getattr(model, 'display_name', None) or model_id} is no longer offered by {_route}. Pick another model.",
    )


def _refuse_a_model_the_policy_bars(db: Session, workspace_id: Any, agent: Agent, model: Any, model_id: str) -> None:
    """PRD-223 W1: policy gate. The Auto agent row is the orchestrator seat;
    quarantined/unapproved models are rejected at write time (the factory
    fail-closes at runtime as the last line)."""
    from core.llm.model_policy import check_model_for_agent

    _is_auto_seat = getattr(agent, "name", "") == "Auto" and bool(getattr(agent, "is_system_agent", False))
    _allowed, _reason = check_model_for_agent(
        db, workspace_id, model_id, orchestrator_seat=_is_auto_seat, provider=model.serving_provider,
    )
    if not _allowed:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Model rejected: {_reason}")


def checked_model_config(db: Session, workspace_id: Any, agent: Agent, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """``model_config`` as the model-config save stores it: a NEW dict on the route
    that serves its model, without a provider-refusal stamp (F141: the catalog just
    vouched for the model, and the form echoes back what it loaded).

    The save replaces the agent's whole model_config, so a body naming no model_id
    is refused (422): it would blank the agent's model, past every check below
    (19 Oct 2026). Any marketplace model, including one only in the OpenRouter
    cache, can be assigned (``_get_or_create_from_cache``)."""
    model_id = model_config.get("model_id")
    if not model_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Pick a model for this agent before saving.")
    from api.llm_marketplace import _get_or_create_from_cache
    from core.llm.model_refusals import without_refusal

    model = _get_or_create_from_cache(db, model_id, model_config.get("provider"))
    if not model:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid model_id: {model_id}")
    _refuse_a_deprecated_route(model, model_id)
    _refuse_a_model_the_policy_bars(db, workspace_id, agent, model, model_id)
    # PRD-236 W1: the stored provider is the ROUTE that serves the row (openrouter /
    # nvidia / openai …), never the vendor: the factory routes to it as tagged.
    return without_refusal({**model_config, "provider": model.serving_provider})
