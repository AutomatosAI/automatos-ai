"""The Auto agent side of the workspace orchestrator settings.

``GET``/``PUT /api/workspaces/current/orchestrator`` (api/workspaces.py) keep the
orchestrator's LLM and Soul on the workspace's Auto agent row, the single source of
truth for both. This module reads that row's model_config as the LLM block the
routes return, and writes a save's LLM and configuration fields to it. Moved out of
api/workspaces.py (PRD-263 PR-1E-a) so the save route fits the 50-line function
rule. Since 19 Oct 2026 a save writes only that row (``sync_auto_llm``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from sqlalchemy.orm import Session

from core.llm.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
from core.models import Agent

logger = logging.getLogger(__name__)

# The fields a save also writes to the Auto agent's configuration, and GET reads back.
AUTO_CONFIG_FIELDS = ("communication_style", "proactive_level", "thinking_level")
# The Auto agent's model_config keys a save writes from the payload's llm.
LLM_KEYS = (
    "provider", "model_id", "temperature", "max_tokens", "top_p", "frequency_penalty",
    "presence_penalty", "stop", "timeout", "fallback_model_id",
)
# The route keys: a save that names no model leaves both as stored.
ROUTE_KEYS = ("provider", "model_id")


def auto_llm_view(mc: Dict[str, Any]) -> Dict[str, Any]:
    """The Auto agent's model_config as the LLM block GET and PUT return."""
    return {
        "provider": mc.get("provider", DEFAULT_LLM_PROVIDER),
        "model_id": mc.get("model_id", DEFAULT_LLM_MODEL),
        "temperature": mc.get("temperature", 0.7),
        "max_tokens": mc.get("max_tokens", 4000),
        "top_p": mc.get("top_p", 1.0),
        "frequency_penalty": mc.get("frequency_penalty", 0.0),
        "presence_penalty": mc.get("presence_penalty", 0.0),
        "stop": mc.get("stop"),
        "timeout": mc.get("timeout"),
        "fallback_model_id": mc.get("fallback_model_id"),
    }


def sync_auto_llm(db: Session, auto_agent: Agent, llm: Any) -> None:
    """LLM config → this workspace's Auto agent model_config, and nothing else.

    The save never writes the platform default (system_settings orchestrator_llm):
    that is a platform admin's setting (System Settings > LLM tiers). Until 19 Oct
    2026 this save wrote its provider and model there too, so one workspace's choice
    became every workspace's default. The row is marked
    (``auto_model_sync.workspace_saved_model_config``): saving the platform default's
    own pair, or leaving a following row's model as it was, keeps it following the
    platform default; any other pair is this workspace's choice. ``db`` stays in the
    signature for api/workspaces.py's call. A save whose model_id is blank (a page
    that could not load the config) names no model: the stored provider and model
    stay, and only the other keys are written (``llm_update``)."""
    if not (llm and isinstance(llm, dict)):
        return
    from sqlalchemy.orm.attributes import flag_modified

    from core.llm.auto_model_sync import workspace_saved_model_config
    from core.llm.new_agent_model import default_route

    update = llm_update(llm)
    auto_agent.model_config = workspace_saved_model_config(auto_agent.model_config or {}, update, default_route())
    flag_modified(auto_agent, "model_config")


def llm_update(llm: Dict[str, Any]) -> Dict[str, Any]:
    """The model_config keys a save of ``llm`` writes: every ``LLM_KEYS`` key it
    carries, less the route keys when it names no model_id, so a blank model is
    never written over the Auto row's."""
    skipped = () if llm.get("model_id") else ROUTE_KEYS
    return {key: llm[key] for key in LLM_KEYS if key in llm and key not in skipped}


def sync_auto_configuration(auto_agent: Agent, payload: Dict[str, Any]) -> None:
    """Configuration fields → the Auto agent's configuration JSONB. personality_mode is
    stored there too (first, as before), so GET can read it back without fragile text
    comparison."""
    personality_mode = payload.get("personality_mode")
    fields = {"personality_mode": personality_mode} if personality_mode else {}
    fields = {**fields, **{key: payload[key] for key in AUTO_CONFIG_FIELDS if key in payload}}
    if not fields:
        return
    from sqlalchemy.orm.attributes import flag_modified

    auto_agent.configuration = {**(auto_agent.configuration or {}), **fields}
    flag_modified(auto_agent, "configuration")
