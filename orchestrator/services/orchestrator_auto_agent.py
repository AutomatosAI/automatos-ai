"""The Auto agent side of the workspace orchestrator settings.

``GET``/``PUT /api/workspaces/current/orchestrator`` (api/workspaces.py) keep the
orchestrator's LLM and Soul on the workspace's Auto agent row, the single source of
truth for both. This module reads that row's model_config as the LLM block the
routes return, and writes a save's LLM and configuration fields to it. Moved out of
api/workspaces.py unchanged (PRD-263 PR-1E-a) so the save route fits the 50-line
function rule.
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
    """LLM config → the Auto agent's model_config, and provider/model → system_settings."""
    if not (llm and isinstance(llm, dict)):
        return
    from sqlalchemy.orm.attributes import flag_modified

    auto_agent.model_config = {**(auto_agent.model_config or {}), **{key: llm[key] for key in LLM_KEYS if key in llm}}
    _sync_orchestrator_llm_settings(db, llm)
    flag_modified(auto_agent, "model_config")


def _sync_orchestrator_llm_settings(db: Session, llm: Dict[str, Any]) -> None:
    """Sync to system_settings so config.LLM_PROVIDER/LLM_MODEL stay in sync (used by
    internal orchestrator operations: agent_factory, chatbot, etc.). Best-effort: a
    failure is logged and the Auto agent's own model_config still saves."""
    # Moved unchanged from api/workspaces.py: the global orchestrator_llm write predates PRD-263.
    try:
        from core.models.system_settings import SystemSetting
        for ss_key, llm_key in [("provider", "provider"), ("model", "model_id")]:
            if llm_key not in llm:
                continue
            ss = db.query(SystemSetting).filter(
                SystemSetting.category == "orchestrator_llm",
                SystemSetting.key == ss_key,
            ).first()
            if ss:
                ss.value = llm[llm_key]
            else:
                db.add(SystemSetting(
                    category="orchestrator_llm",
                    key=ss_key,
                    value=llm[llm_key],
                    default_value=llm[llm_key],
                    value_type="string",
                    is_required=True,
                ))
    except Exception:
        logger.warning("Failed to sync LLM settings to system_settings table", exc_info=True)


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
