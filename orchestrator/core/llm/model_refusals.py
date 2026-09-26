"""F141 — a provider's refusal of a model is remembered, so the next turn goes to
someone who can answer it.

Refresh-3 retest, chat 9c44148f: routed to BEANCOUNTER (agent 302), whose model
OpenRouter refused ("anthropic/claude-sonnet-4-20250514 is not a valid model
ID"). Nothing remembered the refusal, so every turn routed there failed the same
way. Only a definitive refusal counts: the provider's answer names the model and
says it does not offer it (``ProviderModelUnavailableError.definitive``). The LLM
manager records it where the typed error surfaces with the agent known, so a
chat turn, a board task and a mission step all record it, in two places:

- The catalog route (serving provider + model id) becomes ``deprecated``, the
  state PRD-239 S5 defines, for every workspace: a provider's catalog is the same
  for all of them. A catalog sync that lists the model again makes it active
  (Sync is manual today).
- The agent carries ``model_config.unavailable`` when the refused id is its own
  model. That covers an id no catalog row serves, which the first record cannot
  mark (BEANCOUNTER's), and an agent whose stored provider is a legacy vendor
  name. It names the model it is about, so a different model voids it, and a
  checked model write drops it.

Chat routing and the ASSIGN lane read ``unavailable_reason``.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional

from sqlalchemy import text

from core.best_effort import off_loop

logger = logging.getLogger(__name__)

UNAVAILABLE_KEY = "unavailable"
SAID_CHARS = 300


def is_the_agents_model(configured: Optional[str], called: Optional[str]) -> bool:
    """The refused id is the agent's own model as the factory sent it: the same id,
    or a bare id with the vendor prefix the factory adds on OpenRouter. A trial
    pin, a fallback or a tier override called some other model."""
    if not configured or not called:
        return False
    return called == configured or called.endswith("/" + configured)


@off_loop
def record_model_refusal(*, agent_id: Optional[int], provider: str, model: str, said: str) -> None:
    """Record a definitive refusal. Best effort (F105): off the event loop, in its
    own session, and never raised into the call that was refused."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        routes = db.execute(
            text("UPDATE llm_models SET status = 'deprecated', updated_at = now() "
                 "WHERE serving_provider = :provider AND model_id = :model AND status <> 'deprecated'"),
            {"provider": provider, "model": model},
        ).rowcount
        stamped = False
        if agent_id:
            configured = db.execute(
                text("SELECT model_config->>'model_id' FROM agents WHERE id = :id"), {"id": int(agent_id)},
            ).scalar()
            if is_the_agents_model(configured, model):
                stamp = {"model_id": configured, "provider": provider, "called": model, "said": said[:SAID_CHARS]}
                stamped = bool(db.execute(
                    text("UPDATE agents SET model_config = CAST(jsonb_set(CAST(model_config AS jsonb), "
                         "CAST(:path AS text[]), CAST(:stamp AS jsonb) || "
                         "jsonb_build_object('at', CAST(now() AS text)), true) AS json) "
                         "WHERE id = :id AND model_config->>'model_id' = :configured"),
                    {"path": "{" + UNAVAILABLE_KEY + "}", "stamp": json.dumps(stamp),
                     "id": int(agent_id), "configured": configured},
                ).rowcount)
        db.commit()
        logger.warning(
            "[model-refusal] %s refused %s for good (%s): %d route row(s) deprecated; agent %s %s",
            provider, model, said[:200], routes or 0, agent_id, "stamped" if stamped else "not stamped",
        )
    except Exception:  # noqa: BLE001 — a record must never be why a turn fails differently
        db.rollback()
        logger.warning("[model-refusal] could not record %s refusing %s", provider, model, exc_info=True)
    finally:
        db.close()


def unavailable_reason(db: Any, agent: Any) -> Optional[str]:
    """"<agent>'s model <model> is not available from its provider" when its
    provider refused the agent's own model for good, or the catalog route the agent
    names (exactly, never re-resolved) is deprecated. None otherwise, and for a
    session (CLI) agent, which its model_config does not run."""
    from core.cli_runtime import is_cli_agent

    if is_cli_agent(getattr(agent, "configuration", None)):
        return None
    config = getattr(agent, "model_config", None)
    if not isinstance(config, Mapping) or not config.get("model_id"):
        return None
    model = str(config["model_id"])
    stamp = config.get(UNAVAILABLE_KEY)
    refused = isinstance(stamp, Mapping) and stamp.get("model_id") == model
    if not refused and not _route_is_retired(db, config.get("provider"), model):
        return None
    return f"{getattr(agent, 'name', None) or 'The agent'}'s model {model} is not available from its provider"


def _route_is_retired(db: Any, provider: Optional[str], model: str) -> bool:
    from core.llm.providers import normalize_slug
    from core.models.core import LLMModel

    route = normalize_slug(provider)
    if not route:
        return False
    status = (
        db.query(LLMModel.status)
        .filter(LLMModel.serving_provider == route, LLMModel.model_id == model)
        .scalar()
    )
    return status == "deprecated"


def without_refusal(config: Optional[Mapping[str, Any]]) -> dict:
    """A copy of ``config`` with no refusal stamp: a checked model write drops it."""
    return {key: value for key, value in dict(config or {}).items() if key != UNAVAILABLE_KEY}
