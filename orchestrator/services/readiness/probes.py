"""PRD-268 §6.11.3 — one probe per server item, each reading what the feature itself reads.

* ``core`` — the database answers ``SELECT 1``: ``api.system_health_checks``' own check,
  which has no timeout of its own (the runner bounds it).
* ``model_route`` — the platform's ``orchestrator_llm`` setting resolves to a provider, a
  model and a key: the check-in's own manager (``services.proactive.runner.heartbeat_llm``,
  ``LLMManager(service_name='heartbeat')``, K9) is built. Building it reads settings and
  credentials; it makes no model call.
* ``web_search`` — ``services.web_search.resolve_backend``, the engine ``web_search`` would
  use for this workspace, or none.
* ``push`` — the Expo push path: always on SaaS (the platform's own project); on a local
  install only once its operator agreed (``services.push.consent``, read strictly, so a
  read that fails is not a "no").

Each runs on its own session, opened and closed by the runner. A probe returns what it
found; an error it cannot place escapes to the runner, which logs it and reads the item as
``needs_setup`` with ``unreadable``.
"""
from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlparse
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.system_health_checks import _database
from config import config
from services import web_search as search
from services.proactive.runner import heartbeat_llm
from services.push import consent
from services.push.expo import EXPO_SEND_URL
from services.readiness import items as i
from services.readiness.items import NEEDS_SETUP, READY, Finding, Item

logger = logging.getLogger(__name__)

SEARCH_DISABLED = "off"
_SEARCH_DETAILS = {
    search.BACKEND_OPENROUTER: i.SEARCH_OPENROUTER,
    search.BACKEND_COMPOSIO: i.SEARCH_COMPOSIO,
    search.BACKEND_SEARXNG: i.SEARCH_SEARXNG,
}


def _host(url: str) -> Optional[str]:
    """``host:port`` of a URL (never its user or password), or None when it names none."""
    parsed = urlparse(url)
    if not parsed.hostname:
        return None
    return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname


def _database_host(db: Session) -> Optional[str]:
    """Where the session's database is: ``host:port`` (a file database has no host)."""
    url = db.get_bind().engine.url
    if not url.host:
        return None
    return f"{url.host}:{url.port}" if url.port else url.host


def core(db: Session, workspace_id: UUID) -> Finding:
    """The server's database answers."""
    try:
        _database(db)()
    except SQLAlchemyError:
        logger.exception("readiness: the database did not answer")
        return Finding(NEEDS_SETUP, i.CORE_DOWN)
    return Finding(READY, i.CORE_READY, where=_database_host(db))


def model_route(db: Session, workspace_id: UUID) -> Finding:
    """Auto's model, as a check-in resolves it: a provider and a model are set, with a key."""
    try:
        manager = heartbeat_llm(workspace_id, None)
    except ValueError:
        return Finding(NEEDS_SETUP, i.MODEL_UNSET, fix=i.CHOOSE_MODEL)
    if not manager.config.api_key:
        return Finding(NEEDS_SETUP, i.MODEL_NO_KEY, fix=i.ADD_KEY)
    return Finding(READY, i.MODEL_READY)


def web_search(db: Session, workspace_id: UUID) -> Finding:
    """The search engine ``web_search`` would use for this workspace."""
    if (config.WEB_SEARCH_PROVIDER or "").lower() == SEARCH_DISABLED:
        return Finding(NEEDS_SETUP, i.SEARCH_OFF)
    backend = search.resolve_backend(db, workspace_id)
    if backend is None:
        return Finding(NEEDS_SETUP, i.SEARCH_NONE, fix=i.ADD_KEY)
    where = _host(config.SEARXNG_URL) if backend == search.BACKEND_SEARXNG else None
    return Finding(READY, _SEARCH_DETAILS[backend], where=where)


def push(db: Session, workspace_id: UUID) -> Finding:
    """Notifications can leave for Expo's push service: always on SaaS, on a local install once agreed."""
    if not config.IS_LOCAL_EDITION:
        return Finding(READY, i.PUSH_READY)
    state = consent.consent_state()
    if state.agreed:
        return Finding(READY, i.PUSH_READY, where=_host(EXPO_SEND_URL))
    return Finding(NEEDS_SETUP, i.PUSH_DECLINED if state.answered else i.PUSH_UNANSWERED, fix=i.ANSWER_PUSH)


ITEMS = (
    Item(i.CORE, (i.FOR_BRIEF, i.FOR_VOICE, i.FOR_PUSH, i.FOR_CHECK_IN), core),
    Item(i.MODEL_ROUTE, (i.FOR_BRIEF, i.FOR_CHECK_IN), model_route),
    Item(i.WEB_SEARCH, (i.FOR_VOICE,), web_search),
    Item(i.PUSH, (i.FOR_PUSH,), push),
)
