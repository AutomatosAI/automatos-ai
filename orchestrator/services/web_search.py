"""PRD-240 S3 — one ``web_search`` for every route, through whatever engine the
deployment has.

No search engine of our own and no new key: the backends are things the user
already has for another reason —

* **openrouter** — one cheap chat completion carrying OpenRouter's
  ``openrouter:web_search`` server tool; the engine (Exa/Parallel/native) does
  the searching and the pages come back as ``url_citation`` annotations. Works
  for every model, so an agent on a free NVIDIA route still gets the web.
* **composio** — the no-auth ``COMPOSIO_SEARCH`` toolkit, called as an
  internal engine the way ``cloud_sync_service`` calls Composio (no app to
  connect, no agent assignment).
* **searxng** — a self-hosted metasearch container (``--profile search``).

``resolve_backend`` picks the first configured one in that order, or the one
``WEB_SEARCH_PROVIDER`` pins. Every backend returns ``[{title, url, snippet}]``
with the operator's denied hosts filtered out; a backend error raises and the
handler turns it into an honest result.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from config import config
from core.security.web_access import denied_hosts, host_denied

logger = logging.getLogger(__name__)

BACKEND_OPENROUTER = "openrouter"
BACKEND_COMPOSIO = "composio"
BACKEND_SEARXNG = "searxng"
_AUTO_ORDER = (BACKEND_OPENROUTER, BACKEND_COMPOSIO, BACKEND_SEARXNG)

# The Composio Search toolkit's web search action (no auth; the slug the
# platform already uses in unified_executor / the RAG hint).
COMPOSIO_SEARCH_APP = "COMPOSIO_SEARCH"
COMPOSIO_SEARCH_ACTION = "COMPOSIO_SEARCH_WEB"

NO_BACKEND_OPTIONS = [
    "Add an OpenRouter key (Settings → API Keys, or OPENROUTER_API_KEY) — search on any model.",
    "Set COMPOSIO_KEY in .env (free tier) — Composio Search needs no extra setup.",
    "Run the search container: docker compose --profile search up -d, then SEARXNG_URL=http://searxng:8080.",
]

_OPENROUTER_SEARCH_INSTRUCTION = (
    "Use the web search tool once for the query below and reply with a numbered "
    "list of the most relevant results: title — URL — one-line summary. No "
    "commentary, no answer of your own.\n\nQuery: "
)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def openrouter_key_available(db: Any, workspace_id: Any) -> bool:
    """Any OpenRouter key the manager could resolve: env, the operator
    workspace's stored key, or this workspace's own Settings → API Keys row."""
    if (config.OPENROUTER_API_KEY or "").strip():
        return True
    try:
        from core.llm.workspace_keys import get_platform_workspace_key

        if get_platform_workspace_key(BACKEND_OPENROUTER):
            return True
    except Exception:  # noqa: BLE001 — a lookup fault reads as "not configured"
        logger.debug("openrouter platform-key lookup failed", exc_info=True)
    if db is None or workspace_id is None:
        return False
    try:
        from core.models.core import UserApiKey

        row = (
            db.query(UserApiKey.id)
            .filter(
                UserApiKey.workspace_id == workspace_id,
                UserApiKey.provider == BACKEND_OPENROUTER,
                UserApiKey.is_active.is_(True),
            )
            .first()
        )
        return row is not None
    except Exception:  # noqa: BLE001
        logger.debug("openrouter workspace-key lookup failed", exc_info=True)
        return False


def composio_key_available() -> bool:
    from core.composio.client import composio_available

    return composio_available()


def searxng_available() -> bool:
    return bool(config.SEARXNG_URL)


def _available(backend: str, db: Any, workspace_id: Any) -> bool:
    if backend == BACKEND_OPENROUTER:
        return openrouter_key_available(db, workspace_id)
    if backend == BACKEND_COMPOSIO:
        return composio_key_available()
    if backend == BACKEND_SEARXNG:
        return searxng_available()
    return False


def resolve_backend(db: Any = None, workspace_id: Any = None) -> Optional[str]:
    """The backend ``web_search`` will use, or None when there is none.

    ``WEB_SEARCH_PROVIDER=off`` disables search (fetch is unaffected); a pinned
    backend that is not configured resolves to None rather than falling
    through — the operator asked for that one.
    """
    pinned = (config.WEB_SEARCH_PROVIDER or "auto").lower()
    if pinned == "off":
        return None
    if pinned != "auto":
        return pinned if _available(pinned, db, workspace_id) else None
    for backend in _AUTO_ORDER:
        if _available(backend, db, workspace_id):
            return backend
    return None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


async def search(
    query: str,
    *,
    max_results: int,
    backend: str,
    db: Any = None,
    workspace_id: Any = None,
) -> List[Dict[str, Any]]:
    if backend == BACKEND_OPENROUTER:
        results = await _search_openrouter(query, max_results, workspace_id)
    elif backend == BACKEND_COMPOSIO:
        results = await _search_composio(query, max_results, db, workspace_id)
    elif backend == BACKEND_SEARXNG:
        results = await _search_searxng(query, max_results)
    else:
        raise ValueError(f"Unknown web search backend: {backend}")
    return _filter_denied(results)[:max_results]


def _filter_denied(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for r in results:
        host = (urlparse(r.get("url") or "").hostname or "").lower()
        if host and not host_denied(host):
            kept.append(r)
    return kept


def _normalise(title: Any, url: Any, snippet: Any) -> Optional[Dict[str, Any]]:
    u = str(url or "").strip()
    if not u.startswith(("http://", "https://")):
        return None
    return {
        "title": str(title or "").strip() or u,
        "url": u,
        "snippet": " ".join(str(snippet or "").split())[:500],
    }


# --- OpenRouter -------------------------------------------------------------


def _openrouter_server_tool(max_results: int) -> Dict[str, Any]:
    tool: Dict[str, Any] = {
        "type": "openrouter:web_search",
        "max_uses": 1,
        "max_results": max(1, min(int(max_results), 10)),
    }
    deny = list(denied_hosts())
    if deny:
        tool["excluded_domains"] = deny
    return tool


async def _search_openrouter(query: str, max_results: int, workspace_id: Any) -> List[Dict[str, Any]]:
    from core.llm.manager import create_llm_manager

    manager = create_llm_manager(
        service_name="web_search",
        provider=BACKEND_OPENROUTER,
        model=config.WEB_SEARCH_OPENROUTER_MODEL,
        workspace_id=str(workspace_id) if workspace_id else None,
        request_type="web_search",
    )
    messages = [{"role": "user", "content": _OPENROUTER_SEARCH_INSTRUCTION + query}]
    response = await manager.generate_response(messages, tools=[_openrouter_server_tool(max_results)])
    results = [
        r
        for r in (
            _normalise(c.get("title"), c.get("url"), c.get("snippet"))
            for c in (getattr(response, "citations", None) or [])
        )
        if r
    ]
    if results:
        return results
    # No annotations (a model that answered without searching): the text is
    # the only evidence — pull any URLs it listed so the caller still gets links.
    return _urls_from_text(getattr(response, "content", "") or "")


def _urls_from_text(text: str) -> List[Dict[str, Any]]:
    import re

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for line in text.splitlines():
        m = re.search(r"https?://[^\s)\]>\"']+", line)
        if not m:
            continue
        url = m.group(0).rstrip(".,;")
        if url in seen:
            continue
        seen.add(url)
        title = line[: m.start()].strip(" -—:•*0123456789.[]()") or url
        out.append({"title": title, "url": url, "snippet": ""})
    return out


# --- Composio ---------------------------------------------------------------


async def _search_composio(query: str, max_results: int, db: Any, workspace_id: Any) -> List[Dict[str, Any]]:
    from core.composio.entity_manager import EntityManager
    from core.composio.tool_executor import ComposioToolExecutor

    if db is None or workspace_id is None:
        raise ValueError("Composio search needs a workspace session")
    EntityManager(db).get_or_create_entity(workspace_id)
    result = await ComposioToolExecutor(db).execute(
        action=COMPOSIO_SEARCH_ACTION,
        params={"query": query},
        agent_id=0,
        workspace_id=workspace_id,
        app_name=COMPOSIO_SEARCH_APP,
        skip_validation=True,
    )
    if not result.get("success"):
        raise RuntimeError(result.get("error") or "Composio search returned no result")
    return _parse_composio(result.get("data"))[:max_results]


def _parse_composio(data: Any) -> List[Dict[str, Any]]:
    """Composio's search payload is not pinned by a contract here — accept the
    shapes it is known to use (``results`` / ``items`` / ``organic`` lists of
    ``{title, url|link, snippet|description}``) and skip anything else."""
    if isinstance(data, dict):
        for key in ("results", "items", "organic", "organic_results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            inner = data.get("data") if isinstance(data.get("data"), (dict, list)) else None
            if inner is not None:
                return _parse_composio(inner)
            return []
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        r = _normalise(
            item.get("title"),
            item.get("url") or item.get("link"),
            item.get("snippet") or item.get("description") or item.get("content"),
        )
        if r:
            out.append(r)
    return out


# --- SearXNG ----------------------------------------------------------------


async def _search_searxng(query: str, max_results: int) -> List[Dict[str, Any]]:
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(float(config.WEB_FETCH_TIMEOUT_SECONDS))) as client:
        resp = await client.get(
            f"{config.SEARXNG_URL}/search",
            params={"q": query, "format": "json", "safesearch": 1},
        )
    if resp.status_code == 403:
        raise RuntimeError("SearXNG refused the JSON format — add 'json' to search.formats in its settings.yml")
    resp.raise_for_status()
    payload = resp.json()
    out: List[Dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        r = _normalise(item.get("title"), item.get("url"), item.get("content"))
        if r:
            out.append(r)
    return out[:max_results]
