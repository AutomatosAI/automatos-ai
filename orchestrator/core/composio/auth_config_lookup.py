"""A toolkit's Composio auth configs and accounts, every page of them (3 Oct 2026).

Composio lists a project's auth configs 20 to a page, newest first. The resolver
read only that first page. With 62 configs in the project, X's own-app config
("Automatos-X", custom OAUTH2, made in May for the SaaS edition) sat on page 3, so
connecting X found none, tried Composio's managed auth (there is none for X since
February 2026), then a custom config with no credentials, and Composio answered
400 "Missing required field Client id". 23 toolkits had no config on page 1:
those Composio can't manage failed the same way, and a workspace's first connect
to any other made a duplicate config (github had four).

Each toolkit's configs are now asked for by toolkit, every page followed. A
connected account made under an older config of the same toolkit is still the
entity's: the lookups that find one fall back to the whole toolkit (#749).
"""
from __future__ import annotations

from typing import Any, List, Optional

PAGE_SIZE = 100
MAX_PAGES = 20  # 2,000 configs for one toolkit is a runaway, not a project: stop reading
ENABLED = "ENABLED"


def toolkit_auth_configs(composio: Any, app_slug: str) -> List[Any]:
    """Every auth config of one toolkit, newest first."""
    slug = app_slug.lower()
    configs: List[Any] = []
    cursor = None
    for _ in range(MAX_PAGES):
        query = {"toolkit_slug": slug, "limit": PAGE_SIZE, **({"cursor": cursor} if cursor else {})}
        page = composio.auth_configs.list(**query)
        configs.extend(_items(page))
        cursor = getattr(page, "next_cursor", None)
        if not cursor:
            break
    # Composio answers newest first; the order is ours to keep, not its to change.
    ours = [c for c in configs if (getattr(getattr(c, "toolkit", None), "slug", "") or "").lower() == slug]
    return sorted(ours, key=lambda c: str(getattr(c, "created_at", "") or ""), reverse=True)


def newest_enabled(configs: List[Any], preferred_scheme: Optional[str] = None) -> Optional[str]:
    """The id of the newest ENABLED config, of ``preferred_scheme`` when one is asked for."""
    wanted = (preferred_scheme or "").upper()
    for config in configs:
        if getattr(config, "status", ENABLED) != ENABLED:
            continue
        scheme = (getattr(config, "auth_scheme", "") or getattr(config, "authScheme", "") or "").upper()
        if wanted and scheme != wanted:
            continue
        return config.id
    return None


def toolkit_accounts(composio: Any, entity_id: str, app_slug: str, auth_config_id: Optional[str]) -> List[Any]:
    """The entity's connected accounts for the toolkit: those under ``auth_config_id``,
    or, when it has none there, those under any of the toolkit's configs."""
    if auth_config_id:
        under = _items(composio.connected_accounts.list(user_ids=[entity_id], auth_config_ids=[auth_config_id]))
        if under:
            return under
    slug = app_slug.lower()
    every = _items(composio.connected_accounts.list(user_ids=[entity_id], toolkit_slugs=[slug]))
    return [a for a in every if (getattr(getattr(a, "toolkit", None), "slug", "") or "").lower() == slug]


def _items(page: Any) -> List[Any]:
    return list(getattr(page, "items", None) or getattr(page, "data", None) or [])


__all__ = ["newest_enabled", "toolkit_accounts", "toolkit_auth_configs"]
