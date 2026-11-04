"""Composio catalogue payloads as the plain dicts the catalogue sync stores.

Pure mapping, moved out of ``core.composio.client`` so its fetches stay short: one
toolkit from the SDK's ``toolkits.list`` becomes an app record, and one item of the
REST bulk ``/tools`` page becomes an action record. No I/O here; the client keeps
the calls, this module keeps the field fallbacks the SDK and API versions need.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TriggerMap = Dict[str, List[Dict[str, Any]]]

# Toolkits whose trigger lookups are logged, to debug a missing trigger list.
TRIGGER_DEBUG_TOOLKITS = ("slack", "gmail", "github")

# Bulk /tools item keys that become first-class action fields; the rest is metadata.
BULK_ACTION_FIELD_KEYS = frozenset({
    "function", "name", "slug", "toolkit", "toolkit_slug", "toolkitSlug",
    "type", "description", "summary", "parameters", "response_schema",
    "tags", "category", "is_premium",
})


# ---------------------------------------------------------------------------
# Toolkits (SDK ``toolkits.list``) -> app records
# ---------------------------------------------------------------------------


def normalize_triggers(triggers: Any) -> List[Dict[str, Any]]:
    """Normalize triggers into a list of {name, display_name, description} dicts."""
    if not triggers:
        return []
    if isinstance(triggers, list):
        normalized = []
        for trigger in triggers:
            if isinstance(trigger, str):
                normalized.append({"name": trigger, "display_name": trigger, "description": ""})
            elif isinstance(trigger, dict):
                normalized.append({
                    "name": trigger.get("name") or trigger.get("slug") or trigger.get("trigger_name"),
                    "display_name": trigger.get("display_name") or trigger.get("name") or trigger.get("slug"),
                    "description": trigger.get("description", ""),
                })
        return normalized
    return []


def toolkit_app_record(app: Any, trigger_map: TriggerMap) -> Dict[str, Any]:
    """One SDK toolkit as the app record the catalogue sync stores."""
    normalized_triggers = _toolkit_triggers(app, trigger_map)
    return {
        "name": app.slug,  # Use slug as the identifier (e.g. 'github')
        "display_name": app.name,
        "description": _toolkit_description(app),
        "logo_url": _toolkit_logo_url(app),
        "categories": _toolkit_categories(app),
        "auth_schemes": app.auth_schemes or [],
        "action_count": _toolkit_action_count(app),
        "triggers": normalized_triggers,
        "trigger_count": len(normalized_triggers),
    }


def _toolkit_raw_triggers(app: Any, trigger_map: TriggerMap) -> Tuple[Any, str]:
    """The toolkit's triggers from the first source that has any, and that source's name."""
    # Try multiple sources for triggers
    raw_triggers = None
    source = "none"
    if hasattr(app, "meta") and hasattr(app.meta, "triggers"):
        raw_triggers = getattr(app.meta, "triggers", None)
        source = "app.meta.triggers"
    if not raw_triggers and hasattr(app, "triggers"):
        raw_triggers = getattr(app, "triggers", None)
        source = "app.triggers"
    if not raw_triggers:
        raw_triggers = trigger_map.get(app.slug.lower(), [])
        source = f"trigger_map[{app.slug.lower()}]"
    return raw_triggers, source


def _toolkit_triggers(app: Any, trigger_map: TriggerMap) -> List[Dict[str, Any]]:
    """The toolkit's triggers, normalized."""
    raw_triggers, source = _toolkit_raw_triggers(app, trigger_map)
    normalized_triggers = normalize_triggers(raw_triggers)

    # Log if we found triggers (for debugging)
    if normalized_triggers and app.slug.lower() in TRIGGER_DEBUG_TOOLKITS:
        logger.debug(f"Found {len(normalized_triggers)} triggers for {app.slug} (source: {source})")
    return normalized_triggers


def _toolkit_description(app: Any) -> str:
    """Extract description from multiple possible sources."""
    if hasattr(app, "meta"):
        return (
            getattr(app.meta, "description", None)
            or getattr(app.meta, "summary", None)
            or getattr(app, "description", None)
            or ""
        )
    return getattr(app, "description", None) or ""


def _toolkit_action_count(app: Any) -> int:
    """SDK uses tools_count (not actions_count)."""
    if not hasattr(app, "meta"):
        return 0
    return int(
        getattr(app.meta, "tools_count", 0)
        or getattr(app.meta, "actions_count", 0)
        or 0
    )


def _toolkit_logo_url(app: Any) -> Optional[str]:
    """Logo: nested under meta (Pydantic model or dict)."""
    if not hasattr(app, "meta") or app.meta is None:
        return None
    if isinstance(app.meta, dict):
        return app.meta.get("logo")
    return getattr(app.meta, "logo", None)


def _toolkit_categories(app: Any) -> List[str]:
    """Categories: top-level List[str] on AppModel, fallback to meta."""
    categories = getattr(app, "categories", None) or []
    if not categories and hasattr(app, "meta"):
        meta_cats = (
            app.meta.get("categories", [])
            if isinstance(app.meta, dict)
            else getattr(app.meta, "categories", [])
        )
        return [
            (c.name if hasattr(c, "name") else str(c))
            for c in (meta_cats or [])
        ]
    return [str(c) for c in categories]


# ---------------------------------------------------------------------------
# Bulk tools (REST ``/tools`` page items) -> action records
# ---------------------------------------------------------------------------


def bulk_actions_from_items(items: List[Any]) -> List[Dict[str, Any]]:
    """The action records on one page of the bulk ``/tools`` listing, in page order.

    An item that is not a dict, is not a function tool, or names no action or
    toolkit is skipped.
    """
    records = (_bulk_item_action(item) for item in items if isinstance(item, dict))
    return [record for record in records if record is not None]


def _bulk_item_action(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One bulk ``/tools`` item as an action record, or ``None`` when it is skipped."""
    fn = item.get("function") if isinstance(item.get("function"), dict) else None
    if item.get("type") and item.get("type") != "function" and fn is None:
        # Ignore non-function tools unless function payload exists.
        return None

    # Extract display name (human-readable) from function.name or item.name
    display_name_raw = str((fn or {}).get("name") or item.get("name") or "").strip()
    action_name = _bulk_action_name(item, display_name_raw)
    if not action_name:
        return None
    app_name = _bulk_app_name(item, action_name)
    if not app_name:
        return None

    return {
        "app_name": app_name,
        "name": action_name,
        "display_name": display_name_raw or item.get("display_name") or action_name,
        "description": (
            (fn or {}).get("description")
            or item.get("description")
            or item.get("summary")
            or ""
        ),
        "parameters": (fn or {}).get("parameters") or item.get("parameters") or {},
        # Additional metadata fields (if present in API response)
        "response_schema": (fn or {}).get("response_schema") or item.get("response_schema") or {},
        "tags": item.get("tags") or (fn or {}).get("tags") or [],
        "category": item.get("category") or (fn or {}).get("category") or None,
        "is_premium": item.get("is_premium") or (fn or {}).get("is_premium") or False,
        # Store any extra metadata (sanitized to avoid circular refs)
        "metadata": _bulk_extra_metadata(item),
    }


def _bulk_action_name(item: Dict[str, Any], display_name_raw: str) -> str:
    """The API enum identifier (e.g. ``GMAIL_FETCH_EMAILS``); ``""`` when the item has none."""
    # Extract slug and derive API enum identifier (e.g., "GMAIL_FETCH_EMAILS")
    item_slug = str(item.get("slug") or "").strip()
    item_enum = str(item.get("enum") or "").strip()

    # Prefer enum > slug-derived > display-name-derived
    if item_enum:
        return item_enum.upper()
    if item_slug:
        return item_slug.upper().replace("-", "_")
    return display_name_raw


def _bulk_app_name(item: Dict[str, Any], action_name: str) -> str:
    """The toolkit the action belongs to, upper-cased; ``""`` when the item names none."""
    toolkit = item.get("toolkit")
    toolkit_slug = None
    if isinstance(toolkit, dict):
        toolkit_slug = toolkit.get("slug") or toolkit.get("name")
    elif isinstance(toolkit, str):
        toolkit_slug = toolkit
    toolkit_slug = toolkit_slug or item.get("toolkit_slug") or item.get("toolkitSlug")

    app_name = str(toolkit_slug or "").strip()
    if not app_name and "_" in action_name:
        # Fallback: many actions are prefixed with TOOLKIT_*
        app_name = action_name.split("_", 1)[0]

    return app_name.upper().strip()


def _bulk_extra_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """The item's other fields, keeping only JSON-serializable values (no circular refs).

    The REST API may include more metadata than the SDK wrapper exposes.
    """
    extra_metadata = {}
    for k, v in item.items():
        if k in BULK_ACTION_FIELD_KEYS:
            continue
        try:
            # Test if value is JSON-serializable (breaks circular refs)
            json.dumps(v, default=str)
            extra_metadata[k] = v
        except (TypeError, ValueError):
            # Skip non-serializable values (likely circular refs)
            pass
    return extra_metadata
