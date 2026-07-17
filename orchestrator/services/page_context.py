"""PRD-221 S2 — structured page context: sanitize → preamble → inject.

The client may describe WHERE the user is (page key/route, tab, selected
entity, filters, visible ids) — never WHO they are: authz-looking fields are
not in the allow-list and are dropped unread; the server derives roles itself
(PRD-143 discipline). IDs are references Auto re-fetches through platform
tools — page payloads never ride the prompt (PRD-221 decision: references,
not payloads).

Generalises the storefront pattern (integrations/shopify/context_fields.py
allow-list + widget_proactive's system preamble) to platform pages, keyed by
the checked-in page manifest (core/page_manifest.py). One renderer serves both
the structured form and the legacy ``{"page": "<label>"}`` form (PRD-220), so
there is exactly one injection path.

Injection preserves the PRD-220 invariants: the preamble is added AFTER the
clean DB save (caller's responsibility — api/chat.py call placement) and the
history entries are REBUILT, never mutated — ``parts`` is the ORM row's JSONB
list and an in-place append could flush the hint into the ``messages`` table.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.page_manifest import PageEntry, get_page, resolve_route

# Reference-set caps: context carries pointers, never payloads.
_MAX_STR = 128
_MAX_VISIBLE_IDS = 16
_MAX_FILTERS = 8
# Legacy one-line form keeps the historic PRD-220 cap.
_LEGACY_LABEL_MAX_LEN = 80


def _clean_str(value: Any, cap: int = _MAX_STR) -> Optional[str]:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned[:cap] if cleaned else None


def sanitize_page_context(raw: Any) -> Dict[str, Any]:
    """Reduce a client context dict to the allow-listed reference set.

    Unknown keys are dropped unread — authz-looking fields (userRole,
    is_admin, permissions, …) never survive because nothing reads them.
    Garbage input returns ``{}``.
    """
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, Any] = {}

    page = _clean_str(raw.get("page"))
    if page:
        out["page"] = page

    route = _clean_str(raw.get("route"))
    if route and route.startswith("/"):
        out["route"] = route

    tab = _clean_str(raw.get("tab"))
    if tab:
        out["tab"] = tab

    selected = raw.get("selected")
    if isinstance(selected, dict):
        sel_type = _clean_str(selected.get("type"))
        sel_id = _clean_str(selected.get("id"))
        if sel_type and sel_id:
            out["selected"] = {"type": sel_type, "id": sel_id}

    filters = raw.get("filters")
    if isinstance(filters, dict):
        kept: Dict[str, str] = {}
        for key, value in filters.items():
            if len(kept) >= _MAX_FILTERS:
                break
            clean_key = _clean_str(key)
            clean_value = _clean_str(value)
            if clean_key and clean_value is not None:
                kept[clean_key] = clean_value
        if kept:
            out["filters"] = kept

    visible = raw.get("visible_ids")
    if isinstance(visible, (list, tuple)):
        ids = []
        for item in visible:
            if len(ids) >= _MAX_VISIBLE_IDS:
                break
            clean_id = _clean_str(item)
            if clean_id:
                ids.append(clean_id)
        if ids:
            out["visible_ids"] = ids

    # A context with no page and no route references nothing — treat as empty.
    if "page" not in out and "route" not in out:
        return {}
    return out


def _resolve_entry(sanitized: Dict[str, Any]) -> Optional[PageEntry]:
    entry = get_page(sanitized.get("page"))
    if entry is not None:
        return entry
    key = resolve_route(sanitized.get("route"))
    return get_page(key) if key else None


def render_page_preamble(sanitized: Dict[str, Any]) -> str:
    """One renderer for both forms: manifest-grounded block, or legacy line."""
    if not sanitized:
        return ""

    entry = _resolve_entry(sanitized)
    if entry is None:
        label = (sanitized.get("page") or "").strip()[:_LEGACY_LABEL_MAX_LEN]
        if not label:
            return ""
        return f"[Context: the user is currently on the {label} page]"

    bits: List[str] = [
        f'[Page context] The user is on "{entry.title}" ({entry.route}) — {entry.purpose}'
    ]
    tab = sanitized.get("tab")
    if tab:
        bits.append(f"Tab: {tab}.")
    selected = sanitized.get("selected")
    if selected:
        bits.append(f"Selected {selected['type']}: {selected['id']}.")
    filters = sanitized.get("filters")
    if filters:
        rendered = "; ".join(f"{k}={v}" for k, v in filters.items())
        bits.append(f"Active filters: {rendered}.")
    visible = sanitized.get("visible_ids")
    if visible:
        bits.append(f"Visible item ids ({len(visible)}): {', '.join(visible)}.")
    bits.append(
        "Treat these as references — fetch fresh details with platform tools "
        "instead of assuming page contents."
    )
    return " ".join(bits)


def page_actions_from_context(sanitized: Optional[Dict[str, Any]]) -> List[str]:
    """Manifest action names for the page the sanitized context points at.

    Empty list when the context references no known page — the caller then
    gets pure semantic narrowing (unchanged behaviour). Never raises.
    """
    if not sanitized:
        return []
    entry = _resolve_entry(sanitized)
    return list(entry.actions) if entry else []


def merge_into_trace(
    trace: Optional[Dict[str, Any]], sanitized: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Return a NEW trace dict carrying the sanitized page context (PRD-221 S3).

    Empty/absent context returns the trace untouched — identity, including
    ``None``. Never mutates the input trace (it may already be bound to an ORM
    row's JSONB), and only ever receives SANITIZED context — the raw client
    dict is never stored.
    """
    if not sanitized:
        return trace
    return {**(trace or {}), "page_context": sanitized}


def inject_page_preamble(message_history: List[dict], sanitized: Dict[str, Any]) -> List[dict]:
    """Return history with an ephemeral preamble part on the last user message.

    Entries are rebuilt, never mutated (ORM-JSONB flush trap — PRD-220).
    Empty context returns the history unchanged.
    """
    text = render_page_preamble(sanitized)
    if not text:
        return message_history
    for i in range(len(message_history) - 1, -1, -1):
        entry = message_history[i]
        if entry.get("role") != "user":
            continue
        parts = entry.get("parts") if isinstance(entry.get("parts"), list) else []
        hint = {"type": "text", "text": text}
        rebuilt = {**entry, "parts": [*parts, hint]}
        return [*message_history[:i], rebuilt, *message_history[i + 1:]]
    return message_history
