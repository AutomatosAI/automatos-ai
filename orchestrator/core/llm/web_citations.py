"""PRD-240 — provider web-search citations, one shape everywhere.

OpenRouter (and the OpenAI-compatible providers that pass search through)
return the pages a search used as ``url_citation`` annotations on the
assistant message — in the final chunk when streaming, on the message
otherwise. This module turns them into ``[{title, url, snippet}]`` and renders
the Sources footer the chat shows. Pure: no I/O, no config.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

SOURCES_HEADING = "**Sources**"
_MAX_SOURCES_SHOWN = 8


def citations_from_annotations(annotations: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    """Normalise ``url_citation`` annotations; unknown shapes are skipped, not raised.

    Accepts the OpenAI/OpenRouter shape ``{"type": "url_citation",
    "url_citation": {"url", "title", "content", ...}}`` and the flattened
    ``{"url", "title"}`` some providers emit. De-duplicated by URL, order kept.
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for item in annotations or []:
        raw = _as_dict(item)
        if not raw:
            continue
        payload = _as_dict(raw.get("url_citation")) or raw
        url = str(payload.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "title": str(payload.get("title") or "").strip() or url,
                "url": url,
                "snippet": str(payload.get("content") or payload.get("snippet") or "").strip(),
            }
        )
    return out


def sources_markdown(citations: List[Dict[str, Any]]) -> str:
    """The footer appended to a reply that used a web search — empty when none."""
    if not citations:
        return ""
    lines = [f"- [{c.get('title') or c['url']}]({c['url']})" for c in citations[:_MAX_SOURCES_SHOWN]]
    return f"\n\n{SOURCES_HEADING}\n" + "\n".join(lines) + "\n"


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            data = dump()
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 — a foreign object is skipped, never fatal
            return {}
    return {}
