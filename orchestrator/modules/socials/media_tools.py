"""PRD-251B Wave 3 (B10; US-B304): the AI tools a workspace uses, and its defaults per media type.

The Brand kit's AI tools section lists every media toolkit the capability registry can
offer the workspace (the footage and stills recipes, the voice recipes), each connected,
to connect in Composio, or unavailable and why; Templates and Kokoro are built in, free,
and need nothing. Paid tools are connected in Composio only (D15).

The defaults per media type live in ``workspace.settings['media_tools']``:

* ``images``: ``templates`` (free) or a toolkit that makes stills;
* ``ai_images``: a toolkit that makes stills, or ``ask`` (choose each time);
* ``footage``: a toolkit that makes footage, or ``off``;
* ``voice``: ``kokoro`` or a voice toolkit.

A default must be one the workspace is offered now. A render's footage and stills try the
default toolkit first (:func:`prefer_for`). The monthly media cap (``socials`` settings,
D13) and the per-post cap that overrides ``SOCIALS_MEDIA_POST_CAP_USD`` are kept beside.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from core.social_templates import IMAGE_SLOT, VIDEO_SLOT
from modules.socials.recipes.footage import footage_sources
from modules.socials.recipes.voice import voice_sources
from modules.socials.service import InvalidPost

MEDIA_TOOLS_KEY = "media_tools"
TEMPLATES, ASK, OFF, KOKORO = "templates", "ask", "off", "kokoro"
MEDIA_TYPES = ("images", "ai_images", "footage", "voice")
DEFAULTS = {"images": TEMPLATES, "ai_images": ASK, "footage": OFF, "voice": KOKORO}
AVAILABLE = "available"


def _toolkits_making(sources: Mapping[str, Any], kind: str) -> List[Dict[str, str]]:
    return [
        {"value": t["toolkit"], "label": t["label"]}
        for t in sources.get("toolkits") or []
        if t.get("status") == AVAILABLE and kind in (t.get("makes") or [])
    ]


def offered(caps: Any) -> Dict[str, List[Dict[str, str]]]:
    """The choices per media type the workspace has now."""
    footage = footage_sources(caps)
    voices = voice_sources(caps)
    stills = _toolkits_making(footage, IMAGE_SLOT)
    video = _toolkits_making(footage, VIDEO_SLOT)
    speakers = [{"value": s["toolkit"], "label": s["label"]} for s in voices["sources"] if s.get("status") == AVAILABLE and not s.get("builtin")]
    return {
        "images": [{"value": TEMPLATES, "label": "Templates (free)"}, *stills],
        "ai_images": [*stills, {"value": ASK, "label": "Ask each time"}],
        "footage": [*video, {"value": OFF, "label": "Off"}],
        "voice": [{"value": KOKORO, "label": "Kokoro (free)"}, *speakers],
    }


def toolkit_rows(caps: Any) -> List[Dict[str, Any]]:
    """The section's rows: each generation and voice toolkit with its state, then the built-ins."""
    footage = footage_sources(caps)
    voices = voice_sources(caps)
    rows = [{**t, "kind": "Images and footage"} for t in footage.get("toolkits") or []]
    rows += [{**s, "kind": "Voice"} for s in voices["sources"] if not s.get("builtin")]
    rows += [
        {"toolkit": TEMPLATES, "label": "Templates", "kind": "Images", "status": "builtin"},
        {"toolkit": KOKORO, "label": "Kokoro", "kind": "Voice", "status": "builtin"},
    ]
    return rows


def defaults_of(settings: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    raw = (settings or {}).get(MEDIA_TOOLS_KEY) or {}
    return {kind: str(raw.get(kind) or DEFAULTS[kind]) for kind in MEDIA_TYPES}


def validate_defaults(changes: Mapping[str, Any], choices: Mapping[str, List[Dict[str, str]]], current: Mapping[str, str]) -> Dict[str, str]:
    """``current`` with ``changes``, each one of the choices offered for its media type (422 otherwise)."""
    unknown = [kind for kind in changes if kind not in MEDIA_TYPES]
    if unknown:
        raise InvalidPost(f"defaults are set per media type: {', '.join(MEDIA_TYPES)}")
    merged = dict(current)
    for kind, value in changes.items():
        allowed = [choice["value"] for choice in choices.get(kind) or []]
        if value not in allowed:
            raise InvalidPost(f"the {kind.replace('_', ' ')} default must be one of {', '.join(allowed)}: connect a toolkit in Composio first")
        merged[kind] = value
    return merged


def prefer_for(settings: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """The toolkit a render tries first per slot kind: the footage and AI images defaults."""
    defaults = defaults_of(settings)
    prefer = {}
    if defaults["footage"] not in (OFF, TEMPLATES):
        prefer[VIDEO_SLOT] = defaults["footage"]
    if defaults["ai_images"] not in (ASK, TEMPLATES):
        prefer[IMAGE_SLOT] = defaults["ai_images"]
    return prefer
