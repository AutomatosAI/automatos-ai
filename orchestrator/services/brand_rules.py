"""The brand kit, applied by the platform at generation (night 9b, prep for night 10).

Night 9b: drafts and documents came out signed "[Your name]" (#1971, #0095, Auto's
1b39361c), with banned words ("delightful" on #1982 and #1986, "exquisite" in PDF
49c0c2b1) and without the owner's sign-off (#0107, #1964), even when the agent cited
the owner's brand voice paper. Following the brand was left to the agent finding a
document. The brand kit (``workspace.settings['brand_kit']``, one per workspace, read
and written by ``modules.documents.brand_kit``: Deliverables, Brand kit) is now
applied by the platform:

* :func:`brand_rules_block`: the kit's voice as a short block every agent run that
  drafts work is given (``services.brand_hooks``): the name it writes as, its tone
  words, who signs, and the words it never uses. The kit has no length rule; a
  length is the brief's.
* :func:`brand_assets`: the logo, colours and fonts, render-ready, for a renderer
  that has no brand kit of its own (the spreadsheet; the PDF, Word and social renders
  already read the kit, ``DocumentGenerationService._brand_kit_for``).
* :func:`on_brand_text`: a deterministic pass over finished text. A placeholder
  signature ("[Your name]", "[Your Name/Company]", "[Name]" on its own line) becomes
  the kit's sign-off; a banned word is not rewritten but said plainly in a note
  (:data:`BANNED_NOTE_LEAD`). With no sign-off the placeholder stays, so the card's
  leftover-placeholder note still warns.

Who signs (:func:`sign_off_name`): the voice's own sign-off, else the company
contact's name, else the brand's name. Reads are cached per workspace for
:data:`KIT_CACHE_SECONDS`: a run reads the kit several times (prompt, answer,
document), and an owner's edit reaches the next run within that time.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

KIT_CACHE_SECONDS = 30.0
RULES_HEADING = "## The brand's rules"
RULES_LEAD = ("From the owner's brand kit. Everything you write for the owner to send or publish "
              "(an email, a letter, a post, a document) keeps to them:")
BANNED_NOTE_LEAD = "Check before using this answer: it uses words the brand kit bans"
TEXT_KEYS = ("result", "response", "output", "content")

# The sender's name left as a placeholder: anywhere in the text.
SENDER_PLACEHOLDER = re.compile(
    r"\[\s*(?:your|my|sender'?s?)\s+(?:full\s+)?name\b[^\[\]\n]{0,60}\]", re.IGNORECASE)
# A bare "[Name]" is the sender's only where a signature goes: on a line of its own,
# or after a closing ("Best, [Name]"). "Dear [Name]," is the reader's and stays.
BARE_NAME_LINE = re.compile(r"(?im)^([ \t]*)\[\s*name\s*\]([ \t]*)$")
BARE_NAME_AFTER_CLOSING = re.compile(
    r"(?i)\b(best|regards|thanks|thank you|cheers|sincerely|warmly|best wishes|all the best|yours)"
    r"([ \t]*[,.]?[ \t]*)\[\s*name\s*\]")

_cache: Dict[str, Tuple[float, Any]] = {}


def forget_cached_kits() -> None:
    """Drop every cached read (tests, and anything that must see an edit at once)."""
    _cache.clear()


def _cached(workspace_id: Any, read: Callable[[], Any]) -> Any:
    key = str(workspace_id)
    hit = _cache.get(key)
    now = time.monotonic()
    if hit is not None and hit[0] > now:
        return hit[1]
    value = read()
    _cache[key] = (now + KIT_CACHE_SECONDS, value)
    return value


def _workspace_uuid(workspace_id: Any) -> Optional[UUID]:
    """``workspace_id`` as the workspaces table's key; None for anything that is not one."""
    if isinstance(workspace_id, UUID):
        return workspace_id
    try:
        return UUID(str(workspace_id))
    except ValueError:
        return None


def _settings(db: Any, workspace_id: UUID) -> Optional[Dict[str, Any]]:
    from core.models.workspaces import Workspace

    settings = getattr(db.get(Workspace, workspace_id), "settings", None)
    return settings if isinstance(settings, dict) else None


def _read_kit(db: Any, workspace_id: Any) -> Optional[Dict[str, Any]]:
    from modules.documents.brand_kit import BRAND_KIT_SETTINGS_KEY, get_brand_kit

    try:
        settings = _settings(db, workspace_id)
    except SQLAlchemyError:
        logger.exception("[BrandRules] could not read workspace %s's brand kit; going on without it", workspace_id)
        return None
    if not settings or not settings.get(BRAND_KIT_SETTINGS_KEY):
        return None
    return get_brand_kit(settings)


def stored_kit(db: Any, workspace_id: Any) -> Optional[Dict[str, Any]]:
    """The workspace's brand kit (defaults filled in), or None when it never set one, or
    when there is no session to read it with (a caller run without one)."""
    key = _workspace_uuid(workspace_id) if workspace_id else None
    if key is None or not callable(getattr(db, "get", None)):
        return None
    return _cached(key, lambda: _read_kit(db, key))


def sign_off_name(kit: Optional[Dict[str, Any]]) -> Optional[str]:
    """Who signs: the voice's sign-off, else the company contact's name, else the brand's."""
    if not kit:
        return None
    voice = kit.get("voice") or {}
    company = kit.get("company") or {}
    for value in (voice.get("sign_off"), company.get("name"), kit.get("name")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _quoted(words: Sequence[str]) -> str:
    return ", ".join(f'"{word}"' for word in words)


def rules_for_kit(kit: Optional[Dict[str, Any]]) -> Optional[str]:
    """The rules block for ``kit``; None when it says nothing about how to write."""
    if not kit:
        return None
    voice = kit.get("voice") or {}
    name, signer = (kit.get("name") or "").strip(), sign_off_name(kit)
    lines = [f"- Write as {name}." if name else "",
             f"- Tone: {', '.join(voice.get('tone') or [])}." if voice.get("tone") else "",
             (f'- Sign it "{signer}". Never leave a placeholder such as [Your name].' if signer else ""),
             (f"- Never use these words or phrases: {_quoted(voice['banned_phrases'])}."
              if voice.get("banned_phrases") else "")]
    kept = [line for line in lines if line]
    return "\n".join([RULES_HEADING, RULES_LEAD, *kept]) if kept else None


def brand_rules_block(db: Any, workspace_id: Any) -> Optional[str]:
    """The plain-English rules a drafting run is given; None when there is no kit."""
    return rules_for_kit(stored_kit(db, workspace_id))


def with_brand_rules(prompt: str, db: Any, workspace_id: Any) -> str:
    """``prompt`` with the rules after it, once: a prompt that has them already is left."""
    if RULES_HEADING in (prompt or ""):
        return prompt
    block = brand_rules_block(db, workspace_id)
    return f"{prompt}\n\n{block}" if block else prompt


def _first_family(stack: str) -> str:
    """The first family of a CSS font stack, unquoted (a spreadsheet takes one name)."""
    return (stack or "").split(",")[0].strip().strip("'\"")


def brand_assets(db: Any, workspace_id: Any) -> Optional[Dict[str, Any]]:
    """The logo (an uploaded one inlined as a data: URI), the colours and the first font
    of each stack, for a renderer; None when the workspace has no kit, so the renderer
    keeps its own look."""
    from modules.documents.brand_logo import brand_kit_for_render

    stored = stored_kit(db, workspace_id)
    if stored is None:
        return None
    kit = brand_kit_for_render(stored)
    return {
        "name": kit.get("name") or "",
        "logo": kit.get("logo_url") or "",
        "colours": {key: kit.get(f"{key}_color") for key in ("primary", "secondary", "accent", "text")},
        "fonts": {"body": _first_family(kit.get("font_family") or ""),
                  "heading": _first_family(kit.get("heading_font") or kit.get("font_family") or "")},
    }


def fill_sign_off(text: str, name: Optional[str]) -> str:
    """``text`` with a placeholder signature replaced by ``name``; unchanged without one."""
    if not name or not text or "[" not in text:
        return text
    filled = SENDER_PLACEHOLDER.sub(lambda _m: name, text)
    filled = BARE_NAME_LINE.sub(lambda m: f"{m.group(1)}{name}{m.group(2)}", filled)
    return BARE_NAME_AFTER_CLOSING.sub(lambda m: f"{m.group(1)}{m.group(2)}{name}", filled)


def banned_found(text: str, phrases: Sequence[str]) -> List[str]:
    """The banned phrases ``text`` uses, as the kit spells them, whole words, any case."""
    if not text:
        return []
    return [phrase for phrase in phrases
            if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE)]


def banned_note(found: Sequence[str]) -> str:
    """What the card says of the banned words it found; empty when none."""
    return f"{BANNED_NOTE_LEAD}: {_quoted(found)}. Change them before it goes out." if found else ""


def _without_note(text: str) -> str:
    """The text before an earlier banned-words note: the note quotes the words itself."""
    return text.split(BANNED_NOTE_LEAD, 1)[0]


def on_brand_text(text: str, kit: Optional[Dict[str, Any]]) -> str:
    """Finished ``text`` with its placeholder signature filled and, when it uses a banned
    word, the note after it (once)."""
    if not kit or not isinstance(text, str) or not text.strip():
        return text
    filled = fill_sign_off(text, sign_off_name(kit))
    if BANNED_NOTE_LEAD in filled:
        return filled
    note = banned_note(banned_found(_without_note(filled), (kit.get("voice") or {}).get("banned_phrases") or []))
    return f"{filled.rstrip()}\n\n{note}" if note else filled


def on_brand_result(db: Any, workspace_id: Any, result: Any) -> Any:
    """A run's result dict with its text :func:`on_brand_text`; any other result as it was."""
    if not isinstance(result, dict) or result.get("status") in ("error", "cancelled"):
        return result
    key = next((k for k in TEXT_KEYS if isinstance(result.get(k), str) and result.get(k).strip()), None)
    kit = stored_kit(db, workspace_id) if key else None
    if kit is None:
        return result
    text = on_brand_text(result[key], kit)
    if text != result[key]:
        logger.info("[BrandRules] a run's answer was brought to the brand kit in workspace %s", workspace_id)
        return {**result, key: text}
    return result


__all__ = [
    "BANNED_NOTE_LEAD", "KIT_CACHE_SECONDS", "RULES_HEADING", "banned_found", "banned_note", "brand_assets",
    "brand_rules_block", "fill_sign_off", "forget_cached_kits", "on_brand_result", "on_brand_text",
    "rules_for_kit", "sign_off_name", "stored_kit", "with_brand_rules",
]
