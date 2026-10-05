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

Night 10 (F332): session agents had the voice and not the look: three different navies
were guessed in one night, and no agent had the logo, the address or the phone. The
rules block now also carries the kit's colours (hex), its fonts, the company's contact
details and where the logo is, each only when the kit sets it, and says the kit wins
over any document that says otherwise (a brand-voice document kept an old sign-off).

Who signs (:func:`sign_off_name`): the voice's own sign-off, else the company
contact's name, else the brand's name. Reads are cached per workspace for
:data:`KIT_CACHE_SECONDS`: a run reads the kit several times (prompt, answer,
document), and an owner's edit reaches the next run within that time.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

KIT_CACHE_SECONDS = 30.0
RULES_HEADING = "## The brand's rules"
RULES_LEAD = ("From the owner's brand kit. Everything you write for the owner to send or publish "
              "(an email, a letter, a post, a document) keeps to them:")
BANNED_NOTE_LEAD = "Check before using this answer: it uses words the brand kit bans"
KIT_WINS_LINE = "- The brand kit wins over any document that says otherwise."
COLOUR_KEYS = ("primary", "secondary", "accent", "text")
CONTACT_FIELDS = (("address", "Address"), ("phone", "Phone"), ("email", "Email"), ("website", "Website"))
UPLOADED_LOGO_LINE = ("- Logo: the one uploaded to the brand kit. generate_document puts it on the document "
                      "for you; in a session, your ticket file lists its copy.")
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


def _fresh(workspace_id: Any) -> Optional[Tuple[float, Any]]:
    """The cached read for ``workspace_id`` while it is fresh, else None."""
    hit = _cache.get(str(workspace_id))
    return hit if hit is not None and hit[0] > time.monotonic() else None


def _cached(workspace_id: Any, read: Callable[[], Any]) -> Any:
    hit = _fresh(workspace_id)
    if hit is not None:
        return hit[1]
    value = read()
    _cache[str(workspace_id)] = (time.monotonic() + KIT_CACHE_SECONDS, value)
    return value


def _workspace_uuid(workspace_id: Any) -> Optional[UUID]:
    """``workspace_id`` as the workspaces table's key; None for anything that is not one."""
    if isinstance(workspace_id, UUID):
        return workspace_id
    try:
        return UUID(str(workspace_id))
    except ValueError:
        return None


def without_flushing(db: Any) -> Any:
    """A block in which ``db``'s reads never flush the caller's pending changes.

    A read that autoflushed would write them, and take their row locks, in a transaction
    only the caller ends: ``cli_host_service._ticket_prompt`` clears the card's
    review_feedback just before its prompt reads the kit, and a caller that never
    commits then holds the card's row (the PRD-252 Discuss tests hung on it)."""
    return getattr(db, "no_autoflush", None) or nullcontext()


def _settings(db: Any, workspace_id: UUID) -> Optional[Dict[str, Any]]:
    from core.models.workspaces import Workspace

    with without_flushing(db):
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


async def kit_off_loop(db: Any, workspace_id: Any) -> Optional[Dict[str, Any]]:
    """:func:`stored_kit` for async code: a fresh cached read at once, else the read on a
    worker thread, so a pool wait never stops the event loop (F105, F330)."""
    key = _workspace_uuid(workspace_id) if workspace_id else None
    hit = _fresh(key) if key is not None else None
    if hit is not None:
        return hit[1]
    return await asyncio.to_thread(stored_kit, db, workspace_id)


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


def _voice_lines(kit: Dict[str, Any]) -> List[str]:
    """Who the brand writes as, its tone, who signs and the words it never uses."""
    voice = kit.get("voice") or {}
    name, signer = (kit.get("name") or "").strip(), sign_off_name(kit)
    return [f"- Write as {name}." if name else "",
            f"- Tone: {', '.join(voice.get('tone') or [])}." if voice.get("tone") else "",
            (f'- Sign it "{signer}". Never leave a placeholder such as [Your name].' if signer else ""),
            (f"- Never use these words or phrases: {_quoted(voice['banned_phrases'])}."
             if voice.get("banned_phrases") else "")]


def _set(value: Any, default: str = "") -> str:
    """``value`` trimmed when the owner set it: empty for a blank or the platform's default."""
    text = value.strip() if isinstance(value, str) else ""
    return "" if text.lower() == default.lower() else text


def _one_line(text: str) -> str:
    """A multi-line value (an address) on one line, its lines joined by commas."""
    return ", ".join(line.strip().rstrip(",") for line in text.splitlines() if line.strip())


def _colours_line(kit: Dict[str, Any]) -> str:
    """The kit's colours as hex (F332); the neutral defaults are not the brand's."""
    from modules.documents import brand_kit as bk

    defaults = {"primary": bk.DEFAULT_PRIMARY, "secondary": bk.DEFAULT_SECONDARY,
                "accent": bk.DEFAULT_ACCENT, "text": bk.DEFAULT_TEXT}
    values = {key: _set(kit.get(f"{key}_color"), defaults[key]) for key in COLOUR_KEYS}
    colours = [f"{key} {value}" for key, value in values.items() if value]
    return f"- Colours (hex): {', '.join(colours)}." if colours else ""


def _fonts_line(kit: Dict[str, Any]) -> str:
    """The body and heading fonts (F332), by their first family."""
    from modules.documents.brand_kit import DEFAULT_FONT

    body = _first_family(_set(kit.get("font_family"), DEFAULT_FONT))
    heading = _first_family(_set(kit.get("heading_font")))
    if heading and body:
        return f"- Fonts: {heading} for headings, {body} for body text."
    if heading:
        return f"- Fonts: {heading} for headings."
    return f"- Font: {body}, for headings and body text." if body else ""


def _contact_line(kit: Dict[str, Any]) -> str:
    """The company's contact details (F332): only the ones the kit sets."""
    company = kit.get("company") or {}
    parts = [f"{label}: {_one_line(_set(company.get(key)))}" for key, label in CONTACT_FIELDS
             if _set(company.get(key))]
    name = _set(company.get("name"))
    if not parts:
        return f"- Company: {name}." if name else ""
    return f"- Company: {name}. {'. '.join(parts)}." if name else f"- Company details: {'. '.join(parts)}."


def _logo_line(kit: Dict[str, Any]) -> str:
    """Where the logo is (F332): uploaded to the kit, or the URL the kit names."""
    if _set(kit.get("logo_path")) or _set(kit.get("logo_mark_path")):
        return UPLOADED_LOGO_LINE
    url = _set(kit.get("logo_url")) or _set(kit.get("logo_mark_url"))
    return f"- Logo: {url}" if url else ""


def _look_lines(kit: Dict[str, Any]) -> List[str]:
    """How the brand looks and who it is (F332): colours, fonts, contact details, logo."""
    return [_colours_line(kit), _fonts_line(kit), _contact_line(kit), _logo_line(kit)]


def rules_for_kit(kit: Optional[Dict[str, Any]]) -> Optional[str]:
    """The rules block for ``kit``; None when it says nothing about how to write or look.
    Every rule is a "- " line, so a block an answer repeats reads as one block."""
    if not kit:
        return None
    kept = [line for line in [*_voice_lines(kit), *_look_lines(kit)] if line]
    return "\n".join([RULES_HEADING, RULES_LEAD, *kept, KIT_WINS_LINE]) if kept else None


def brand_rules_block(db: Any, workspace_id: Any) -> Optional[str]:
    """The plain-English rules a drafting run is given; None when there is no kit."""
    return rules_for_kit(stored_kit(db, workspace_id))


def prompt_with_rules(prompt: str, kit: Optional[Dict[str, Any]]) -> str:
    """``prompt`` with ``kit``'s rules after it, once: a prompt that has them already is left."""
    if RULES_HEADING in (prompt or ""):
        return prompt
    block = rules_for_kit(kit)
    return f"{prompt}\n\n{block}" if block else prompt


def with_brand_rules(prompt: str, db: Any, workspace_id: Any) -> str:
    """``prompt`` with the workspace's rules after it, once."""
    if RULES_HEADING in (prompt or ""):
        return prompt
    return prompt_with_rules(prompt, stored_kit(db, workspace_id))


async def with_brand_rules_off_loop(prompt: str, db: Any, workspace_id: Any) -> str:
    """:func:`with_brand_rules` for async code: the kit read never waits on the loop."""
    if RULES_HEADING in (prompt or ""):
        return prompt
    return prompt_with_rules(prompt, await kit_off_loop(db, workspace_id))


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


def _text_key(result: Any) -> Optional[str]:
    """Which of a finished run's result keys holds its answer; None for a failed run or none."""
    if not isinstance(result, dict) or result.get("status") in ("error", "cancelled"):
        return None
    return next((k for k in TEXT_KEYS if isinstance(result.get(k), str) and result.get(k).strip()), None)


def on_brand_result(db: Any, workspace_id: Any, result: Any) -> Any:
    """A run's result dict with its text :func:`on_brand_text`; any other result as it was."""
    kit = stored_kit(db, workspace_id) if _text_key(result) else None
    return result_on_brand(result, kit, workspace_id)


async def on_brand_result_off_loop(db: Any, workspace_id: Any, result: Any) -> Any:
    """:func:`on_brand_result` for async code: the kit read never waits on the loop."""
    kit = await kit_off_loop(db, workspace_id) if _text_key(result) else None
    return result_on_brand(result, kit, workspace_id)


def result_on_brand(result: Any, kit: Optional[Dict[str, Any]], workspace_id: Any = None) -> Any:
    """``result`` with its answer :func:`on_brand_text` by ``kit``: a new dict when it changed."""
    key = _text_key(result)
    if kit is None or key is None:
        return result
    text = on_brand_text(result[key], kit)
    if text != result[key]:
        logger.info("[BrandRules] a run's answer was brought to the brand kit in workspace %s", workspace_id)
        return {**result, key: text}
    return result


__all__ = [
    "BANNED_NOTE_LEAD", "KIT_CACHE_SECONDS", "KIT_WINS_LINE", "RULES_HEADING", "banned_found", "banned_note",
    "brand_assets", "brand_rules_block", "fill_sign_off", "forget_cached_kits", "kit_off_loop", "on_brand_result",
    "on_brand_result_off_loop", "on_brand_text", "prompt_with_rules", "result_on_brand", "rules_for_kit",
    "sign_off_name", "stored_kit", "with_brand_rules", "with_brand_rules_off_loop", "without_flushing",
]
