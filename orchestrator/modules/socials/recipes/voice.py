"""PRD-251 S1.5 (D11): a post's script spoken by the workspace's Composio voice toolkit.

Kokoro, inside media-render, is the default and needs nothing here: the render
bundle carries the script's text and the renderer speaks it. A post may instead
choose a voice toolkit the workspace has connected in Composio: Fish Audio
(``fish_audio``) or ElevenLabs (``elevenlabs``). Its render then speaks the same
script through that toolkit (:func:`speak`):

* one call per script line, with the toolkit's ``tts`` action as the media
  capability registry offers it (connected, allowlisted, in the cached schemas,
  not denied: ``modules/socials/capabilities.py``), made through
  ``ComposioToolExecutor.execute``, which checks the Wave 0 deny list again;
* each line's audio is copied into our storage the moment it returns
  (``social-media/{workspace}/{post}/voice-<line>.<ext>``, D9): a toolkit's
  file link expires, and Automatos holds no provider key (D15);
* the render bundle's lines then name those files instead of their text
  (``core.media_render_bundle.with_voice_files``), and media-render fetches them
  from our storage like any media. The same script, the same line ids and start
  times: the scene renders with no other change, and media-render fits each
  line into its script window (services/media-render/media_render/fit.py);
* money (D13): Fish Audio bills credit, so its balance is read before the first
  line and after the last (the ``balance`` action) and the difference is booked
  on the media lane against the post. One credit window per workspace and
  toolkit runs at a time across every worker process (a Postgres advisory lock,
  ``toolkit.credit_window``), so two renders' readings never overlap and neither
  books the other's spend. ElevenLabs bills the customer's own plan and has no balance
  action on the allowlist: its lines are booked as units (characters) at $0.

The voice picker reads :func:`voice_sources` (Kokoro always; each voice toolkit
the workspace can speak with; each allowlisted one it has not connected, to
connect) and :func:`list_voices` (a toolkit's voices, through its ``voices``
action). No recordings, no Automatos-held voice keys (D11): the Composio
connection is the key.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from config import config
from core.composio.tool_executor import ComposioToolExecutor
from core.llm.usage_context import LANE_MEDIA, usage_scope
from core.llm.usage_tracker import UsageTracker
from modules.socials import service
from modules.socials.capabilities import BALANCE, TTS, VOICES, MediaCapabilities, OfferedAction
from modules.socials.media_store import MediaNameError, MediaStore, media_key
from modules.socials.recipes.files import FileOutputError, fetch, returned_file
from modules.socials.recipes.toolkit import (
    ToolkitSchemaChanged,
    balance_of,
    call,
    credit_window,
    error_of,
    params_for,
)

logger = logging.getLogger(__name__)

EXECUTION_PREFIX = "social_post:"
KOKORO_LABEL = "Kokoro (built in)"
AVAILABLE, CONNECT, UNAVAILABLE = "available", "connect", "unavailable"
# The keys a voice listing's items carry, in the order they are trusted.
VOICE_ID_KEYS = ("voice_id", "_id", "id", "model_id")
VOICE_NAME_KEYS = ("name", "title")
VOICE_DESCRIPTION_CHARS = 200
# Audio by its first bytes → (extension, content type). ADTS is read before MPEG
# audio, which shares its sync bits.
AUDIO_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
}


class VoiceError(Exception):
    """A voice toolkit could not speak the script, or its lines could not be kept."""


class VoiceUnavailable(service.InvalidPost):
    """The post's voice is not one this workspace can speak with now (422)."""


class VoiceToolError(VoiceError):
    """The toolkit answered with an error, or with no audio (502 when listing voices)."""


def _utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


@dataclass(frozen=True)
class VoiceRecipe:
    """How one voice toolkit speaks a line and lists its voices (D12: one small recipe per toolkit)."""

    toolkit: str
    label: str
    # The speech action's parameters: the line's text, and the voice (a list for Fish Audio).
    text_param: str
    voice_param: str
    voice_as_list: bool
    speak_extra: Mapping[str, Any] = field(default_factory=dict)
    # The voices action's parameters: a title filter and a page size, where the toolkit takes them.
    list_query_param: Optional[str] = None
    list_limit_param: Optional[str] = None
    list_extra: Mapping[str, Any] = field(default_factory=dict)
    # What the toolkit bills a line by, booked as units on the media lane.
    units: Callable[[str], int] = len
    # A credit-billed toolkit books its balance difference (D13): the keys its
    # balance action carries the money under, in the order they are trusted.
    credit_billed: bool = False
    balance_keys: Tuple[str, ...] = ()


# Parameter names as docs.composio.dev lists them (FISH_AUDIO_SYNTHESIZE_SPEECH,
# FISH_AUDIO_LIST_VOICE_MODELS checked 2026-09-26; ELEVENLABS_TEXT_TO_SPEECH 2026-09-26).
# The action slugs are the registry's (the allowlist is data); a parameter the
# cached schema does not list is left out.
RECIPES: Mapping[str, VoiceRecipe] = {
    "elevenlabs": VoiceRecipe(
        toolkit="elevenlabs",
        label="ElevenLabs",
        text_param="text",
        voice_param="voice_id",
        voice_as_list=False,
        units=len,
    ),
    "fish_audio": VoiceRecipe(
        toolkit="fish_audio",
        label="Fish Audio",
        text_param="text",
        voice_param="voice_model_ids",
        voice_as_list=True,
        speak_extra={"format": "mp3"},
        list_query_param="title",
        list_limit_param="page_size",
        list_extra={"sort_by": "score"},
        units=_utf8_bytes,
        credit_billed=True,
        # Fish Audio's API credit (dollars) is "credit"; its package allowance
        # ("balance", "total") is free quota, not money, and is never booked.
        balance_keys=("credit", "api_credit"),
    ),
}


@dataclass(frozen=True)
class VoicePlan:
    """What a render needs to speak its script through a voice toolkit, resolved as it starts."""

    toolkit: str
    label: str
    voice_id: str
    speak_action: OfferedAction
    balance_action: Optional[OfferedAction] = None


@dataclass(frozen=True)
class SpokenLine:
    """One line's audio, copied into our storage."""

    line_id: str
    key: str
    extension: str
    content_type: str
    bytes: int
    sha256: str


# ── what a workspace can speak with ─────────────────────────────────────────
def _why_unavailable(recipe: VoiceRecipe, caps: MediaCapabilities) -> Optional[str]:
    """Why the workspace cannot speak with ``recipe``'s toolkit now; ``None`` when it can."""
    if caps.problem:
        return caps.problem
    if recipe.toolkit not in caps.connected:
        return f"{recipe.label} is not connected in this workspace: connect it in Composio, or choose Kokoro."
    if not caps.actions(recipe.toolkit, TTS):
        return (
            f"{recipe.label}'s speech action is not available here (not on the Socials media allowlist, "
            "not in its synced actions, or blocked): choose Kokoro."
        )
    if recipe.credit_billed and not caps.actions(recipe.toolkit, BALANCE):
        return (
            f"{recipe.label} bills credit, and its balance action is not available here, so what it spends "
            "could not be booked: sync its actions in Composio, or choose Kokoro."
        )
    return None


def plan_for(voice: Optional[Mapping[str, Any]], caps: MediaCapabilities) -> Optional[VoicePlan]:
    """The plan for the post's ``voice`` (``service.validate_voice``'s shape): ``None`` for
    Kokoro; :class:`VoiceUnavailable` when the workspace cannot speak with it now."""
    if not voice or voice.get("toolkit") in (None, service.KOKORO):
        return None
    toolkit = str(voice["toolkit"])
    recipe = RECIPES.get(toolkit)
    if recipe is None:
        raise VoiceUnavailable(f"{toolkit} is not a voice Socials can speak with: choose Kokoro or a voice toolkit.")
    why = _why_unavailable(recipe, caps)
    if why:
        raise VoiceUnavailable(why)
    balances = caps.actions(toolkit, BALANCE) if recipe.credit_billed else ()
    return VoicePlan(
        toolkit=toolkit,
        label=recipe.label,
        voice_id=str(voice["voice_id"]),
        speak_action=caps.actions(toolkit, TTS)[0],
        balance_action=balances[0] if balances else None,
    )


def voice_sources(caps: MediaCapabilities) -> Dict[str, Any]:
    """The voice picker's choices (D11, D15): Kokoro, always; each voice toolkit
    the workspace can speak with; each allowlisted one it has not connected,
    with a link to the Composio connect flow; a connected one it cannot use, and why."""
    sources: List[Dict[str, Any]] = [
        {"toolkit": service.KOKORO, "label": KOKORO_LABEL, "status": AVAILABLE, "builtin": True, "lists_voices": False}
    ]
    connectable = set(caps.connectable(TTS))
    for toolkit, recipe in sorted(RECIPES.items()):
        entry = {"toolkit": toolkit, "label": recipe.label, "builtin": False, "lists_voices": bool(caps.actions(toolkit, VOICES))}
        why = _why_unavailable(recipe, caps)
        if why is None:
            sources.append({**entry, "status": AVAILABLE})
        elif toolkit in connectable and not caps.problem:
            sources.append({**entry, "status": CONNECT})
        elif toolkit in caps.connected:
            sources.append({**entry, "status": UNAVAILABLE, "reason": why})
    return {"sources": sources, "problem": caps.problem}


# ── calling the toolkit ─────────────────────────────────────────────────────
def _params(action: OfferedAction, wanted: Mapping[str, Any], required: Sequence[str]) -> Dict[str, Any]:
    """``wanted`` as the action's cached schema takes it (``toolkit.params_for``);
    a required parameter the schema no longer lists is a :class:`VoiceToolError`."""
    try:
        return params_for(action, wanted, required)
    except ToolkitSchemaChanged as exc:
        raise VoiceToolError(str(exc)) from None


def audio_extension(data: bytes) -> Optional[str]:
    """The audio format of ``data`` by its first bytes; ``None`` when it is not audio."""
    head = data[:12]
    if head.startswith(b"ID3"):
        return "mp3"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"fLaC":
        return "flac"
    if head[4:8] == b"ftyp":
        return "m4a"
    if len(head) >= 2 and head[0] == 0xFF and head[1] & 0xF6 == 0xF0:
        return "aac"  # ADTS: sync, then layer 00
    if len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0:
        return "mp3"  # MPEG audio frame sync
    return None


async def _read_balance(executor: Any, workspace_id: UUID, plan: VoicePlan) -> Optional[Decimal]:
    if plan.balance_action is None:
        return None
    try:
        result = await call(executor, workspace_id, plan.balance_action, {})
    except Exception:  # noqa: BLE001 — a balance that cannot be read is said so by the caller
        logger.exception("[SocialsVoice] %s's balance could not be read", plan.label)
        return None
    if not result.get("success"):
        logger.warning("[SocialsVoice] %s's balance could not be read: %s", plan.label, error_of(result))
        return None
    return balance_of(result.get("data"), RECIPES[plan.toolkit].balance_keys)


# ── the voices a toolkit lists ──────────────────────────────────────────────
def _first_text(item: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
            return str(value).strip()
    return None


def parse_voices(response: Any, *, limit: int, query: Optional[str] = None) -> List[Dict[str, str]]:
    """The first list of voices in a listing (items with an id and a name), as
    ``[{id, name, description?}]``; filtered by ``query`` in the name when the
    toolkit could not filter, at most ``limit``."""
    frontier = [response]
    for _ in range(6):
        for item in frontier:
            if not isinstance(item, list):
                continue
            voices = []
            for entry in item:
                if not isinstance(entry, Mapping):
                    continue
                voice_id, name = _first_text(entry, VOICE_ID_KEYS), _first_text(entry, VOICE_NAME_KEYS)
                if voice_id and name:
                    voice = {"id": voice_id, "name": name}
                    description = entry.get("description")
                    if isinstance(description, str) and description.strip():
                        voice["description"] = description.strip()[:VOICE_DESCRIPTION_CHARS]
                    voices.append(voice)
            if voices:
                wanted = (query or "").strip().lower()
                return [v for v in voices if not wanted or wanted in v["name"].lower()][:limit]
        frontier = [
            v
            for item in frontier
            for v in (item.values() if isinstance(item, dict) else item if isinstance(item, list) else [])
        ]
        if not frontier:
            break
    return []


async def list_voices(
    db: Any,
    workspace_id: UUID,
    toolkit: str,
    *,
    caps: MediaCapabilities,
    limit: int,
    query: Optional[str] = None,
    executor: Any = None,
) -> List[Dict[str, str]]:
    """The voices ``toolkit`` offers the workspace, through its ``voices`` action."""
    recipe = RECIPES.get(toolkit)
    if recipe is None:
        raise VoiceUnavailable(f"{toolkit} is not a voice Socials can speak with.")
    why = _why_unavailable(recipe, caps)
    if why:
        raise VoiceUnavailable(why)
    listers = caps.actions(toolkit, VOICES)
    if not listers:
        raise VoiceUnavailable(f"{recipe.label} does not list its voices here: enter a voice id instead.")
    action = listers[0]
    wanted: Dict[str, Any] = dict(recipe.list_extra)
    if recipe.list_limit_param:
        wanted[recipe.list_limit_param] = limit
    if recipe.list_query_param and query and query.strip():
        wanted[recipe.list_query_param] = query.strip()
    params = _params(action, wanted, ())
    # A toolkit that does not take the filter lists everything: filter here instead.
    filtered_by_toolkit = bool(recipe.list_query_param) and recipe.list_query_param in params
    result = await call(executor or ComposioToolExecutor(db), workspace_id, action, params)
    if not result.get("success"):
        raise VoiceToolError(f"{recipe.label} did not list its voices: {error_of(result)}")
    return parse_voices(result.get("data"), limit=limit, query=None if filtered_by_toolkit else query)


# ── speaking the script ─────────────────────────────────────────────────────
def voice_file_name(line_id: str, extension: str) -> str:
    return f"voice-{line_id.lower()}.{extension}"


async def _audio_of(result: Mapping[str, Any], plan: VoicePlan, line_id: str) -> Tuple[bytes, str]:
    """The line's audio from the speech action's answer, fetched now: provider links expire."""
    returned = returned_file(result.get("data"))
    if returned is None:
        raise VoiceToolError(f"{plan.label} returned no audio for line {line_id}.")
    try:
        data = returned.data or await fetch(
            returned.url,
            max_bytes=config.SOCIALS_VOICE_LINE_MAX_BYTES,
            timeout_seconds=config.SOCIALS_MEDIA_FETCH_TIMEOUT_SECONDS,
        )
    except FileOutputError as exc:
        raise VoiceToolError(f"{plan.label}'s audio for line {line_id} could not be fetched: {exc}.") from exc
    if len(data) > config.SOCIALS_VOICE_LINE_MAX_BYTES:
        raise VoiceToolError(f"{plan.label}'s audio for line {line_id} is larger than the {config.SOCIALS_VOICE_LINE_MAX_BYTES}-byte limit.")
    extension = audio_extension(data)
    if extension is None:
        raise VoiceToolError(f"{plan.label} returned something that is not audio for line {line_id}.")
    return data, extension


def _store(store: MediaStore, key: str, data: bytes, content_type: str) -> None:
    with tempfile.TemporaryDirectory(prefix="socials-voice-") as scratch:
        path = Path(scratch) / "line"
        path.write_bytes(data)
        store.put_file(key, path, content_type)


async def _speak_line(
    executor: Any, store: MediaStore, plan: VoicePlan, recipe: VoiceRecipe,
    workspace_id: UUID, post_id: UUID, line_id: str, text: str,
) -> SpokenLine:
    voice = [plan.voice_id] if recipe.voice_as_list else plan.voice_id
    wanted = {recipe.text_param: text, recipe.voice_param: voice, **recipe.speak_extra}
    params = _params(plan.speak_action, wanted, (recipe.text_param, recipe.voice_param))
    result = await call(executor, workspace_id, plan.speak_action, params)
    if not result.get("success"):
        raise VoiceToolError(f"{plan.label} could not speak line {line_id}: {error_of(result)}")
    data, extension = await _audio_of(result, plan, line_id)
    try:
        key = media_key(workspace_id, post_id, voice_file_name(line_id, extension))
    except MediaNameError as exc:
        raise VoiceError(f"line {line_id} cannot be stored: {exc}") from exc
    content_type = AUDIO_TYPES[extension]
    try:
        await asyncio.to_thread(_store, store, key, data, content_type)
    except Exception as exc:  # noqa: BLE001 — any storage error fails the render, loudly
        logger.exception("[SocialsVoice] storing %s failed", key)
        raise VoiceError(f"{plan.label}'s audio for line {line_id} could not be stored.") from exc
    return SpokenLine(
        line_id=line_id, key=key, extension=extension, content_type=content_type,
        bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )


def _book(plan: VoicePlan, recipe: VoiceRecipe, *, workspace_id: UUID, post_id: UUID, texts: Sequence[str],
          before: Optional[Decimal], after: Optional[Decimal], latency_ms: int) -> None:
    """Book what speaking cost on the media lane (D13), against the post."""
    units = sum(recipe.units(text) for text in texts)
    usd, problem = Decimal(0), None
    if recipe.credit_billed:
        if before is None or after is None:
            problem = f"{plan.label}'s balance could not be read after speaking: its spend is not known"
            logger.error("[SocialsVoice] %s (post %s)", problem, post_id)
        else:
            usd = max(before - after, Decimal(0))
    if not units and not usd:
        return
    with usage_scope(request_type=LANE_MEDIA, execution_id=f"{EXECUTION_PREFIX}{post_id}", workspace_id=workspace_id):
        UsageTracker.track_media(
            provider=plan.toolkit,
            model_id=plan.speak_action.slug.lower(),
            units=units,
            usd=float(usd),
            latency_ms=latency_ms,
            error_message=problem,
        )


async def speak(
    plan: VoicePlan,
    *,
    workspace_id: UUID,
    post_id: UUID,
    lines: Sequence[Tuple[str, str]],
    session_factory: Callable[[], Any],
    store: MediaStore,
) -> Dict[str, SpokenLine]:
    """Speak each ``(line id, text)`` through the plan's toolkit, one call per
    line, and copy each line's audio into our storage as it returns. What it
    spent is booked on the media lane, whatever happened. :class:`VoiceError`
    when a line cannot be spoken or kept."""
    recipe = RECIPES[plan.toolkit]
    spoken: Dict[str, SpokenLine] = {}
    attempted: List[str] = []
    db = session_factory()
    try:
        executor = ComposioToolExecutor(db)
        # Only a credit-billed toolkit reads a balance difference: only it needs the window.
        window = credit_window(session_factory, workspace_id, plan.toolkit) if recipe.credit_billed else nullcontext()
        async with window:
            started = time.monotonic()
            before = await _read_balance(executor, workspace_id, plan)
            if recipe.credit_billed and before is None:
                raise VoiceToolError(
                    f"{plan.label}'s balance could not be read, so what it would spend could not be booked: "
                    "nothing was spoken."
                )
            try:
                for line_id, text in lines:
                    attempted.append(text)
                    spoken[line_id] = await _speak_line(executor, store, plan, recipe, workspace_id, post_id, line_id, text)
            finally:
                after = await _read_balance(executor, workspace_id, plan) if recipe.credit_billed else None
                _book(
                    plan, recipe, workspace_id=workspace_id, post_id=post_id, texts=attempted,
                    before=before, after=after, latency_ms=int((time.monotonic() - started) * 1000),
                )
    finally:
        db.close()
    return spoken
