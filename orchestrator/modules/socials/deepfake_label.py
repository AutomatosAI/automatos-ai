"""PRD-264 EUAI-08a (EU AI Act Art. 50(4) ¶1): Socials labels realistic AI media as
AI-generated, on by default.

**The setting.** "Realistic AI people, places, products or voice: label as AI-generated"
(:data:`SETTING_NAME`) is on by default whenever the post carries AI media (AI footage, AI
stills or a synthetic voice: ``ai_media.py``, EUAI-05). A person may switch it off for one
post, and the composer shows why that is theirs to decide: :data:`OFF_REASON`. The choice is
an ``ai_label`` entry in the post's ``review_log`` (who, when, ``on``), so the post's history
shows who switched the label off and when; the newest entry is the setting
(:func:`setting_of`), and a post with none follows its media. Only a person sets it,
through ``PUT /api/socials/posts/{id}/ai-label`` (:func:`set_label`): no agent tool takes it,
so Automatos's own posts, and every post an agent or a plan makes, use the default.

**Approved content.** Once set, the setting is in the content hash (:func:`hashed`, read by
``service._content_of``), so changing it voids an approval and the post goes back for one,
as any content edit does. A post nobody set hashes as before.

**When it is on:**

* a video shows a visible label (:func:`render_label`, the render bundle's ``ai_label``) for
  its first ``SOCIALS_AI_LABEL_SECONDS`` and again every
  ``SOCIALS_AI_LABEL_INTERVAL_SECONDS``: media-render burns it in
  (``services/media-render/media_render/ai_label.py``). The render record keeps whether it
  did (``ai_media.render_record``); a video whose label disagrees with the setting is neither
  approved nor published until it is rendered again (``ai_media.label_mismatch``). A still
  and a carousel carry the caption label only;
* every channel's caption starts with "AI-generated." in the post's language
  (:data:`CAPTION_LABELS`, :func:`caption_for`), fitted to the channel's limit
  (``copy_limits``) so the label is never the part cut off;
* an audio-only post opens with the spoken line :data:`SPOKEN_LINE` (:func:`with_spoken_line`).

**The post's language.** A post carries no language field, so the language is read from the
text being labelled (:func:`language_of`): the language whose common words it uses most, of
English, Portuguese, Spanish, French, German and Italian; English when none stands out.

Pure but for the config the render label reads: no database, no Composio.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Optional

from config import config
from core.media_render_bundle import AI_INPUTS
from modules.socials import copy_limits

FIELD = "ai_label"  # the content hash's key, the render bundle's key and the render record's
ACTION = "ai_label"  # the review_log action of a person setting the label for a post
ON = "on"  # the entry's flag: True on, False off
AI_FLAGS = ("footage", "stills", "voice")  # the ai_inputs flags that make a render's media AI-made
STILL = "still"  # a render bundle's key when it makes images (core/media_render_bundle.build_bundle)
VIDEO_FORMAT = "video"  # a post's format whose render is a video

SETTING_NAME = "Realistic AI people, places, products or voice: label as AI-generated"
OFF_REASON = "You are responsible for disclosing deep fakes."
SWITCHED_ON = "AI-generated label switched on."
SWITCHED_OFF = "AI-generated label switched off: you are responsible for disclosing deep fakes."
NOT_A_FLAG = "on must be true or false"

# The caption label by language (the language subtag); English is the fallback.
DEFAULT_LANGUAGE = "en"
CAPTION_LABELS: Mapping[str, str] = MappingProxyType({
    "en": "AI-generated.",
    "pt": "Gerado por IA.",
    "es": "Generado por IA.",
    "fr": "Généré par IA.",
    "de": "KI-generiert.",
    "it": "Generato dall'IA.",
})
# Common words of each language, none shared with another's list: a text is in the
# language it uses most of them, at least MIN_LANGUAGE_WORDS times and more than any other.
LANGUAGE_WORDS: Mapping[str, FrozenSet[str]] = MappingProxyType({
    "en": frozenset({"the", "and", "with", "for", "you", "your", "our", "this", "is", "are", "of", "to", "from"}),
    "pt": frozenset({"não", "você", "vocês", "nosso", "nossa", "uma", "com", "são", "também", "muito", "isso", "os", "seu"}),
    "es": frozenset({"el", "los", "las", "usted", "nuestro", "nuestra", "también", "muy", "esto", "y", "hoy", "pero"}),
    "fr": frozenset({"des", "les", "une", "avec", "pour", "vous", "votre", "notre", "nous", "est", "sont", "et", "cette"}),
    "de": frozenset({"der", "die", "dem", "und", "mit", "für", "ist", "sind", "ein", "eine", "nicht", "wir", "auch"}),
    "it": frozenset({"nel", "gli", "della", "per", "sono", "nostro", "nostra", "anche", "oggi", "molto", "questo", "che", "è"}),
})
MIN_LANGUAGE_WORDS = 2
_WORD = re.compile(r"[^\W\d_]+")

# An audio-only post opens with this line, spoken by the post's own voice; its script
# moves on by SPOKEN_LEAD_SECONDS to make room (the line is six words, about 2.4 s at
# Kokoro's pace). Socials has no audio-only format yet (``SOCIAL_POST_FORMATS``): a post
# whose format is one of AUDIO_ONLY_FORMATS gets the line.
SPOKEN_LINE = "This audio was generated by AI."
SPOKEN_LINE_ID = "ai-label"
SPOKEN_LEAD_SECONDS = 2.5
AUDIO_ONLY_FORMATS: FrozenSet[str] = frozenset({"audio"})

# What the receipt says about the setting (``summary``).
ON_LINE = '{name}: on. The caption starts with "{label}"'  # the label ends in its own full stop
OFF_LINE = "{name}: off for this post. You are responsible for disclosing deep fakes."
VIDEO_LINE = ' The video shows "{text}" for its first {seconds:g} s and again every {interval:g} s.'
RENDER_AGAIN = " The video was rendered with the label {was}: render it again before it is approved or published."


def _service() -> Any:
    from modules.socials import service  # the service imports this module

    return service


# ── the setting ─────────────────────────────────────────────────────────────
def setting_of(post: Any) -> Optional[bool]:
    """The label setting a person chose for the post (the newest ``ai_label`` entry of its
    review_log), or ``None``: nobody chose, so it follows the post's media."""
    log = getattr(post, "review_log", None)
    for entry in reversed(log if isinstance(log, list) else []):
        if isinstance(entry, Mapping) and entry.get("action") == ACTION and isinstance(entry.get(ON), bool):
            return bool(entry[ON])
    return None


def is_on(setting: Optional[bool], carries: bool) -> bool:
    """Whether the post is labelled: as a person set it, else whenever it carries AI media."""
    return bool(carries) if setting is None else setting


def hashed(content: Mapping[str, Any], post: Any) -> Dict[str, Any]:
    """A new copy of the content hash's input carrying the post's label setting once a
    person set it; a post nobody set hashes as before, so no existing approval moves."""
    setting = setting_of(post)
    return dict(content) if setting is None else {**content, FIELD: setting}


def set_label(post: Any, actor: str, on: Any) -> Any:
    """``actor`` switches the post's label on or off: a new ``ai_label`` entry in its
    review_log, then the edit's own path (``service.update_post``): the setting is in the
    hash, so an approved or scheduled post goes back for approval. Setting it to what it is
    already changes nothing. A post that cannot be edited now (rendering, publishing,
    published) refuses it."""
    svc = _service()
    if not isinstance(on, bool):
        raise svc.InvalidPost(NOT_A_FLAG)
    if post.status not in svc.EDITABLE_STATUSES:
        raise svc.IllegalTransition(post.status, svc.ACTION_EDIT)
    if setting_of(post) is on:
        return post
    entry = {
        "at": datetime.now(timezone.utc).isoformat(), "by": actor, "action": ACTION,
        "comment": SWITCHED_ON if on else SWITCHED_OFF, ON: on,
    }
    post.review_log = [*(post.review_log or []), entry]
    return svc.update_post(post, actor, {})


# ── the caption ─────────────────────────────────────────────────────────────
def language_of(text: Any) -> str:
    """The language of ``text`` (a CAPTION_LABELS key): the one whose common words it uses
    most, at least MIN_LANGUAGE_WORDS times and more than any other; else English."""
    words = _WORD.findall(text.casefold()) if isinstance(text, str) else []
    counts = {language: sum(1 for word in words if word in common) for language, common in LANGUAGE_WORDS.items()}
    best = max(counts.values(), default=0)
    leaders = [language for language, count in counts.items() if count == best]
    return leaders[0] if best >= MIN_LANGUAGE_WORDS and len(leaders) == 1 else DEFAULT_LANGUAGE


def caption_label(text: Any) -> str:
    """The caption label in the language of ``text``: "AI-generated.", "Gerado por IA.", ..."""
    return CAPTION_LABELS.get(language_of(text), CAPTION_LABELS[DEFAULT_LANGUAGE])


def video_label(text: Any) -> str:
    """The label a video shows, in the language of ``text``: the caption label without its full stop."""
    return caption_label(text).rstrip(".")


def channel_copy(copy: Any, toolkit: str) -> str:
    """The channel's own copy, else the base copy."""
    copy = copy if isinstance(copy, Mapping) else {}
    channels = copy.get("channels") if isinstance(copy.get("channels"), Mapping) else {}
    return str(channels.get(toolkit) or copy.get("base") or "")


@dataclass(frozen=True)
class Caption:
    """What a channel publishes as its caption, and the label it starts with (``""``: none).
    ``added`` says the label was put there; ``False`` with a label: the text began with it."""

    text: str
    label: str = ""
    added: bool = False


def _starts_labelled(text: str) -> bool:
    start = text.lstrip().casefold()
    return any(start.startswith(label.rstrip(".").casefold()) for label in CAPTION_LABELS.values())


def caption_for(copy: Any, toolkit: str, on: bool) -> Caption:
    """The channel's caption (``channel_copy``), starting with the caption label when the
    post is labelled. Text that already starts with a label keeps it, once. Over the
    channel's limit (``copy_limits``) the text is cut at a word, never the label."""
    text = channel_copy(copy, toolkit)
    if not on:
        return Caption(text)
    label = caption_label(text)
    if _starts_labelled(text):
        return Caption(text, label=label)
    limit = (copy_limits.limits_for(toolkit) or {}).get(copy_limits.TEXT)
    body = text.strip()
    if limit is not None and len(label) + 1 + len(body) > limit:
        body = copy_limits.trim_at_word(body, limit - len(label) - 1)
    return Caption(f"{label} {body}" if body else label, label=label, added=True)


# ── the render ──────────────────────────────────────────────────────────────
def render_carries(bundle: Mapping[str, Any]) -> bool:
    """Whether a render bundle's media are AI-made, by its ``ai_inputs`` (EUAI-06a)."""
    flags = bundle.get(AI_INPUTS)
    return isinstance(flags, Mapping) and any(flags.get(name) is True for name in AI_FLAGS)


def render_label(post: Any, bundle: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The label media-render burns into a video (the bundle's ``ai_label``): its text, how
    long it shows at the start and how often it shows again. ``None`` when the post is not
    labelled, or the bundle is a still (its caption carries the label)."""
    if STILL in bundle or not is_on(setting_of(post), render_carries(bundle)):
        return None
    return {
        "text": video_label(channel_copy(getattr(post, "copy", None), "")),
        "seconds": config.SOCIALS_AI_LABEL_SECONDS,
        "interval": config.SOCIALS_AI_LABEL_INTERVAL_SECONDS,
    }


def audio_only(post: Any) -> bool:
    """Whether the post is audio and nothing else: it says it is AI in words, at the start."""
    return getattr(post, "format", None) in AUDIO_ONLY_FORMATS


def with_spoken_line(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """A new copy of ``bundle`` that opens with :data:`SPOKEN_LINE`, its script moved on by
    SPOKEN_LEAD_SECONDS to make room."""
    audio = dict(bundle.get("audio") or {})
    voice = dict(audio.get("voice") or {})
    later = [
        {**line, "at": round(float(line["at"]) + SPOKEN_LEAD_SECONDS, 3)}
        for line in voice.get("lines") or [] if isinstance(line, Mapping) and "at" in line
    ]
    opening = {"id": SPOKEN_LINE_ID, "at": 0, "text": SPOKEN_LINE}
    return {**dict(bundle), "audio": {**audio, "voice": {**voice, "lines": [opening, *later]}}}


def with_label(bundle: Mapping[str, Any], post: Any) -> Dict[str, Any]:
    """A new copy of a render ``bundle`` with the post's label: the video's
    (:func:`render_label`) and, for an audio-only post, the spoken line."""
    label = render_label(post, bundle)
    if label is None:
        return dict(bundle)
    labelled = {**dict(bundle), FIELD: label}
    return with_spoken_line(labelled) if audio_only(post) else labelled


# ── the receipt ─────────────────────────────────────────────────────────────
def _video_note(post: Any, on: bool, labelled: Optional[bool]) -> str:
    """The receipt's word on a video: that its render disagrees with the setting (``labelled``:
    whether the rendered video shows the label, ``None`` when there is none), or what it shows."""
    if labelled is not None and labelled != on:
        return RENDER_AGAIN.format(was="on" if labelled else "off")
    if getattr(post, "format", None) != VIDEO_FORMAT or not on:
        return ""
    text = video_label(channel_copy(getattr(post, "copy", None), ""))
    return VIDEO_LINE.format(
        text=text, seconds=config.SOCIALS_AI_LABEL_SECONDS, interval=config.SOCIALS_AI_LABEL_INTERVAL_SECONDS,
    )


def summary(post: Any, carries: bool, labelled: Optional[bool] = None) -> Dict[str, Any]:
    """The setting as the composer and the receipt show it. ``carries``: the post carries AI
    media (the default); ``labelled``: whether its rendered video shows the label (``None``: it
    carries no rendered video; ``ai_media.burned_label``)."""
    setting = setting_of(post)
    on = is_on(setting, carries)
    label = caption_label(channel_copy(getattr(post, "copy", None), ""))
    line = (ON_LINE if on else OFF_LINE).format(name=SETTING_NAME, label=label)
    return {
        "name": SETTING_NAME, "setting": setting, "default": bool(carries), "on": on, "label": label,
        "reason": OFF_REASON, "video_labelled": labelled, "line": line + _video_note(post, on, labelled),
    }
