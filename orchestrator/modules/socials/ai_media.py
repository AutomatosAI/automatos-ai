"""PRD-264 EUAI-05 (EU AI Act Art. 50): whether a post carries AI media, and the AI
label each network gets at publish.

**AI media** is what an AI made for the post:

* **AI footage and AI stills**: a slot a render filled from the workspace's Composio
  generation toolkit (D12): a ``footage`` record marked done whose ``toolkit`` is not the
  person's own file (``service.OWN_FILE_TOOLKITS``: an upload or a Library file). A still
  is one whose file is an image.
* **A synthetic voice**: a render that spoke its script. Kokoro (the template's own
  voice, the post's ``voice`` None) and every voice toolkit (fish_audio, elevenlabs, ...)
  are synthetic.

A post whose media are all the person's own uploads and Library files, the template's
own visuals and music, carries none.

**The render record.** A finished render records what it was made from in its
``render_done`` entry of ``review_log``, under ``report.ai_media`` (:func:`with_record`):
the slots above, the voice that spoke, and the sha256 of each file it made. At publish,
:func:`for_post` reads the record whose files the post still carries, so the label can't
drift from what was rendered. Like the voice, the record is outside the content hash: the
rendered files' digests carry what it changed. A post with no such record (never
rendered, or rendered before the record existed) is judged from its own fields by the
same rule: its footage records, and the voice a rendered video post speaks with.

**Each network's label** (Composio's action schemas, docs.composio.dev/toolkits, read
19 Oct 2026): ``TIKTOK_UPLOAD_VIDEO`` takes ``is_aigc`` ("Set true to label the video as
AI-generated content"), set through ``$generated`` in ``channel_adapters.py``.
``YOUTUBE_UPLOAD_VIDEO`` and ``YOUTUBE_UPDATE_VIDEO`` take no ``status`` and no
``containsSyntheticMedia``, so the receipt asks for YouTube Studio's setting. LinkedIn's
``LINKEDIN_CREATE_LINKED_IN_POST`` / ``LINKEDIN_CREATE_VIDEO_POST``, X's
``TWITTER_CREATION_OF_A_POST`` / ``TWITTER_UPLOAD_MEDIA`` / ``TWITTER_UPLOAD_LARGE_MEDIA``
and Instagram's ``INSTAGRAM_POST_IG_USER_MEDIA`` /
``INSTAGRAM_CREATE_CAROUSEL_CONTAINER`` / ``INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH`` take
no AI flag, so the caption carries the disclosure.

**The render's AI inputs** (PRD-264 EUAI-06a): every render bundle carries ``ai_inputs``
(:func:`with_inputs`), the same rule applied to what the render is about to show: a slot it
fills is AI footage or an AI still unless it holds the person's own file, a render that
speaks its script speaks with a synthetic voice, and the template's own visuals are always
there. media-render marks each file it makes from it (IPTC DigitalSourceType in XMP,
``services/media-render/media_render/provenance.py``), so a post and its files agree.

**The AI-generated label** (PRD-264 EUAI-08a, ``deepfake_label.py``): on by default while
the post carries AI media. The render record keeps whether the render burned it into the
video (``labelled``); a caption channel's receipt says the caption label was added while
it is on, and asks the person to add it while it is off; the AI-labels answer carries the
setting (``deepfake_label.summary``).

Pure: no database, no Composio.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from core.media_render_bundle import AI_INPUTS, ai_inputs, voice_script
from core.social_templates import IMAGE_SLOT
from modules.socials import deepfake_label
from modules.socials.channel_adapters import CHANNEL_ADAPTERS
from modules.socials.deepfake_label import Caption
from modules.socials.kokoro_voices import KOKORO

RECORD = "ai_media"  # a render_done entry's report key: the render record
LABEL = "ai_label"  # a target's action_plan key: the AI label line its last publish attempt used
RENDER_DONE = "render_done"  # service.ACTION_RENDER_DONE (the service imports this module)
VIDEO_FORMAT = "video"  # a post's format whose render speaks its script
IMAGE_TYPE = "image/"
FOOTAGE, STILLS, OWN, VOICE, FILES = "footage", "stills", "own", "voice", "files"
LABELLED = "labelled"  # the render record's: the render burned the AI-generated label in (EUAI-08a)
INPUTS = AI_INPUTS  # a render bundle's key: which inputs are AI-made (media-render's bundle.py)

# How each channel shows a post's AI label through Composio (the schemas above).
NETWORK_FLAG, STUDIO_ONLY, CAPTION_ONLY = "network_flag", "studio_only", "caption_only"
CHANNEL_AI_LABEL: Mapping[str, str] = MappingProxyType({
    "tiktok": NETWORK_FLAG,
    "youtube": STUDIO_ONLY,
    "linkedin": CAPTION_ONLY,
    "twitter": CAPTION_ONLY,
    "instagram": CAPTION_ONLY,
})
FLAG_OPTION = "is_aigc"  # the target option a person may set the network flag with

NO_AI_MEDIA = "No AI media in this post: no AI label is needed."
FLAG_ON = "{channel}'s AI-generated label is on: the post carries {what}."
FLAG_CHOSEN = "{channel}'s AI-generated label is on: you chose it. The post carries no AI media."
FLAG_OFF = "No AI media in this post: {channel}'s AI-generated label stays off."
STUDIO_LINE = (
    "This video carries {what}, and {channel}'s upload takes no AI label. "
    "Set 'Altered or synthetic content' in YouTube Studio for this video."
)
CAPTION_LINE = (
    "This post carries {what}, and {channel} takes no AI label through its API, so the caption "
    "carries the disclosure: start it with \"AI-generated.\""
)
# PRD-264 EUAI-08a: while the post is labelled (deepfake_label.py), the caption carries it already.
CAPTION_ADDED = (
    "This post carries {what}, and {channel} takes no AI label through its API, so the caption "
    "carries the disclosure: \"{label}\" was added at its start."
)
CAPTION_HAS = (
    "This post carries {what}, and {channel} takes no AI label through its API, so the caption "
    "carries the disclosure: it starts with \"{label}\"."
)
CAPTION_CHOSEN = "The post carries no AI media, and you chose to label it: \"{label}\" starts its caption."
NO_CAPTION = Caption("")  # a caption with no label: the person adds the disclosure


@dataclass(frozen=True)
class AiMedia:
    """What AI made for a post: the slots holding AI footage or AI stills, the synthetic
    voice its render spoke with (``None``: nothing was spoken), and the slots holding the
    person's own files. ``recorded`` says the decision came from the render record."""

    footage: Tuple[str, ...] = ()
    stills: Tuple[str, ...] = ()
    voice: Optional[str] = None
    own: Tuple[str, ...] = ()
    recorded: bool = False

    @property
    def carries(self) -> bool:
        """Whether the post carries any AI media: what every channel's AI label reads."""
        return bool(self.footage or self.stills or self.voice)

    def to_dict(self) -> Dict[str, Any]:
        """As the AI-labels route answers it."""
        return {
            "carries": self.carries, FOOTAGE: list(self.footage), STILLS: list(self.stills), VOICE: self.voice,
            OWN: list(self.own), "recorded": self.recorded,
        }


def _service() -> Any:
    from modules.socials import service  # the service imports this module

    return service


def _done_slots(footage: Any) -> Dict[str, Tuple[str, ...]]:
    """The slots a render filled, by who made them: AI footage, AI stills, the person's own."""
    svc = _service()
    found: Dict[str, List[str]] = {FOOTAGE: [], STILLS: [], OWN: []}
    records = footage.items() if isinstance(footage, Mapping) else ()
    for slot, record in records:
        if not isinstance(record, Mapping) or record.get("status") != svc.FOOTAGE_DONE:
            continue
        if record.get("toolkit") in svc.OWN_FILE_TOOLKITS:
            found[OWN].append(str(slot))
        elif str(record.get("content_type") or "").startswith(IMAGE_TYPE):
            found[STILLS].append(str(slot))
        else:
            found[FOOTAGE].append(str(slot))
    return {kind: tuple(sorted(slots)) for kind, slots in found.items()}


def footage_generated(footage: Any) -> bool:
    """Whether a render recorded footage or a still an AI toolkit made for one of the
    post's slots (D12). The person's own picture in a photo slot (an upload or a Library
    file) is not AI-made. The content hash reads it (``service._content_of``); the voice
    stays outside the hash."""
    slots = _done_slots(footage)
    return bool(slots[FOOTAGE] or slots[STILLS])


def voice_name(voice: Any) -> str:
    """The synthetic voice a render speaks with: the post's voice toolkit, else Kokoro."""
    toolkit = voice.get("toolkit") if isinstance(voice, Mapping) else None
    return toolkit if isinstance(toolkit, str) and toolkit else KOKORO


def _digests(media: Any) -> Set[str]:
    """The sha256 of each rendered file record in ``media`` (``{aspect: [records]}``)."""
    groups = media.values() if isinstance(media, Mapping) else ()
    return {
        str(record["sha256"])
        for records in groups if isinstance(records, (list, tuple))
        for record in records if isinstance(record, Mapping) and record.get("sha256")
    }


def render_record(post: Any, bundle: Any, media: Any) -> Dict[str, Any]:
    """What a finished render was made from: the post's filled slots (a rendering post is
    never edited, so they are the render's), the voice that spoke the bundle's script, and
    the sha256 of each file in ``media``, the render's output, and whether it burned the
    AI-generated label in (PRD-264 EUAI-08a: the bundle's ``ai_label``)."""
    slots = _done_slots(getattr(post, "footage", None))
    spoken = bool(voice_script(bundle)) if isinstance(bundle, Mapping) else False
    return {
        FOOTAGE: list(slots[FOOTAGE]), STILLS: list(slots[STILLS]), OWN: list(slots[OWN]),
        VOICE: voice_name(getattr(post, "voice", None)) if spoken else None,
        FILES: sorted(_digests(media)),
        LABELLED: isinstance(bundle, Mapping) and bool(bundle.get(deepfake_label.FIELD)),
    }


def _ai_still(slot: str, done: Mapping[str, Tuple[str, ...]], specs: Mapping[str, Any]) -> bool:
    """A filled slot's file is a still by its record; one still to be made, by its template slot."""
    if slot in done[STILLS] or slot in done[FOOTAGE]:
        return slot in done[STILLS]
    spec = specs.get(slot)
    return isinstance(spec, Mapping) and spec.get("kind") == IMAGE_SLOT


def render_inputs(post: Any, shown: Sequence[str], slots: Any, *, speaks: bool) -> Dict[str, bool]:
    """Which inputs of a render are AI-made, as media-render's ``ai_inputs`` takes them: each
    slot in ``shown`` (the slots it fills) is the person's own file when its record says so
    (an upload or a Library file), else AI footage or an AI still (``slots``, the template's,
    tell a still to be made); a render that ``speaks`` its script speaks with a synthetic
    voice; the template's own visuals are always in it."""
    done = _done_slots(getattr(post, "footage", None))
    specs = slots if isinstance(slots, Mapping) else {}
    made = [slot for slot in shown if slot not in done[OWN]]
    stills = [slot for slot in made if _ai_still(slot, done, specs)]
    return ai_inputs(
        footage=len(stills) < len(made), stills=bool(stills), voice=speaks,
        uploads=any(slot in done[OWN] for slot in shown),
    )


def with_inputs(bundle: Mapping[str, Any], post: Any, shown: Sequence[str], slots: Any) -> Dict[str, Any]:
    """A new copy of the render ``bundle`` carrying its ``ai_inputs`` (:func:`render_inputs`)."""
    return {**dict(bundle), INPUTS: render_inputs(post, shown, slots, speaks=bool(voice_script(bundle)))}


def own_file_inputs() -> Dict[str, bool]:
    """``ai_inputs`` of a render showing only the person's own file (a crop of their still)."""
    return ai_inputs(uploads=True)


def with_record(report: Optional[Mapping[str, Any]], post: Any, bundle: Any, media: Any) -> Dict[str, Any]:
    """A new copy of a finished render's ``report`` carrying its render record."""
    return {**dict(report or {}), RECORD: render_record(post, bundle, media)}


def _record_of(entry: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(entry, Mapping) or entry.get("action") != RENDER_DONE:
        return None
    report = entry.get("report")
    record = report.get(RECORD) if isinstance(report, Mapping) else None
    return record if isinstance(record, Mapping) else None


def _record_for(log: Any, carried: Set[str]) -> Optional[Mapping[str, Any]]:
    """The newest render record whose files the post still carries."""
    entries = log if isinstance(log, list) else []
    for entry in reversed(entries):
        record = _record_of(entry)
        if record is not None and carried & {str(sha) for sha in record.get(FILES) or ()}:
            return record
    return None


def _names(value: Any) -> Tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, (list, tuple)) else ()


def _from_record(record: Mapping[str, Any]) -> AiMedia:
    voice = record.get(VOICE)
    return AiMedia(
        footage=_names(record.get(FOOTAGE)), stills=_names(record.get(STILLS)), own=_names(record.get(OWN)),
        voice=str(voice) if isinstance(voice, str) and voice else None, recorded=True,
    )


def _from_post(post: Any, *, rendered: bool) -> AiMedia:
    """The same rule over the post's own fields: a rendered video post spoke its script."""
    slots = _done_slots(getattr(post, "footage", None))
    spoke = rendered and getattr(post, "format", None) == VIDEO_FORMAT
    return AiMedia(
        footage=slots[FOOTAGE], stills=slots[STILLS], own=slots[OWN],
        voice=voice_name(getattr(post, "voice", None)) if spoke else None,
    )


def for_post(post: Any) -> AiMedia:
    """The AI media of the post as it is now: from the render record of the files it
    carries, else from its own fields."""
    record = _render_record_of(post)
    return _from_record(record) if record is not None else _from_post(post, rendered=bool(_digests(getattr(post, "media", None))))


def describe(ai: AiMedia) -> str:
    """The AI media in words, for a receipt: "AI footage, an AI still and a synthetic voice (Kokoro)"."""
    parts = []
    if ai.footage:
        parts.append("AI footage")
    if ai.stills:
        parts.append("an AI still" if len(ai.stills) == 1 else "AI stills")
    if ai.voice:
        parts.append(f"a synthetic voice ({ai.voice.replace('_', ' ').title()})")
    if not parts:
        return "no AI media"
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def channel_label(toolkit: str) -> str:
    """The channel's name as the Socials tab shows it (its adapter's label), else its toolkit's."""
    adapter = CHANNEL_ADAPTERS.get(toolkit)
    return str(adapter["label"]) if adapter else toolkit.replace("_", " ").title()


def _caption_line(channel: str, ai: AiMedia, caption: Caption) -> str:
    """A caption channel's line: the label its caption starts with (PRD-264 EUAI-08a, while
    the post is labelled), else the person is asked to add it."""
    what = describe(ai)
    if not caption.label:
        return CAPTION_LINE.format(channel=channel, what=what) if ai.carries else NO_AI_MEDIA
    if not ai.carries:
        return CAPTION_CHOSEN.format(label=caption.label)
    line = CAPTION_ADDED if caption.added else CAPTION_HAS
    return line.format(channel=channel, what=what, label=caption.label)


def label_line(toolkit: str, ai: AiMedia, options: Mapping[str, Any], caption: Caption = NO_CAPTION) -> str:
    """The channel's AI label decision, as its receipt states it. A channel outside the
    table (a generic one) takes no AI flag: the caption carries the disclosure, and
    ``caption`` (``deepfake_label.caption_for``) says whether it starts with the label."""
    how = CHANNEL_AI_LABEL.get(toolkit, CAPTION_ONLY)
    channel = channel_label(toolkit)
    if how == NETWORK_FLAG:
        if ai.carries:
            return FLAG_ON.format(channel=channel, what=describe(ai))
        return FLAG_CHOSEN.format(channel=channel) if options.get(FLAG_OPTION) is True else FLAG_OFF.format(channel=channel)
    if how == CAPTION_ONLY:
        return _caption_line(channel, ai, caption)
    return STUDIO_LINE.format(channel=channel, what=describe(ai)) if ai.carries else NO_AI_MEDIA


def _target_label(target: Any, ai: AiMedia, post: Any, labelled: bool) -> Dict[str, Any]:
    plan = target.action_plan if isinstance(target.action_plan, Mapping) else {}
    stored = plan.get(LABEL)
    at_publish = isinstance(stored, str) and bool(stored)
    caption = deepfake_label.caption_for(getattr(post, "copy", None), target.toolkit, labelled)
    return {
        "id": str(target.id), "toolkit": target.toolkit, "post_kind": target.post_kind,
        "ai_label": stored if at_publish else label_line(target.toolkit, ai, target.options, caption),
        "at_publish": at_publish,
    }


def _render_record_of(post: Any) -> Optional[Mapping[str, Any]]:
    carried = _digests(getattr(post, "media", None))
    return _record_for(getattr(post, "review_log", None), carried) if carried else None


def labels_for(post: Any) -> Dict[str, Any]:
    """The post's AI media, its AI-generated label setting (PRD-264 EUAI-08a,
    ``deepfake_label.summary``) and each target's AI label: the line its publish used, for
    a target a publish attempt ran, else the line its publish will use."""
    ai = for_post(post)
    record = _render_record_of(post)
    labelled = record.get(LABELLED) if record is not None and isinstance(record.get(LABELLED), bool) else None
    setting = deepfake_label.summary(post, ai.carries, labelled)
    targets = sorted(getattr(post, "targets", None) or (), key=lambda t: (t.toolkit, t.post_kind))
    return {
        "ai_media": ai.to_dict(), "deepfake_label": setting,
        "targets": [_target_label(target, ai, post, setting["on"]) for target in targets],
    }
