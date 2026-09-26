"""The credit a rendered file's music asks for (PRD-251 S1.6).

media-render's job report names the library track a render mixed (``music``):
its title, its licence and its attribution line, and whether the licence asks
for credit (``services/media-render/media_render/music.py``). A CC BY track's
line must go wherever the file is published, so:

* a Socials post's render appends it to the post's copy as the render finishes
  (``modules/socials/service.finish_render``);
* every rendered file records its music on its Deliverable (``extra.music``,
  :func:`deliverable_credit`), and a post whose media names that Deliverable
  carries the line too (``modules/socials/credits.py``);
* ``generate_document`` with a social format records it the same way.

A report that asks for credit but carries no usable line fails the render, and
so does one that does not name the track the render's bundle asked for
(:func:`credit_for_render`): a CC BY track never reaches a post without its
credit, whatever either service's version. Pure: no database, no IO.
Shared by ``modules/socials`` and ``modules/documents``, which may not import
each other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

# A credit line is one line of text; longer is not a credit line.
CREDIT_MAX_CHARS = 300
TRACK_MAX_CHARS = 64
LABEL_MAX_CHARS = 200


class MusicCreditMissing(ValueError):
    """The render's music asks for credit, and the report gives no usable line."""


@dataclass(frozen=True)
class MusicCredit:
    """The music one rendered file carries, and the line a post that uses it must carry."""

    track: str
    title: str
    licence: str
    # None when the licence asks for no credit (CC0).
    line: Optional[str]

    def extra(self) -> Dict[str, Any]:
        """What the file's Deliverable records in ``extra.music``."""
        return {"track": self.track, "title": self.title, "licence": self.licence, "credit": self.line}


def _label(value: Any, limit: int) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def credit_line(value: Any) -> Optional[str]:
    """``value`` as one credit line (whitespace folded), or ``None`` when it cannot be one."""
    if not isinstance(value, str):
        return None
    line = " ".join(value.split())
    return line if line and len(line) <= CREDIT_MAX_CHARS else None


def credit_from_report(report: Any) -> Optional[MusicCredit]:
    """The music a finished render mixed, from its job report; ``None`` when it had none.

    :class:`MusicCreditMissing` when the licence asks for credit and the report
    carries no usable attribution line.
    """
    music = report.get("music") if isinstance(report, Mapping) else None
    if not isinstance(music, Mapping):
        return None
    track = _label(music.get("track"), TRACK_MAX_CHARS)
    if not track:
        return None
    line = credit_line(music.get("attribution"))
    required = music.get("credit_required") is True
    if required and line is None:
        raise MusicCreditMissing(f"the music {track} needs its credit line, and the renderer gave none")
    return MusicCredit(
        track=track,
        title=_label(music.get("title"), LABEL_MAX_CHARS),
        licence=_label(music.get("licence"), LABEL_MAX_CHARS),
        line=line if required else None,
    )


def bundle_track(bundle: Any) -> Optional[str]:
    """The library track a render bundle asks media-render to mix, if any."""
    audio = bundle.get("audio") if isinstance(bundle, Mapping) else None
    music = audio.get("music") if isinstance(audio, Mapping) else None
    track = music.get("track") if isinstance(music, Mapping) else None
    return track if isinstance(track, str) and track else None


def credit_for_render(bundle: Any, report: Any) -> Optional[MusicCredit]:
    """:func:`credit_from_report`, checked against what ``bundle`` asked for: a
    render whose bundle named a track must report that track, or
    :class:`MusicCreditMissing` (fail closed across the service boundary)."""
    credit = credit_from_report(report)
    asked = bundle_track(bundle)
    if asked and (credit is None or credit.track != asked):
        said = credit.track if credit is not None else "no music"
        raise MusicCreditMissing(f"the render was asked to mix {asked}, and its report names {said}")
    return credit


def deliverable_credit(extra: Any) -> Optional[str]:
    """The credit line a Deliverable's ``extra.music`` asks for, if any."""
    music = extra.get("music") if isinstance(extra, Mapping) else None
    return credit_line(music.get("credit")) if isinstance(music, Mapping) else None
