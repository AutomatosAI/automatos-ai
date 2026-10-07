"""F377 (night 11, 7 Oct): a render the voice check refuses names the field, not the line.

media-render refuses a voice line it cannot fit into its moment, even sped up
(``services/media-render/media_render/pipeline.py`` ``voice_findings``): "line l02 is still
speaking at 4.19 s, when line l05 starts", or "line l09 runs from … past the end". The owner
never sees a line id, and night 11 could not tell which field "line l05" was.

:func:`spoken_labels` maps each voice line of a template to the labels of the fields it speaks;
the render job carries the map (``render.RenderJob.spoken_fields``), and :func:`named` turns
such a refusal into what the owner can act on: "The spoken 'Problem: the line' is too long for
its moment: shorten it." The renderer's own findings stay in the report as they were.

Pure: no IO.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar

from core.social_templates import is_bundle_variable, placeholders, voice_lines

# The voice check's findings whose message names the line that is too long first.
VOICE_TOO_LONG_CODES = ("voice_lines_overlap", "voice_line_overruns")
_FIRST_LINE = re.compile(r"^line (?P<id>[\w-]+) ")
QUOTE = "'{}'"
ONE_FIELD = "The spoken {fields} is too long for its moment: shorten it, or choose a longer video. Nothing was rendered."
FIELDS = "The spoken {fields} are too long for their moment: shorten them, or choose a longer video. Nothing was rendered."

Failure = TypeVar("Failure")


def spoken_labels(blocks: Optional[Mapping[str, Any]]) -> Dict[str, Tuple[str, ...]]:
    """Each voice line's id with the labels of the fields it speaks (a field without a label by
    its name); a line that speaks only the brand is left out."""
    schema = (blocks or {}).get("variables_schema")
    schema = schema if isinstance(schema, Mapping) else {}
    labels: Dict[str, Tuple[str, ...]] = {}
    for line in voice_lines((blocks or {}).get("audio_plan")):
        names = [name for name in placeholders(str(line.get("text") or "")) if not is_bundle_variable(name)]
        if line.get("id") is not None and names:
            labels[str(line["id"])] = tuple(_label(schema.get(name), name) for name in names)
    return labels


def _label(spec: Any, name: str) -> str:
    label = spec.get("label") if isinstance(spec, Mapping) else None
    return label if isinstance(label, str) and label.strip() else name


def _too_long_lines(findings: Any) -> List[str]:
    """The ids of the lines the voice check found too long, in the order it found them, each once."""
    ids: List[str] = []
    for finding in findings if isinstance(findings, list) else []:
        if not isinstance(finding, Mapping) or finding.get("code") not in VOICE_TOO_LONG_CODES:
            continue
        first = _FIRST_LINE.match(str(finding.get("message") or ""))
        line_id = first.group("id") if first else finding.get("line")
        if line_id and str(line_id) not in ids:
            ids.append(str(line_id))
    return ids


def owner_message(report: Optional[Mapping[str, Any]], labels: Mapping[str, Sequence[str]]) -> Optional[str]:
    """What the owner is told when the voice check found a line too long, naming its fields;
    ``None`` when the report has no such finding, or its lines speak no field."""
    found = _too_long_lines((report or {}).get("findings"))
    names = list(dict.fromkeys(label for line_id in found for label in labels.get(line_id, ())))
    if not names:
        return None
    quoted = [QUOTE.format(name) for name in names]
    joined = quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]
    return (ONE_FIELD if len(quoted) == 1 else FIELDS).format(fields=joined)


def named(failure: Failure, labels: Mapping[str, Sequence[str]]) -> Failure:
    """``failure`` (a ``render.RenderFailure``) told in the owner's words when the voice check
    refused a line (:func:`owner_message`); any other failure as it is."""
    message = owner_message(getattr(failure, "report", None), labels)
    if message is None:
        return failure
    return type(failure)(failure.code, message, failure.report)


__all__ = ["VOICE_TOO_LONG_CODES", "named", "owner_message", "spoken_labels"]
