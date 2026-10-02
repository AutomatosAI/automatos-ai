"""PRD-251 D8 (US-301, US-304): the source grammar of a channel step's params.

A param's value in the adapter data (``channel_adapters.py``) is a literal, a list,
a choice (``{"choose", "among", "else"}``), or ``$`` sources joined by ``|``:
``$copy``, ``$title``, ``$media``, ``$media[]``, ``$thumbnail``, ``$generated``,
``$idempotency_key``, ``$option.<name>`` and ``$steps.<id>[.<name>]`` (what an
earlier step returned). ``capabilities.py`` checks the data against it when it
loads; ``publish_sources.py`` resolves it at publish.
"""
from __future__ import annotations

import re
from typing import Any, FrozenSet, Mapping

STEPS_SOURCE = "$steps."
SOURCE = re.compile(
    r"\$(?:copy|title|thumbnail|idempotency_key|generated|media(?:\[\]|\.content_type|\.bytes)?"
    r"|option\.[a-z][a-z0-9_]*|steps\.[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?)"
)
# A choice (US-304): the ``choose`` source's value when ``among`` holds it, else the
# first of ``else`` that ``among`` holds (``else``'s first when ``among`` is unknown).
CHOICE_KEYS = frozenset({"choose", "among", "else"})
# The sources that name a media FILE: a param reading one takes a file or a link.
FILE_SOURCES = ("$media", "$media[]", "$thumbnail")


def _check_choice(value: Mapping[str, Any], earlier: FrozenSet[str], where: str) -> None:
    fallback = value.get("else")
    if set(value) != CHOICE_KEYS or not isinstance(fallback, list) or not all(isinstance(v, str) for v in fallback):
        raise ValueError(f"{where}: a choice is {sorted(CHOICE_KEYS)}, its else a list of values")
    check_source(value["choose"], earlier, where)
    check_source(value["among"], earlier, where)


def check_source(value: Any, earlier: FrozenSet[str], where: str) -> None:
    """A list of sources, a choice, a literal, or ``$`` sources joined by ``|``: each
    one the grammar knows, and a step source naming an EARLIER step and what it
    returns (``earlier``: ``<id>`` for a step that returns an id, ``<id>.<name>`` for
    each name it returns). Raises ``ValueError`` naming ``where``."""
    if isinstance(value, list):
        for item in value:
            check_source(item, earlier, where)
        return
    if isinstance(value, Mapping):
        _check_choice(value, earlier, where)
        return
    for ref in value.split("|") if isinstance(value, str) and value.startswith("$") else ():
        if not SOURCE.fullmatch(ref) or (ref.startswith(STEPS_SOURCE) and ref[len(STEPS_SOURCE):] not in earlier):
            raise ValueError(f"{where}: {ref!r} is not a source (a step source names an earlier step that returns an id)")


def _reads(value: Any, wanted) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_reads(item, wanted) for item in value)
    return isinstance(value, str) and any(wanted(ref) for ref in value.split("|"))


def file_params(params: Mapping[str, Any]) -> FrozenSet[str]:
    """The params whose source reads a media file (``$media``, ``$media[]``, ``$thumbnail``)."""
    return frozenset(name for name, value in params.items() if _reads(value, lambda ref: ref in FILE_SOURCES))


def reads_steps(source: Any) -> bool:
    """Whether a param's source reads an earlier step's output (``$steps.``); a
    choice has its own fallback and does not count."""
    return _reads(source, lambda ref: ref.startswith(STEPS_SOURCE))
