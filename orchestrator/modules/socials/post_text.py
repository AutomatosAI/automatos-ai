"""F378 (night 11, 7 Oct): a post is never saved with a placeholder in its words.

Night 11 saved post 802ab561 with the caption "We sold {retail_bags_sold} retail bags,
grew our Harvest Club to {harvest_club_subscribers} subscribers…": the braces were saved
as they were, and would have been published so (B15). The post's validators
(``service._validate_copy`` and ``service._validate_variables``) now hand their checked
value through :func:`without_placeholders`, which refuses copy or a variable's text that
holds a template placeholder, or a variable that is a bare ``true``/``false``/``null``
(B-i3-5), naming the field (``core/social_text_values.py``). Every save path goes through
those validators: the editor, the composer's draft, Auto's and the agents' tools.

The caller passes its own error type (``service.InvalidPost``, a 422 with the message),
so this module needs nothing of the service it serves.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Mapping, Tuple, Type

from core.social_text_values import is_literal_word, placeholder_in

COPY, VARIABLES = "copy", "variables"
PLACEHOLDER_MESSAGE = "{where} holds the placeholder {found}: write the real words before saving"
LITERAL_MESSAGE = "{where} is the word {word!r}, not words for the post: write the text, or leave it empty"


def _copy_texts(copy: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    yield "copy.base", copy.get("base")
    channels = copy.get("channels") if isinstance(copy.get("channels"), Mapping) else {}
    for toolkit, text in channels.items():
        yield f"copy.channels.{toolkit}", text


def _variable_texts(variables: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    for name, spec in variables.items():
        if isinstance(spec, Mapping):
            yield f"variables.{name}", spec.get("value")


def problem_in(field: str, value: Mapping[str, Any]) -> str:
    """What is wrong with the words of ``value`` (the post's ``copy`` or ``variables``), or ``""``."""
    texts = _copy_texts(value) if field == COPY else _variable_texts(value)
    for where, text in texts:
        found = placeholder_in(text)
        if found:
            return PLACEHOLDER_MESSAGE.format(where=where, found=found)
        if field == VARIABLES and is_literal_word(text):
            return LITERAL_MESSAGE.format(where=where, word=text.strip())
    return ""


def without_placeholders(field: str, value: Dict[str, Any], error: Type[Exception]) -> Dict[str, Any]:
    """``value`` as it is when its words hold no placeholder (and no variable is a bare
    literal); ``error`` with the message naming the field otherwise."""
    problem = problem_in(field, value)
    if problem:
        raise error(problem)
    return value
