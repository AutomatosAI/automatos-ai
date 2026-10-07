"""F383 (night 11, 7 Oct): a date in a document's data prints in the kit's date style.

Night 11's first Branded Invoice printed its dates as ISO ("2026-11-06") although the
kit's date style was "d MMMM yyyy": the style reached only the ``date.long`` chip
(``variables.resolver``), and every value of the call's data printed as sent, through
the block chips and tables (``amounts.field_text``), the no-template fallback
(``blocks.data_details``) and the legacy Jinja templates (``templates/invoice.html``).

:func:`dates_follow_the_kit` wraps ``DocumentGenerationService.generate``: before any
renderer reads the data, each value that is wholly an ISO date ("2026-11-06") or an ISO
date and time ("2026-11-06T09:30:00Z") becomes that day in the kit's style
(``locale_text.long_date`` over ``locale_text.date_style_of``: "6 November 2026", or
"November 6, 2026"), in a new copy of the data, rows of a table included. Any other
value (an invoice number, a code, "6 Nov") is left as sent. Only a PDF or a Word file
is changed: a spreadsheet keeps its dates as data, and a social render its own.

``services`` is imported when a document is made, never when this module loads.
"""
from __future__ import annotations

import functools
import re
from datetime import date
from typing import Any, Awaitable, Callable, Optional

from .locale_text import date_style_of, long_date

Async = Callable[..., Awaitable[Any]]
# The formats whose data prints as text on a page.
DATED_FORMATS = frozenset({"pdf", "docx"})
_ISO_DATE = re.compile(
    r"^\s*(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}(?::?\d{2})?)?)?\s*$")


def iso_day(value: Any) -> Optional[date]:
    """The day ``value`` is, when it is wholly an ISO date or date and time; else None. Pure."""
    found = _ISO_DATE.match(value) if isinstance(value, str) else None
    if found is None:
        return None
    try:
        return date(int(found["year"]), int(found["month"]), int(found["day"]))
    except ValueError:
        return None  # "2026-13-40" is not a day: printed as sent


def kit_dated(value: Any, style: Any) -> Any:
    """``value`` with every ISO date in it printed in ``style``, as a new copy. Pure."""
    if isinstance(value, dict):
        return {key: kit_dated(item, style) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [kit_dated(item, style) for item in value]
    day = iso_day(value)
    return long_date(day, style) if day is not None else value


def has_iso_dates(value: Any) -> bool:
    """Whether ``value`` (rows of a table too) holds an ISO date anywhere. Pure."""
    if isinstance(value, dict):
        return any(has_iso_dates(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_iso_dates(item) for item in value)
    return iso_day(value) is not None


def dates_follow_the_kit(generate: Async) -> Async:
    """Wrap ``DocumentGenerationService.generate`` (called with keywords): a PDF's or a Word
    file's data prints its ISO dates in the workspace kit's date style."""
    @functools.wraps(generate)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        data, fmt = kwargs.get("data"), str(kwargs.get("format") or "").lower()
        if fmt in DATED_FORMATS and isinstance(data, dict) and has_iso_dates(data):
            from services.brand_rules import kit_off_loop

            workspace_id = kwargs.get("workspace_id") or getattr(self, "workspace_id", None)
            kit = await kit_off_loop(getattr(self, "db", None), workspace_id)
            kwargs = {**kwargs, "data": kit_dated(data, date_style_of(kit))}
        return await generate(self, *args, **kwargs)
    return wrapped


__all__ = ["DATED_FORMATS", "dates_follow_the_kit", "has_iso_dates", "iso_day", "kit_dated"]
