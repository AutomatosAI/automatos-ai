"""A document whose amounts print with no currency sign says so (F367, night 10c; PRD-255 FR-7).

F367 (6 Oct): with the kit's currency blank (c1's derived kit), an invoice built
from plain numbers printed "Total due 269.00", and nothing warned anyone. A
document never invents a currency. The kit's country gives one (7 Oct:
``country_locale``), and a kit with neither a currency nor a country makes the
platform ask: :func:`unpriced_amounts_are_said` wraps
``DocumentGenerationService.generate`` and, when the workspace kit has no
currency and the data sent an amount as a bare number, records those keys on the
result (``GeneratedDocument.unpriced_keys``). The generate_document tool hands
them to the agent with one instruction: ask the owner once which country (or
currency), save it to the brand kit, make the document again. Once the kit has a currency, every
format prints it the same way: PDF and Word through ``amounts.field_text``, the
spreadsheet through ``xlsx_letterhead.money_format``, all from
``locale_text.currency_of``.

``services`` is imported when a document is made, never when this module loads.
"""
from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import Any, Awaitable, Callable, List, Mapping, Optional

from .amounts import bare_amount_keys
from .locale_text import currency_of

Async = Callable[..., Awaitable[Any]]
# The formats whose amounts take the kit's currency.
PRICED_FORMATS = frozenset({"pdf", "docx", "xlsx"})


def amounts_without_currency(kit: Optional[Mapping[str, Any]], data: Any, fmt: str) -> List[str]:
    """The amount keys ``data`` sent as bare numbers when ``kit`` has no currency; empty otherwise. Pure."""
    if str(fmt or "").lower() not in PRICED_FORMATS or currency_of(kit):
        return []
    return bare_amount_keys(data)


def unpriced_amounts_are_said(generate: Async) -> Async:
    """Wrap ``DocumentGenerationService.generate``: its result names the amounts that printed
    with no currency sign because the workspace's kit has none (a copy of the result)."""
    signature = inspect.signature(generate)

    @functools.wraps(generate)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        from services.brand_rules import kit_off_loop

        result = await generate(self, *args, **kwargs)
        call = signature.bind_partial(self, *args, **kwargs).arguments
        data, fmt = call.get("data"), call.get("format")
        if not bare_amount_keys(data) or not dataclasses.is_dataclass(result):
            return result
        workspace_id = call.get("workspace_id") or getattr(self, "workspace_id", None)
        kit = await kit_off_loop(getattr(self, "db", None), workspace_id)
        unpriced = amounts_without_currency(kit, data, fmt)
        return dataclasses.replace(result, unpriced_keys=unpriced) if unpriced else result
    return wrapped


__all__ = ["PRICED_FORMATS", "amounts_without_currency", "unpriced_amounts_are_said"]
