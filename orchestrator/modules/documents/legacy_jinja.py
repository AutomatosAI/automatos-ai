"""The filters a legacy (template_content) document's Jinja environment carries (F347, F356).

F347 (night 10b): the seeded "Invoice" put a hardcoded "$" before every amount,
whatever the business's currency, through ``"%.2f" | format(...)`` (which also
fails on an amount sent as text, "1,500.00"). It now prints its amounts with
``{{ total | amount }}``: a bare number with two decimals, anything else as
given, and never a currency the data did not carry (``modules.documents.amounts``).

F356 (5 Oct): the seeded Jinja starters (Basic Report, Invoice, Executive Summary)
had their own CSS, colours and sizes, no letterhead and, on two of them, no
footer. ``{{ brand | document_style }}`` prints the block documents' stylesheet
(``blocks.page_style``, with the kit's uploaded fonts from ``blocks.page_fonts``)
for the kit the page is rendered under, so a legacy starter has the same type
scale, spacing, colours, letterhead and footer as a Branded one. It is a filter,
not a global, so the parser that lists the names a template reads does not count
it as a data field.

PRD-255 (US-004): a legacy template gets the kit's design system as variables, as
the block documents print with it: :func:`legacy_brand` adds ``brand.palette.*``
(the effective colour roles, ``core.brand_palette.derive_palette``) and
``brand.type.*`` (the type scale, each step ``{size_pt, line_pt, weight}``) to the
kit it is rendered under, and ``amount`` prints a bare number in the kit's
currency when it has one (FR-7).

Both environments a legacy template meets carry the filters: the sandboxed one it
renders in (``generation_service``, PRD-156 S4) and the parser that lists the
names it reads (``data_coverage``), so a template that uses them parses there too.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Mapping

from jinja2 import Environment, pass_context
from jinja2.runtime import Context
from markupsafe import Markup

from core.brand_palette import derive_palette
from modules.documents.amounts import amount_text
from modules.documents.blocks.design_tokens import type_scale
from modules.documents.blocks.page_fonts import font_css
from modules.documents.blocks.page_style import build_styles
from modules.documents.locale_text import currency_of

AMOUNT_FILTER = "amount"
STYLE_FILTER = "document_style"
BRAND_VARIABLE = "brand"
PALETTE_VARIABLE, TYPE_VARIABLE = "palette", "type"


def legacy_brand(kit: Mapping[str, Any]) -> Dict[str, Any]:
    """``kit`` as a legacy template's ``brand``: with ``palette`` (every effective colour role)
    and ``type`` (every step of the type scale). Pure."""
    bk = dict(kit or {})
    steps = {name: asdict(step) for name, step in type_scale(bk).items()}
    return {**bk, PALETTE_VARIABLE: derive_palette(bk), TYPE_VARIABLE: steps}


@pass_context
def _amount(context: Context, value: Any) -> str:
    """``amount_text`` in the currency of the kit the page is rendered under."""
    brand = context.get(BRAND_VARIABLE)
    return amount_text(value, currency_of(brand if isinstance(brand, Mapping) else None))


def document_style(brand: Any) -> Markup:
    """The block documents' stylesheet for ``brand``, safe to print inside ``<style>`` as it is."""
    kit = dict(brand) if isinstance(brand, Mapping) else {}
    return Markup(build_styles(kit) + font_css(kit))  # noqa: S704 — built from validated kit values (page_style)


def with_document_filters(environment: Environment) -> Environment:
    """``environment`` (one the caller just made) with the document filters added; returned for chaining."""
    environment.filters = {**environment.filters, AMOUNT_FILTER: _amount, STYLE_FILTER: document_style}
    return environment


__all__ = ["AMOUNT_FILTER", "STYLE_FILTER", "document_style", "legacy_brand", "with_document_filters"]
