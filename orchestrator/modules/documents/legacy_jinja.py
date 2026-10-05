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

Both environments a legacy template meets carry the filters: the sandboxed one it
renders in (``generation_service``, PRD-156 S4) and the parser that lists the
names it reads (``data_coverage``), so a template that uses them parses there too.
"""
from __future__ import annotations

from typing import Any, Mapping

from jinja2 import Environment
from markupsafe import Markup

from modules.documents.amounts import amount_text
from modules.documents.blocks.page_fonts import font_css
from modules.documents.blocks.page_style import build_styles

AMOUNT_FILTER = "amount"
STYLE_FILTER = "document_style"


def document_style(brand: Any) -> Markup:
    """The block documents' stylesheet for ``brand``, safe to print inside ``<style>`` as it is."""
    kit = dict(brand) if isinstance(brand, Mapping) else {}
    return Markup(build_styles(kit) + font_css(kit))  # noqa: S704 — built from validated kit values (page_style)


def with_document_filters(environment: Environment) -> Environment:
    """``environment`` (one the caller just made) with the document filters added; returned for chaining."""
    environment.filters = {**environment.filters, AMOUNT_FILTER: amount_text, STYLE_FILTER: document_style}
    return environment


__all__ = ["AMOUNT_FILTER", "STYLE_FILTER", "document_style", "with_document_filters"]
