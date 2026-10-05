"""The filters a legacy (template_content) document's Jinja environment carries (F347).

F347 (night 10b): the seeded "Invoice" put a hardcoded "$" before every amount,
whatever the business's currency, through ``"%.2f" | format(...)`` (which also
fails on an amount sent as text, "1,500.00"). It now prints its amounts with
``{{ total | amount }}``: a bare number with two decimals, anything else as
given, and never a currency the data did not carry (``modules.documents.amounts``).

Both environments a legacy template meets carry the filter: the sandboxed one it
renders in (``generation_service``, PRD-156 S4) and the parser that lists the
names it reads (``data_coverage``), so a template that uses it parses there too.
"""
from __future__ import annotations

from jinja2 import Environment

from modules.documents.amounts import amount_text

AMOUNT_FILTER = "amount"


def with_document_filters(environment: Environment) -> Environment:
    """``environment`` (one the caller just made) with the document filters added; returned for chaining."""
    environment.filters = {**environment.filters, AMOUNT_FILTER: amount_text}
    return environment


__all__ = ["AMOUNT_FILTER", "with_document_filters"]
