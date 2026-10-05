"""Every key a no-template document's data carries is printed (F331, night 10).

F331 (5 Oct): eleven invoice PDFs came out as a title on an empty page. Auto had
sent the invoice as ``sections`` plus ``invoice_number``, ``customer_name`` and an
``item_details`` list; agents sent ``line_items``, ``subtotal`` and ``total``. The
no-template render printed only the report shape (title, author, date, sections,
metrics, highlights, recommendations) and dropped the rest without a word, while
the tool's markdown answer listed every key, so each card said "done".

The keys the report shape does not print now print here, by what they hold:

* text, numbers and dates: one "Details" table, a row per key;
* a list of objects (line items): a table, a column per field;
* a list of text: a list, one line per item;
* an object: a table of its fields.

Pure: blocks only, no IO.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Tuple

from .markdown_body import blocks_from_markdown, section_text
from .schema import HeadingBlock, TableBlock, TextRun

# The tool's title argument, injected into data; it is the page's heading already.
TITLE_KEY = "title"
# Keys the platform adds for itself (``_charts``) are not the caller's content.
INTERNAL_PREFIX = "_"
DETAILS_HEADING = "Details"
LIST_SEPARATOR = ", "
FIELD_SEPARATOR = ": "

Ids = Callable[[str], str]


def is_carried(value: Any) -> bool:
    """A value that holds something to print: not None, not empty or blank text."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def carried_keys(data: Dict[str, Any], skip: Iterable[str] = ()) -> List[str]:
    """The caller's keys that hold a value, in the order sent; never the title or ``_`` keys. Pure."""
    skipped = frozenset(skip) | {TITLE_KEY}
    return [
        key for key, value in data.items()
        if key not in skipped and not str(key).startswith(INTERNAL_PREFIX) and is_carried(value)
    ]


def label(key: Any) -> str:
    """``invoice_number`` as a reader sees it: "Invoice number"."""
    text = str(key).replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def cell_text(value: Any) -> str:
    """A value as one table cell: a list joined, an object as its ``field: value`` pairs."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return LIST_SEPARATOR.join(f"{label(k)}{FIELD_SEPARATOR}{cell_text(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return LIST_SEPARATOR.join(cell_text(item) for item in value)
    return str(value)


def _row(cells: Iterable[str], bold_first: bool = False) -> List[List[TextRun]]:
    """One table row; a details row's label in bold."""
    return [
        [TextRun(text=text, marks=["bold"] if bold_first and index == 0 else [])]
        for index, text in enumerate(cells)
    ]


def _heading(text: str, bid: Ids) -> HeadingBlock:
    return HeadingBlock(id=bid("h"), level=2, content=[TextRun(text=text)])


def _pairs_table(pairs: List[Tuple[Any, Any]], bid: Ids) -> TableBlock:
    rows = [_row([label(key), cell_text(value)], bold_first=True) for key, value in pairs]
    return TableBlock(id=bid("tbl"), header=False, rows=rows)


def _columns(items: List[Dict[str, Any]]) -> List[Any]:
    """Every field any row has, in the order first seen."""
    seen: Dict[Any, None] = {}
    for item in items:
        for key in item:
            seen.setdefault(key, None)
    return list(seen)


def _records_table(items: List[Dict[str, Any]], bid: Ids) -> TableBlock:
    columns = _columns(items)
    rows = [_row([label(column) for column in columns])]
    rows += [_row([cell_text(item.get(column)) for column in columns]) for item in items]
    return TableBlock(id=bid("tbl"), header=True, rows=rows)


def structured_blocks(key: Any, value: Any, bid: Ids) -> List[Any]:
    """One list or object key, under its own heading."""
    if isinstance(value, dict):
        body: List[Any] = [_pairs_table(list(value.items()), bid)]
    elif all(isinstance(item, dict) for item in value):
        body = [_records_table(list(value), bid)]
    else:
        text = section_text([cell_text(item) for item in value]) or ""
        body = blocks_from_markdown(text, bid("md"))
    return [_heading(label(key), bid), *body]


def extra_blocks(data: Dict[str, Any], printed: FrozenSet[str], bid: Ids) -> Tuple[List[Any], List[Any]]:
    """The keys ``printed`` leaves out: ``(details table, list and object blocks)``. Pure."""
    keys = carried_keys(data, skip=printed)
    scalars = [(key, data[key]) for key in keys if not isinstance(data[key], (list, tuple, dict))]
    details = [_heading(DETAILS_HEADING, bid), _pairs_table(scalars, bid)] if scalars else []
    structured = [
        block for key in keys if isinstance(data[key], (list, tuple, dict))
        for block in structured_blocks(key, data[key], bid)
    ]
    return details, structured


__all__ = ["carried_keys", "cell_text", "extra_blocks", "is_carried", "label", "structured_blocks"]
