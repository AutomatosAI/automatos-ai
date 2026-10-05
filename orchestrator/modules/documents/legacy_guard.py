"""A legacy template is held to the block lane's promise: no field it needs goes out unfilled (F345).

F345 (night 10b): the seeded Jinja templates (Invoice, Executive Summary, Basic
Report) and any template uploaded or written as HTML had no required check. The
missing required fields of their ``data_schema`` were backfilled with an empty
value, so the template's own ``default(...)`` printed instead: an Invoice said
"BILL TO Client Name", and an Executive Summary without highlights came out as a
title on a blank page.

Now, before a legacy template renders:

* every field its ``data_schema`` requires must hold a value, the same blank rule
  as a block chip (``is_blank``: whitespace fills nothing). Objects and lists are
  read inside: a required key of an object, and of every row of a list, is checked
  too (``data.client.name``, ``data.line_items[row 2].total``);
* a page that shows none of the data is refused: the template is rendered with the
  data and with the title alone, and when the two read the same the document is
  empty (``EmptyDocumentError``), naming the fields the template reads.

Both raise :class:`UnresolvedDeliverableError`, which every caller already answers
with the fields to fill. The checks run in the decorator
:func:`legacy_fields_are_required`, so the render functions are unchanged.
"""
from __future__ import annotations

import functools
import html
import inspect
import logging
import re
from typing import Any, Callable, Dict, List, Optional

import jinja2
import jsonschema

from modules.documents.blocks import legacy_render_data
from modules.documents.blocks.table_cells import cell_path
from modules.documents.data_coverage import legacy_template_keys
from modules.documents.models import EmptyDocumentError, UnresolvedDeliverableError
from modules.documents.variables.catalog import is_blank

logger = logging.getLogger(__name__)

DATA_PREFIX = "data"
TITLE_KEY = "title"
BRAND_KEY = "brand"
# Keys every legacy page gets without the caller: the title (the tool's argument) and the brand kit.
NOT_THE_CALLERS = frozenset({TITLE_KEY, BRAND_KEY})
_HIDDEN = re.compile(r"(?is)<(head|style|script)\b.*?</\1\s*>")
_TAG = re.compile(r"(?s)<[^>]+>")
SECTIONS_KEY = "sections"
# The column each legacy lane renders from.
LEGACY_PDF_SOURCE = "template_content"
LEGACY_DOCX_SOURCE = "template_file_path"
# A bare render can fail where the template formats a number it was not given.
_BARE_RENDER_FAILURES = (jinja2.TemplateError, TypeError, ValueError, AttributeError)


def _empty(value: Any) -> bool:
    """Blank, or an object or list holding only blanks."""
    if isinstance(value, dict):
        return all(_empty(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_empty(item) for item in value)
    return is_blank(value)


def missing_required(data: Any, schema: Any, prefix: str = DATA_PREFIX) -> List[str]:
    """The required fields of ``schema`` that ``data`` leaves blank, as ``data.*`` paths. Pure."""
    if not isinstance(schema, dict) or not isinstance(data, dict):
        return []
    properties = schema.get("properties") or {}
    found: List[str] = []
    for key in schema.get("required") or []:
        value = data.get(key)
        path = f"{prefix}.{key}"
        if _empty(value):
            found.append(path)
        else:
            found.extend(_missing_inside(value, properties.get(key), path))
    return found


def _missing_inside(value: Any, schema: Any, path: str) -> List[str]:
    """A filled object's own required keys, and each list row's."""
    if isinstance(value, dict):
        return missing_required(value, schema, path)
    items = schema.get("items") if isinstance(schema, dict) else None
    if not isinstance(value, list) or not isinstance(items, dict):
        return []
    found: List[str] = []
    for number, row in enumerate(value, start=1):
        for key in missing_required(row, items, DATA_PREFIX):
            found.append(cell_path(path, number, key[len(DATA_PREFIX) + 1:]))
    return found


def page_text(page: str) -> str:
    """The words a reader sees on an HTML page: no head, styles or tags."""
    return " ".join(html.unescape(_TAG.sub(" ", _HIDDEN.sub(" ", page))).split())


def _render(env: Any, source: str, context: Dict[str, Any]) -> Optional[str]:
    try:
        return env.from_string(source).render(**context)
    except _BARE_RENDER_FAILURES as exc:
        logger.info("[DocGen] the empty-page check could not render the template (%s): skipped", exc)
        return None


def refuse_an_empty_page(env: Any, source: str, data: Dict[str, Any]) -> None:
    """Refuse a legacy page that shows nothing of ``data`` beyond its title."""
    wanted = sorted(set(legacy_template_keys(source, data) or ()) - NOT_THE_CALLERS)
    if not wanted:
        return  # the template reads no data: what it prints is all of it
    # Both renders get the same (empty) brand kit, so only the data differs between them.
    full = _render(env, source, {**data, BRAND_KEY: {}})
    bare = _render(env, source, {TITLE_KEY: data.get(TITLE_KEY), BRAND_KEY: {}})
    if full is None or bare is None or page_text(full) != page_text(bare):
        return
    raise EmptyDocumentError([f"{DATA_PREFIX}.{key}" for key in wanted])


def renders_legacy(template: Any, page: bool) -> bool:
    """Whether the legacy lane renders ``template``: its Jinja HTML for a PDF (``page``),
    its uploaded .docx for a Word file. A block template, or none, renders elsewhere."""
    if template is None or getattr(template, "blocks", None):
        return False
    return bool(getattr(template, LEGACY_PDF_SOURCE if page else LEGACY_DOCX_SOURCE, None))


def _as_checked(lane: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """The fields the caller is held to. F298: a body sent as ``content`` alone prints as one
    untitled section the platform made, so its blank title is not a field left unfilled; it
    is checked as the document's own title, which is required in its own right."""
    if data.get(SECTIONS_KEY) or not lane.get(SECTIONS_KEY):
        return lane
    titled = [{**row, TITLE_KEY: lane.get(TITLE_KEY)} for row in lane[SECTIONS_KEY]]
    return {**lane, SECTIONS_KEY: titled}


def check_legacy_template(env: Any, template: Any, data: Dict[str, Any], title: str, page: bool) -> None:
    """Refuse a legacy (non-block) template's render that would leave a field unfilled."""
    if not renders_legacy(template, page):
        return
    source = template.template_content if page else None
    # As generate() hands it to the template: the title argument, and the prose as the page prints it.
    lane = {TITLE_KEY: title, **(legacy_render_data(data) if page else data)}
    schema = getattr(template, "data_schema", None)
    missing = missing_required(_as_checked(lane, data), schema)
    if missing:
        raise UnresolvedDeliverableError(unresolved=missing)
    _log_schema_drift(lane, schema)
    if source:
        refuse_an_empty_page(env, source, lane)


def _log_schema_drift(data: Dict[str, Any], schema: Any) -> None:
    """A value of the wrong type is logged, as before; the template still tries to print it."""
    if not schema:
        return
    try:
        jsonschema.validate(instance=data, schema=schema)
    except (jsonschema.ValidationError, jsonschema.SchemaError) as exc:
        logger.warning("[DocGen] Schema validation warning: %s", exc.message)


def legacy_fields_are_required(*, page: bool) -> Callable:
    """Decorates ``generate_pdf`` (``page=True``: also refuses an empty page) and
    ``generate_docx``: a legacy template's render runs only with its fields filled."""

    def decorate(render: Callable) -> Callable:
        signature = inspect.signature(render)

        @functools.wraps(render)
        async def checked(*args: Any, **kwargs: Any) -> Any:
            call = signature.bind(*args, **kwargs)
            call.apply_defaults()
            given = call.arguments
            service = given["self"]
            check_legacy_template(service._jinja_env, given["template"], given["data"] or {}, given["title"], page)
            return await render(*args, **kwargs)

        return checked

    return decorate


__all__ = [
    "check_legacy_template", "legacy_fields_are_required", "missing_required", "page_text", "refuse_an_empty_page",
    "renders_legacy",
]
