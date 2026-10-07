"""Which of a document's data keys its template has no place for (F331, night 10).

F331 (5 Oct): with no template named, a PDF was filled by the workspace's seeded
"Basic Report", a legacy Jinja template that prints a title, a byline, metrics and
sections. An invoice's number, customer and line items were dropped without a
word, and the calling agent, which cannot see the page, said "done" eleven times.

Two uses:

* :func:`template_for` decides whether that default may fill a call at all;
  when it would drop a key, the block fallback renders instead, which prints every
  key (``blocks.data_details``). A call that names a template gets that template or a
  refusal, never the default (F383, ``template_lookup``).
* :func:`unused_data_keys` is what a template the caller DID name has no place
  for. It rides the result, so the tool tells the agent which keys the page lacks.

A block template reads ``data.<key>`` chips and ``data_table`` paths; a legacy
template reads the names its Jinja source uses (parsed, never rendered). An
uploaded .docx, a spreadsheet and a social render are not read: they report
nothing. No IO of its own: :func:`template_for` reads through the template service.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, List, Optional

from jinja2 import Environment, TemplateSyntaxError, meta

from modules.documents.blocks import collect_variable_paths, validate_blocks
from modules.documents.blocks.data_details import carried_keys
from modules.documents.legacy_jinja import with_document_filters
from modules.documents.variables.catalog import DYNAMIC_PREFIX, SIGN_OFF_PATH, SIGNER_KEY

logger = logging.getLogger(__name__)

# The formats whose template's data keys can be read.
PDF_FORMAT = "pdf"
CHECKED_FORMATS = frozenset({PDF_FORMAT, "docx"})
# A legacy template that prints sections prints a body sent as content alone too
# (``blocks.legacy_render_data`` makes it the one section).
SECTIONS_KEY = "sections"
BODY_KEY = "content"
# The seeded legacy template a PDF that names no template is filled by, when it fits.
DEFAULT_PDF_TEMPLATE = "Basic Report"
# Parses a legacy template's source to list the names it reads; renders nothing.
_PARSER = with_document_filters(Environment(autoescape=True))


def block_template_keys(blocks: Any) -> FrozenSet[str]:
    """The top-level ``data`` keys a block template's chips and tables read; one that prints the
    sign-off reads ``signer`` too (F364: a named signer signs in its place)."""
    paths = collect_variable_paths(validate_blocks(blocks))
    signer = {SIGNER_KEY} if SIGN_OFF_PATH in paths else set()
    return frozenset(
        {path[len(DYNAMIC_PREFIX):].split(".")[0] for path in paths if path.startswith(DYNAMIC_PREFIX)} | signer
    )


def legacy_template_keys(source: str, data: Dict[str, Any]) -> Optional[FrozenSet[str]]:
    """The names a legacy Jinja template reads; None when its source does not parse."""
    try:
        names = set(meta.find_undeclared_variables(_PARSER.parse(source)))
    except TemplateSyntaxError:
        logger.warning("[DocGen] a legacy template's source does not parse; its data keys are not checked")
        return None
    if SECTIONS_KEY in names and not data.get(SECTIONS_KEY):
        names.add(BODY_KEY)
    return frozenset(names)


def template_keys(template: Any, data: Dict[str, Any], fmt: str) -> Optional[FrozenSet[str]]:
    """The data keys ``template`` reads; None when it prints every key or cannot be read."""
    if template is None:
        return None  # the no-template block fallback prints every key
    blocks = getattr(template, "blocks", None)
    if blocks:
        return block_template_keys(blocks)
    source = getattr(template, "template_content", None)
    if fmt == PDF_FORMAT and source:
        return legacy_template_keys(source, data)
    return None


def unused_data_keys(template: Any, data: Dict[str, Any], fmt: str) -> List[str]:
    """The keys ``data`` carries that ``template`` has no place for, in the order sent. Pure."""
    if fmt not in CHECKED_FORMATS:
        return []
    used = template_keys(template, data, fmt)
    if used is None:
        return []
    return [key for key in carried_keys(data) if key not in used]


def prints_everything(template: Any, data: Dict[str, Any], fmt: str) -> bool:
    """Whether ``template`` has a place for every key ``data`` carries."""
    return not unused_data_keys(template, data, fmt)


def template_for(templates: Any, workspace_id: Any, fmt: str, data: Dict[str, Any],
                 template_id: Any = None, template_name: Optional[str] = None) -> Any:
    """The template a call names; a PDF that names none takes the workspace's
    :data:`DEFAULT_PDF_TEMPLATE` only when it has a place for every key ``data``
    carries, else none: the block fallback renders and prints them all.

    F383 (night 11): a name is read with case and spacing aside, an id kept from before
    an edit is the template's current version, and a named template that is not in the
    workspace raises ``template_lookup.TemplateNotFound`` with its closest names: never
    Basic Report or the letterhead page in its place.

    ``templates`` is the workspace's ``DocumentTemplateService``.
    """
    from modules.documents.template_lookup import named_template, template_by_id

    if template_id:
        return template_by_id(templates, workspace_id, template_id)
    if template_name:
        return named_template(templates, workspace_id, template_name)
    if fmt != PDF_FORMAT:
        return None
    default = templates.get_template_by_name(workspace_id, DEFAULT_PDF_TEMPLATE)
    left_out = unused_data_keys(default, data, fmt)
    if not left_out:
        return default
    logger.info("[DocGen] %s has no place for %s: the block fallback renders every key", DEFAULT_PDF_TEMPLATE, left_out)
    return None


__all__ = [
    "DEFAULT_PDF_TEMPLATE", "block_template_keys", "legacy_template_keys", "prints_everything", "template_for",
    "template_keys", "unused_data_keys",
]
