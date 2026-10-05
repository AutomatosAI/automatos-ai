"""The file types a document template can make (F348, night 10b).

F348 (5 Oct): asking for an xlsx from a PDF template answered a bare 400,
"Invalid document generation request": the spreadsheet path ignores the template,
found no ``columns`` in the template's data and raised, and the route hid why.
Nothing said which formats the template does make, and the gallery did not say
it either.

What a template makes follows from its body, the way the renderers read it
(``generation_service``):

* a block body renders to a PDF and to a Word file;
* a legacy Jinja HTML body (``template_content``) renders to a PDF only;
* an uploaded ``.docx`` (``template_file_path``) renders to a Word file only;
* a spreadsheet template and a social template make their own format only;
* a template with no body at all (the "Meeting Notes" starter) renders its data
  under the brand kit's letterhead, as a PDF or a Word file.

:func:`refuse_unsupported_format` stops a request for anything else before it
renders, in words that name what the template does make. A social format asked
of a template keeps its own refusal (``generate_social``). Pure: no DB, no IO.
"""
from __future__ import annotations

from typing import Any, List, Optional

from core.social_templates import is_social_format

PDF = "pdf"
DOCX = "docx"
XLSX = "xlsx"
# What a block body (or no body) renders to.
BODY_FORMATS = (PDF, DOCX)
UNSUPPORTED = (
    "The template '{name}' makes {supported} files, not {requested}. Ask for {supported}, "
    "or pick a template that makes {requested} (each template lists its supported_formats)."
)
XLSX_HINT = " A plain spreadsheet needs no template: send xlsx with 'columns' and 'rows' in data."


class UnsupportedTemplateFormat(ValueError):
    """A template asked for a format it cannot make; the message names the ones it can."""

    def __init__(self, name: str, requested: str, supported: List[str]) -> None:
        self.requested = requested
        self.supported = list(supported)
        message = UNSUPPORTED.format(name=name, supported=" or ".join(supported), requested=requested)
        super().__init__(message + (XLSX_HINT if requested == XLSX else ""))


def supported_formats(template: Any) -> List[str]:
    """The formats ``template`` renders to, in the order the gallery shows them."""
    fmt = getattr(template, "format", None)
    if fmt == XLSX or is_social_format(fmt):
        return [fmt]
    if getattr(template, "blocks", None):
        return list(BODY_FORMATS)
    bodies = ((PDF, getattr(template, "template_content", None)), (DOCX, getattr(template, "template_file_path", None)))
    made = [made_format for made_format, body in bodies if body]
    return made or list(BODY_FORMATS)


def refuse_unsupported_format(template: Optional[Any], requested: str) -> None:
    """Raise :class:`UnsupportedTemplateFormat` when ``template`` cannot make ``requested``.

    No template, or a social format (which ``generate_social`` checks with its own
    words), is never refused here.
    """
    if template is None or is_social_format(requested):
        return
    supported = supported_formats(template)
    if requested not in supported:
        raise UnsupportedTemplateFormat(str(getattr(template, "name", None) or "this template"), requested, supported)


__all__ = ["BODY_FORMATS", "UnsupportedTemplateFormat", "refuse_unsupported_format", "supported_formats"]
