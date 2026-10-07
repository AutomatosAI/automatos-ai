"""F383 (night 11, 7 Oct): a template a call names is found, or the call is refused; never swapped.

Night 11's invoice regenerations 3 and 4 skipped the invoice template: the first two used
"Branded Invoice", and the later calls named it in a way the exact, case-sensitive name
lookup missed (or by an id no longer active). ``template_for`` then fell back without a
word to Basic Report or, when that had no place for the data, to the block fallback under
the kit's letterhead: a generic two-page document with the kit's contact block as its
heading, which the agent, which cannot see the page, reported as the invoice.

A named template is now found by its name as people write it (exactly first, then with
case and spacing aside), and an id kept from before the template was replaced resolves
to the current version of the same name. A template that still cannot be found is a
:class:`TemplateNotFound`, a ``ValueError`` whose message the generate_document tool
passes back to the model, with the workspace's closest names: nothing is made, and
nothing else is made in its place. ``template_tools.find_template`` (the schema tool)
reads names the same way.

``templates`` is a ``DocumentTemplateService`` or anything with its ``get_template`` /
``get_template_by_name`` (``list_templates`` and ``get_template_version`` are read when
present). No IO of its own.
"""
from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional

CLOSE_NAMES_SHOWN = 5
CLOSE_NAME_CUTOFF = 0.5
NAMES_WHEN_NONE_CLOSE = 8

NO_SUCH_NAME = ("No template {name!r} in this workspace, so nothing was made. {names} Name one exactly as "
                "platform_list_templates lists it.")
NO_SUCH_ID = ("No template with id {template_id} in this workspace, so nothing was made: it was removed, or the "
              "id is not one platform_list_templates gave. {names}")
CLOSE_NAMES = "Close names: {names}."
SOME_NAMES = "Its templates include: {names}."
NO_TEMPLATES = "The workspace has no templates."


class TemplateNotFound(ValueError):
    """The template a call names is not in the workspace; the message lists close names."""


def folded(name: Any) -> str:
    """A template name with case and spacing aside, as people write one."""
    return " ".join(str(name or "").split()).casefold()


def _listed(templates: Any, workspace_id: Any) -> List[Any]:
    """The workspace's active templates (each name's latest version first), when the service lists them."""
    listing = getattr(templates, "list_templates", None)
    return list(listing(workspace_id)) if callable(listing) else []


def names_line(wanted: str, rows: List[Any]) -> str:
    """The workspace's names closest to ``wanted``, else some of its names. Pure."""
    by_fold: Dict[str, str] = {}
    for row in rows:
        by_fold.setdefault(folded(getattr(row, "name", "")), str(getattr(row, "name", "") or ""))
    by_fold.pop("", None)
    if not by_fold:
        return NO_TEMPLATES
    close = difflib.get_close_matches(folded(wanted), list(by_fold), n=CLOSE_NAMES_SHOWN, cutoff=CLOSE_NAME_CUTOFF)
    if close:
        return CLOSE_NAMES.format(names=", ".join(by_fold[key] for key in close))
    return SOME_NAMES.format(names=", ".join(list(by_fold.values())[:NAMES_WHEN_NONE_CLOSE]))


def named_template(templates: Any, workspace_id: Any, name: str) -> Any:
    """The latest active template of the workspace called ``name``, case and spacing
    aside; :class:`TemplateNotFound` with the closest names when there is none."""
    exact = templates.get_template_by_name(workspace_id, name)
    if exact is not None:
        return exact
    rows = _listed(templates, workspace_id)
    wanted = folded(name)
    match = next((row for row in rows if folded(getattr(row, "name", "")) == wanted), None)
    if match is not None:
        return match
    raise TemplateNotFound(NO_SUCH_NAME.format(name=name, names=names_line(name, rows)))


def _current_version(templates: Any, workspace_id: Any, template_id: Any) -> Optional[Any]:
    """The active template of the same name as the version ``template_id`` names, if any."""
    version = getattr(templates, "get_template_version", None)
    earlier = version(template_id, workspace_id) if callable(version) else None
    name = getattr(earlier, "name", None)
    return templates.get_template_by_name(workspace_id, name) if name else None


def template_by_id(templates: Any, workspace_id: Any, template_id: Any) -> Any:
    """The active template ``template_id`` names, or the current version of the one it
    named; :class:`TemplateNotFound` when neither is in the workspace."""
    found = templates.get_template(template_id, workspace_id)
    if found is not None:
        return found
    current = _current_version(templates, workspace_id, template_id)
    if current is not None:
        return current
    rows = _listed(templates, workspace_id)
    names = names_line("", rows) if rows else NO_TEMPLATES
    raise TemplateNotFound(NO_SUCH_ID.format(template_id=template_id, names=names))


__all__ = ["TemplateNotFound", "folded", "named_template", "names_line", "template_by_id"]
