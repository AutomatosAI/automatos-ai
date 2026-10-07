"""F380 (night 11, 7 Oct): a ticket's answer links a generated file by its Deliverable, never by storage.

#2160's answer linked its card as ``localhost:9000/…/generated-documents/…png``: the
object-storage address of the file, which answered "Access Denied". generate_document
gives an agent two links (F298, ``generated_document_summary``): the owner's link, the
Deliverables page opened on the file, and a signed share link for people outside the
workspace, "never on a card". The agent put the storage address on the card anyway,
without its signature.

Before a ticket closes, every object-storage address of a generated file in its answer
(signed or not, any host) becomes the owner's link to that file's Deliverable
(``generate_document_tool.deliverable_open_url``), found in the workspace by the file's
name; a file with no Deliverable gets the Deliverables page. The app's own route
(``/api/documents/generated/<file>``) is left as it is.
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import UUID

from sqlalchemy import text

logger = logging.getLogger(__name__)

Async = Callable[..., Awaitable[Any]]

# An object-storage address of a generated file: host, bucket and key, the signature optional.
_STORAGE_LINK = re.compile(
    r"(?:https?://)?[\w.\-]+(?::\d+)?/(?:[\w.\-]+/)*generated-documents/"
    r"(?P<name>[\w\-]+(?:\.[\w\-]+)*)(?:\?[^\s)\]>\"'`]*)?")
_DELIVERABLE_BY_NAME = text(
    "SELECT id FROM deliverables WHERE workspace_id = CAST(:ws AS uuid) AND file_name = :name "
    "AND deleted_at IS NULL ORDER BY created_at DESC LIMIT 1")


def storage_links(answer: str) -> Dict[str, str]:
    """{each storage address in ``answer``: the file name it points at}. Pure."""
    return {found.group(0): found.group("name") for found in _STORAGE_LINK.finditer(answer or "")}


def _deliverable_id(db: Any, workspace_id: Any, name: str) -> Optional[str]:
    """The newest Deliverable of the workspace for the generated file ``name``."""
    row = db.execute(_DELIVERABLE_BY_NAME, {"ws": str(UUID(str(workspace_id))), "name": name}).fetchone()
    return str(row[0]) if row else None


def with_owners_links(db: Any, workspace_id: Any, answer: str) -> str:
    """``answer`` with each storage address of a generated file made the owner's link to it."""
    from modules.tools.execution.generate_document_tool import deliverable_open_url

    links = storage_links(answer)
    if not links:
        return answer
    owners = {name: deliverable_open_url(_deliverable_id(db, workspace_id, name)) for name in set(links.values())}
    return _STORAGE_LINK.sub(lambda found: owners[found.group("name")], answer)


def _linked_kwargs(db: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """``finalize_board_task_run``'s keywords with the answer's storage links made the owner's."""
    exec_result = kwargs.get("exec_result") or {}
    answer = exec_result.get("result")
    if not isinstance(answer, str) or not storage_links(answer):
        return kwargs
    try:
        linked = with_owners_links(db, kwargs.get("workspace_id"), answer)
    except Exception:  # noqa: BLE001 — logged; the answer keeps the links it had
        logger.exception("[F380] the document links of ticket %s could not be resolved", kwargs.get("task_id"))
        return kwargs
    return {**kwargs, "exec_result": {**exec_result, "result": linked}}


def a_cards_document_links_open_in_the_app(finalize: Async) -> Async:
    """Wrap ``api.board_tasks.finalize_board_task_run`` (keywords after ``db``): a generated
    file the answer links by its storage address is linked by its Deliverable instead."""
    @functools.wraps(finalize)
    async def wrapped(db: Any, *args: Any, **kwargs: Any) -> Any:
        return await finalize(db, *args, **_linked_kwargs(db, kwargs))
    return wrapped


__all__ = ["a_cards_document_links_open_in_the_app", "storage_links", "with_owners_links"]
