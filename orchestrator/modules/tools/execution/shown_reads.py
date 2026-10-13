"""F413, F420: a read the owner asked to see is theirs to see: one rule for every read.

Night 13 (F413): platform_get_deliverable read the Guji blog draft, its receipt said done,
and the reply held none of it. Night 14 (F420, 12–13 Oct, Auto on Haiku 5.5): 3 of 3 "show
me the file" misses went through the other reads, one a workspace_read_file of
guji-shakiso-natural-blog-draft.md, done and no text. So the rule N13-6 gave the deliverable
fetch lives here, once, for every read the owner may ask to see (``READS``): in the owner's
chat a read's answer says first that its text is the owner's to see (``SHOW_IT_NOTE``, first
so it survives the loop's cut), and a text past DELIVERABLE_SHOWN_MAX_CHARS is cut there with
the link to the rest. An agent's run, and a session's read_step_file, read the whole text
with no note, as before. ``fetched_work`` reads the same table to nudge a reply that shows
none of it.

Stdlib only at import, like the loop.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, Optional
from urllib.parse import urlencode

FILE_READ = "workspace_read_file"
# Each read the owner may ask to see, and the key its answer holds the read under (None: the answer itself).
READS: Dict[str, Optional[str]] = {
    "platform_get_deliverable": "deliverable",
    "platform_read_document": None,
    FILE_READ: None,
    "read_file": None,                     # workspace_read_file by its file tool's name (exec_file_ops)
}
SHOW_IT_NOTE = (
    "The owner sees none of this text until you put it in your reply. When they asked to see, read or "
    "show it, put the text itself in your reply as it is here, never a description of it."
)
CUT_NOTE = (
    " This is its first {shown:,} of {total:,} characters: put those in your reply, then the link to the "
    "rest: {link}"
)
# Where the rest of a cut read opens in the app: a workspace file in the Explorer, a document on its page.
EXPLORER_APP_PATH = "/deliverables/explorer"
EXPLORER_PATH_PARAM = "path"
DOCUMENTS_APP_PATH = "/documents"

Found = Dict[str, Any]
LinkOf = Callable[[Found], str]
Read = Callable[..., Awaitable[Any]]


def shown_chars() -> Optional[int]:
    """How much of a text a read hands back: in the owner's chat DELIVERABLE_SHOWN_MAX_CHARS, for
    the reply to show; elsewhere (an agent's run, a session's read_step_file) all of it, as before."""
    from config import config
    from core.llm.usage_context import LANE_CHAT, current_usage_scope

    if current_usage_scope().get("request_type") != LANE_CHAT:
        return None
    return max(1, int(config.DELIVERABLE_SHOWN_MAX_CHARS))


def _app_url(path: str) -> str:
    from config import config

    return f"{(config.FRONTEND_URL or '').rstrip('/')}{path}"


def deliverable_link(found: Found) -> str:
    """The rest of a deliverable: the Deliverables page opened on it (F298)."""
    from modules.tools.execution.generate_document_tool import deliverable_open_url

    return deliverable_open_url(found.get("id"))


def file_link(found: Found) -> str:
    """The rest of a workspace file: the Workspace Explorer opened on its path."""
    path = found.get("path")
    query = f"?{urlencode({EXPLORER_PATH_PARAM: path})}" if path else ""
    return _app_url(EXPLORER_APP_PATH) + query


def document_link(_found: Found) -> str:
    """The rest of a knowledge-base document: the Documents page (it opens no single document)."""
    return _app_url(DOCUMENTS_APP_PATH)


def text_to_show(data: Found, max_chars: Optional[int], link_of: LinkOf) -> Dict[str, Any]:
    """The content fields of a read. In the owner's chat (``max_chars`` set) a text comes with the
    note that it is the owner's to see, and is cut at ``max_chars`` with ``link_of``'s link to the rest."""
    shown: Dict[str, Any] = {"content_truncated": bool(data.get("content_truncated", False))}
    if data.get("content_error"):
        shown["content_error"] = data["content_error"]
    content = data.get("content")
    if not isinstance(content, str) or max_chars is None:
        return {"content": content, **shown}
    if len(content) <= max_chars:
        return {"show_the_owner": SHOW_IT_NOTE, "content": content, **shown}
    link = link_of(data)
    note = SHOW_IT_NOTE + CUT_NOTE.format(shown=max_chars, total=len(content), link=link)
    return {"show_the_owner": note, "content": content[:max_chars], **shown, "content_truncated": True,
            "content_chars": len(content), "rest_at": link}


def shown_read(result: Any, link_of: LinkOf) -> Any:
    """A read whose answer is the read itself, as the owner's chat gets it: the note first, then
    the answer with its text cut; a failure, an empty text and an agent's run pass as they are."""
    max_chars = shown_chars()
    content = result.get("content") if isinstance(result, dict) else None
    if max_chars is None or not isinstance(content, str) or not content.strip() or result.get("success") is False:
        return result
    fields = text_to_show(result, max_chars, link_of)
    return {"show_the_owner": fields["show_the_owner"], **result, **fields}


def shows_the_owner(link_of: LinkOf) -> Callable[[Read], Read]:
    """Decorates a read handler whose answer is the read itself (platform_read_document)."""
    def wrap(read: Read) -> Read:
        @functools.wraps(read)
        async def run(*args: Any, **kwargs: Any) -> Any:
            return shown_read(await read(*args, **kwargs), link_of)
        return run
    return wrap


def shows_a_read_file(execute: Read) -> Read:
    """Decorates ``exec_workspace.execute_workspace_action``: its workspace_read_file answer, by that
    name or the read_file tool's (both reach it), is shown like every read; other tools pass."""
    @functools.wraps(execute)
    async def run(*args: Any, **kwargs: Any) -> Any:
        result = await execute(*args, **kwargs)
        tool_name = kwargs.get("tool_name", args[1] if len(args) > 1 else None)
        return shown_read(result, file_link) if tool_name == FILE_READ else result
    return run


__all__ = ["CUT_NOTE", "FILE_READ", "READS", "SHOW_IT_NOTE", "deliverable_link", "document_link", "file_link",
           "shown_chars", "shown_read", "shows_a_read_file", "shows_the_owner", "text_to_show"]
