"""F413 (night 13): work Auto fetches for the owner to see is shown in its reply.

Night 13 (11–12 Oct, Auto on Haiku 5.5): "show me the Guji blog draft" ran
platform_get_deliverable, its receipt said done, and the reply held none of the draft, 12
times; asked for it "between two lines of dashes", Auto printed the whole file every time.

Night 14 (F420): 3 of 3 "show me the file" misses went through workspace_read_file and
platform_read_document, so the rule covers every read in ``shown_reads.READS``, one table.

Now a read's answer in the owner's chat says first that the text is the owner's to see
and to be put in the reply (``shown_reads.SHOW_IT_NOTE``), cut at
DELIVERABLE_SHOWN_MAX_CHARS with the link to the rest. And when the owner's own words ask
to see, read or show something, a final reply after a fetch that succeeded which quotes
none of the fetched text is nudged once (``shows_fetched_work``, on the loop's final-reply
check, after that check's own nudges): the FX-017 retry seam, so the nudged draft is
retracted when the retry replaces it and stands when the retry comes back blank. A fetch
of the details only (no ``include_content``) is asked for the text.

Stdlib only at import, like the loop.
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .nudges import is_nudge, nudge_about
from .shown_reads import READS
from .turn_account import auto_speaks

logger = logging.getLogger(__name__)

SHOW_FETCHED_MSG = (
    "The owner asked to see {title} and you fetched it, but your reply shows none of its text. Write your "
    "reply again with the text itself in it, as the tool returned it{cut}. Do not describe or summarise it "
    "instead."
)
CUT_PART = ": its first {shown:,} characters, then the link to the rest ({link})"
FETCH_THE_TEXT_MSG = (
    "The owner asked to see {title}, but you fetched only its details, not its text. Fetch it again with "
    "include_content=true, in this response, and put its text in your reply."
)
UNTITLED = "the deliverable"
# What a read is called when it has no title: a document's file name, a workspace file's name or path.
TITLE_KEYS = ("title", "filename", "name", "path")
TITLE_SHOWN_CHARS = 120
# The owner asks to see a thing: "show me the draft", "can I see it", "read me the post", "open the report".
# F420: every read is covered now, so "read the brief and…" and "see if…" (work on a text) are no ask to see it.
_ASKS_TO_SEE = re.compile(r"\b(?:show|see(?!\s+(?:if|whether)\b)|read\s+(?:me|out|aloud|back|it\s+to\s+me)|open|view"
                          r"|print|display|paste|pull\s+up|bring\s+up)\b", re.IGNORECASE)
# A quote is the opening of one of the text's lines, its marks (markdown, rules) and spacing aside.
QUOTE_PROBE_CHARS = 40
# Lines shorter than this ("Hi", "---") prove nothing when the text has longer ones.
QUOTE_MIN_CHARS = 12
_MARKS = re.compile(r"[*_#>`|~=\-]+")

Outcome = Sequence[Any]          # (action, its params, its result)
Recover = Callable[..., Awaitable[Any]]


def _plain(text: str) -> str:
    return " ".join(_MARKS.sub(" ", text).split()).lower()


def quotes_any(reply: str, text: str) -> bool:
    """Whether ``reply`` holds a run of ``text``'s own words: the opening of one of its lines."""
    lines = [line for line in (_plain(raw) for raw in text.splitlines()) if line]
    if not lines:
        return True                                  # nothing in it to show
    said = _plain(reply)
    probes = [line[:QUOTE_PROBE_CHARS] for line in lines if len(line) >= QUOTE_MIN_CHARS] or lines
    return any(probe in said for probe in probes)


def _said(content: Any) -> str:
    """A message's words: its text, or the text parts of a message with attachments."""
    if isinstance(content, str):
        return content
    parts = content if isinstance(content, list) else []
    return " ".join(part.get("text") or "" for part in parts if isinstance(part, dict))


def owners_words(messages: List[Dict[str, Any]]) -> str:
    """The latest message the person wrote, never a nudge of the loop's."""
    return next((_said(m.get("content")) for m in reversed(messages)
                 if m.get("role") == "user" and not is_nudge(m)), "")


def _fetched(action: str, result: Any) -> Optional[Dict[str, Any]]:
    """What a read that succeeded returned (the chat wraps the action's answer as raw_result): a
    deliverable, with or without its text; a document or a workspace file, only with its text."""
    if action not in READS or not isinstance(result, dict):
        return None
    raw = result.get("raw_result") if isinstance(result.get("raw_result"), dict) else result
    key = READS[action]
    found = raw.get(key) if key else raw
    if raw.get("success") is False or not isinstance(found, dict):
        return None
    return found if key or isinstance(found.get("content"), str) else None


def _title(found: Dict[str, Any]) -> str:
    named = next((found[key] for key in TITLE_KEYS if found.get(key)), "")
    title = " ".join(str(named).split())[:TITLE_SHOWN_CHARS]
    return f'"{title}"' if title else UNTITLED


def _nudge_for(found: Dict[str, Any]) -> str:
    """What the retry is told about one fetch: show its text, or fetch the text first."""
    content = found.get("content")
    if not isinstance(content, str):
        return FETCH_THE_TEXT_MSG.format(title=_title(found))
    link = found.get("rest_at")
    cut = CUT_PART.format(shown=len(content), link=link) if link else ""
    return SHOW_FETCHED_MSG.format(title=_title(found), cut=cut)


def unshown_fetch(reply: str, outcomes: Sequence[Outcome], messages: List[Dict[str, Any]]) -> Optional[str]:
    """The nudge for a reply that shows none of the work the owner asked to see; None when the
    owner asked for nothing to see, nothing was fetched, or the reply quotes a fetched text."""
    if not _ASKS_TO_SEE.search(owners_words(messages)):
        return None
    fetched = [found for action, _params, result in outcomes
               for found in [_fetched(action, result)] if found is not None]
    if not fetched:
        return None
    texts = [found["content"] for found in fetched if isinstance(found.get("content"), str)]
    if any(quotes_any(reply, text) for text in texts):
        return None
    return _nudge_for(fetched[-1])


def shows_fetched_work(recover: Recover) -> Recover:
    """Wraps the loop's final-reply check (``ToolLoopExecutor._recover_claimed_action``): when it
    has nothing to recover, a reply in the owner's chat that shows none of the work the owner
    asked to see is nudged once, through ``nudge_about`` like the check's own nudges."""
    @functools.wraps(recover)
    async def check(loop: Any, current: Any, messages: List[Dict[str, Any]],
                    tools: Optional[List[Dict[str, Any]]]) -> Any:
        retry = await recover(loop, current, messages, tools)
        if retry is not None or not tools or getattr(current, "tool_calls", None) or not auto_speaks(loop.promises):
            return retry
        text = unshown_fetch(getattr(current, "content", "") or "", loop.tracker.outcomes, messages)
        if not text:
            return None
        logger.warning("[tool-loop] the owner asked to see fetched work and the reply shows none of it — nudging once")
        return await nudge_about(loop._llm, current, messages, tools, text)
    return check


__all__ = ["FETCH_THE_TEXT_MSG", "SHOW_FETCHED_MSG", "owners_words", "quotes_any",
           "shows_fetched_work", "unshown_fetch"]
