"""PRD-268 §6.11.5, K15 — what a calibration reads: titles and kinds only.

Four reads, each scoped to the caller's workspace, newest first (ties by id, so the cut and the refs hold still), at most ``PER_KIND`` rows:

* **Deliverables**: the workspace's ``v_workspace_outputs`` rows not deleted (every member sees
  the Deliverables page, ``api/deliverables.py``), its title and ``artifact_type``. An agent's
  heartbeat report is left out, as the Deliverables feed leaves it out
  (``frontend/hooks/use-deliverables-api.ts``): it is an agent's status, not the person's work;
  so is an archived one (an archived post, or the view's word for a report whose run failed).
* **Knowledge**: the workspace's ``documents`` that are not an agent's output (the Knowledge
  page's ``_knowledge_documents``, ``api/documents.py``), shared or the person's own upload; its
  file name and ``file_type``.
* **Conversations**: the person's own chats (``kind`` 'user' and their ``user_id``), never
  another member's, Auto's thread, or a widget visitor's (``widget_key_id``: those are started
  under a workspace's key, not by the person); its title. A chat still called "New Chat" says
  nothing and is left out.
* **Skills**: the names of the skills installed in the workspace (``workspace_enabled_skills``,
  as ``handlers_marketplace.list_workspace_skills`` reads them) and their category; a skill
  switched off (``is_active`` false) is nothing Auto can do.

No read selects a body, a summary, a description, a chunk, a file path or a file: the
statements name each column they read, and none of those (``tests/test_prd268w3_calibration_build.py``
spies on them). No call reads a Deliverable's or a document's contents (the Deliverable service's
content read, the documents' content routes). Every title is the person's data: the prompt quotes it,
never follows it (``services.calibration.prompt``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.tools.discovery.suggestion_fields import plain_text
from services.calibration.model import CONVERSATION, DELIVERABLE, DOCUMENT, SKILL, WORK_TYPES, CalibrationSource

__all__ = ["LABEL_CHARS", "PER_KIND", "Read", "has_work", "read_sources"]

# At most this many of each kind, the newest.
PER_KIND = 12
# A title as the prompt and the card show it, cut to this many characters.
LABEL_CHARS = 120
# api/chat.py's _UNTITLED_CHAT: a chat nobody has named yet.
UNTITLED_CHAT = "New Chat"
# A Deliverable's origin that is an agent's status, not work (the feed's exclusion).
HEARTBEAT_ORIGIN = "heartbeat"
# A Deliverable put away: an archived post, or an agent's report whose run failed (the view's mapping).
ARCHIVED = "archived"
# documents.source_type of an agent's output (api/documents.py AGENT_OUTPUT_SOURCE_TYPE).
AGENT_OUTPUT = "agent_output"

_DELIVERABLES = text("""
SELECT CAST(o.id AS text) AS id, o.title AS label, o.artifact_type AS kind
FROM v_workspace_outputs o
WHERE o.workspace_id = CAST(:workspace_id AS uuid) AND o.deleted_at IS NULL
  AND o.source_type IS DISTINCT FROM :heartbeat AND o.status IS DISTINCT FROM :archived
ORDER BY o.created_at DESC, CAST(o.id AS text) DESC
LIMIT :limit
""")

_DOCUMENTS = text("""
SELECT CAST(d.id AS text) AS id, COALESCE(NULLIF(d.original_filename, ''), d.filename) AS label,
       d.file_type AS kind
FROM documents d
WHERE d.workspace_id = CAST(:workspace_id AS uuid)
  AND d.source_type IS DISTINCT FROM :agent_output
  AND (d.is_shared IS NOT FALSE OR d.created_by_user_id = :user_id)
ORDER BY d.upload_date DESC NULLS LAST, d.id DESC
LIMIT :limit
""")

_CONVERSATIONS = text("""
SELECT CAST(c.id AS text) AS id, c.title AS label, c.kind AS kind
FROM chats c
WHERE c.workspace_id = CAST(:workspace_id AS uuid) AND c.user_id = :user_id AND c.kind = 'user'
  AND c.widget_key_id IS NULL AND c.title <> :untitled
ORDER BY c.updated_at DESC, c.id DESC
LIMIT :limit
""")

_SKILLS = text("""
SELECT CAST(s.id AS text) AS id, s.name AS label, s.category AS kind
FROM workspace_enabled_skills w
JOIN skills s ON s.id = w.skill_id
WHERE w.workspace_id = CAST(:workspace_id AS uuid) AND s.is_active IS NOT FALSE
ORDER BY w.enabled_at DESC, s.id DESC
LIMIT :limit
""")

# Each read, the source type it gives, and the parameters it binds beyond the workspace and limit.
_READS = (
    (DELIVERABLE, _DELIVERABLES, ("heartbeat", "archived")),
    (DOCUMENT, _DOCUMENTS, ("agent_output", "user_id")),
    (CONVERSATION, _CONVERSATIONS, ("user_id", "untitled")),
    (SKILL, _SKILLS, ()),
)


@dataclass(frozen=True)
class Read:
    """One source as read: what a line may cite, and its kind (an artifact type, a file type,
    a skill's category), which only the prompt sees."""

    source: CalibrationSource
    kind: str


def _label(value: Any) -> str:
    return plain_text(str(value or ""))[:LABEL_CHARS]


def _rows(db: Session, source_type: str, statement: Any, params: dict) -> Tuple[Read, ...]:
    rows = db.execute(statement, params).fetchall()
    return tuple(
        Read(CalibrationSource(type=source_type, id=str(row.id), label=_label(row.label)), _label(row.kind))
        for row in rows if _label(row.label)
    )


def read_sources(db: Session, workspace_id: Any, user_id: int) -> Tuple[Read, ...]:
    """Every source the person's calibration may read, by kind in ``_READS``' order, each kind
    newest first. Writes nothing."""
    values = {
        "workspace_id": str(UUID(str(workspace_id))), "user_id": int(user_id), "limit": PER_KIND,
        "heartbeat": HEARTBEAT_ORIGIN, "archived": ARCHIVED, "agent_output": AGENT_OUTPUT, "untitled": UNTITLED_CHAT,
    }
    found: Tuple[Read, ...] = ()
    for source_type, statement, names in _READS:
        params = {key: values[key] for key in ("workspace_id", "limit", *names)}
        found = (*found, *_rows(db, source_type, statement, params))
    return found


def has_work(reads: Tuple[Read, ...]) -> bool:
    """Whether anything read is the person's work (a skill alone is not)."""
    return any(read.source.type in WORK_TYPES for read in reads)
