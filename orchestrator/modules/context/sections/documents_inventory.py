"""F085-B (night 3): the prompt says the workspace has documents, and which.

Night 3's RAG test: the product's own 54-page manual sat in the workspace and
Auto searched it once in 40 questions. Nothing in its prompt said documents
existed, while two tool texts said every capability was a catalog search away.
This section names what is there — the owner's documents (newest titles first)
and the reports the agents saved — and says to search them for questions.

Volatile (rendered after the cached prompt prefix): the counts move every time
an agent saves a report, and a cache-stable block would be re-read each time.

F077/F078 (refresh-2 retest): with one database connected, Auto answered shop
questions from its memory and the documents, or asked "which database?". Nothing
named the database, and the documents sentence claimed every question about the
business. The section now names the workspace's active databases and sends a
number question to ``platform_query_data``; the documents sentence keeps what a
document says. Without a database the F085 text is unchanged.

F155: on a public widget turn the section names only what the widget key's
scopes may read: the documents with documents:read (the key's team's and the
shared ones, never the reports the agents saved), the databases with
data:query; nothing without either.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)

TITLES_SHOWN = 10
DATABASES_SHOWN = 5          # PRD-231: past five, the count alone; platform_query_data lists them
_READY = "status IN ('completed', 'processed')"
_ASK_DOCUMENTS = ("For a question about the business, a document, or how the product works, "
                  "search them with search_knowledge first and name the file you used.")
_ASK_DOCUMENTS_BESIDE_DATA = ("For what a document says or how the product works, "
                              "search them with search_knowledge first and name the file you used.")
# F181 (night 6): #1115 totalled a club's orders from the first pages of the
# export (47.06 kg; the file says about 100.3). A spreadsheet is counted in code.
_COUNT_SPREADSHEETS = ("A spreadsheet (CSV or Excel) is counted or totalled with code, never searched: "
                       "platform_read_document gives its copy's workspace_path and its row count.")


def connected_databases(db: Any, workspace_id: Any) -> List[str]:
    """Names of the workspace's active database sources, newest first: the filter
    nl2sql ``active_sources`` resolves ``platform_query_data`` against."""
    from sqlalchemy import text

    rows = db.execute(
        text("SELECT name FROM database_knowledge_sources WHERE workspace_id = CAST(:ws AS uuid) "
             "AND is_active IS TRUE ORDER BY created_at DESC, id DESC"),
        {"ws": str(workspace_id)},
    ).fetchall()
    return [str(row[0]) for row in rows]


def databases_sentence(names: List[str]) -> str:
    """Two sentences: what is connected, and how to ask it for a number. The tool
    is smart_query_database, the one Auto holds first-class: the refresh-3 retest
    named platform_query_data, reachable only through platform_execute, and Auto
    called neither (0 of 7) while the graph action got params={} (F027)."""
    if len(names) == 1:
        return (f"This workspace has 1 connected database ({names[0]}). For numbers about the business "
                "(counts, totals, rankings, trends), call smart_query_database with the question; with one "
                "database no name is needed.")
    listed = f" ({', '.join(names)})" if len(names) <= DATABASES_SHOWN else ""
    return (f"This workspace has {len(names)} connected databases{listed}. For numbers about the business "
            "(counts, totals, rankings, trends), call smart_query_database with the question and the "
            "database's name (it lists them when none is named).")


def documents_summary(db: Any, workspace_id: Any) -> Optional[str]:
    """The section's text, or None when the workspace holds no documents and no database."""
    databases = connected_databases(db, workspace_id) if _key_may("data:query") else []
    held = _documents_held(db, workspace_id) if _key_may("documents:read") else None
    if not held and not databases:
        return None
    count = f" {_COUNT_SPREADSHEETS}" if held and _holds_a_spreadsheet(db, workspace_id) else ""
    if not databases:
        return f"## Documents in this workspace\nThis workspace holds {held}. {_ASK_DOCUMENTS}{count}"
    lines = [f"This workspace holds {held}. {_ASK_DOCUMENTS_BESIDE_DATA}{count}"] if held else []
    return "\n".join(["## Documents and data in this workspace", *lines, databases_sentence(databases)])


def _key_may(scope: str) -> bool:
    """Any turn but a widget one; a widget turn only when its key holds ``scope``."""
    from core.security.surface import widget_scopes, widget_turn

    return not widget_turn() or scope in widget_scopes()


def _documents_held(db: Any, workspace_id: Any) -> Optional[str]:
    """"2 of the owner's documents (a, b) and 1 report its agents saved", or None."""
    # Imported where it queries, like the other sections: the package imports
    # every section eagerly, so a module-level import binds whatever
    # `sqlalchemy` is in sys.modules at that moment.
    from sqlalchemy import text

    from core.security.surface import widget_team, widget_turn
    from core.team_access import TEAM_FILTER_CLAUSE

    # F155: a widget key with a team lock sees its team's documents and the shared ones.
    lock = widget_team()
    team_filter = f" {TEAM_FILTER_CLAUSE}" if lock else ""
    params = {"ws": str(workspace_id), **({"team": lock} if lock else {})}
    counts = db.execute(
        text(f"SELECT (source_type IS NOT DISTINCT FROM 'agent_output') AS report, count(*) FROM documents "
             f"WHERE workspace_id = CAST(:ws AS uuid) AND {_READY}{team_filter} GROUP BY 1"),
        params,
    ).fetchall()
    by_kind = {bool(report): int(n) for report, n in counts}
    # The reports the agents saved are the owner's to read, never a widget visitor's.
    owners, reports = by_kind.get(False, 0), (0 if widget_turn() else by_kind.get(True, 0))
    if owners + reports == 0:
        return None
    titles = [row[0] for row in db.execute(
        text(f"SELECT filename FROM documents WHERE workspace_id = CAST(:ws AS uuid) AND {_READY}{team_filter} "
             "AND source_type IS DISTINCT FROM 'agent_output' ORDER BY upload_date DESC NULLS LAST, id DESC "
             "LIMIT :n"),
        {**params, "n": TITLES_SHOWN},
    ).fetchall()]
    held = []
    if owners:
        more = f", +{owners - len(titles)} more" if owners > len(titles) else ""
        held.append(f"{owners} of the owner's document{'s' if owners != 1 else ''} ({', '.join(titles)}{more})")
    if reports:
        held.append(f"{reports} report{'s' if reports != 1 else ''} its agents saved")
    return " and ".join(held)


def _holds_a_spreadsheet(db: Any, workspace_id: Any) -> bool:
    """Whether a ready CSV or Excel file is among the documents it may read."""
    from sqlalchemy import text

    from core.security.surface import widget_team
    from core.team_access import TEAM_FILTER_CLAUSE

    lock = widget_team()
    team_filter = f" {TEAM_FILTER_CLAUSE}" if lock else ""
    params = {"ws": str(workspace_id), **({"team": lock} if lock else {})}
    return bool(db.execute(
        text(f"SELECT EXISTS (SELECT 1 FROM documents WHERE workspace_id = CAST(:ws AS uuid) AND {_READY}"
             f"{team_filter} AND file_type IN ('csv', 'spreadsheet'))"),
        params,
    ).scalar())


class DocumentsInventorySection(BaseSection):
    """What the workspace's knowledge base holds, and where a number is asked for."""

    name: str = "documents_inventory"
    priority: int = 4
    # The budget truncates past this, from the end: the database sentences come last,
    # so the cap holds ten long titles and five database names whole.
    max_tokens: Optional[int] = 320

    async def render(self, ctx: SectionContext) -> str:
        if ctx.db_session is None or not ctx.workspace_id:
            return ""
        try:
            return documents_summary(ctx.db_session, ctx.workspace_id) or ""
        except Exception:  # noqa: BLE001 — a section that cannot read is left out
            logger.warning("DocumentsInventorySection.render failed", exc_info=True)
            return ""
