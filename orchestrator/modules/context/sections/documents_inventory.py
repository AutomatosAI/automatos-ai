"""F085-B (night 3): the prompt says the workspace has documents, and which.

Night 3's RAG test: the product's own 54-page manual sat in the workspace and
Auto searched it once in 40 questions. Nothing in its prompt said documents
existed, while two tool texts said every capability was a catalog search away.
This section names what is there — the owner's documents (newest titles first)
and the reports the agents saved — and says to search them for questions.

Volatile (rendered after the cached prompt prefix): the counts move every time
an agent saves a report, and a cache-stable block would be re-read each time.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)

TITLES_SHOWN = 10
_READY = "status IN ('completed', 'processed')"


def documents_summary(db: Any, workspace_id: Any) -> Optional[str]:
    """The section's text, or None when the workspace holds no documents."""
    counts = db.execute(
        text(f"SELECT (source_type IS NOT DISTINCT FROM 'agent_output') AS report, count(*) FROM documents "
             f"WHERE workspace_id = CAST(:ws AS uuid) AND {_READY} GROUP BY 1"),
        {"ws": str(workspace_id)},
    ).fetchall()
    by_kind = {bool(report): int(n) for report, n in counts}
    owners, reports = by_kind.get(False, 0), by_kind.get(True, 0)
    if owners + reports == 0:
        return None
    titles = [row[0] for row in db.execute(
        text(f"SELECT filename FROM documents WHERE workspace_id = CAST(:ws AS uuid) AND {_READY} "
             "AND source_type IS DISTINCT FROM 'agent_output' ORDER BY upload_date DESC NULLS LAST, id DESC "
             "LIMIT :n"),
        {"ws": str(workspace_id), "n": TITLES_SHOWN},
    ).fetchall()]
    held = []
    if owners:
        more = f", +{owners - len(titles)} more" if owners > len(titles) else ""
        held.append(f"{owners} of the owner's document{'s' if owners != 1 else ''} ({', '.join(titles)}{more})")
    if reports:
        held.append(f"{reports} report{'s' if reports != 1 else ''} its agents saved")
    return ("## Documents in this workspace\n"
            f"This workspace holds {' and '.join(held)}. For a question about the business, a document, "
            "or how the product works, search them with search_knowledge first and name the file you used.")


class DocumentsInventorySection(BaseSection):
    """What the workspace's knowledge base holds, in two sentences."""

    name: str = "documents_inventory"
    priority: int = 4
    max_tokens: Optional[int] = 220

    async def render(self, ctx: SectionContext) -> str:
        if ctx.db_session is None or not ctx.workspace_id:
            return ""
        try:
            return documents_summary(ctx.db_session, ctx.workspace_id) or ""
        except Exception:  # noqa: BLE001 — a section that cannot read is left out
            logger.warning("DocumentsInventorySection.render failed", exc_info=True)
            return ""
