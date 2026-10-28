"""PRD-268 P268W1-RVW-1: what brings each read action that keeps a record of its own to that record.

The empty-database sweep (``test_prd268_read_actions_write_nothing.py``) could not see these
writes: each is reached only when something exists. A seed puts that something in place at
the edge of the action's own code, and the action's code runs from there to its write:

- ``platform_search_packages``: a package that matches, in a workspace mid-onboarding: the
  funnel's ``package_offered`` stamp, committed on the caller's session;
- ``platform_codegraph_get_symbol`` and ``platform_codegraph_search``: an indexed project:
  its ``codegraph_query_logs`` row, committed on the caller's session (the service is built
  without its embedding manager and parser, which a name search does not use; the search is
  by name, since a search by meaning first sends the query to the embedding provider);
- ``platform_read_document``: a spreadsheet whose workspace copy is missing: the copy, written
  over HTTP to the workspace worker;
- ``platform_query_data``: a connected database: its audit row and its training example. A
  stand-in for ``DatabaseKnowledgeService`` writes them on a session of its own, as the real
  one does (``write_nl_audit``; ``query_database``'s ``auto_train``);
- ``platform_search_documents``: a search that finds a passage: its document's
  ``last_accessed`` and the substrate tile's event, each on a session of its own (a stand-in
  ``retrieve`` ends as the real one does, with the same two calls);
- ``platform_render_preview``: a session's ticket and a template's page: the picture, written
  over HTTP to the session's folder.

``Seed.marks`` are what each writes outside a check-in, so a seed that stops reaching its
write fails its own test instead of passing quietly. ``Seed.answers`` is part of what each
answers in a check-in, so a call that never reached its handler cannot pass either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace as NS
from typing import Any, Callable, Dict, Tuple

from modules.tools.discovery.handlers_render_preview import NOT_IN_A_CHECK_IN
from tests.helpers_write_spies import WriteLog, seeded_session, spy_session

WORKER = "http://worker.test:8081"
DOC_ID = 4127
TEMPLATE_ID = "6f0d2c8e-3a51-4b7e-9d2a-0c4f1e7b5a93"
TICKET = 5

SessionFor = Callable[[WriteLog], Any]


@dataclass(frozen=True)
class Seed:
    """One read action, what it is called with, what to plant, what it writes outside a
    check-in and part of what it answers in one. ``context`` is what the server puts in the
    executor's ``caller_context``."""

    action: str
    params: Dict[str, Any]
    plant: Callable[[Any, Any], SessionFor]
    marks: Tuple[str, ...]
    answers: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)


def _to_the_worker(monkeypatch: Any) -> None:
    """The workspace worker answers at ``WORKER``, through a client made for this test."""
    import core.workspace_client as workspace_client
    from config import config

    monkeypatch.setattr(config, "WORKER_INTERNAL_URL", WORKER, raising=False)
    monkeypatch.setattr(workspace_client, "_client", None)


def _packages(monkeypatch: Any, tmp_path: Any) -> SessionFor:
    from modules.tools.discovery import handlers_packages
    from services import marketplace_packages

    package = NS(slug="cafe-starter", name="Cafe starter", description="A team for a cafe.",
                 members=[{"type": "agent", "ref": "barista"}], setup_manifest={}, showcase=False)
    monkeypatch.setattr(marketplace_packages, "list_packages", lambda *a, **k: [package])
    monkeypatch.setattr(marketplace_packages, "match_by_signals",
                        lambda *a, **k: [NS(package=package, score=0.9, reasons=["cafe"])])
    monkeypatch.setattr(handlers_packages, "_load_workspace",
                        lambda db, workspace_id: NS(id=workspace_id, onboarding={"stage": "proposal"}))
    return spy_session


def _bare_codegraph_service(db: Any) -> Any:
    from modules.codegraph.codegraph_service import CodeGraphService

    service = CodeGraphService.__new__(CodeGraphService)
    service.db = db
    return service


def _codegraph(monkeypatch: Any, tmp_path: Any) -> SessionFor:
    from modules.tools.discovery import handlers_codegraph

    monkeypatch.setattr(handlers_codegraph, "_service", _bare_codegraph_service)
    return lambda log: seeded_session(log, rows={"FROM codegraph_projects WHERE name = :name": [NS(id=7)]})


def _read_document(monkeypatch: Any, tmp_path: Any) -> SessionFor:
    from core.models import Document
    from modules.rag import retrieval_filters
    from modules.tools.discovery import read_document_lookup

    upload = tmp_path / "orders.csv"
    upload.write_text("order,bags\n1,3\n2,5\n")
    document = NS(id=DOC_ID, file_type="csv", file_path=str(upload), original_filename="orders.csv",
                  filename="orders.csv", upload_date=None, last_accessed=None)
    monkeypatch.setattr(read_document_lookup, "resolve", lambda db, workspace_id, params: (DOC_ID, None))
    monkeypatch.setattr(retrieval_filters, "allowed_document_ids", lambda db, ids, filters: {str(DOC_ID)})
    _to_the_worker(monkeypatch)
    return lambda log: seeded_session(log, found={Document: document})


class AuditRow:
    """What ``write_nl_audit`` adds: a ``DatabaseQueryAudit`` row."""


class TrainingExample:
    """What ``query_database`` adds when it trains on a first-try success."""


def _written_on_its_own_session(row: Any) -> None:
    from core.database import database

    session = database.SessionLocal()
    try:
        session.add(row)
        session.commit()
    finally:
        session.close()


class _ConnectedDatabase:
    """``DatabaseKnowledgeService`` for a workspace with one database connected."""

    async def resolve_source_id(self, workspace_id: Any, reference: Any, db_session: Any = None) -> str:
        return "7"

    async def query_database(self, **kwargs: Any) -> Dict[str, Any]:
        if kwargs.get("auto_train", True):
            _written_on_its_own_session(TrainingExample())
        return {"success": True, "sql": "SELECT count(*) AS orders FROM orders", "data": [{"orders": 412}],
                "columns": ["orders"], "row_count": 1}

    async def write_nl_audit(self, **kwargs: Any) -> None:
        _written_on_its_own_session(AuditRow())


def _query_data(monkeypatch: Any, tmp_path: Any) -> SessionFor:
    import modules.nl2sql as nl2sql
    from modules.nl2sql import agent_answer

    async def no_schema(service: Any, source_id: Any, workspace_id: Any) -> Dict[str, Any]:
        return {}

    monkeypatch.setattr(nl2sql, "get_database_knowledge_service", lambda: _ConnectedDatabase())
    monkeypatch.setattr(agent_answer, "ground_source", no_schema)
    return spy_session


def _search_documents(monkeypatch: Any, tmp_path: Any) -> SessionFor:
    from core.observability.substrate_metrics import SEAM_DOCUMENTS, STATUS_HIT
    from modules.rag import service as rag_service

    passage = {"document_id": DOC_ID, "content": "The November price is 4.20 a bag.", "source_file": "prices.pdf",
               "chunk_index": 0, "metadata": {"document_id": DOC_ID}}

    async def retrieve(self: Any, query: str = "", **kwargs: Any) -> Any:
        workspace_id = kwargs.get("workspace_id")
        rag_service.record_substrate_search_nowait(seam=SEAM_DOCUMENTS, workspace_id=workspace_id, candidates=1,
                                                   latency_ms=1.0, status=STATUS_HIT)
        self._schedule_access_tracking([passage], workspace_id)
        return NS(chunks=[passage], sources_map=[{"document_id": DOC_ID, "score": 0.9}], sources=["prices.pdf"])

    monkeypatch.setattr(rag_service.RAGService, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(rag_service.RAGService, "retrieve", retrieve)
    return spy_session


def _render_preview(monkeypatch: Any, tmp_path: Any) -> SessionFor:
    from modules.tools.discovery import handlers_render_preview

    monkeypatch.setattr(handlers_render_preview, "_template_job",
                        lambda db, workspace_id, request: ((lambda: b"\x89PNG page one"), ".png"))
    _to_the_worker(monkeypatch)
    return spy_session


SEEDS: Tuple[Seed, ...] = (
    Seed("platform_search_packages", {"text": "a cafe that roasts its own coffee"}, _packages, ("commit",),
         {"success": True, "count": 1}),
    Seed("platform_codegraph_get_symbol", {"project": "shop", "symbol": "retry", "limit": 5}, _codegraph,
         ("INSERT INTO codegraph_query_logs", "commit"), {"success": True, "count": 0}),
    Seed("platform_codegraph_search", {"project": "shop", "query": "retry", "mode": "fuzzy", "limit": 5},
         _codegraph, ("INSERT INTO codegraph_query_logs", "commit"), {"success": True, "mode": "fuzzy", "count": 0}),
    Seed("platform_read_document", {"document_id": DOC_ID}, _read_document, ("HTTP POST worker.test",),
         {"success": False, "error": "Document has no readable content yet"}),
    Seed("platform_query_data", {"question": "How many orders this month?"}, _query_data,
         ("add(TrainingExample)", "add(AuditRow)"), {"success": True, "row_count": 1}),
    Seed("platform_search_documents", {"query": "November price"}, _search_documents,
         ("UPDATE documents", "add(SubstrateMetricEvent)"), {"success": True, "count": 1}),
    Seed("platform_render_preview", {"template_id": TEMPLATE_ID, "_session_task_id": TICKET}, _render_preview,
         ("HTTP POST worker.test",), {"success": False, "error": NOT_IN_A_CHECK_IN},
         context={"session_task_id": TICKET}),
)

__all__ = ["AuditRow", "DOC_ID", "SEEDS", "Seed", "TICKET", "TrainingExample", "WORKER"]
