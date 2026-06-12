"""PRD-157 S2 — read_document + grep_documents agent tools.

Pure layer: reachability (registered + wired) and the no-DB validation paths.
Integration layer (marked): paged reading past the first chunk + team isolation.
"""

from __future__ import annotations

import uuid

import pytest

from modules.tools.discovery.action_registry import ActionRegistry
from modules.tools.discovery.actions_documents import register_documents_actions

_TOOLS = ["platform_read_document", "platform_grep_documents"]


def _registry() -> ActionRegistry:
    reg = ActionRegistry()
    register_documents_actions(reg)
    return reg


class TestReachability:
    @pytest.mark.parametrize("name", _TOOLS)
    def test_registered_read_only_operator_tier(self, name):
        action = _registry().get(name)
        assert action is not None, f"{name} not registered"
        assert action.permission_level == "read"
        assert action.super_admin_only is False   # operator tier (PRD-143)
        assert action.workspace_scoped is True     # multitenancy isolation

    @pytest.mark.parametrize("name", _TOOLS)
    def test_handler_wired_in_executor(self, name):
        from modules.tools.discovery.platform_executor import PlatformActionExecutor

        ex = PlatformActionExecutor(db=None, workspace_id=None)
        assert name in ex._handlers
        assert callable(ex._handlers[name])


class TestValidationNoDB:
    """Validation paths that return before any DB access (db=None is never used)."""

    @pytest.mark.asyncio
    async def test_read_document_missing_id(self):
        from modules.tools.discovery.handlers_documents import read_document

        res = await read_document(db=None, workspace_id="ws", params={})
        assert res["success"] is False and "document_id" in res["error"]

    @pytest.mark.asyncio
    async def test_read_document_non_int_id(self):
        from modules.tools.discovery.handlers_documents import read_document

        res = await read_document(db=None, workspace_id="ws", params={"document_id": "abc"})
        assert res["success"] is False

    @pytest.mark.asyncio
    async def test_grep_missing_pattern(self):
        from modules.tools.discovery.handlers_documents import grep_documents

        res = await grep_documents(db=None, workspace_id="ws", params={})
        assert res["success"] is False and "pattern" in res["error"]

    @pytest.mark.asyncio
    async def test_grep_invalid_regex(self):
        from modules.tools.discovery.handlers_documents import grep_documents

        res = await grep_documents(db=None, workspace_id="ws", params={"pattern": "("})
        assert res["success"] is False and "regular expression" in res["error"]


# --------------------------------------------------------------------------- #
# Integration (real Postgres) — seed documents + chunks
# --------------------------------------------------------------------------- #

def _seed_doc(db, workspace_id, team_access="{}", n_chunks=12):
    from sqlalchemy import text

    doc_id = db.execute(
        text(
            """
            INSERT INTO documents (filename, workspace_id, team_access, status, upload_date)
            VALUES (:fn, CAST(:ws AS uuid), :ta, 'processed', NOW())
            RETURNING id
            """
        ),
        {"fn": "seed.txt", "ws": workspace_id, "ta": team_access},
    ).scalar()
    for i in range(n_chunks):
        db.execute(
            text(
                """
                INSERT INTO document_chunks (document_id, chunk_index, content, workspace_id)
                VALUES (:doc, :idx, :content, :ws)
                """
            ),
            {"doc": doc_id, "idx": i, "content": f"paragraph {i} " * 200, "ws": workspace_id},
        )
    db.flush()
    return doc_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_document_pages_past_first_chunk(db_session):
    from modules.tools.discovery.handlers_documents import read_document

    ws = str(uuid.uuid4())
    doc_id = _seed_doc(db_session, ws, n_chunks=12)

    page0 = await read_document(db_session, ws, {"document_id": doc_id, "page": 0})
    assert page0["success"] is True
    assert page0["total_pages"] >= 2 and page0["has_more"] is True
    assert page0["source_id"] == doc_id

    page1 = await read_document(db_session, ws, {"document_id": doc_id, "page": 1})
    assert page1["success"] is True
    # reading past the first page surfaces different content (past char 500)
    assert page1["content"] != page0["content"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_document_team_isolation(db_session, monkeypatch):
    import modules.tools.discovery.handlers_documents as h

    ws = str(uuid.uuid4())
    doc_id = _seed_doc(db_session, ws, team_access="{sales}", n_chunks=3)

    # A 'support' agent must not read a 'sales'-only document.
    monkeypatch.setattr(h, "_resolve_agent_team", lambda db, aid: "support")
    denied = await h.read_document(db_session, ws, {"document_id": doc_id, "_agent_id": 1})
    assert denied["success"] is False

    # The owning team can.
    monkeypatch.setattr(h, "_resolve_agent_team", lambda db, aid: "sales")
    allowed = await h.read_document(db_session, ws, {"document_id": doc_id, "_agent_id": 1})
    assert allowed["success"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_grep_documents_matches_and_scopes(db_session):
    from modules.tools.discovery.handlers_documents import grep_documents
    from sqlalchemy import text

    ws = str(uuid.uuid4())
    doc_id = db_session.execute(
        text(
            """
            INSERT INTO documents (filename, workspace_id, team_access, status, upload_date)
            VALUES ('grep.txt', CAST(:ws AS uuid), '{}', 'processed', NOW())
            RETURNING id
            """
        ),
        {"ws": ws},
    ).scalar()
    db_session.execute(
        text(
            """
            INSERT INTO document_chunks (document_id, chunk_index, content, workspace_id)
            VALUES (:doc, 0, 'the error code is ERR_TIMEOUT here', :ws)
            """
        ),
        {"doc": doc_id, "ws": ws},
    )
    db_session.flush()

    res = await grep_documents(db_session, ws, {"pattern": r"ERR_[A-Z]+"})
    assert res["success"] is True
    assert res["count"] == 1
    assert res["matches"][0]["document_id"] == doc_id
    assert res["matches"][0]["chunk_index"] == 0
