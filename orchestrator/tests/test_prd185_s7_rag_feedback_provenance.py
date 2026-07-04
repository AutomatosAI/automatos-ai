"""PRD-185 S7 — give the RAG feedback loop a mouth.

Chat votes must write ``rag_feedback`` rows carrying the turn's retrieved
document ids, so the PRD-179 live ranker (which reads ``UNNEST(document_ids)``)
learns from thumbs instead of reading an empty table.

Pure tests — no DB, network, or app import chain — per ``feedback-no-local-servers``.
They exercise the two self-contained modules the vote path delegates to:
``modules.rag.retrieval_provenance`` (capture) and ``modules.rag.feedback_writer``
(write). The DB session is mocked at the boundary.
"""
from unittest.mock import MagicMock

from modules.rag.retrieval_provenance import (
    is_retrieval_tool,
    collect_doc_ids_from_tool_result,
    build_retrieval_context,
    RETRIEVAL_TOOL_NAMES,
)
from modules.rag.feedback_writer import (
    write_rag_feedback,
    feedback_from_retrieval_context,
)


# ── capture: which tools count as retrieval ───────────────────────────────────

def test_is_retrieval_tool_gates_to_document_tools():
    for name in ("search_documents", "search_knowledge", "semantic_search"):
        assert is_retrieval_tool(name) is True
        assert name in RETRIEVAL_TOOL_NAMES
    # A DB tool can surface a column literally named document_id — must not count.
    assert is_retrieval_tool("query_database") is False
    assert is_retrieval_tool("smart_query_database") is False
    assert is_retrieval_tool(None) is False
    assert is_retrieval_tool("") is False


# ── capture: extract ids from a tool result ───────────────────────────────────

def test_collect_doc_ids_from_nested_result():
    result = {
        "success": True,
        "raw_result": {
            "results": [
                {"document_id": 12, "chunk_id": 881},
                {"document_id": 47, "chunk_ids": [903, 904]},
            ]
        },
        "frontend_data": {"sources": [{"document_id": 12}]},
    }
    docs, chunks = collect_doc_ids_from_tool_result(result)
    assert docs == {12, 47}
    assert chunks == {881, 903, 904}


def test_collect_doc_ids_coerces_int_strings_and_drops_junk():
    result = {"raw_result": {
        "document_ids": ["12", 47, "not-an-id", None, True, 3.5],
        "chunk_id": "not-int",
    }}
    docs, chunks = collect_doc_ids_from_tool_result(result)
    # "12"->12, 47 kept; "not-an-id"/None/True(bool)/3.5(float) dropped
    assert docs == {12, 47}
    assert chunks == set()


def test_collect_doc_ids_handles_non_dict_safely():
    assert collect_doc_ids_from_tool_result(None) == (set(), set())
    assert collect_doc_ids_from_tool_result("nope") == (set(), set())
    assert collect_doc_ids_from_tool_result({}) == (set(), set())


# ── capture: build the stored provenance blob ─────────────────────────────────

def test_build_retrieval_context_none_when_empty():
    assert build_retrieval_context(set(), set(), "q") is None


def test_build_retrieval_context_sorts_ids_and_keeps_query():
    ctx = build_retrieval_context({47, 12}, {904, 881}, "why is the sky blue?")
    assert ctx == {
        "document_ids": [12, 47],
        "chunk_ids": [881, 904],
        "query": "why is the sky blue?",
    }


def test_build_retrieval_context_bounds_query_length():
    ctx = build_retrieval_context({1}, set(), "x" * 5000)
    assert len(ctx["query"]) == 2000
    assert ctx["document_ids"] == [1]


def test_build_retrieval_context_omits_empty_query():
    ctx = build_retrieval_context({1}, set(), "")
    assert "query" not in ctx


# ── write: the shared rag_feedback INSERT ─────────────────────────────────────

def _mock_db(returned_id=999):
    db = MagicMock()
    row = MagicMock()
    row.id = returned_id
    db.execute.return_value.fetchone.return_value = row
    return db


def test_write_rag_feedback_inserts_and_commits():
    db = _mock_db(returned_id=101)
    fid = write_rag_feedback(
        db,
        query="q",
        workspace_id="ws-1",
        user_id=7,
        document_ids=[12, 47],
        chunk_ids=[881],
        feedback_type="thumbs_down",
    )
    assert fid == 101
    db.commit.assert_called_once()
    params = db.execute.call_args.args[1]
    assert params["document_ids"] == [12, 47]
    assert params["chunk_ids"] == [881]
    assert params["feedback_type"] == "thumbs_down"
    assert params["workspace_id"] == "ws-1"
    assert params["user_id"] == 7
    assert params["query"] == "q"


def test_write_rag_feedback_empty_arrays_and_query_normalised():
    db = _mock_db()
    write_rag_feedback(db, query=None, workspace_id="ws-1", document_ids=[], chunk_ids=[])
    params = db.execute.call_args.args[1]
    # empty arrays -> SQL NULL, not []; None query -> "" (column is TEXT NOT NULL)
    assert params["document_ids"] is None
    assert params["chunk_ids"] is None
    assert params["query"] == ""


def test_write_rag_feedback_commit_false_leaves_txn_open():
    db = _mock_db()
    write_rag_feedback(db, query="q", workspace_id="ws-1", commit=False)
    db.commit.assert_not_called()


# ── the S7 acceptance: a chat vote writes rag_feedback with retrieved doc ids ──

def test_chat_vote_writes_rag_feedback():
    """Casting a thumbs-down on a message with retrieval provenance lands a
    rag_feedback row carrying that turn's retrieved document ids and the
    down-vote polarity — exactly what the PRD-179 ranker consumes."""
    db = _mock_db(returned_id=555)
    retrieval_context = {
        "document_ids": [12, 47],
        "chunk_ids": [881, 903],
        "query": "how do refunds work?",
    }
    fid = feedback_from_retrieval_context(
        db,
        retrieval_context=retrieval_context,
        is_upvoted=False,
        workspace_id="ws-9",
        user_id=7,
    )
    assert fid == 555
    params = db.execute.call_args.args[1]
    assert params["document_ids"] == [12, 47]
    assert params["chunk_ids"] == [881, 903]
    assert params["feedback_type"] == "thumbs_down"
    assert params["query"] == "how do refunds work?"
    assert params["workspace_id"] == "ws-9"
    assert params["user_id"] == 7


def test_chat_vote_upvote_maps_to_thumbs_up():
    db = _mock_db()
    feedback_from_retrieval_context(
        db,
        retrieval_context={"document_ids": [3], "chunk_ids": [], "query": "q"},
        is_upvoted=True,
        workspace_id="ws-9",
    )
    params = db.execute.call_args.args[1]
    assert params["feedback_type"] == "thumbs_up"
    assert params["document_ids"] == [3]


def test_vote_without_retrieval_context_writes_nothing():
    # A pure-chat turn (no retrieval) must not create a hollow rag_feedback row.
    db = _mock_db()
    assert feedback_from_retrieval_context(
        db, retrieval_context=None, is_upvoted=False, workspace_id="ws-9",
    ) is None
    assert feedback_from_retrieval_context(
        db, retrieval_context={"document_ids": [], "chunk_ids": []},
        is_upvoted=False, workspace_id="ws-9",
    ) is None
    db.execute.assert_not_called()


def test_vote_feedback_filters_non_int_ids_before_write():
    db = _mock_db()
    feedback_from_retrieval_context(
        db,
        retrieval_context={"document_ids": [12, "x", None], "chunk_ids": ["y"], "query": "q"},
        is_upvoted=False,
        workspace_id="ws-9",
    )
    params = db.execute.call_args.args[1]
    assert params["document_ids"] == [12]
    assert params["chunk_ids"] is None  # only non-int chunk ids -> NULL
