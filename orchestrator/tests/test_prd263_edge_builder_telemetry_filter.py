"""PRD-263 US-018 — EdgeBuilder reads production, synthetic-signal and untagged rows, never an ask.

At 697ca7d6e the nightly EdgeBuilder's read (core/services/edge_builder_logs) had no
filter: rows tagged ``eval`` or ``replay`` fed the Knowledge Graph builder, and a wait for
the owner's click, which build_edges counts as a failure like every status but
``success``, became a ``fails_for_intent`` for its tool. The read now keeps
``telemetry_source`` production, ``synthetic_signal`` (the PRD-232 gap and shown markers)
or NULL, so any other tag (``synthetic``, seed_telemetry's eval-query rows, included)
stays out, and never a row whose status is ``asked``.

Rows go into the transactional ``db_session``. The build_edges case reads them with the
real loader and fakes the rest: the embedder, the cluster write (every loaded log in one
intent cluster) and the edge and affinity writes, which it records.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta
from itertools import count
from uuid import UUID

from sqlalchemy import text

import core.services.edge_builder as edge_builder
import core.services.edge_builder_logs as edge_builder_logs
from core.models.composio_cache import ToolExecutionLog
from tests.test_prd256_fix_edge_builder_off_the_loop import _FakeDB, _FakeEmbedder

WINDOW = timedelta(days=30)
QUERY = "send the weekly sales summary to the team"
ASKED_ACTION = "platform_update_agent"
DIMENSION = 16
CLUSTER = 1


def _log(db, ws, *, status="success", source="production", action="GMAIL_SEND_EMAIL", app="GMAIL"):
    row = ToolExecutionLog(workspace_id=UUID(ws), app_name=app, action_name=action, status=status,
                           user_query=QUERY, telemetry_source=source,
                           executed_at=datetime.utcnow() - timedelta(hours=1))
    db.add(row)
    db.flush()
    if source is None:  # the column defaults to 'production' on insert
        db.execute(text("UPDATE tool_execution_logs SET telemetry_source = NULL WHERE id = :id"), {"id": row.id})
    return row.id


class _Vectors(dict):
    """Every query embeds to the same vector."""

    def __missing__(self, _query):
        return [1.0] * DIMENSION


def _wire(monkeypatch, db_session):
    """build_edges over the real loader on ``db_session``; returns what it loaded and wrote."""
    seen = {"loaded": [], "affinities": []}
    store = {"open": 0, "open_at_embed": [], "clusters": [], "ids": count(1)}

    @contextmanager
    def _session():
        yield _FakeDB(store)

    def _load(_db, cutoff, workspace_id=None):
        seen["loaded"] = edge_builder_logs.load_logs(db_session, cutoff, workspace_id=workspace_id)
        return seen["loaded"]

    def _write_affinities(_db, affinities):
        seen["affinities"].extend(affinities)
        return len(affinities)

    monkeypatch.setattr(edge_builder, "get_db_session", _session)
    monkeypatch.setattr(edge_builder, "_load_logs", _load)
    monkeypatch.setattr(edge_builder, "get_embedding_manager", lambda: _FakeEmbedder(_Vectors(), store))
    monkeypatch.setattr(edge_builder, "upsert_clusters",
                        lambda _db, _embedded: {index: CLUSTER for index in range(len(seen["loaded"]))})
    monkeypatch.setattr(edge_builder, "_upsert_edges", lambda _db, edges: len(edges))
    monkeypatch.setattr(edge_builder, "_upsert_global_edges", lambda _db, _edges: 0)
    monkeypatch.setattr(edge_builder, "_upsert_failed_after_edges", lambda _db, edges: len(edges))
    monkeypatch.setattr(edge_builder, "_upsert_affinities", _write_affinities)
    return seen


def test_eval_replay_seeded_and_asked_rows_are_left_out_and_signal_and_untagged_rows_are_read(
        db_session, seed_workspace):
    ws = seed_workspace(name="edge-builder-filter")
    read = {
        _log(db_session, ws),
        _log(db_session, ws, source="synthetic_signal", status="gap", action="__tool_gap__", app="PLATFORM"),
        _log(db_session, ws, source=None, status="error"),
    }
    for source in ("eval", "replay", "synthetic"):
        _log(db_session, ws, source=source, status="error")
    _log(db_session, ws, status="asked", action=ASKED_ACTION, app="PLATFORM")

    logs = edge_builder_logs.load_logs(db_session, datetime.utcnow() - WINDOW, workspace_id=ws)

    assert {log["id"] for log in logs} == read


def test_an_asked_production_row_gives_no_log_and_no_fails_for_intent(db_session, seed_workspace, monkeypatch):
    ws = seed_workspace(name="edge-builder-asked")
    asked = {_log(db_session, ws, status="asked", action=ASKED_ACTION, app="PLATFORM") for _ in range(3)}
    _log(db_session, ws)  # one call that ran, so the window is not empty
    assert len(asked) == 3 >= edge_builder._SAMPLE_FLOOR  # enough to make an affinity, were they read
    seen = _wire(monkeypatch, db_session)

    summary = asyncio.run(edge_builder.build_edges(window=WINDOW, workspace_id=ws))

    assert not summary.failed and summary.logs_processed == 1
    assert not asked & {log["id"] for log in seen["loaded"]}
    assert [a for a in seen["affinities"] if a["affinity_type"] == "fails_for_intent"] == []
    assert all(a["action_name"] != ASKED_ACTION for a in seen["affinities"])
