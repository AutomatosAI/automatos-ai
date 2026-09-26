"""F181 (night 6) — an uploaded spreadsheet is copied into the workspace, where an
agent counts and totals it with code.

Ticket #1115 totalled the club's October orders from the first pages of the
export and said 47.06 kg. The file (440 rows) says 341 bags, about 100.3 kg of
green coffee, about 40.3 kg short. The file was only in the knowledge base,
read a page of chunks at a time, and python3 in the workspace found no file.
Now the upload writes documents/<name> (an Excel file as one CSV per sheet).
read_document names that copy with a row count made in code, and makes the
copy from the stored upload when it is missing. The documents inventory says
a spreadsheet is counted with code. The fixture is generated here, so no night
data is committed.
"""
from __future__ import annotations

import asyncio
import csv
import io
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

NAME = "harbourline-club-members-export-2026-09-26.csv"
COPY = "documents/harbourline-club-members-export-2026-09-26.csv"


def _club_export(rows=440) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["member_id", "name", "order_date", "coffee", "bags"])
    for i in range(rows):
        writer.writerow([1000 + i, f"Member {i}", "2026-10-05" if i < 380 else "2026-09-28",
                         "Kiambu" if i % 3 else "Harbour Blend", 1 + i % 2])
    return out.getvalue().encode()


@pytest.fixture
def worker(monkeypatch):
    """The workspace worker: what is written, and what documents/ holds."""
    import core.workspace_client as wc

    state = NS(written={}, listed=[], down=False)

    async def write_file(self, path, content):
        if state.down:
            return {"success": False, "error": "Workspace worker unreachable"}
        state.written[path] = content
        return {"success": True, "path": path}

    async def list_dir(self, path="."):
        return {"path": path, "entries": [{"name": name, "type": "file"} for name in state.listed]}

    monkeypatch.setattr(wc.WorkspaceClient, "write_file", write_file)
    monkeypatch.setattr(wc.WorkspaceClient, "list_dir", list_dir)
    return state


# ── the upload ──────────────────────────────────────────────────────────────

class _Upload:
    def __init__(self, filename, data):
        self.filename, self._data = filename, data

    async def read(self):
        return self._data

    async def seek(self, _pos):
        pass


class _Db:
    def __init__(self):
        self.added = []

    def query(self, *_a):
        return self

    def filter(self, *_a):
        return self

    def first(self):
        return None

    def add(self, obj):
        obj.id = 1007
        self.added.append(obj)

    def commit(self):
        pass

    def refresh(self, obj):
        obj.status = "completed"


class _Manager:
    async def _process_document(self, *args, **kwargs):
        pass


def _upload(monkeypatch, tmp_path, filename, data):
    import api.documents as documents
    from services import document_versions as dv

    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(documents, "get_document_manager", lambda ws: _Manager())
    monkeypatch.setattr(dv, "replaceable_document", lambda *a, **k: None)
    ctx = NS(workspace_id=uuid4(), user=NS(clerk_user_id="user_2"))
    return asyncio.run(documents.handle_request(ctx=ctx, file=_Upload(filename, data), description=None,
                                                tags=None, team_access=None, db=_Db()))


def test_an_uploaded_csv_is_copied_into_the_workspace_as_it_came(monkeypatch, tmp_path, worker):
    data = _club_export()
    out = _upload(monkeypatch, tmp_path, NAME, data)

    assert out.status == "completed"
    assert worker.written == {COPY: data.decode()}


def test_an_upload_the_worker_cannot_take_still_succeeds(monkeypatch, tmp_path, worker):
    worker.down = True
    out = _upload(monkeypatch, tmp_path, NAME, _club_export())
    assert out.status == "completed" and worker.written == {}


def test_an_excel_workbook_becomes_one_csv_per_sheet(worker):
    import openpyxl

    from services.spreadsheet_workspace import copy_to_workspace

    book = openpyxl.Workbook()
    book.active.title = "Orders"
    book.active.append(["member_id", "bags"])
    book.active.append([1000, 2])
    stock = book.create_sheet("Green stock (kg)")
    stock.append(["coffee", "kg"])
    stock.append(["Kiambu", 60])
    stock.append(["Huila", 12.5])
    out = io.BytesIO()
    book.save(out)

    copied = asyncio.run(copy_to_workspace(uuid4(), "Club orders Oct.xlsx", out.getvalue()))

    assert copied == [{"workspace_path": "documents/Club_orders_Oct.Orders.csv", "row_count": 1},
                      {"workspace_path": "documents/Club_orders_Oct.Green_stock_kg.csv", "row_count": 2}]
    assert worker.written["documents/Club_orders_Oct.Green_stock_kg.csv"] == "coffee,kg\nKiambu,60\nHuila,12.5\n"


# ── read_document ───────────────────────────────────────────────────────────

class _ReadDb:
    def __init__(self, doc, chunks):
        self.doc, self.chunks = doc, chunks

    def query(self, *_a):
        return self

    def filter(self, *_a):
        return self

    def first(self):
        return self.doc

    def execute(self, *_a, **_k):
        return NS(fetchall=lambda: self.chunks)


def _read(monkeypatch, tmp_path, *, chunks):
    import modules.rag.retrieval_filters as filters
    from modules.tools.discovery import handlers_documents

    stored = tmp_path / "3f9c.csv"
    stored.write_bytes(_club_export())
    doc = NS(id=1007, file_type="csv", file_path=str(stored), original_filename=NAME, filename=NAME,
             upload_date=None, last_accessed=None)
    monkeypatch.setattr(filters, "allowed_document_ids", lambda db, ids, f: {str(i) for i in ids})
    monkeypatch.setattr(handlers_documents, "_resolve_agent_team", lambda db, agent_id: None)
    return asyncio.run(handlers_documents.read_document(_ReadDb(doc, chunks), uuid4(), {"document_id": 1007}))


FIRST_PAGE = [NS(chunk_index=0, content="member_id,name,order_date,coffee,bags\n1000,Member 0,2026-10-05,…")]


def test_read_document_names_the_copy_and_counts_the_rows_in_code(monkeypatch, tmp_path, worker):
    result = _read(monkeypatch, tmp_path, chunks=FIRST_PAGE)

    assert (result["workspace_path"], result["row_count"]) == (COPY, 440)
    assert result["count_with_code"].startswith("This is a spreadsheet. Count or total it with code")
    assert worker.written == {COPY: _club_export().decode()}          # 1007 had no copy: made from its upload


def test_a_copy_already_there_is_not_written_again(monkeypatch, tmp_path, worker):
    worker.listed = [COPY.split("/", 1)[1]]
    result = _read(monkeypatch, tmp_path, chunks=FIRST_PAGE)
    assert result["workspace_path"] == COPY and worker.written == {}


def test_a_spreadsheet_not_yet_in_the_knowledge_base_still_has_its_copy(monkeypatch, tmp_path, worker):
    result = _read(monkeypatch, tmp_path, chunks=[])
    assert result["success"] is True and (result["workspace_path"], result["row_count"]) == (COPY, 440)


# ── the documents inventory ─────────────────────────────────────────────────

@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        session.execute(text("DROP TABLE IF EXISTS pg_temp.documents"))
        session.execute(text("CREATE TEMP TABLE documents (LIKE public.documents INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        session.execute(text("DROP TABLE IF EXISTS pg_temp.documents"))
        session.commit()
        session.close()


def test_the_inventory_says_a_spreadsheet_is_counted_with_code(db):
    from modules.context.sections.documents_inventory import documents_summary

    ws = str(uuid4())
    db.execute(text("INSERT INTO documents (id, workspace_id, filename, status, file_type) "
                    "VALUES (1, CAST(:ws AS uuid), 'brand-voice.md', 'completed', 'markdown')"), {"ws": ws})
    assert "counted or totalled with code" not in documents_summary(db, ws)
    db.execute(text("INSERT INTO documents (id, workspace_id, filename, status, file_type) "
                    "VALUES (2, CAST(:ws AS uuid), :name, 'completed', 'csv')"), {"ws": ws, "name": NAME})
    assert documents_summary(db, ws).endswith(
        "A spreadsheet (CSV or Excel) is counted or totalled with code, never searched: "
        "platform_read_document gives its copy's workspace_path and its row count.")
