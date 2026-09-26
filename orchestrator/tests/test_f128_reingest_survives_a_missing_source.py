"""F128 — one missing source never stops the re-ingest run.

Refresh-2 retest, 09-23: ``reingest_documents.py --apply`` died on the first
document whose copy object storage no longer held. ``download_file`` raised the
HeadObject 404 as an uncaught ClientError, the traceback exited 1, and every
document after it (workspace 47aee314…, then the whole RAG workspace) was never
reached, in pass 1b and again in pass 1c. The plan had already failed to read
that source and still listed it to re-ingest. Now a key object storage doesn't
hold is NO SOURCE, with the key, like an upload missing from disk: the plan lists
it, ``--apply`` skips it before clearing anything, re-ingests the rest, and
exits 1 with the count.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reingest_documents.py"
spec = importlib.util.spec_from_file_location("reingest_documents_f128", _SCRIPT)
rd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rd          # dataclasses resolve their module by name
spec.loader.exec_module(rd)

REPORT = "s3://bucket/workspaces/ws/documents/900_report.md"
GONE = "s3://bucket/workspaces/ws/documents/901_gone.md"
BRIEF = "s3://bucket/workspaces/ws/documents/902_brief.md"


def key(uri):
    return uri.split("/", 3)[3]


ROWS = [
    (900, "ws", "report.md", "md", REPORT, ["alpha"]),
    (901, "ws", "gone.md", "md", GONE, ["beta"]),
    (902, "ws", "brief.md", "md", BRIEF, ["one"]),
]
OBJECTS = {key(REPORT): "alpha beta gamma delta", key(BRIEF): "one two three four"}


class Storage:
    """download_file as boto3 answers it: HeadObject first, a 404 for a key it doesn't hold."""

    def __init__(self, objects, code="404"):
        self.objects, self.code = objects, code

    def download_file(self, bucket, object_key, dest):
        if object_key not in self.objects:
            raise ClientError({"Error": {"Code": self.code, "Message": "Not Found"}}, "HeadObject")
        Path(dest).write_text(self.objects[object_key])


class Manager:
    s3_bucket = "bucket"

    def __init__(self, storage):
        self.s3_client, self.calls = storage, []

    def clear_chunks(self, document_id):
        self.calls.append(("clear", document_id))

    async def _process_document(self, document_id, path, file_type, s3_key=None, filename=None,
                                update_graph=True):
        self.calls.append(("process", document_id))


class Processor:
    def extract_text_from_file(self, path):
        return Path(path).read_text()


def stack_over(objects, code="404"):
    """The script's own stack (local_copy, measure) over a fake object storage."""
    stack = rd._Stack.__new__(rd._Stack)
    stack._managers = {"ws": Manager(Storage(objects, code))}
    stack._processor = Processor()
    return stack


class Session:
    def __init__(self, engine):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run_main(monkeypatch, argv, stack, rows=ROWS):
    import sqlalchemy
    import sqlalchemy.orm

    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *a, **k: None)
    monkeypatch.setattr(sqlalchemy.orm, "Session", Session)
    monkeypatch.setattr(rd, "_Stack", lambda: stack)
    monkeypatch.setattr(rd, "_rows", lambda db, ws, ids: [r for r in rows if ids is None or r[0] in ids])
    monkeypatch.setattr(sys, "argv", ["reingest_documents.py", "--below", "98", *argv])
    return rd.main()


# ── the plan: a key object storage doesn't hold is NO SOURCE ────────────────

def test_the_plan_lists_a_key_object_storage_no_longer_holds_as_no_source():
    found = rd.plan(ROWS, below=98, measure=stack_over(OBJECTS).measure)
    assert [(c.id, c.kept, c.source) for c in found] == [
        (900, 25, "object storage"), (901, None, "none"), (902, 25, "object storage")]


def test_the_dry_run_counts_it_as_no_source_names_the_key_and_changes_nothing(monkeypatch, capsys):
    stack = stack_over(OBJECTS)
    assert run_main(monkeypatch, [], stack) == 0
    assert stack._managers["ws"].calls == []
    out = capsys.readouterr().out
    assert "documents: 3 · under 98%: 2 to re-ingest · no source: 1" in out
    assert f"#901 [ws] gone.md — NO SOURCE (object storage has no {GONE}): the owner must upload it again" in out


# ── --apply: skip it untouched, re-ingest the rest, exit 1 with the count ───

def test_apply_skips_the_missing_source_reingests_the_rest_and_exits_1(monkeypatch, capsys):
    stack = stack_over(OBJECTS)
    assert run_main(monkeypatch, ["--apply"], stack) == 1
    assert stack._managers["ws"].calls == [("clear", 900), ("process", 900), ("clear", 902), ("process", 902)]
    out = capsys.readouterr().out
    assert f"#901 [ws] gone.md — NO SOURCE (object storage has no {GONE})" in out
    assert out.rstrip().endswith("no source: 1 not re-ingested (#901)")


def test_a_source_gone_between_the_plan_and_the_apply_is_skipped_before_anything_is_cleared(monkeypatch, capsys):
    stack = stack_over({**OBJECTS, key(GONE): "beta gamma"})
    real_plan = rd.plan

    def plan_then_lose_it(*args, **kwargs):
        found = real_plan(*args, **kwargs)
        del stack._managers["ws"].s3_client.objects[key(GONE)]
        return found

    monkeypatch.setattr(rd, "plan", plan_then_lose_it)
    assert run_main(monkeypatch, ["--apply"], stack) == 1
    assert stack._managers["ws"].calls == [("clear", 900), ("process", 900), ("clear", 902), ("process", 902)]
    out = capsys.readouterr().out
    assert f"#901 [ws] gone.md — NO SOURCE (object storage has no {GONE})" in out
    assert out.rstrip().endswith("no source: 1 not re-ingested (#901)")


def test_an_upload_missing_from_disk_or_never_recorded_counts_the_same(monkeypatch, capsys, tmp_path):
    lost = tmp_path / "lost.pdf"            # never written
    rows = [ROWS[0], (1, "ws", "lost.pdf", "pdf", str(lost), []), (2, "ws", "orphan.txt", "txt", None, [])]
    stack = stack_over(OBJECTS)
    assert run_main(monkeypatch, ["--apply"], stack, rows) == 1
    out = capsys.readouterr().out
    assert f"#1 [ws] lost.pdf — NO SOURCE (no file at {lost}): the owner must upload it again" in out
    assert "#2 [ws] orphan.txt — NO SOURCE (no path recorded)" in out
    assert out.rstrip().endswith("no source: 2 not re-ingested (#1, #2)")


def test_an_apply_with_every_source_present_exits_0(monkeypatch):
    stack = stack_over(OBJECTS)
    assert run_main(monkeypatch, ["--apply"], stack, [ROWS[0], ROWS[2]]) == 0
    assert [c for c in stack._managers["ws"].calls if c[0] == "process"] == [("process", 900), ("process", 902)]


# ── only a missing key is NO SOURCE: a refusal still stops the run ──────────

@pytest.mark.parametrize("code", ["AccessDenied", "NoSuchBucket", "InternalError"])
def test_a_download_refused_for_any_other_reason_still_stops_the_run(code):
    stack = stack_over({}, code)
    with pytest.raises(ClientError):
        with stack.local_copy("ws", GONE):
            pass
