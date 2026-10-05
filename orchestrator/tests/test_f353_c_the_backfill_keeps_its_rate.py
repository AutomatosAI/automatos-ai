"""F353 (issue #947): the backfill draws the documents made before F353, at a set rate.

* At most ``per_minute`` renders start a minute: a sleep of 60/per_minute seconds
  between two renders, none before the first.
* A Deliverable that has a picture is skipped; one whose render failed is skipped
  unless ``retry_failed``; a report counts as drawn when its picture is stored.
* ``dry_run`` draws nothing; one render that blows up is counted and the rest go on.
* It is a script run by hand, not a boot step.
"""
from __future__ import annotations

from pathlib import Path

from modules.documents.thumbnails import backfill as bf
from modules.documents.thumbnails.backfill import Candidate, backfill

WS = "00000000-0000-0000-0000-0000000000c1"


def _doc(n: int, extra=None) -> Candidate:
    return Candidate(f"00000000-0000-0000-0000-00000000000{n}", WS, "document", f"generated/{n}.pdf", extra or {})


def _run(candidates, **kwargs):
    sleeps, drawn = [], []

    def render_one(session_factory, workspace_id, output_id):
        drawn.append(output_id)
        return "rendered"

    outcomes = backfill(object, candidates, sleep=sleeps.append, render_one=render_one, **kwargs)
    return outcomes, sleeps, drawn


def test_renders_are_spaced_to_the_rate():
    outcomes, sleeps, drawn = _run([_doc(1), _doc(2), _doc(3), _doc(4)], per_minute=30)
    assert outcomes == {"rendered": 4}
    assert len(drawn) == 4
    assert sleeps == [2.0, 2.0, 2.0]


def test_drawn_and_failed_ones_are_skipped_unless_retrying_failures():
    done = _doc(1, {"thumbnail": {"file": "thumb_1.png"}})
    failed = _doc(2, {"thumbnail": {"failed": "not a readable PDF"}})
    fresh = _doc(3)
    outcomes, _, drawn = _run([done, failed, fresh])
    assert drawn == [fresh.id]
    assert outcomes == {"has-picture": 2, "rendered": 1}
    _, _, drawn = _run([done, failed, fresh], retry_failed=True)
    assert drawn == [failed.id, fresh.id]


def test_a_report_counts_as_drawn_when_its_picture_is_stored(monkeypatch):
    stored = {"00000000-0000-0000-0000-000000000001"}
    monkeypatch.setattr(bf, "load_thumbnail", lambda ws, oid: b"png" if oid in stored else None)
    reports = [Candidate(oid, WS, "report", "reports/a/x.md", {}) for oid in
               ("00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002")]
    _, _, drawn = _run(reports)
    assert drawn == ["00000000-0000-0000-0000-000000000002"]


def test_a_dry_run_draws_nothing_and_sleeps_not_at_all():
    outcomes, sleeps, drawn = _run([_doc(1), _doc(2)], dry_run=True)
    assert outcomes == {"would-draw": 2}
    assert sleeps == [] and drawn == []


def test_one_render_that_blows_up_is_counted_and_the_rest_go_on():
    def render_one(session_factory, workspace_id, output_id):
        if output_id.endswith("1"):
            raise RuntimeError("worker unreachable")
        return "rendered"

    outcomes = backfill(object, [_doc(1), _doc(2)], sleep=lambda s: None, render_one=render_one)
    assert outcomes == {"error": 1, "rendered": 1}


def test_the_backfill_is_a_script_never_a_boot_step():
    orchestrator = Path(__file__).resolve().parent.parent
    assert (orchestrator / "scripts" / "backfill_f353_document_thumbnails.py").is_file()
    assert "thumbnails.backfill" not in (orchestrator / "main.py").read_text(encoding="utf-8")
