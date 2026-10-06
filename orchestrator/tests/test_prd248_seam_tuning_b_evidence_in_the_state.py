"""PRD-248 tuning (6 Oct): session_end and report_triage read the evidence, not just the account.

Night 4: P(complete) was 0.92 on wrong runs and 0.73 on right ones, because the
state held only the agent's final message. Ticket #673 was asked for 17 Sep and
did 22 Sep, and read as complete at 0.93. Now the state carries the brief's dates
and the dates found in the work (and the ones missing from it), the deliverables'
names and first lines, the files written and the commands refused; "complete"
asks about that evidence, and a brief with dates gets its own dates question.
A report's state carries its action items, recommendations, attachments and the
tickets it links. The evidence is read in the shadow task, after the platform's
own decision, so nothing on the request path waits for it.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from config import config
from core.llm.decisions import evidence
from core.llm.decisions import judgements as J
from core.llm.decisions.questions import DecisionAnswer, DecisionResult
from services import decision_evidence

WS = "11111111-1111-1111-1111-111111111111"
BRIEF_673 = "Write the weekly facts post for 17 Sep and schedule it."


def _result(**answers: DecisionAnswer) -> DecisionResult:
    return DecisionResult(answers=answers, provider="fake", model="fake-1", latency_ms=200)


def _noul(p: float) -> DecisionAnswer:
    return DecisionAnswer(type="noul", noul=p)


class _Engine:
    def __init__(self, result=None):
        self.result, self.calls, self.rows = result, [], []

    async def decide(self, **kw):
        self.calls.append(kw)
        return self.result

    def record_shadow(self, row):
        self.rows.append(dict(row))


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #


def test_dates_are_read_in_the_forms_people_write_them():
    assert evidence.dates_in("For 17 Sep, Sept 18th, 2026 and 2026-09-19") == ["17 Sep", "18 Sep 2026", "19 Sep 2026"]
    assert evidence.dates_in("you may 2 things; on 2 May") == ["2 May"]
    assert evidence.dates_in("we decided 3 options") == []


def test_ticket_673_shows_the_asked_date_missing_from_the_work():
    found = evidence.date_evidence(BRIEF_673, ["Posted the weekly facts for 22 Sep.", "weekly-facts-2026-09-22.md"])
    assert found["brief_dates"] == ["17 Sep"]
    assert found["dates_in_the_work"] == ["22 Sep", "22 Sep 2026"]
    assert found["brief_dates_missing_from_the_work"] == ["17 Sep"]
    assert evidence.date_evidence("Due Sep 17 2026", ["for 17 Sep"])["brief_dates_missing_from_the_work"] == []
    assert evidence.date_evidence("Write the facts post", ["anything"]) == {}


# --------------------------------------------------------------------------- #
# session_end
# --------------------------------------------------------------------------- #


def _state(**over: Any) -> Dict[str, Any]:
    base = dict(
        title="Weekly facts", description=BRIEF_673, final_text="Done. Posted for 22 Sep.", attempt=1,
        exit_reason="success", files=["/ws/sessions/673/weekly-facts.md"],
        denials=[{"tool": "Bash", "subject": "curl x", "reason": "outside the allowlist"}],
        deliverables=[{"name": "weekly-facts.md", "type": "document", "first_lines": "# Facts for 22 Sep"}],
        asked_on="15 Sep 2026", finished_on="22 Sep 2026",
    )
    return J.session_end_state(**{**base, **over})


def test_the_session_state_carries_the_evidence_beside_the_final_message():
    state = _state()
    assert state["brief"] == BRIEF_673 and state["asked_on"] == "15 Sep 2026" and state["finished_on"] == "22 Sep 2026"
    assert state["deliverables"] == [{"name": "weekly-facts.md", "type": "document", "first_lines": "# Facts for 22 Sep"}]
    assert state["files_written"] == ["673/weekly-facts.md"]
    assert state["commands_refused"] == ["Bash: curl x (outside the allowlist)"]
    assert state["brief_dates_missing_from_the_work"] == ["17 Sep"]
    assert state["final_message"].startswith("Done.")


def test_complete_asks_about_the_evidence_and_dates_get_their_own_question():
    plain = J.session_end_questions()
    assert set(plain) == {"work_complete", "nothing_done", "needs_owner"}
    assert "deliverables" in plain["work_complete"].instructions and "final message" not in plain["work_complete"].instructions
    dated = J.session_end_questions(brief_has_dates=True)
    assert dated["dates_match"].to_wire()["type"] == "noul"


def test_wrong_dates_make_the_verdict_incomplete():
    done_wrong_day = _result(work_complete=_noul(0.9), nothing_done=_noul(0.1), needs_owner=_noul(0.1), dates_match=_noul(0.2))
    assert J.session_end_verdict(done_wrong_day) == "incomplete"
    done_right_day = _result(work_complete=_noul(0.9), nothing_done=_noul(0.1), needs_owner=_noul(0.1), dates_match=_noul(0.8))
    assert J.session_end_verdict(done_right_day) == "complete"


@pytest.mark.asyncio
async def test_the_session_row_keeps_its_fields_and_adds_the_evidence_counts():
    engine = _Engine(_result(work_complete=_noul(0.9), nothing_done=_noul(0.1), needs_owner=_noul(0.1), dates_match=_noul(0.1)))
    row = await J.shadow_session_end(
        engine, workspace_id=WS, task_id=673, attempt=1, title="Weekly facts", description=BRIEF_673,
        final_text="Done. Posted for 22 Sep.", exit_reason="success", files=["a.md", "b.md"], denials=[],
        deliverables=[{"name": "a.md", "type": "document"}], asked_on="15 Sep 2026", finished_on="22 Sep 2026",
        platform_status="success",
    )
    assert {"task_id", "attempt", "title", "final_preview", "exit_reason", "platform_status"} <= set(row)
    assert row["files_touched"] == 2 and row["denials"] == 0 and row["deliverables"] == 1
    assert row["brief_dates"] == ["17 Sep"] and row["brief_dates_missing_from_the_work"] == ["17 Sep"]
    assert row["jev_verdict"] == "incomplete" and row["jev_dates_match"] == 0.1
    assert set(engine.calls[0]["questions"]) == {"work_complete", "nothing_done", "needs_owner", "dates_match"}


def test_a_deliverables_first_lines_are_read_from_the_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    folder = tmp_path / WS / "sessions" / "673"
    folder.mkdir(parents=True)
    (folder / "facts.md").write_text("\n# Facts for 22 Sep\n\nCoffee is a fruit.\nMore\nAnd more\n")
    (folder / "page.html").write_text("<html><body><h1>Price list</h1><p>Espresso 3.00</p></body></html>")
    (folder / "invoice.pdf").write_bytes(b"%PDF-1.7 binary")

    made = decision_evidence.session_deliverables(WS, [
        {"file_path": "sessions/673/facts.md", "title": "facts.md", "artifact_type": "document"},
        {"file_path": "sessions/673/page.html", "title": "page.html", "artifact_type": "document"},
        {"file_path": "sessions/673/invoice.pdf", "title": "invoice.pdf", "artifact_type": "document"},
        {"file_path": "sessions/673/gone.md", "title": "gone.md", "artifact_type": "document"},
    ])
    assert made[0]["first_lines"] == "# Facts for 22 Sep\nCoffee is a fruit.\nMore"
    assert "Price list" in made[1]["first_lines"] and "<h1>" not in made[1]["first_lines"]
    assert made[2]["first_lines"] == "" and made[3]["first_lines"] == ""


def test_the_session_is_judged_once_its_result_has_landed_with_its_deliverables(monkeypatch):
    """``apply_result`` is wrapped: the judgement reads what the ticket records after the
    result landed (deliverables, files, refusals), never runs for a result that hands the
    ticket back to the queue, and never runs with the dial off."""
    from services import cli_host_service as chs
    from services import session_end_shadow as ses

    assert chs.apply_result.__wrapped__  # the decorator, not a call inside apply_result
    captured: List[Any] = []

    class _On:
        def __init__(self, mode="shadow"):
            self.mode = mode

        def dials(self):
            return SimpleNamespace(session_end_mode=self.mode)

        def shadow(self, coro, purpose="shadow"):
            captured.append(coro.cr_frame.f_locals)
            coro.close()
            return True

    task = SimpleNamespace(
        id=673, workspace_id=WS, title="Weekly facts", description=BRIEF_673,
        created_at=datetime(2026, 9, 15, 9, tzinfo=timezone.utc),
        runtime_ref={"files_touched": ["a.md"], "permission_denials": [{"tool": "Bash", "subject": "curl x"}],
                     "deliverables": [{"file_path": "sessions/673/a.md", "title": "a.md", "artifact_type": "document"}]},
    )

    class _Db:
        def query(self, model):
            return self

        def filter(self, *criteria):
            return self

        def first(self):
            return task

    async def landed(db, host, task_id, payload):
        return {"applied": True, "status": "done"}

    import core.llm.decisions as pkg

    wrapped = ses.judged_after_landing(landed)
    monkeypatch.setattr(pkg, "get_decision_engine", lambda: _On())
    asyncio.run(wrapped(_Db(), None, 673, {"result_text": "Done", "attempt": 1, "status": "success"}))
    asyncio.run(wrapped(_Db(), None, 673, {"status": "usage_limit"}))
    monkeypatch.setattr(pkg, "get_decision_engine", lambda: _On("off"))
    asyncio.run(wrapped(_Db(), None, 673, {"result_text": "Done", "status": "success"}))

    (frame,) = captured
    values = frame["values"]
    assert frame["deliverable_refs"][0]["file_path"] == "sessions/673/a.md"
    assert values["asked_on"] == "15 Sep 2026" and values["files"] == ["a.md"]
    assert values["denials"][0]["subject"] == "curl x" and values["platform_status"] == "success"


# --------------------------------------------------------------------------- #
# report_triage
# --------------------------------------------------------------------------- #


def test_the_report_state_carries_what_it_asks_attaches_and_was_written_for():
    state = J.report_state(
        kind="report", title="Weekly facts done", summary="Posted the facts for 22 Sep.", status="ok",
        agent_name="Writer", report_type="task", action_items=[{"title": "Approve the post"}, "Check the image"],
        recommendations=[{"text": "Post on Fridays"}], attachments=[{"name": "facts.md"}], requires_approval=True,
        linked_tickets=[{"title": "Weekly facts", "brief": BRIEF_673, "asked_on": "15 Sep 2026"}],
    )
    assert state["action_items"] == ["Approve the post", "Check the image"]
    assert state["recommendations"] == ["Post on Fridays"] and state["attachments"] == ["facts.md"]
    assert state["waits_for_approval"] is True
    assert state["linked_tickets"][0]["asked_on"] == "15 Sep 2026"
    assert state["brief_dates_missing_from_the_work"] == ["17 Sep"]
    assert J.report_state(kind="heartbeat:agent", title="t", summary="s", status="ok", agent_name="A",
                          report_type=None, action_items=2)["action_items"] == 2  # a count still reads


def test_a_report_with_tickets_asks_whether_it_matches_them():
    assert "matches_the_ask" not in J.report_questions()
    assert J.report_questions(has_tickets=True)["matches_the_ask"].to_wire()["type"] == "noul"


def test_the_report_call_site_reads_the_linked_tickets_in_the_shadow_task(monkeypatch):
    engine = _Engine(_result(needs_attention=_noul(0.7), matches_the_ask=_noul(0.2)))
    monkeypatch.setattr(decision_evidence, "linked_tickets",
                        lambda ws, ids: [{"title": "Weekly facts", "brief": BRIEF_673, "asked_on": "15 Sep 2026"}] if ids else [])
    row = asyncio.run(decision_evidence.shadow_report_triage(
        engine, linked_task_ids=[673], workspace_id=WS, kind="report", subject_id="9", title="Weekly facts done",
        summary="Posted for 22 Sep.", status="ok", agent_name="Writer", report_type="task",
        platform_action="report_submitted",
    ))
    assert row["linked_tickets"] == 1 and row["jev_matches_the_ask"] == 0.2
    assert engine.calls[0]["state"]["linked_tickets"][0]["title"] == "Weekly facts"


def test_a_report_is_triaged_once_it_is_written_with_its_lists(monkeypatch):
    """``create_report`` is wrapped: the triage gets the report's id and its lists from the
    call, after the report is written, and only for a report that was written."""
    from services import report_service as rs

    assert rs.ReportService.create_report.__wrapped__
    seen: List[Dict[str, Any]] = []
    monkeypatch.setattr(rs, "_shadow_report_triage", lambda **kw: seen.append(kw))

    async def create_report(self, agent_id, agent_name, title, content, report_type="standup", status="ok",
                            summary=None, metrics=None, attachments=None, heartbeat_result_id=None,
                            recommendations=None, action_items=None, linked_task_ids=None, requires_approval=False):
        return {"success": title != "fails", "report_id": "9"}

    wrapped = rs._triaged_after_report(create_report)
    me = SimpleNamespace(workspace_id=WS)
    asyncio.run(wrapped(me, 3, "Writer", "Weekly facts done", "Posted for 22 Sep.", action_items=[{"title": "Approve"}],
                        linked_task_ids=[673], requires_approval=True))
    asyncio.run(wrapped(me, 3, "Writer", "fails", "x"))

    (call,) = seen
    assert call["report_id"] == "9" and call["linked_task_ids"] == [673] and call["action_items"] == [{"title": "Approve"}]
    assert call["summary"] == "Posted for 22 Sep." and call["requires_approval"] is True

