"""PRD-248 S5 — four more decision points, judged beside the platform in shadow.

The question shapes for each hook; the engine's fire-and-forget helper on a
running loop and from a plain sync call (a daemon thread); each hook's row —
what the platform decided next to what the engine would have — and its
agreement rule; the dials (no live mode: ``live`` reads as ``shadow``); the
scorer's summary for the new purposes; and, at the call sites, that with the
dial off the matcher, grant creation, report creation and heartbeat dispatch
never call the engine.

PURE tests: fake engines and backends, no HTTP, no database.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from config import config
from core.llm.decisions import judgements as J
from core.llm.decisions.engine import DecisionEngine, _SHADOW_TASKS
from core.llm.decisions.questions import DecisionAnswer, DecisionResult

WS = "11111111-1111-1111-1111-111111111111"


def _result(**answers: DecisionAnswer) -> DecisionResult:
    return DecisionResult(answers=answers, provider="fake", model="fake-1", latency_ms=240, input_tokens=700)


def _choice(option: str, p: float = 0.9) -> DecisionAnswer:
    return DecisionAnswer(type="choice", choice=option, probabilities={option: p}, confidence=p)


def _noul(p: float) -> DecisionAnswer:
    return DecisionAnswer(type="noul", noul=p)


def _score(index: float, p: float = 0.8) -> DecisionAnswer:
    return DecisionAnswer(type="score", score=index, probabilities=None, confidence=p)


class _Engine:
    """A fake engine: canned decide(), rows captured, dials as given."""

    def __init__(self, result: Optional[DecisionResult] = None, exc: Optional[Exception] = None, **modes: str):
        self.result, self.exc = result, exc
        self.rows: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []
        self._dials = SimpleNamespace(
            ticket_assign_mode="shadow", session_end_mode="shadow",
            hold_risk_mode="shadow", report_triage_mode="shadow", **modes,
        )

    def dials(self):
        return self._dials

    async def decide(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.result

    def record_shadow(self, row):
        self.rows.append(dict(row))

    def shadow(self, coro, purpose="shadow"):
        # run inline on a private loop so tests can assert immediately
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()
        return True


# --------------------------------------------------------------------------- #
# Question shapes
# --------------------------------------------------------------------------- #


def test_assignment_questions_are_one_choice_over_the_roster_plus_none():
    q = J.assignment_questions([("Jim", "Writes board packs"), ("Atlas", ""), ("jim", "dup"), ("", "skip")])
    assert list(q) == ["assignee"]
    assert q["assignee"].options == ["Jim", "Atlas", "none"]
    assert q["assignee"].criteria["Jim"] == "Writes board packs" and q["assignee"].criteria["Atlas"] is None
    with pytest.raises(ValueError):
        J.assignment_questions([])


def test_session_hold_and_report_questions_have_the_documented_shapes():
    s = J.session_end_questions()
    assert set(s) == {"work_complete", "nothing_done", "needs_owner"} and all(q.to_wire()["type"] == "noul" for q in s.values())
    h = J.hold_questions()
    assert h["risk"].to_wire()["type"] == "score" and len(h["risk"].criteria) == 5
    assert set(h["intent"].options) == set(J.INTENT_CRITERIA) and h["owner_can_judge"].to_wire()["type"] == "noul"
    r = J.report_questions()
    assert r["needs_attention"].to_wire()["type"] == "noul" and set(r["severity"].options) == set(J.SEVERITY_CRITERIA)


def test_states_carry_only_the_fields_the_question_needs():
    st = J.assignment_state(title="Draft the board pack", description="x" * 5000, role="writer", required_tools=["docs"])
    assert st["task"] == "Draft the board pack" and len(st["details"]) == J.TEXT_MAX_CHARS
    assert st["role_wanted"] == "writer" and st["tools_needed"] == ["docs"]
    se = J.session_end_state(title="t", description="d", final_text="done", attempt=3, exit_reason="success", files_touched=2, denials=1)
    assert se["attempt"] == 3 and se["files_written"] == 2 and se["commands_refused"] == 1
    hs = J.hold_state(kind="question", subject_type="board_task", tool_name="bash", question_md="Allow `comm -23 a b`?", options=["allow", "deny"], reason="outside the allowlist")
    assert hs["tool"] == "bash" and hs["answers_offered"] == ["allow", "deny"] and "allowlist" in hs["why_it_was_raised"]
    rs = J.report_state(kind="report", title="Weekly", summary="all fine", status="ok", agent_name="OPS", report_type="standup", action_items=2)
    assert rs["from"] == "OPS" and rs["action_items"] == 2 and "recommendations" not in rs


# --------------------------------------------------------------------------- #
# The engine's fire-and-forget helper
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_shadow_runs_as_a_task_on_the_running_loop():
    done = asyncio.Event()

    async def work():
        done.set()

    engine = DecisionEngine(settings_reader=lambda c, k, d: None)
    assert engine.shadow(work(), purpose="t") is True
    await asyncio.wait_for(done.wait(), timeout=2)


def test_shadow_runs_on_a_daemon_thread_from_a_sync_call():
    flag = threading.Event()

    async def work():
        flag.set()

    engine = DecisionEngine(settings_reader=lambda c, k, d: None)
    assert engine.shadow(work(), purpose="t") is True
    assert flag.wait(timeout=2)


def test_shadow_never_raises_when_the_work_fails():
    async def boom():
        raise RuntimeError("down")

    engine = DecisionEngine(settings_reader=lambda c, k, d: None)
    assert engine.shadow(boom(), purpose="t") is True
    time.sleep(0.2)  # the thread swallows the error


def test_dials_have_no_live_mode_for_the_extra_hooks():
    values = {"ticket_assign_mode": "live", "session_end_mode": "shadow", "hold_risk_mode": "nonsense", "report_triage_mode": "OFF"}
    d = DecisionEngine(settings_reader=lambda c, k, default: values.get(k)).dials()
    assert d.ticket_assign_mode == "shadow" and d.session_end_mode == "shadow"
    assert d.hold_risk_mode == "off" and d.report_triage_mode == "off" and d.any_on is True
    assert DecisionEngine(settings_reader=lambda c, k, d: None).dials().any_on is False


# --------------------------------------------------------------------------- #
# The four rows
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_assignment_row_records_agreement_and_platform_rank():
    engine = _Engine(result=_result(assignee=_choice("Atlas", 0.8)))
    row = await J.shadow_assignment(
        engine, workspace_id=WS, task_id=41, title="Draft the board pack", description="", role="writer",
        required_tools=[], candidates=[("Jim", "Writes"), ("Atlas", "Plans")], platform_ranked=["Jim", "Atlas"],
    )
    assert row["purpose"] == "ticket_assign" and row["platform_top"] == "Jim" and row["candidates"] == 2
    assert row["jev_pick"] == "Atlas" and row["agree"] is False and row["jev_pick_platform_rank"] == 2
    assert row["jev_confidence"] == 0.8 and row["latency_ms"] == 240 and engine.rows == [row]
    assert engine.calls[0]["purpose"] == "ticket_assign" and set(engine.calls[0]["questions"]) == {"assignee"}

    agree = await J.shadow_assignment(
        _Engine(result=_result(assignee=_choice("jim"))), workspace_id=WS, task_id=1, title="t", description="",
        role=None, required_tools=[], candidates=[("Jim", "")], platform_ranked=["Jim"],
    )
    assert agree["agree"] is True and agree["jev_pick_platform_rank"] == 1

    none = await J.shadow_assignment(
        _Engine(result=_result(assignee=_choice("none"))), workspace_id=WS, task_id=1, title="t", description="",
        role=None, required_tools=[], candidates=[("Jim", "")], platform_ranked=["Jim"],
    )
    assert none["agree"] is False and none["jev_pick_platform_rank"] is None

    miss = await J.shadow_assignment(
        _Engine(result=None), workspace_id=WS, task_id=1, title="t", description="", role=None,
        required_tools=[], candidates=[("Jim", "")], platform_ranked=["Jim"],
    )
    assert miss["error"] == "no_result" and "agree" not in miss


@pytest.mark.asyncio
async def test_session_end_row_derives_a_verdict():
    result = _result(work_complete=_noul(0.2), nothing_done=_noul(0.9), needs_owner=_noul(0.1))
    engine = _Engine(result=result)
    row = await J.shadow_session_end(
        engine, workspace_id=WS, task_id=177, attempt=110, title="Ethiopian blog post", description="...",
        final_text="110th dispatch, ticket unchanged. #177 is complete; no action taken.", exit_reason="success",
        files_touched=0, denials=0, platform_status="success",
    )
    assert row["purpose"] == "session_end" and row["jev_verdict"] == "nothing_done" and row["attempt"] == 110
    assert row["platform_status"] == "success" and row["final_preview"].startswith("110th dispatch")
    assert J.session_end_verdict(_result(work_complete=_noul(0.9), nothing_done=_noul(0.1), needs_owner=_noul(0.2))) == "complete"
    assert J.session_end_verdict(_result(work_complete=_noul(0.1), nothing_done=_noul(0.1), needs_owner=_noul(0.9))) == "needs_owner"
    assert J.session_end_verdict(_result(work_complete=_noul(0.3), nothing_done=_noul(0.2), needs_owner=_noul(0.2))) == "incomplete"
    assert J.session_end_verdict(_result()) is None


@pytest.mark.asyncio
async def test_hold_row_carries_the_grant_id_risk_level_and_intent():
    engine = _Engine(result=_result(risk=_score(0.2, 0.9), intent=_choice("inspect_files"), owner_can_judge=_noul(0.15)))
    row = await J.shadow_hold(
        engine, workspace_id=WS, grant_id=275, kind="question", subject_type="board_task", subject_id=165,
        tool_name="bash", risk_tier=None, question_md="Allow `comm -23 declared.txt used.txt`?",
        options=["allow", "deny"], reason="'comm' is outside this ticket's Bash allowlist", agent_id=58,
    )
    assert row["purpose"] == "hold_risk" and row["grant_id"] == 275 and row["subject_id"] == "165"
    assert row["platform_decision"] == "ask" and row["jev_risk_level"] == 1.2 and row["jev_intent"] == "inspect_files"
    assert row["jev_owner_can_judge"] == 0.15 and row["jev_risk_confidence"] == 0.9
    state = engine.calls[0]["state"]
    assert state["kind"] == "question" and state["tool"] == "bash" and state["answers_offered"] == ["allow", "deny"]


@pytest.mark.asyncio
async def test_report_row_records_attention_and_severity_beside_the_platform_action():
    engine = _Engine(result=_result(needs_attention=_noul(0.85), severity=_choice("needs_a_decision", 0.7)))
    row = await J.shadow_report_triage(
        engine, workspace_id=WS, kind="heartbeat:agent", subject_id=763, title="OPS Heartbeat",
        summary="doc 455 ingest ok (18 chunks)", status="ok", agent_name="OPS", agent_id=267,
        report_type=None, platform_action="report_to=workspace",
    )
    assert row["purpose"] == "report_triage" and row["jev_needs_attention"] == 0.85
    assert row["jev_severity"] == "needs_a_decision" and row["platform_action"] == "report_to=workspace"
    failed = await J.shadow_report_triage(
        _Engine(exc=RuntimeError("down")), workspace_id=WS, kind="report", subject_id="9", title="t", summary="s",
        status="ok", agent_name="A", report_type="standup", platform_action="report_submitted",
    )
    assert "RuntimeError" in failed["error"]


# --------------------------------------------------------------------------- #
# The scorer knows the new purposes
# --------------------------------------------------------------------------- #


def test_scorer_summarises_the_extra_purposes():
    from scripts.eval.decision_shadow import score as scorer

    rows = [
        {"ts": 1.0, "purpose": "ticket_assign", "provider": "fake", "model": "m", "latency_ms": 200, "agree": True,
         "jev_pick_platform_rank": 1, "answers": {"assignee": {"type": "choice", "choice": "Jim"}}},
        {"ts": 2.0, "purpose": "ticket_assign", "provider": "fake", "model": "m", "latency_ms": 400, "agree": False,
         "jev_pick_platform_rank": 3, "answers": {"assignee": {"type": "choice", "choice": "Atlas"}}},
        {"ts": 3.0, "purpose": "session_end", "provider": "fake", "model": "m", "latency_ms": 300, "platform_status": "success",
         "jev_verdict": "nothing_done", "answers": {"nothing_done": {"type": "noul", "noul": 0.9}, "work_complete": {"type": "noul", "noul": 0.2}}},
        {"ts": 4.0, "purpose": "hold_risk", "error": "no_result"},
        {"ts": 5.0, "purpose": "hold_risk", "provider": "fake", "model": "m", "latency_ms": 250, "platform_decision": "ask",
         "jev_intent": "inspect_files", "answers": {"risk": {"type": "score", "score": 0.2}, "intent": {"type": "choice", "choice": "inspect_files"}}},
    ]
    d = scorer.summary_dict(rows)
    assert set(d["purposes"]) == {"ticket_assign", "session_end", "hold_risk"}
    ta = d["purposes"]["ticket_assign"]
    assert ta["agreement"] == {"rate": 0.5, "n": 2} and ta["jev_pick_platform_rank_mean"] == 2.0
    assert ta["questions"]["assignee"]["picks"] == {"Jim": 1, "Atlas": 1}
    se = d["purposes"]["session_end"]
    assert se["questions"]["nothing_done"]["share_yes"] == 1.0 and se["jev_verdict"] == {"nothing_done": 1}
    hr = d["purposes"]["hold_risk"]
    assert hr["rows"] == 2 and hr["scored"] == 1 and hr["questions"]["risk"]["mean_score"] == 0.2
    json.dumps(d)
    text = scorer.summarize_purpose("ticket_assign", rows[:2])
    assert "agreement with the platform: 50%" in text and "platform rank 2.0" in text


# --------------------------------------------------------------------------- #
# Call sites: off means no engine call
# --------------------------------------------------------------------------- #


class _OffEngine:
    def __init__(self):
        self.calls = 0
        self._dials = SimpleNamespace(
            ticket_assign_mode="off", session_end_mode="off", hold_risk_mode="off", report_triage_mode="off",
        )

    def dials(self):
        return self._dials

    def shadow(self, coro, purpose="shadow"):
        self.calls += 1
        coro.close()
        return True


class _ShadowEngine(_OffEngine):
    def __init__(self):
        super().__init__()
        self._dials = SimpleNamespace(
            ticket_assign_mode="shadow", session_end_mode="shadow", hold_risk_mode="shadow", report_triage_mode="shadow",
        )
        self.purposes: List[str] = []

    def shadow(self, coro, purpose="shadow"):
        self.purposes.append(purpose)
        return super().shadow(coro, purpose)


def _install(monkeypatch, engine):
    import core.llm.decisions as pkg

    monkeypatch.setattr(pkg, "get_decision_engine", lambda: engine)


def test_matcher_call_site_only_shadows_when_the_dial_is_on(monkeypatch):
    from modules.coordination import agent_matcher as am

    ranked = [am.MatchResult(agent_id=7, agent_name="Jim", total_score=0.9, tool_coverage=1, skill_match=1, model_fit=1, availability=1, history=0)]
    task = SimpleNamespace(id=41, title="t", description="d")
    agents = [SimpleNamespace(id=7, name="Jim", description="Writes", workspace_id=WS)]

    off = _OffEngine()
    _install(monkeypatch, off)
    am._shadow_assignment(task, agents, ranked, "writer", [])
    assert off.calls == 0

    on = _ShadowEngine()
    _install(monkeypatch, on)
    am._shadow_assignment(task, agents, ranked, "writer", [])
    assert on.purposes == ["ticket_assign"]
    am._shadow_assignment(task, agents, [], "writer", [])  # nothing ranked → nothing to judge
    assert on.purposes == ["ticket_assign"]


def test_grant_creation_call_site_only_shadows_when_the_dial_is_on(monkeypatch):
    from core.services import approval_grants as ag

    class _Db:
        def add(self, obj):
            pass

        def flush(self):
            pass

    off = _OffEngine()
    _install(monkeypatch, off)
    grant = ag.create_grant(_Db(), WS, subject_type="board_task", subject_id="165", kind="question", question_md="Allow?", options=["allow", "deny"])
    assert grant.kind == "question" and off.calls == 0

    on = _ShadowEngine()
    _install(monkeypatch, on)
    ag.create_grant(_Db(), WS, subject_type="board_task", subject_id="165", kind="question", question_md="Allow?")
    assert on.purposes == ["hold_risk"]


def test_session_end_report_and_heartbeat_call_sites_only_shadow_when_on(monkeypatch):
    from services import cli_host_service as chs
    from services import heartbeat_service as hb
    from services import report_service as rs

    task = SimpleNamespace(id=177, workspace_id=WS, title="t", description="d")
    payload = {"result_text": "done", "attempt": 2, "exit_reason": "success"}
    exec_result = {"status": "success"}

    off = _OffEngine()
    _install(monkeypatch, off)
    chs._shadow_session_end(task, {}, payload, exec_result, [], [])
    rs._shadow_report_triage(workspace_id=WS, report_id="9", agent_id=1, agent_name="A", title="t", summary="s", status="ok", report_type="standup", requires_approval=False, action_items=0, recommendations=0)
    hb._shadow_heartbeat_triage(workspace_id=WS, link_id=1, agent_id=1, agent_name="A", title="t", message="m", status="ok", source_type="agent", platform_action="report_to=workspace")
    assert off.calls == 0

    on = _ShadowEngine()
    _install(monkeypatch, on)
    chs._shadow_session_end(task, {}, payload, exec_result, ["a.md"], [{"tool": "bash"}])
    rs._shadow_report_triage(workspace_id=WS, report_id="9", agent_id=1, agent_name="A", title="t", summary="s", status="ok", report_type="standup", requires_approval=True, action_items=1, recommendations=0)
    hb._shadow_heartbeat_triage(workspace_id=WS, link_id=1, agent_id=1, agent_name="A", title="t", message="m", status="ok", source_type="agent", platform_action="report_to=auto")
    assert on.purposes == ["session_end", "report_triage", "report_triage"]
