"""PRD-204 S6 -- run-level verdict: judge-stubbed scoring math, threshold
boundaries (0.79 fail / 0.80 pass), output-hash cache hit, cost attribution
(request_type='watch' + watch-scoped execution_id), and the DB-backed
collect/apply round-trip.

The judge is ALWAYS stubbed (no live model anywhere in this suite). Pure
tests run without a DB; the collect/apply tests use the stage-1 DB fixture
pattern and skip cleanly when Postgres is absent.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from modules.coordination.run_verdict import (
    ALL_DIMENSIONS,
    LLM_DIMENSIONS,
    MECHANICS_DIMENSION,
    RunOutputBundle,
    RunVerdict,
    RunVerdictService,
    _default_llm_factory,
    build_run_judge_prompt,
    mission_mechanics,
    weighted_mean,
)

FROZEN_NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


def _bundle(text="The quarterly report: revenue up 12%.", state="completed", mech=1.0):
    return RunOutputBundle(
        text=text,
        kind="playbook_execution",
        terminal_state=state,
        mechanics_reliability=mech,
        executor_model="openai/gpt-4o-mini",
        empty=not text.strip(),
    )


def _watch_stub(**overrides):
    base = dict(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        target_type="playbook_execution",
        target_id="exec-abc123",
        title="Watch: quarterly report",
        success_criteria="Produce the quarterly revenue report",
        quality_threshold=0.8,
        created_by=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _StubResponse:
    def __init__(self, content, total_tokens=321):
        self.content = content
        self.usage = {"total_tokens": total_tokens}


class _StubLLM:
    """Judge stub: returns queued payloads, counts calls."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    async def generate_response(self, messages):
        self.calls += 1
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return _StubResponse(payload)


def _judge_json(**dims):
    import json

    body = {
        "business_usefulness": 0.9,
        "completeness": 0.9,
        "evidence_quality": 0.9,
        "clarity": 0.9,
        "actionability": 0.9,
        "confidence": 0.95,
        "reasoning": "Solid, grounded output.",
        "caveats": ["Numbers not independently verified"],
    }
    body.update(dims)
    return json.dumps(body)


def _score(service, watch, bundle, llm):
    return asyncio.run(
        service.score_run(None, watch, bundle=bundle, llm_factory=lambda w, m: llm)
    )


# ---------------------------------------------------------------------------
# Pure scoring math
# ---------------------------------------------------------------------------


def test_weighted_mean_equal_weights_and_clamping():
    flat = {dim: 0.6 for dim in ALL_DIMENSIONS}
    assert weighted_mean(flat) == pytest.approx(0.6)

    # Out-of-range judge values clamp; missing dims score 0.
    assert weighted_mean({"business_usefulness": 9.0}) == pytest.approx(1 / 6, abs=1e-4)
    assert weighted_mean({}) == 0.0


def test_reliability_dimension_is_mechanics_not_llm():
    """The judge never sets reliability -- it is folded in from mechanics."""
    service = RunVerdictService()
    watch = _watch_stub()
    llm = _StubLLM([_judge_json()])
    verdict = _score(service, watch, _bundle(mech=0.25), llm)

    assert verdict.dimension_scores[MECHANICS_DIMENSION] == 0.25
    expected = weighted_mean({**{d: 0.9 for d in LLM_DIMENSIONS}, MECHANICS_DIMENSION: 0.25})
    assert verdict.score == pytest.approx(expected)


def test_mission_mechanics_heuristic():
    all_good = [{"state": "verified", "attempts": 1}] * 4
    assert mission_mechanics(all_good, "completed") == pytest.approx(1.0)

    half_failed = [
        {"state": "verified", "attempts": 1},
        {"state": "failed", "attempts": 3},
    ]
    score = mission_mechanics(half_failed, "failed")
    assert 0.0 <= score < 0.6
    # Empty task list stays neutral-ish, never crashes.
    assert 0.0 <= mission_mechanics([], "completed") <= 1.0


# ---------------------------------------------------------------------------
# Threshold boundaries (PRD-204 S6: 0.79 fail / 0.80 pass)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,passes",
    [(0.79, False), (0.80, True), (0.8000001, True), (0.0, False), (1.0, True)],
)
def test_threshold_boundary(score, passes):
    verdict = RunVerdict(score=score)
    assert verdict.passes(0.8) is passes


def test_judge_failed_verdict_never_passes():
    verdict = RunVerdict(score=None, judge_failed=True)
    assert verdict.passes(0.0) is False
    assert "scoring unavailable" in verdict.as_text().lower()


# ---------------------------------------------------------------------------
# Judge flow: parse, retry on non-JSON, fail-soft after retries
# ---------------------------------------------------------------------------


def test_judge_scores_parsed_and_verdict_text_built():
    service = RunVerdictService()
    watch = _watch_stub()
    llm = _StubLLM([_judge_json(business_usefulness=0.7)])
    verdict = _score(service, watch, _bundle(), llm)

    assert llm.calls == 1
    assert verdict.judge_failed is False
    assert verdict.dimension_scores["business_usefulness"] == 0.7
    assert verdict.tokens_used == 321
    assert "Solid, grounded output." in verdict.as_text()
    assert "Caveats:" in verdict.as_text()


def test_non_json_then_valid_retries_once():
    service = RunVerdictService()
    watch = _watch_stub()
    llm = _StubLLM(["sorry, no JSON here", _judge_json()])
    verdict = _score(service, watch, _bundle(text="output " + uuid.uuid4().hex), llm)
    assert llm.calls == 2
    assert verdict.score is not None


def test_judge_exhaustion_returns_judge_failed():
    service = RunVerdictService()
    watch = _watch_stub()
    llm = _StubLLM([RuntimeError("model down"), RuntimeError("model down")])
    verdict = _score(service, watch, _bundle(text="output " + uuid.uuid4().hex), llm)
    assert verdict.judge_failed is True
    assert verdict.score is None


def test_empty_output_scores_deterministic_floor_without_llm():
    service = RunVerdictService()
    watch = _watch_stub()
    llm = _StubLLM([])  # any call would IndexError
    verdict = _score(service, watch, _bundle(text="", mech=0.6), llm)
    assert llm.calls == 0
    assert verdict.dimension_scores[MECHANICS_DIMENSION] == 0.6
    assert all(verdict.dimension_scores[d] == 0.0 for d in LLM_DIMENSIONS)
    assert verdict.score == pytest.approx(weighted_mean(verdict.dimension_scores))


# ---------------------------------------------------------------------------
# Output-hash cache (identical output -> zero LLM calls)
# ---------------------------------------------------------------------------


def test_cache_hit_on_identical_output_hash():
    service = RunVerdictService()
    watch = _watch_stub()
    text = "identical output " + uuid.uuid4().hex
    llm = _StubLLM([_judge_json()])

    first = _score(service, watch, _bundle(text=text), llm)
    second = _score(service, watch, _bundle(text=text), llm)

    assert llm.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.score == first.score

    # Different output on the SAME watch -> fresh judge call.
    llm.payloads = [_judge_json()]
    third = _score(service, watch, _bundle(text=text + " changed"), llm)
    assert llm.calls == 2
    assert third.cached is False

    assert RunVerdictService.clear_cache(watch.id) >= 2


# ---------------------------------------------------------------------------
# Cost attribution: request_type='watch', watch-scoped execution_id
# ---------------------------------------------------------------------------


def test_default_llm_factory_sets_watch_cost_attribution(monkeypatch):
    captured = {}

    def _fake_create_llm_manager(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(_tracking_ctx=dict(kwargs))

    import core.llm

    monkeypatch.setattr(core.llm, "create_llm_manager", _fake_create_llm_manager)

    watch = _watch_stub()
    llm = _default_llm_factory(watch, "openai/gpt-4o-mini")

    assert captured["service_name"] == "watch_verdict"
    assert captured["request_type"] == "watch"
    assert captured["workspace_id"] == watch.workspace_id
    assert llm._tracking_ctx["request_type"] == "watch"
    # Watch-scoped execution id: supervision cost lands on 'watch-<id>' --
    # NOT on the watched execution's rollup (the S7 rerun estimate sums that).
    assert llm._tracking_ctx["execution_id"] == f"watch-{watch.id}"


def test_judge_prompt_contains_criteria_and_output():
    prompt = build_run_judge_prompt(
        success_criteria="Ship the report", bundle=_bundle(text="THE OUTPUT")
    )
    assert "Ship the report" in prompt
    assert "THE OUTPUT" in prompt
    assert "reliability" in prompt  # mechanics context note
    assert '"business_usefulness"' in prompt


# ---------------------------------------------------------------------------
# DB-backed: collect (mission + playbook) and apply_verdict round-trip
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text

    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM watches LIMIT 1"))
            c.execute(text("SELECT 1 FROM recipe_executions LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"run verdict suite needs a reachable Postgres with schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(new_session):
    from sqlalchemy import text

    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": "prd204-run-verdict"},
    )
    s.commit()
    s.close()

    yield ws_id

    s = new_session.sweep()
    for stmt in (
        "DELETE FROM watch_events WHERE watch_id IN "
        "(SELECT id FROM watches WHERE workspace_id = CAST(:w AS uuid))",
        "DELETE FROM watches WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM recipe_executions WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM orchestration_tasks WHERE run_id IN "
        "(SELECT id FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid))",
        "DELETE FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
    ):
        s.execute(text(stmt), {"w": ws_id})
    s.commit()
    s.close()


def test_collect_playbook_output_and_mechanics(workspace, new_session):
    from core.models import WorkflowTemplate
    from core.models.core import RecipeExecution

    s = new_session()
    recipe = WorkflowTemplate(
        template_id=f"prd204-rv-{uuid.uuid4().hex[:10]}",
        name="verdict test playbook",
        description="prd204 run verdict",
        workspace_id=workspace,
        template_definition={"steps": []},
        steps=[{"step_id": "s1", "order": 1}],
        created_by="user_test",  # NOT NULL on workflow_recipes
    )
    s.add(recipe)
    s.commit()

    execution = RecipeExecution(
        execution_id=f"prd204-rv-{uuid.uuid4().hex[:10]}",
        recipe_id=recipe.id,
        workspace_id=workspace,
        status="completed",
        input_data={},
        output_data={"final_output": "Report delivered: revenue up 12%."},
        step_results=[
            {"step_id": "s1", "order": 1, "status": "completed",
             "agent_name": "writer", "output": "done", "retries": 0},
        ],
    )
    s.add(execution)
    s.commit()

    watch = SimpleNamespace(
        id=uuid.uuid4(),
        workspace_id=workspace,
        target_type="playbook_execution",
        target_id=execution.execution_id,
        title="w",
        success_criteria="deliver report",
    )
    bundle = RunVerdictService.collect_run_output(s, watch)
    assert bundle is not None
    assert bundle.kind == "playbook_execution"
    assert "revenue up 12%" in bundle.text
    assert bundle.terminal_state == "completed"
    assert 0.0 <= bundle.mechanics_reliability <= 1.0
    assert bundle.empty is False

    # Missing target -> None (caller parks the watch).
    gone = SimpleNamespace(
        id=uuid.uuid4(), workspace_id=workspace,
        target_type="playbook_execution", target_id="exec-nope",
    )
    assert RunVerdictService.collect_run_output(s, gone) is None
    s.close()


def test_collect_mission_output(workspace, new_session):
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    s = new_session()
    run = OrchestrationRun(
        workspace_id=workspace,
        goal="produce the launch plan",
        state="completed",
        created_by="user_test",
        output_summary={"headline": "Launch plan produced"},
    )
    s.add(run)
    s.commit()
    task = OrchestrationTask(
        run_id=run.id,
        title="draft plan",
        description="draft it",
        sequence_number=1,
        agent_role="writer",
        state="verified",
        state_type="blocked",  # TASK_STATE_TYPE[VERIFIED]
        output="The launch plan: phase one...",
    )
    s.add(task)
    s.commit()

    watch = SimpleNamespace(
        id=uuid.uuid4(),
        workspace_id=workspace,
        target_type="mission",
        target_id=str(run.id),
        title="w",
        success_criteria="produce the launch plan",
    )
    bundle = RunVerdictService.collect_run_output(s, watch)
    assert bundle is not None
    assert bundle.kind == "mission"
    assert "Launch plan produced" in bundle.text
    assert "The launch plan: phase one" in bundle.text
    assert bundle.mechanics_reliability == pytest.approx(1.0)
    s.close()


def test_apply_verdict_writes_score_verdict_and_scored_event(workspace, new_session):
    from core.models.watches import WatchEvent
    from core.models.watch_enums import WatchEventType
    from services.watch_service import WatchService

    s = new_session()
    watch = WatchService.create_watch(
        s,
        workspace_id=workspace,
        watch_type="playbook_execution",
        target_type="playbook_execution",
        target_id="exec-verdict-1",
        title="Watch: apply verdict",
        success_criteria="deliver",
        now=FROZEN_NOW,
    )
    s.commit()

    verdict = RunVerdict(
        score=0.83,
        dimension_scores={d: 0.83 for d in ALL_DIMENSIONS},
        reasoning="Good output.",
        caveats=["minor caveat"],
        output_hash="a" * 64,
    )
    event = RunVerdictService.apply_verdict(s, watch, verdict)
    s.commit()

    assert event is not None
    s.refresh(watch)
    assert watch.final_score == pytest.approx(0.83)
    assert "Good output." in watch.final_verdict
    assert "minor caveat" in watch.final_verdict

    # Idempotent per output hash: same verdict applied twice -> one event.
    again = RunVerdictService.apply_verdict(s, watch, verdict)
    s.commit()
    assert again is None
    events = (
        s.query(WatchEvent)
        .filter(
            WatchEvent.watch_id == watch.id,
            WatchEvent.event_type == WatchEventType.SCORED.value,
        )
        .all()
    )
    assert len(events) == 1
    assert events[0].score == pytest.approx(0.83)
    assert events[0].snapshot["dimension_scores"]["reliability"] == pytest.approx(0.83)
    s.close()


# ---------------------------------------------------------------------------
# Verdict notification formatting (display edge is the ONLY x10 place)
# ---------------------------------------------------------------------------


def test_verdict_notification_message_formats_score_x10():
    from services.watch_notifications import build_verdict_message, format_score_display

    assert format_score_display(0.83) == "8.3/10"
    assert format_score_display(None) == "unscored"

    watch = _watch_stub(quality_threshold=0.8)
    msg = build_verdict_message(watch, score=0.79, explanation="Close but incomplete.")
    assert "7.9/10" in msg
    assert "8.0/10" in msg  # the bar
    assert "Close but incomplete." in msg

    unscored = build_verdict_message(watch, score=None, explanation=None, terminal_state="failed")
    assert "failed" in unscored


def test_notify_watch_verdict_dispatches_through_shared_seam(monkeypatch):
    import services.watch_notifications as wn

    sent = []

    async def _capture(db, watch, *, event_type, title, message, status="ok"):
        sent.append({"event_type": event_type, "title": title,
                     "message": message, "status": status})
        return True

    monkeypatch.setattr(wn, "dispatch_watch_notification", _capture)

    watch = _watch_stub()
    ok = asyncio.run(
        wn.notify_watch_verdict(
            db=None, watch=watch, score=0.83, explanation="Nice.", passed=True
        )
    )
    assert ok is True
    assert sent[0]["event_type"] == "watch_verdict"
    assert "8.3/10" in sent[0]["message"]
    assert sent[0]["status"] == "ok"
