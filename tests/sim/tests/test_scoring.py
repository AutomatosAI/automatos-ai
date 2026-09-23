"""Scores are explainable, findings are trace-backed, the store round-trips."""

import pytest

from tests.sim import score, store
from tests.sim.results import Check, ScenarioResult

T0, T1 = "2026-09-18T01:00:00+00:00", "2026-09-18T01:05:00+00:00"


def _task(**kw):
    base = dict(id="t", kind="task", started_at=T0, ended_at=T1, outcome="done", ok=True,
                artifacts={"deliverables": [{"id": 1, "content": "x"}], "reports": []},
                checks=(Check("outcome", True, ""),))
    return ScenarioResult(**{**base, **kw})


def test_clean_task_scores_full_marks_without_a_judge():
    s = score.score_scenario(_task(), {"cost_usd": 0.01, "calls": 3}, None)
    assert s["outcome"] == 1.0 and s["usability"] == 1.0 and s["deductions"] == []
    assert s["quality"] is None and "no judge" in s["quality_basis"]
    assert s["usefulness"] == 1.0 and s["usefulness_basis"] == "artifact+outcome"
    assert s["cost_usd"] == 0.01 and s["calls"] == 3 and s["duration_s"] == 300.0


def test_usability_deductions_are_listed():
    res = _task(errors=("boom",), notes=("dispatcher had not claimed the ticket after 60s; run-now sent",),
                asks=({"a": 1}, {"a": 2}, {"a": 3}), checks=(Check("outcome", True), Check("contains:x", False, "not found")))
    s = score.score_scenario(res, {}, None)
    assert s["usability"] == 0.2  # 1 - (0.25 error + 0.20 two extra asks + 0.25 nudge + 0.10 failed check)
    assert len(s["deductions"]) == 4
    assert any("run-now" in d for d in s["deductions"]) and any("questions" in d for d in s["deductions"])


def test_empty_chat_reply_is_half_the_usability():
    res = ScenarioResult(id="c", kind="chat", started_at=T0, ended_at=T1, outcome="ok", ok=True, chat=({"text": ""},))
    s = score.score_scenario(res, {}, None)
    assert s["usability"] == 0.5 and s["usefulness"] == 1.0


def test_judge_verdict_drives_quality_and_usefulness():
    s = score.score_scenario(_task(), {}, {"quality": 5, "usefulness": 3})
    assert (s["quality"], s["quality_basis"]) == (1.0, "judge") and (s["usefulness"], s["usefulness_basis"]) == (0.5, "judge")
    s2 = score.score_scenario(_task(checks=(Check("contains:a", True), Check("contains:b", False))), {}, None)
    assert s2["quality"] == 0.5 and s2["quality_basis"] == "must_contain"


def test_findings_are_named_and_trace_backed():
    kinds = {f["kind"] for f in score.findings(_task(outcome="timeout", ok=False, timeline=({"status": "in_progress"},)), {})}
    assert "stall" in kinds
    kinds = {f["kind"] for f in score.findings(_task(outcome="review", artifacts={"review_feedback": "needs work"}), {})}
    assert "review" in kinds
    bare = _task(artifacts={"deliverables": [], "reports": []}, asks=({"grant_id": 1},),
                 checks=(Check("contains:x", False, "not found"),))
    found = score.findings(bare, {"errors": 2, "by_model": {}})
    assert {f["kind"] for f in found} >= {"missing_artifact", "asks", "model_errors", "check"}
    assert all(f["basis"] == "trace" and f["scenario"] == "t" for f in found)


def test_pack_score_ignores_missing_rows_and_renders():
    scenarios = [
        {"id": "a", "kind": "task", "outcome": "done", "ok": True, "scores": {"outcome": 1.0, "usability": 1.0, "quality": None, "usefulness": 1.0, "cost_usd": 0.01, "calls": 2, "duration_s": 10}},
        {"id": "b", "kind": "chat", "outcome": "error", "ok": False, "scores": {"outcome": 0.0, "usability": 0.5, "quality": 0.5, "usefulness": 0.0, "cost_usd": 0.02, "calls": 1, "duration_s": 5}},
    ]
    card = score.score_pack(scenarios, 0.05)
    assert card == {"scenarios": 2, "ok": 1, "outcome_rate": 0.5, "usability": 0.75, "quality": 0.5, "usefulness": 0.5,
                    "cost_usd": 0.05, "attributed_usd": 0.03, "calls": 3, "duration_s": 15.0}
    run = {"run_id": "r", "pack": "smoke", "started_at": T0, "status": "completed", "model": {"model_id": "m"},
           "workspace": {"slug": "sim-smoke", "purged": True}, "scorecard": card, "cost": {"source": "llm_usage", "models_seen": ["m"]},
           "scenarios": scenarios, "findings": [{"kind": "stall", "text": "a: stuck"}], "notes": ["kept"]}
    md = score.render_markdown(run)
    assert "# Simulation night" in md and "| a | task | done |" in md and "**stall**" in md and "(purged)" in md


def test_run_lock_is_exclusive_and_released(tmp_path):
    first = store.acquire_run_lock(tmp_path)
    try:
        with pytest.raises(store.RunLocked, match="held by another run"):
            store.acquire_run_lock(tmp_path)
    finally:
        first.close()
    second = store.acquire_run_lock(tmp_path)  # released by close()
    second.close()


def test_store_round_trip(tmp_path):
    run_id, run_dir = store.new_run_dir("smoke", tmp_path)
    assert run_dir.exists() and run_id.endswith("-smoke")
    run = {"run_id": run_id, "pack": "smoke", "started_at": T0, "ended_at": T1, "status": "completed",
           "workspace": {"id": "w"}, "model": {"model_id": "m"},
           "scorecard": {"cost_usd": 0.1, "usability": 1.0, "quality": None, "usefulness": 0.5, "outcome_rate": 1.0, "scenarios": 1, "ok": 1},
           "scenarios": [{"id": "a", "kind": "task", "outcome": "done", "ok": True, "scores": {"duration_s": 1, "cost_usd": 0.1, "calls": 1, "usability": 1.0, "quality": None, "usefulness": 0.5}}]}
    store.write_run(run_dir, run)
    assert (run_dir / "run.json").exists() and not (run_dir / "run.json.tmp").exists()
    db = tmp_path / "campaign.sqlite"
    store.record_campaign(run, run_dir, db)
    store.record_campaign(run, run_dir, db)  # idempotent
    rows = store.previous_runs("smoke", db_path=db)
    assert len(rows) == 1 and rows[0]["run_id"] == run_id and rows[0]["cost_usd"] == 0.1
    store.update_latest(run_dir, tmp_path)
    assert store.latest_run_dir(tmp_path) == run_dir.resolve()
