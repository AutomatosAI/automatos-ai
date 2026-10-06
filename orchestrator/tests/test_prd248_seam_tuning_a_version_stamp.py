"""PRD-248 tuning (6 Oct): every shadow row carries the seam version.

The rows are scored against graded nights before and after each change to the
questions or the state, so each row says which version of the seam wrote it. The
2026-10-06 baseline has no stamp (read as 1); this batch writes 2. Every field a row
already had is kept.
"""
from __future__ import annotations

import json

from config import config
from core.llm.decisions import SEAM_VERSION, DecisionEngine


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_every_shadow_row_is_stamped_and_keeps_its_fields(monkeypatch, tmp_path):
    path = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(config, "DECISION_SHADOW_LOG_PATH", str(path))
    engine = DecisionEngine(settings_reader=lambda *_: None)

    engine.record_shadow({"purpose": "session_end", "task_id": 673, "files_touched": 2, "jev_verdict": "complete"})
    engine.record_shadow({"purpose": "tool_rerank", "kept": ["a"]})

    first, second = _rows(path)
    assert first["seam_version"] == second["seam_version"] == SEAM_VERSION == 2
    assert {"ts", "purpose", "task_id", "files_touched", "jev_verdict"} <= set(first)
    assert second["kept"] == ["a"]


def test_the_scorer_splits_rows_by_seam_version():
    from scripts.eval.decision_shadow import score as scorer

    rows = [
        {"purpose": "session_end", "ts": 1.0},  # the baseline, before the stamp
        {"purpose": "session_end", "ts": 2.0, "seam_version": 2},
        {"purpose": "session_end", "ts": 3.0, "seam_version": 2},
    ]
    assert [scorer.seam_version_of(r) for r in rows] == [1, 2, 2]
    assert len(scorer.filter_seam(rows, 1)) == 1 and len(scorer.filter_seam(rows, 2)) == 2
    assert scorer.filter_seam(rows, None) == rows
    assert scorer.summary_dict(rows)["seam_versions"] == {"1": 1, "2": 2}
    assert scorer.summary_dict(rows, seam_version=2)["rows"] == 2
