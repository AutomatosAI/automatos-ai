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
