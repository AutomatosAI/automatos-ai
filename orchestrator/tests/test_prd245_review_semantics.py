"""PRD-245 S0.3 — review only on holds (decision D6).

Every ticket of the Phase 1 run landed in review because ``force_review`` was
``bool(denials)``: a refused Read of a sibling ticket, a ToolSearch attempt and
a denied TUI prompt counted the same as "could not run the tests". Now a
refusal has a ``kind``, and only a HELD command the operator never allowed (no
answer in time, or denied) — or a refusal the backend cannot place, fail
closed — turns an auto ``done`` into ``review``.

Pure units: ``apply_result`` over a fake session with the board's completion
writer captured; the classifier on the host's own sentences; and a parity check
that those sentences are still what the host says.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from services import cli_host_service as svc  # noqa: E402
from services.session_denials import (  # noqa: E402
    HOLD_REASON_MARKERS,
    PROMPT_STAGE,
    READ_OUTSIDE_MARKER,
    UNKNOWN_TOOL_MARKER,
    classify_denial,
    forces_review,
    group_denials_by_kind,
)

HOST_PKG = _ORCH.parent / "services" / "cli-host" / "automatos_cli_host"
WS = uuid4()

# The four refusal shapes of 2026-09-17, in the host's own words.
HOLD = {"tool": "Bash", "stage": "PreToolUse",
        "reason": "'pip --version' is outside this ticket's Bash allowlist — no answer from the operator within 120 s",
        "input": {"command": "pip --version"}}
DENIED = {"tool": "Bash", "stage": "PreToolUse",
          "reason": "'pip --version' is outside this ticket's Bash allowlist — denied by the operator",
          "input": {"command": "pip --version"}}
READ_OUTSIDE = {"tool": "Read", "stage": "PreToolUse",
                "reason": "Read outside the session directory: /Users/x/.automatos/cli-host/sessions/117/ticket.md",
                "input": {"file_path": "/Users/x/.automatos/cli-host/sessions/117/ticket.md"}}
UNKNOWN_TOOL = {"tool": "ToolSearch", "stage": "PreToolUse", "reason": "tool 'ToolSearch' is not enabled for session tickets"}
PROMPT = {"tool": "AskUserQuestion", "stage": "PermissionRequest",
          "reason": "a permission prompt reached the TUI — sessions are policy-gated, not prompted"}


class _Query:
    def __init__(self, result=None):
        self._result = result

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._result

    def all(self):
        return []          # no open hold rows on these tickets


class _DB:
    def __init__(self):
        self.commits = 0

    def query(self, model):
        return _Query(None)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def _ticket(**over):
    base = dict(
        id=116, workspace_id=WS, assigned_agent_id=57, status="in_progress", review_mode="auto",
        title="Find things out", runtime_ref={"runtime": "cli", "host_id": "h1", "session_id": "s-116", "attempt": 1},
    )
    base.update(over)
    return SimpleNamespace(**base)


def _land(monkeypatch, task, denials):
    """``apply_result`` with the ONE completion writer captured → (terminal status, force_review it was given)."""
    import api.board_tasks as board

    seen = {}

    async def _finalize(db, **kw):
        seen.update(kw)
        return "review" if (kw["force_review"] or kw["review_mode"] != "auto") else "done"

    monkeypatch.setattr(board, "finalize_board_task_run", _finalize)
    monkeypatch.setattr(svc, "_owned_task", lambda db, host, tid: task)
    monkeypatch.setattr(svc, "_register_session_deliverables", lambda *a, **k: [])
    monkeypatch.setattr(svc, "publish_canvas_events", lambda *a, **k: 0)
    host = SimpleNamespace(id="h1", workspace_id=WS)
    payload = {"attempt": 1, "status": "success", "result_text": "done what I could", "permission_denials": denials}
    out = asyncio.run(svc.apply_result(_DB(), host, task.id, payload))
    return out["status"], seen["force_review"]


# ── apply_result ─────────────────────────────────────────────────────────────

def test_reads_outside_unknown_tools_and_prompts_never_force_review(monkeypatch):
    task = _ticket()
    status, forced = _land(monkeypatch, task, [READ_OUTSIDE, UNKNOWN_TOOL, PROMPT])
    assert forced is False and status == "done"
    assert [d["kind"] for d in task.runtime_ref["permission_denials"]] == ["read_outside", "unknown_tool", "prompt"]
    assert task.runtime_ref["denials"] == 3   # recorded, not blamed


def test_one_expired_hold_forces_review(monkeypatch):
    task = _ticket()
    status, forced = _land(monkeypatch, task, [READ_OUTSIDE, HOLD])
    assert forced is True and status == "review"
    assert [d["kind"] for d in task.runtime_ref["permission_denials"]] == ["read_outside", "hold"]
    assert task.runtime_ref["permission_denials"][1]["subject"] == "pip --version"


def test_a_hold_the_operator_denied_forces_review_too(monkeypatch):
    status, forced = _land(monkeypatch, _ticket(), [DENIED])
    assert forced is True and status == "review"


def test_no_refusals_no_review(monkeypatch):
    status, forced = _land(monkeypatch, _ticket(), [])
    assert forced is False and status == "done"


def test_a_refusal_the_backend_cannot_place_fails_closed(monkeypatch):
    task = _ticket()
    status, forced = _land(monkeypatch, task, [{"tool": "Bash", "stage": "PreToolUse", "reason": "something the host never said before"}])
    assert forced is True and status == "review"
    assert task.runtime_ref["permission_denials"][0]["kind"] == "other"


def test_a_hold_past_the_kept_cap_still_forces_review(monkeypatch):
    """The ticket keeps the first MAX_DENIALS_KEPT summaries; the verdict reads them all."""
    task = _ticket()
    status, forced = _land(monkeypatch, task, [READ_OUTSIDE] * svc.MAX_DENIALS_KEPT + [HOLD])
    assert forced is True and status == "review"
    assert len(task.runtime_ref["permission_denials"]) == svc.MAX_DENIALS_KEPT


def test_manual_review_mode_is_untouched(monkeypatch):
    status, forced = _land(monkeypatch, _ticket(review_mode="manual"), [READ_OUTSIDE])
    assert forced is False and status == "review"


# ── the classifier ───────────────────────────────────────────────────────────

def test_classification_reads_the_hosts_wording_in_order():
    assert classify_denial("PreToolUse", HOLD["reason"]) == "hold"
    assert classify_denial("PreToolUse", DENIED["reason"]) == "hold"
    assert classify_denial("PermissionRequest", "anything at all") == "prompt"
    assert classify_denial("PreToolUse", READ_OUTSIDE["reason"]) == "read_outside"
    assert classify_denial("PreToolUse", UNKNOWN_TOOL["reason"]) == "unknown_tool"
    assert classify_denial("", None) == "other"
    # a hold's ending wins over any other marker in the same sentence (S0.2 wording on a held command)
    assert classify_denial("PreToolUse", "cat outside the session directory — denied by the operator") == "hold"


def test_the_verdict_and_the_grouping_are_pure():
    assert forces_review([{"kind": "read_outside"}, {"kind": "prompt"}, {"kind": "unknown_tool"}]) is False
    assert forces_review([{"kind": "read_outside"}, {"kind": "hold"}]) is True
    assert forces_review([{"kind": "other"}]) is True
    assert forces_review([{"tool": "Bash"}]) is True          # an older summary without a kind → a hold
    assert forces_review([]) is False and forces_review(None) is False
    grouped = group_denials_by_kind([{"kind": "prompt"}, {"kind": "hold"}, {"tool": "x"}, {"kind": "hold", "tool": "y"}])
    assert list(grouped) == ["hold", "other", "prompt"] and len(grouped["hold"]) == 2


def test_markers_are_still_what_the_host_says():
    """The classifier reads the host's sentences. A reworded host must fail HERE,
    not silently turn every hold into ``other`` (which would put every ticket
    back into review — safe, but exactly what S0.3 exists to stop)."""
    session_py = (HOST_PKG / "session.py").read_text(encoding="utf-8")
    policy_py = (HOST_PKG / "policy.py").read_text(encoding="utf-8")
    for marker in HOLD_REASON_MARKERS:
        assert marker in session_py, f"session.py no longer says {marker!r}"
    assert f'"{PROMPT_STAGE}"' in session_py
    assert READ_OUTSIDE_MARKER in policy_py, "policy.py no longer uses the Read tool's wording for a path outside the roots"
    assert UNKNOWN_TOOL_MARKER in policy_py
