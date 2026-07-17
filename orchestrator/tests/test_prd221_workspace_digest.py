"""PRD-221 S8 — digest snapshot builder tests (pure: ActivityService mocked).

Locks the state-hash contract (stable under feed reordering; changes when a
needs-attention item appears) and the plain-language shape (names + reasons,
no raw event payloads), plus the deterministic fallback digest.
"""
from __future__ import annotations

from unittest.mock import MagicMock

# CI collection-order guard (see PR #434).
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers.")
                  or n == "services" or n.startswith("services."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

import services.workspace_digest as wd  # noqa: E402


def _mock_service(monkeypatch, items, stats=None):
    svc = MagicMock()
    svc.get_stats.return_value = stats or {
        "working_now": 0, "completed_today": 0,
        "needs_attention": 0, "channels_live": 0, "period": "1d",
    }
    svc.get_feed.return_value = {"items": items, "total": len(items),
                                 "limit": 50, "offset": 0}
    monkeypatch.setattr(wd, "ActivityService", lambda db, ws: svc)
    return svc


def _item(id_, type_, status, name="thing", **extra):
    return {"id": id_, "type": type_, "status": status, "name": name, **extra}


def test_digest_hash_stable_under_reordering(monkeypatch):
    items_a = [_item("1", "mission", "running"), _item("2", "task", "completed")]
    items_b = list(reversed(items_a))
    _mock_service(monkeypatch, items_a)
    h1 = wd.build_digest_snapshot(MagicMock(), "ws")["state_hash"]
    _mock_service(monkeypatch, items_b)
    h2 = wd.build_digest_snapshot(MagicMock(), "ws")["state_hash"]
    assert h1 == h2


def test_digest_hash_changes_on_new_attention(monkeypatch):
    base = [_item("1", "mission", "running")]
    _mock_service(monkeypatch, base)
    h1 = wd.build_digest_snapshot(MagicMock(), "ws")["state_hash"]
    _mock_service(monkeypatch, base + [_item("2", "mission", "failed")])
    h2 = wd.build_digest_snapshot(MagicMock(), "ws")["state_hash"]
    assert h1 != h2


def test_digest_hash_changes_on_status_flip(monkeypatch):
    _mock_service(monkeypatch, [_item("1", "mission", "running")])
    h1 = wd.build_digest_snapshot(MagicMock(), "ws")["state_hash"]
    _mock_service(monkeypatch, [_item("1", "mission", "failed")])
    h2 = wd.build_digest_snapshot(MagicMock(), "ws")["state_hash"]
    assert h1 != h2


def test_digest_snapshot_is_plain(monkeypatch):
    items = [
        _item("1", "mission", "failed", name="Outreach",
              error_message="No email account connected",
              parts=[{"secret": "raw event payload"}]),
        _item("2", "task", "running", name="Research"),
        _item("3", "task", "completed", name="Summary"),
    ]
    _mock_service(monkeypatch, items, stats={
        "working_now": 1, "completed_today": 1,
        "needs_attention": 1, "channels_live": 2, "period": "1d",
    })
    snap = wd.build_digest_snapshot(MagicMock(), "ws")

    assert snap["needs_attention"] == [
        {"name": "Outreach", "reason": "No email account connected"}
    ]
    assert snap["needs_attention_count"] == 1
    assert {"name": "Research", "type": "task"} in snap["active"]
    assert {"name": "Summary", "type": "task"} in snap["recent_completions"]
    # no raw event payload leaks into the snapshot
    assert "raw event payload" not in repr(snap)
    assert snap["counts"]["working_now"] == 1


def test_attention_reason_falls_back_when_no_error(monkeypatch):
    _mock_service(monkeypatch, [_item("1", "mission", "failed", name="X")])
    snap = wd.build_digest_snapshot(MagicMock(), "ws")
    assert snap["needs_attention"][0]["reason"] == "Stopped and needs a look."


def test_fallback_digest_names_attention_item():
    snap = {
        "counts": {"working_now": 2, "completed": 1},
        "needs_attention": [{"name": "Outreach", "reason": "No email account connected"}],
    }
    text = wd.render_fallback_digest(snap)
    assert "Outreach" in text
    assert "No email account connected" in text
    assert "2 items working now" in text


def test_fallback_digest_quiet_workspace():
    text = wd.render_fallback_digest({"counts": {"working_now": 0, "completed": 0},
                                      "needs_attention": []})
    assert "Nothing is running" in text
    assert "Nothing needs your attention" in text
