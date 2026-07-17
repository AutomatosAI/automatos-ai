"""PRD-221 S2 — page-context sanitizer + preamble tests (pure: no DB, no network).

Locks the trust boundary: authz-looking fields never survive sanitization,
caps hold, one renderer serves both the manifest-grounded block and the
legacy label line, and injection rebuilds history entries instead of
mutating them (the ORM-JSONB flush trap from PRD-220).
"""
from __future__ import annotations

# CI collection-order guard: earlier-collected tests stub modules.*/consumers.*
# in sys.modules (bare ModuleType, no __spec__). On Linux collection order the
# stubs are still live HERE, so the real imports below resolve against them and
# die at collection ("unknown location" ImportError — see PR #434 CI). Purge
# origin-less entries so the real packages import fresh; conftest's autouse
# repair fixture re-binds everything else at test time.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers.")
                  or n == "services" or n.startswith("services."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

from services.page_context import (  # noqa: E402
    inject_page_preamble,
    merge_into_trace,
    render_page_preamble,
    sanitize_page_context,
)


def test_sanitize_drops_unknown_and_authz_fields():
    raw = {
        "page": "command-center",
        "userRole": "admin",
        "user_role": "owner",
        "is_admin": True,
        "permissions": ["*"],
        "evil": "<script>",
    }
    sanitized = sanitize_page_context(raw)
    assert set(sanitized) == {"page"}
    preamble = render_page_preamble(sanitized)
    for leaked in ("admin", "owner", "permissions", "<script>"):
        assert leaked not in preamble


def test_sanitize_caps_hold():
    raw = {
        "page": "x" * 500,
        "filters": {f"k{i}": f"v{i}" for i in range(12)},
        "visible_ids": [f"id{i}" for i in range(25)],
    }
    sanitized = sanitize_page_context(raw)
    assert len(sanitized["page"]) == 128
    assert len(sanitized["filters"]) == 8
    assert len(sanitized["visible_ids"]) == 16


def test_sanitize_garbage_returns_empty():
    assert sanitize_page_context(None) == {}
    assert sanitize_page_context("command-center") == {}
    assert sanitize_page_context([1, 2]) == {}
    # tab/filters alone reference no page — empty
    assert sanitize_page_context({"tab": "board"}) == {}
    # route must look like a route
    assert sanitize_page_context({"route": "not-a-route"}) == {}


def test_preamble_structured_from_manifest():
    sanitized = sanitize_page_context({
        "page": "command-center",
        "tab": "watchlist",
        "selected": {"type": "watch", "id": "w_1"},
        "filters": {"period": "7d"},
        "visible_ids": ["m_1", "m_2"],
    })
    preamble = render_page_preamble(sanitized)
    assert preamble.startswith('[Page context] The user is on "Command Centre"')
    assert "workspace pulse" in preamble
    assert "Selected watch: w_1." in preamble
    assert "period=7d" in preamble
    assert "Visible item ids (2): m_1, m_2." in preamble
    assert "fetch fresh details with platform tools" in preamble


def test_preamble_resolves_route_subpath():
    sanitized = sanitize_page_context({"route": "/missions/abc-123"})
    preamble = render_page_preamble(sanitized)
    assert '"Missions"' in preamble


def test_preamble_legacy_label_same_renderer():
    sanitized = sanitize_page_context({"page": "Agent Management"})
    assert render_page_preamble(sanitized) == (
        "[Context: the user is currently on the Agent Management page]"
    )


def test_preamble_empty_context_is_empty():
    assert render_page_preamble({}) == ""
    history = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
    assert inject_page_preamble(history, {}) is history


def test_inject_rebuilds_never_mutates():
    original_entry = {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
    history = [original_entry]
    out = inject_page_preamble(history, {"page": "command-center"})
    assert out is not history
    assert out[0] is not original_entry
    # the ORM row's list is untouched; the rebuilt entry carries the hint
    assert len(original_entry["parts"]) == 1
    assert len(out[0]["parts"]) == 2
    assert out[0]["parts"][1]["text"].startswith("[Page context]")


def test_context_trace_includes_page_context():
    trace = {"mode": "auto", "sections": [{"name": "memory", "tokens": 120}]}
    sanitized = sanitize_page_context({"page": "command-center", "tab": "board"})
    merged = merge_into_trace(trace, sanitized)
    assert merged["page_context"] == {"page": "command-center", "tab": "board"}
    # original trace keys preserved; input dict not mutated (ORM-bound JSONB)
    assert merged["mode"] == "auto"
    assert "page_context" not in trace
    assert merge_into_trace(None, sanitized) == {"page_context": sanitized}


def test_context_trace_never_raw():
    dirty = {"page": "agents", "userRole": "admin", "permissions": ["*"]}
    merged = merge_into_trace({}, sanitize_page_context(dirty))
    assert merged["page_context"] == {"page": "agents"}
    assert "userRole" not in repr(merged)
    assert "permissions" not in repr(merged)


def test_merge_into_trace_identity_when_empty():
    trace = {"mode": "auto"}
    assert merge_into_trace(trace, {}) is trace
    assert merge_into_trace(None, {}) is None
    assert merge_into_trace(None, None) is None


def test_inject_targets_last_user_message():
    history = [
        {"role": "user", "parts": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "reply"}]},
        {"role": "user", "parts": [{"type": "text", "text": "second"}]},
    ]
    out = inject_page_preamble(history, {"page": "missions"})
    assert len(out[0]["parts"]) == 1
    assert len(out[2]["parts"]) == 2
    assert out[1] is history[1]
