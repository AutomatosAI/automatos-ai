"""PRD-223 Wave 0 — model policy gate, attachment truth marker, cost estimate.

Pure unit tests: no app boot, no DB, no LLM. The policy predicate, the
resolver's failure capture + marker injection, and the shared cost estimator
are the load-bearing pieces the route/factory checks compose.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from core.llm.model_policy import check_model_for_agent, check_orchestrator_model
from core.llm.manager import estimate_cost_usd
from modules.attachments.resolver import (
    AttachmentResolver,
    build_unavailable_marker,
)
from modules.attachments.store import AttachmentNotFoundError, MediaType


# ---------------------------------------------------------------------------
# check_orchestrator_model
# ---------------------------------------------------------------------------

def _policy(quarantine="[]", allowlist="[]"):
    """Patch the settings read underneath the policy module."""
    def fake_get(category, key, default=None):
        assert category == "model_policy"
        if key == "orchestrator_quarantine":
            return quarantine
        if key == "orchestrator_allowlist":
            return allowlist
        return default
    return patch("core.llm.manager.get_system_setting", side_effect=fake_get)


class TestCheckOrchestratorModel:
    def test_quarantined_model_blocked(self):
        with _policy(quarantine='["openai/gpt-5.6-sol-pro"]'):
            allowed, reason = check_orchestrator_model("openai/gpt-5.6-sol-pro")
        assert allowed is False
        assert "quarantined" in reason

    def test_non_quarantined_passes_in_quarantine_only_mode(self):
        with _policy(quarantine='["openai/gpt-5.6-sol-pro"]'):
            allowed, _ = check_orchestrator_model("openai/gpt-5.5")
        assert allowed is True

    def test_strict_allowlist_blocks_unlisted(self):
        with _policy(allowlist='["openai/gpt-5.5"]'):
            allowed, reason = check_orchestrator_model("anthropic/claude-opus-5")
        assert allowed is False
        assert "allowlist" in reason

    def test_strict_allowlist_passes_listed(self):
        with _policy(allowlist='["openai/gpt-5.5"]'):
            allowed, _ = check_orchestrator_model("openai/gpt-5.5")
        assert allowed is True

    def test_quarantine_beats_allowlist(self):
        with _policy(
            quarantine='["openai/gpt-5.6-sol-pro"]',
            allowlist='["openai/gpt-5.6-sol-pro"]',
        ):
            allowed, reason = check_orchestrator_model("openai/gpt-5.6-sol-pro")
        assert allowed is False
        assert "quarantined" in reason

    def test_empty_model_blocked(self):
        with _policy():
            allowed, _ = check_orchestrator_model("")
        assert allowed is False

    def test_malformed_policy_json_treated_as_empty(self):
        # Fail-open on a broken policy VALUE (infra problem), never a crash.
        with _policy(quarantine="not json at all", allowlist="{}"):
            allowed, _ = check_orchestrator_model("openai/gpt-5.5")
        assert allowed is True

    def test_whitespace_entries_ignored(self):
        with _policy(quarantine='["  ", ""]'):
            allowed, _ = check_orchestrator_model("openai/gpt-5.5")
        assert allowed is True


# ---------------------------------------------------------------------------
# check_model_for_agent (W1: workspace approval row + platform policy)
# ---------------------------------------------------------------------------

def _db_with_row(row):
    """MagicMock Session whose query().join().filter().first() returns row."""
    db = MagicMock()
    db.query.return_value.join.return_value.filter.return_value.first.return_value = row
    return db


def _row(status="unreviewed", roles=None):
    return SimpleNamespace(approval_status=status, approved_roles=roles)


class TestCheckModelForAgent:
    def test_workspace_quarantine_blocks_regular_agent(self):
        db = _db_with_row(_row(status="quarantined"))
        with _policy():
            allowed, reason = check_model_for_agent(
                db, uuid4(), "openai/gpt-5.6-sol-pro", orchestrator_seat=False,
            )
        assert allowed is False
        assert "quarantined in this workspace" in reason

    def test_workspace_quarantine_blocks_orchestrator_seat(self):
        db = _db_with_row(_row(status="quarantined"))
        with _policy():
            allowed, _ = check_model_for_agent(
                db, uuid4(), "openai/gpt-5.5", orchestrator_seat=True,
            )
        assert allowed is False

    def test_role_grants_restrict_orchestrator_seat(self):
        db = _db_with_row(_row(status="approved", roles=["research", "drafting"]))
        with _policy():
            allowed, reason = check_model_for_agent(
                db, uuid4(), "openai/gpt-5.6-sol-pro-lite", orchestrator_seat=True,
            )
        assert allowed is False
        assert "orchestrator role" in reason

    def test_role_grants_do_not_block_regular_agents(self):
        db = _db_with_row(_row(status="approved", roles=["research"]))
        with _policy():
            allowed, _ = check_model_for_agent(
                db, uuid4(), "openai/gpt-5.6-sol-pro-lite", orchestrator_seat=False,
            )
        assert allowed is True

    def test_empty_roles_defer_to_platform_policy(self):
        db = _db_with_row(_row(status="approved", roles=[]))
        with _policy():
            allowed, _ = check_model_for_agent(
                db, uuid4(), "openai/gpt-5.5", orchestrator_seat=True,
            )
        assert allowed is True

    def test_platform_quarantine_is_orchestrator_scoped(self):
        # D2: platform quarantine bars the SEAT, not every agent — 5.6 may
        # still serve narrow non-orchestrator roles.
        db = _db_with_row(None)
        with _policy(quarantine='["openai/gpt-5.6-sol-pro"]'):
            seat_allowed, _ = check_model_for_agent(
                db, uuid4(), "openai/gpt-5.6-sol-pro", orchestrator_seat=True,
            )
            agent_allowed, _ = check_model_for_agent(
                db, uuid4(), "openai/gpt-5.6-sol-pro", orchestrator_seat=False,
            )
        assert seat_allowed is False
        assert agent_allowed is True

    def test_workspace_grant_cannot_override_platform_quarantine(self):
        # Q5: a workspace may further restrict, never loosen.
        db = _db_with_row(_row(status="approved", roles=["orchestrator"]))
        with _policy(quarantine='["openai/gpt-5.6-sol-pro"]'):
            allowed, reason = check_model_for_agent(
                db, uuid4(), "openai/gpt-5.6-sol-pro", orchestrator_seat=True,
            )
        assert allowed is False
        assert "quarantined" in reason

    def test_approval_lookup_error_fails_open_to_platform(self):
        db = MagicMock()
        db.query.side_effect = RuntimeError("db down")
        with _policy():
            allowed, _ = check_model_for_agent(
                db, uuid4(), "openai/gpt-5.5", orchestrator_seat=True,
            )
        assert allowed is True


# ---------------------------------------------------------------------------
# build_unavailable_marker
# ---------------------------------------------------------------------------

class TestUnavailableMarker:
    def test_marker_names_files_and_instructs_honesty(self):
        marker = build_unavailable_marker([
            {"attachment_id": "a-1", "filename": "Q3-board-pack.pdf", "reason": "not found or expired"},
            {"attachment_id": "b-2", "filename": None, "reason": "text could not be extracted"},
        ])
        assert marker["type"] == "text"
        text = marker["text"]
        assert "[ATTACHMENT UNAVAILABLE]" in text
        assert "Q3-board-pack.pdf" in text
        assert "attachment b-2" in text  # id fallback when filename unknown
        assert "do NOT have the contents" in text
        assert "Do not infer" in text


# ---------------------------------------------------------------------------
# AttachmentResolver.resolve — failure capture (PRD-223 S0.3)
# ---------------------------------------------------------------------------

def _ref(name, media=MediaType.DOCUMENT):
    return SimpleNamespace(
        attachment_id=uuid4(),
        workspace_id=uuid4(),
        media_type=media,
        mime="application/pdf",
        filename=name,
        size_bytes=100,
        s3_key="k",
    )


class _StubStore:
    """Store where some ids resolve and some raise AttachmentNotFoundError."""

    def __init__(self, known: dict):
        self._known = known

    async def get(self, aid, workspace_id):
        if aid in self._known:
            return self._known[aid]
        raise AttachmentNotFoundError(str(aid))


@pytest.mark.asyncio
async def test_missing_attachment_yields_failure_and_marker():
    good = _ref("readable.pdf")
    missing_id = uuid4()
    resolver = AttachmentResolver(store=_StubStore({good.attachment_id: good}))
    resolver._resolve_document = AsyncMock(return_value=({"type": "text", "text": "### readable.pdf\nhello"}, 10))

    parts, failures = await resolver.resolve(
        attachment_ids=[good.attachment_id, missing_id],
        workspace_id=uuid4(),
        model_id="openai/gpt-5.5",
    )

    assert len(failures) == 1
    assert failures[0]["attachment_id"] == str(missing_id)
    assert failures[0]["reason"] == "not found or expired"
    # Marker is the LAST part, after the successfully resolved document.
    assert parts[-1]["type"] == "text"
    assert "[ATTACHMENT UNAVAILABLE]" in parts[-1]["text"]
    assert parts[0]["text"].startswith("### readable.pdf")


@pytest.mark.asyncio
async def test_all_attachments_missing_still_returns_marker():
    resolver = AttachmentResolver(store=_StubStore({}))
    missing = [uuid4(), uuid4()]

    parts, failures = await resolver.resolve(
        attachment_ids=missing,
        workspace_id=uuid4(),
        model_id="openai/gpt-5.5",
    )

    assert len(failures) == 2
    assert len(parts) == 1
    assert "[ATTACHMENT UNAVAILABLE]" in parts[0]["text"]


@pytest.mark.asyncio
async def test_extraction_failure_is_reported_not_silent():
    ref = _ref("corrupt.pdf")
    resolver = AttachmentResolver(store=_StubStore({ref.attachment_id: ref}))
    resolver._resolve_document = AsyncMock(return_value=(None, 0))

    parts, failures = await resolver.resolve(
        attachment_ids=[ref.attachment_id],
        workspace_id=uuid4(),
        model_id="openai/gpt-5.5",
    )

    assert failures == [{
        "attachment_id": str(ref.attachment_id),
        "filename": "corrupt.pdf",
        "reason": "text could not be extracted",
    }]
    assert len(parts) == 1
    assert "corrupt.pdf" in parts[0]["text"]


@pytest.mark.asyncio
async def test_clean_resolution_has_no_marker_no_failures():
    ref = _ref("fine.pdf")
    resolver = AttachmentResolver(store=_StubStore({ref.attachment_id: ref}))
    resolver._resolve_document = AsyncMock(return_value=({"type": "text", "text": "### fine.pdf\nok"}, 5))

    parts, failures = await resolver.resolve(
        attachment_ids=[ref.attachment_id],
        workspace_id=uuid4(),
        model_id="openai/gpt-5.5",
    )

    assert failures == []
    assert len(parts) == 1
    assert "[ATTACHMENT UNAVAILABLE]" not in parts[0]["text"]


@pytest.mark.asyncio
async def test_empty_ids_short_circuit():
    resolver = AttachmentResolver(store=_StubStore({}))
    parts, failures = await resolver.resolve(
        attachment_ids=[], workspace_id=uuid4(), model_id="m",
    )
    assert (parts, failures) == ([], [])


# ---------------------------------------------------------------------------
# estimate_cost_usd (shared audit-log + governor pricing)
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_known_model_uses_mapped_rates(self):
        # gpt-4o: (0.0025, 0.010) per 1k
        cost = estimate_cost_usd("openai/gpt-4o", 10_000, 1_000)
        assert cost == pytest.approx(10 * 0.0025 + 1 * 0.010)

    def test_unknown_model_conservative_default(self):
        # The 2026-07-31 incident pricing path: unknown slug → $0.003/1k blended.
        cost = estimate_cost_usd("openai/gpt-5.6-sol-pro", 170_000, 4_000)
        assert cost == pytest.approx(174 * 0.003)

    def test_none_model_does_not_crash(self):
        assert estimate_cost_usd(None, 1000, 0) == pytest.approx(0.003)
