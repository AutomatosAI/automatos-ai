"""PRD-251 Wave 1, US-103 (S4.4) — the ``media`` usage lane.

* a footage job's dollars (the toolkit's estimate) book on the ``media`` lane,
  attributed to the post by the enclosing ``usage_scope``, as BYOK: the
  workspace's own Composio toolkit paid (D15);
* a render books its rendered seconds as units at $0 on the platform's
  renderer, rounded up so the render quota never under-counts;
* a negative or non-finite amount never books below zero, and a row with no
  workspace is a warning, never a silent loss;
* the budget gate needs no change: ``spend_to_date`` sums every lane (no
  ``request_type`` filter), so media spend tips the gate. Proven on a real
  ``llm_usage`` table (in-memory SQLite), then through ``PolicyGate.check``;
* Analytics reads the renderer as "Media render", billed free. The lane label
  itself is the vitest ``frontend/lib/__tests__/socials-media-lane.test.ts``.
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper, so LLMUsage configures)
from core import best_effort  # noqa: E402
from core.llm import usage_context as uc  # noqa: E402
from core.llm import usage_tracker as ut  # noqa: E402
from core.llm.providers import MEDIA_RENDER_PROVIDER, describe_usage_provider  # noqa: E402
from core.models.core import LLMUsage  # noqa: E402
from modules.policy import budget  # noqa: E402
from modules.policy import gate as gate_mod  # noqa: E402
from modules.policy import policy_document as pd  # noqa: E402
from modules.policy.gate import PolicyGate, ToolCall  # noqa: E402
from modules.policy.types import Decision  # noqa: E402

# Hex with letters: SQLite stores a UUID column as text only when it is not a
# well-formed number.
WS = uuid.UUID("00000000-0000-0000-0000-0000000000c1")
OTHER_WS = uuid.UUID("00000000-0000-0000-0000-0000000000c2")
POST_REF = "social_post:5f0c2b7e-4a1d-4c3e-9b8a-0d6e1f2a3b4c"
FOOTAGE_MODEL = "fal-ai/fixture-video"


# ── the tracker (a fake session records the row it would write) ──────────────

class _Query:
    def filter(self, *clauses):
        return self

    def first(self):
        return None


class _Session:
    added: list = []

    def query(self, *entities):
        return _Query()

    def add(self, row):
        _Session.added.append(row)

    def commit(self):
        pass

    def close(self):
        pass


@pytest.fixture
def fake_db(monkeypatch):
    _Session.added = []
    monkeypatch.setitem(
        sys.modules, "core.database.database", types.SimpleNamespace(SessionLocal=_Session)
    )
    return _Session


class _Log:
    """Stands in for the tracker's logger: its handlers may not reach caplog."""

    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def test_a_footage_job_books_its_estimate_on_the_media_lane(fake_db):
    with uc.usage_scope(workspace_id=WS, agent_id=7, execution_id=POST_REF):
        # What the fal recipe books after FAL_AI_ESTIMATE_PRICING quoted a 5 s clip.
        ut.UsageTracker.track_media(provider="fal_ai", model_id=FOOTAGE_MODEL, units=5, usd=0.46, latency_ms=41_000)

    (row,) = fake_db.added
    assert row.request_type == uc.LANE_MEDIA == "media"
    assert row.provider == "fal_ai" and row.model_id == FOOTAGE_MODEL
    assert row.total_cost == pytest.approx(0.46)
    assert row.input_cost == pytest.approx(0.46) and row.output_cost == 0.0
    assert row.tier == "direct" and row.is_byok is True     # the workspace's own toolkit paid
    assert row.workspace_id == WS and row.agent_id == 7 and row.execution_id == POST_REF
    assert row.input_tokens == 5 and row.output_tokens == 0 and row.latency_ms == 41_000


def test_a_render_books_its_rendered_seconds_as_units_at_zero(fake_db):
    with uc.usage_scope(workspace_id=WS, execution_id=POST_REF):
        ut.UsageTracker.track_media(provider=MEDIA_RENDER_PROVIDER, model_id="hyperframes", units=39.5)

    (row,) = fake_db.added
    assert row.request_type == "media" and row.provider == "media_render"
    assert row.input_tokens == 40                            # 39.5 s rounds UP: a quota never under-counts
    assert row.total_cost == 0.0 and row.input_cost == 0.0
    assert row.tier == "direct" and row.is_byok is False    # the platform's renderer, not a toolkit
    assert row.execution_id == POST_REF


@pytest.mark.parametrize(
    "usd, units, cost, count",
    [
        (-0.25, 5, 0.0, 5),             # a balance difference read the wrong way round
        (float("nan"), 5, 0.0, 5),
        ("n/a", 5, 0.0, 5),
        (0.46, -3, 0.46, 0),
        (0.46, float("inf"), 0.46, 0),
    ],
)
def test_a_bad_amount_books_zero_and_warns_never_below_zero(fake_db, monkeypatch, usd, units, cost, count):
    log = _Log()
    monkeypatch.setattr(ut, "logger", log)
    with uc.usage_scope(workspace_id=WS):
        ut.UsageTracker.track_media(provider="fal_ai", model_id=FOOTAGE_MODEL, units=units, usd=usd)

    (row,) = fake_db.added
    assert row.total_cost == pytest.approx(cost) and row.input_tokens == count
    assert any("booked as 0" in w for w in log.warnings)


@pytest.fixture
def no_workspace(monkeypatch):
    """No workspace anywhere: no usage scope, no request context, the SaaS
    edition. The request ContextVar is pinned because a test earlier in the
    process can leave it set (``_enrich_log_context`` sets it and never resets)."""
    from config import config as cfg
    from core.monitoring.automatos_logging import workspace_id_var

    monkeypatch.setattr(cfg, "AUTH_EDITION", "saas", raising=False)
    monkeypatch.setattr(uc, "current_usage_scope", lambda: {})
    token = workspace_id_var.set("")
    yield
    workspace_id_var.reset(token)


def test_no_workspace_books_nothing_and_says_so(fake_db, no_workspace, monkeypatch):
    log = _Log()
    monkeypatch.setattr(ut, "logger", log)
    ut.UsageTracker.track_media(provider="fal_ai", model_id=FOOTAGE_MODEL, units=5, usd=0.46)

    assert fake_db.added == []
    assert any("media usage not booked" in w and "0.4600" in w for w in log.warnings)


def test_attribution_survives_the_hop_off_the_event_loop(fake_db):
    async def render_finished():
        with uc.usage_scope(workspace_id=WS, execution_id=POST_REF):
            ut.UsageTracker.track_media(provider=MEDIA_RENDER_PROVIDER, model_id="hyperframes", units=12)

    asyncio.run(render_finished())
    assert best_effort.drain(timeout=5)

    (row,) = fake_db.added
    assert row.workspace_id == WS and row.execution_id == POST_REF and row.input_tokens == 12


def test_analytics_reads_the_renderer_as_a_free_media_provider():
    assert describe_usage_provider(MEDIA_RENDER_PROVIDER) == {
        "slug": "media_render", "label": "Media render", "kind": "media", "billing": "free",
    }


# ── the budget gate counts the lane (a real llm_usage table) ─────────────────

@pytest.fixture
def ledger():
    """``llm_usage`` on in-memory SQLite, rows written through the model. The
    table is raw DDL over the model's own column names, untyped: SQLAlchemy
    2.0.23 cannot compile the Postgres UUID type for SQLite (the approach of
    test_tool_routing_models.py)."""
    engine = sa.create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    columns = ", ".join(
        f"{c.name} INTEGER PRIMARY KEY" if c.primary_key else c.name
        for c in LLMUsage.__table__.columns
    )
    with engine.begin() as conn:
        conn.exec_driver_sql(f"CREATE TABLE {LLMUsage.__tablename__} ({columns})")
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _usage(workspace_id, lane, usd, units, *, provider="openrouter", model_id="moonshotai/kimi-k3"):
    return LLMUsage(
        workspace_id=workspace_id, model_id=model_id, provider=provider, tier="direct",
        request_type=lane, input_tokens=units, output_tokens=0, total_tokens=units,
        cache_read_tokens=0, cache_write_tokens=0, input_cost=usd, output_cost=0.0, total_cost=usd,
        is_byok=False, status="success",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )


def _footage(workspace_id, usd):
    return _usage(workspace_id, "media", usd, 5, provider="fal_ai", model_id=FOOTAGE_MODEL)


def test_spend_to_date_counts_the_media_lane(ledger):
    ledger.add_all([
        _usage(WS, "chat", 0.50, 100),
        _footage(WS, 0.46),
        _usage(WS, "media", 0.0, 40, provider=MEDIA_RENDER_PROVIDER, model_id="hyperframes"),
        _footage(OTHER_WS, 5.00),                            # another workspace's spend never counts
    ])
    ledger.commit()

    spent = budget.spend_to_date(ledger, WS, "day")
    assert spent["cost_usd"] == pytest.approx(0.96)
    assert spent["total_tokens"] == 145                      # units ride in the token sum, as rerank's do


def test_media_spend_is_what_tips_the_budget_ceiling(ledger, monkeypatch):
    monkeypatch.setattr(budget, "load_budget", lambda db, ws: {"window": "day", "max_cost_usd": 1.0})
    ledger.add(_usage(WS, "chat", 0.50, 100))
    ledger.commit()
    assert budget.check_budget(ledger, WS, projected_cost_usd=0.10).allowed is True

    ledger.add(_footage(WS, 0.46))
    ledger.commit()
    decision = budget.check_budget(ledger, WS, projected_cost_usd=0.10)
    assert decision.allowed is False and decision.dimension == "cost_usd"
    assert decision.spent == pytest.approx(0.96)


def test_the_policy_gate_denies_once_media_spend_crosses_the_ceiling(ledger, monkeypatch):
    monkeypatch.setattr(budget, "load_budget", lambda db, ws: {"window": "day", "max_cost_usd": 0.90})
    monkeypatch.setattr(
        gate_mod._policy_doc, "load_policy_document",
        lambda db, ws: pd.PolicyDocument(pd.BALANCED, False, {}),
    )
    gate = PolicyGate(ledger)
    monkeypatch.setattr(gate, "_lookup_action", lambda name: None)
    monkeypatch.setattr(gate, "_full_autonomy", lambda ws: False)
    call = ToolCall(tool_name="platform_list_agents", parameters={}, workspace_id=WS)

    ledger.add(_usage(WS, "chat", 0.50, 100))
    ledger.commit()
    assert gate.check(call).decision is Decision.ALLOW

    ledger.add(_footage(WS, 0.46))
    ledger.commit()
    verdict = gate.check(call)
    assert verdict.decision is Decision.DENY
    assert verdict.error.code == "budget_exceeded"
