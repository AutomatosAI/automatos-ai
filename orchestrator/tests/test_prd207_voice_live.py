"""PRD-207 — Auto Live: real-time voice + the presence orb.

* S3/S4 — voice_calls lifecycle model + migration chain, the settings plane
  (platform toggle + Retell creds from DB system_settings, never env), the
  workspace voice_live shape, the cap-gate formula with active-call
  reservation, and the fail-closed settings whitelist.
* S1 — the web-call mint: gates in order with honest refusals, dynamic vars,
  the row born at mint.
* S2 — the webhook trust boundary: mint-row cross-validation, binding to the
  on-screen chat + real user, fail-closed fallback, phone lane.
* S3 — lifecycle events idempotency, loud orphans, HMAC refusal, the meter.
* S8 — telemetry parity (voice_turns written by the live path).
* Guard — the live path never imports the 120s-TTS pod client.

Pure logic is tested without a DB; DB seams follow the PRD-205 idiom
(skip cleanly without Postgres); vendor HTTP is always mocked.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))


# ---------------------------------------------------------------------------
# S3/S4 · migration chain — single-parent on the current head
# ---------------------------------------------------------------------------

def test_migration_chains_on_prd206_chat_summary():
    mig = (_orchestrator_root / "alembic" / "versions" / "prd207_voice_live.py").read_text()
    assert 'revision = "prd207_voice_live"' in mig
    # chains directly on the current single head (PRD-206 S2), no second join
    assert 'down_revision = "prd206_chat_summary"' in mig
    assert "voice_calls" in mig and "call_id" in mig


def test_no_other_migration_chains_on_prd206_head():
    """prd207_voice_live must be the ONLY child of prd206_chat_summary —
    a second child would re-fork main's migration history (the #545/#548
    parallel-merge-heads lesson)."""
    versions = _orchestrator_root / "alembic" / "versions"
    children = [
        p.name
        for p in versions.glob("*.py")
        if 'down_revision = "prd206_chat_summary"' in p.read_text()
    ]
    assert children == ["prd207_voice_live.py"]


# ---------------------------------------------------------------------------
# S4 · workspace voice_live shape (pure)
# ---------------------------------------------------------------------------

def test_parse_workspace_voice_live_defaults_fail_closed():
    from modules.voice.live_settings import parse_workspace_voice_live

    for raw in (None, {}, {"voice_live": None}, {"voice_live": "on"}, {"voice_live": []}):
        view = parse_workspace_voice_live(raw)  # type: ignore[arg-type]
        assert view.enabled is False  # never live by accident
        assert view.monthly_cap_minutes > 0  # config default applies
        assert view.retell_voice_id is None


def test_parse_workspace_voice_live_reads_values():
    from modules.voice.live_settings import parse_workspace_voice_live

    view = parse_workspace_voice_live(
        {"voice_live": {"enabled": True, "monthly_cap_minutes": 250, "retell_voice_id": "retell-Cimo"}}
    )
    assert view.enabled is True
    assert view.monthly_cap_minutes == 250
    assert view.retell_voice_id == "retell-Cimo"


def test_parse_workspace_voice_live_bad_cap_falls_back():
    from config import config
    from modules.voice.live_settings import parse_workspace_voice_live

    for bad in ("many", -3, 0, None):
        view = parse_workspace_voice_live({"voice_live": {"enabled": True, "monthly_cap_minutes": bad}})
        assert view.monthly_cap_minutes == int(config.VOICE_LIVE_DEFAULT_MONTHLY_CAP_MINUTES)


def test_validate_voice_live_update_matrix():
    from modules.voice.live_settings import validate_voice_live_update

    # happy path normalizes
    ok = validate_voice_live_update(
        {"enabled": True, "monthly_cap_minutes": 60, "retell_voice_id": " v1 "}
    )
    assert ok == {"enabled": True, "monthly_cap_minutes": 60, "retell_voice_id": "v1"}

    for bad in (
        "on",                                   # not an object
        {"enabled": "yes"},                     # non-bool enabled
        {"monthly_cap_minutes": True},          # bool is not an int here
        {"monthly_cap_minutes": 0},             # zero cap
        {"monthly_cap_minutes": 200_000},       # absurd cap
        {"retell_voice_id": "x" * 65},          # oversized voice id
        {"retell_voice_id": "key_" + "a1b2"},   # an API key is never a voice id
        {"surprise": 1},                        # unknown key fail-closed
    ):
        with pytest.raises(ValueError):
            validate_voice_live_update(bad)


# ---------------------------------------------------------------------------
# S4 · the cap formula (pure) — reservation bounds the two-tabs race
# ---------------------------------------------------------------------------

def test_cap_formula_boundary_and_reservation():
    from modules.voice.voice_meter import MeterReading, cap_allows_mint

    # under cap, no active calls → allowed
    ok, _ = cap_allows_mint(MeterReading(80, 0, 10), cap_minutes=100)
    assert ok

    # exactly at cap → refused, honest reason
    refused, reason = cap_allows_mint(MeterReading(100, 0, 10), cap_minutes=100)
    assert not refused
    assert "100/100" in reason

    # the second simultaneous mint sees the first call's reservation:
    # 95 ended + 1 active × 10 reserve = 105 ≥ 100 → refused
    refused2, reason2 = cap_allows_mint(MeterReading(95, 1, 10), cap_minutes=100)
    assert not refused2
    assert "reserved" in reason2


def test_month_window_utc_covers_year_rollover():
    from datetime import datetime, timezone

    from modules.voice.voice_meter import month_window_utc

    start, nxt = month_window_utc(datetime(2026, 12, 15, 9, 30, tzinfo=timezone.utc))
    assert (start.year, start.month, start.day) == (2026, 12, 1)
    assert (nxt.year, nxt.month) == (2027, 1)


# ---------------------------------------------------------------------------
# S4 · settings plane reads system_settings, never env
# ---------------------------------------------------------------------------

def test_platform_toggle_reads_system_settings(monkeypatch):
    import modules.voice.live_settings as ls

    calls = []

    def fake_get(category, key, default=None):
        calls.append((category, key))
        return {"live_enabled": "true", "retell_api_key": "k", "retell_webhook_secret": "s",
                "retell_agent_id": "a"}.get(key, default)

    monkeypatch.setattr(ls, "get_system_setting", fake_get)
    assert ls.voice_live_enabled() is True
    creds = ls.retell_credentials()
    assert creds.armed
    assert ("voice", "live_enabled") in calls  # DB settings, not config/env


def test_credentials_not_armed_when_any_missing(monkeypatch):
    import modules.voice.live_settings as ls

    monkeypatch.setattr(
        ls, "get_system_setting",
        lambda c, k, d=None: {"retell_api_key": "k", "retell_webhook_secret": ""}.get(k, d),
    )
    assert ls.retell_credentials().armed is False


# ---------------------------------------------------------------------------
# S4 · workspace-settings whitelist (the PRD-143 S11 fail-closed surface)
# ---------------------------------------------------------------------------

def _mock_ws_db(initial_settings=None):
    ws = SimpleNamespace(settings=dict(initial_settings or {}), id=uuid.uuid4())
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = ws
    return db, ws


def test_whitelist_still_refuses_unknown_keys():
    from modules.tools.discovery.handlers_workspace import update_workspace_settings

    db, _ = _mock_ws_db()
    out = asyncio.run(
        update_workspace_settings(db, uuid.uuid4(), {"key": "integrations", "value": {}})
    )
    assert out["success"] is False
    assert "voice_live" in out["error"]  # the whitelist names its members


def test_whitelist_accepts_and_merges_voice_live():
    from modules.tools.discovery.handlers_workspace import update_workspace_settings

    db, ws = _mock_ws_db({"voice_live": {"enabled": False, "retell_voice_id": "keep-me"}})
    out = asyncio.run(
        update_workspace_settings(
            db, uuid.uuid4(), {"key": "voice_live", "value": {"enabled": True}}
        )
    )
    assert out["success"] is True
    assert ws.settings["voice_live"]["enabled"] is True
    # merge, not replace: the untouched key survives
    assert ws.settings["voice_live"]["retell_voice_id"] == "keep-me"


def test_whitelist_refuses_malformed_voice_live():
    from modules.tools.discovery.handlers_workspace import update_workspace_settings

    db, ws = _mock_ws_db({"voice_live": {"enabled": False}})
    out = asyncio.run(
        update_workspace_settings(
            db, uuid.uuid4(), {"key": "voice_live", "value": {"enabled": "yes"}}
        )
    )
    assert out["success"] is False
    assert ws.settings["voice_live"]["enabled"] is False  # nothing written


# ---------------------------------------------------------------------------
# S1 · the Retell payload (pure)
# ---------------------------------------------------------------------------

def test_build_web_call_payload_strings_vars_and_nests_override():
    from modules.voice.retell_api import build_web_call_payload

    payload = build_web_call_payload(
        agent_id="agent_1",
        dynamic_variables={"workspace_id": uuid.UUID(int=7), "user_id": 42, "chat_id": None},
        voice_id="retell-Cimo",
        max_call_minutes=30,
    )
    dv = payload["retell_llm_dynamic_variables"]
    assert dv["user_id"] == "42"  # Retell demands string values
    assert "chat_id" not in dv  # Nones dropped, never the string 'None'
    # the verified nesting: agent_override.agent.{voice_id, max_call_duration_ms}
    assert payload["agent_override"]["agent"]["voice_id"] == "retell-Cimo"
    assert payload["agent_override"]["agent"]["max_call_duration_ms"] == 30 * 60_000


def test_build_web_call_payload_omits_override_when_empty():
    from modules.voice.retell_api import build_web_call_payload

    payload = build_web_call_payload(
        agent_id="agent_1", dynamic_variables={"workspace_id": "w"}, voice_id=None, max_call_minutes=None
    )
    assert "agent_override" not in payload


# ---------------------------------------------------------------------------
# S1 · the mint endpoint — gates in order, honest refusals, row born at mint
# ---------------------------------------------------------------------------

def _mint_ctx(ws_id=None, user_int=7):
    return SimpleNamespace(
        workspace_id=ws_id or uuid.uuid4(),
        user=SimpleNamespace(id=user_int) if user_int is not None else None,
        auth_type="clerk",
    )


def _mint_db(workspace=None, chat="unqueried"):
    """A query-dispatching mock db: Workspace / Chat lookups by model arg."""
    from core.models.core import Chat as ChatModel
    from core.models.workspaces import Workspace as WorkspaceModel

    db = MagicMock()

    def dispatch(model_arg):
        q = MagicMock()
        if model_arg is WorkspaceModel:
            q.filter.return_value.first.return_value = workspace
        elif model_arg is ChatModel:
            result = None if chat == "unqueried" else chat
            q.filter.return_value.first.return_value = result
            # The welcome-path find-or-create queries .filter(...).order_by(...).first()
            q.filter.return_value.order_by.return_value.first.return_value = result
        else:
            q.filter.return_value.first.return_value = None
        return q

    db.query.side_effect = dispatch
    return db


def _run_mint(body=None, ctx=None, db=None):
    from api.voice_retell import MintWebCallRequest, mint_web_call

    return asyncio.run(
        mint_web_call(body=body or MintWebCallRequest(), ctx=ctx or _mint_ctx(), db=db or _mint_db())
    )


def test_web_call_mint_gates_in_order(monkeypatch):
    from fastapi import HTTPException

    import api.voice_retell as vr
    from modules.voice.live_settings import RetellCredentials, WorkspaceVoiceLive
    from modules.voice.voice_meter import MeterReading

    vendor_calls = []

    async def never_vendor(*a, **k):
        vendor_calls.append(1)
        raise AssertionError("vendor must not be reached by a refused mint")

    monkeypatch.setattr(vr.retell_api, "create_web_call", never_vendor)

    # Gate 1 — platform toggle OFF → 503, nothing else consulted.
    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: False)
    with pytest.raises(HTTPException) as e1:
        _run_mint()
    assert e1.value.status_code == 503
    assert "platform-wide" in e1.value.detail

    # Gate 2 — ON but unarmed → 503 with the arming reason.
    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(
        vr.live_settings, "retell_credentials", lambda: RetellCredentials("", "", "")
    )
    with pytest.raises(HTTPException) as e2:
        _run_mint()
    assert e2.value.status_code == 503
    assert "not armed" in e2.value.detail

    # Gate 3 — armed but the workspace toggle is off → 403.
    monkeypatch.setattr(
        vr.live_settings, "retell_credentials", lambda: RetellCredentials("k", "s", "a")
    )
    ws = SimpleNamespace(settings={"voice_live": {"enabled": False}})
    with pytest.raises(HTTPException) as e3:
        _run_mint(db=_mint_db(workspace=ws))
    assert e3.value.status_code == 403
    assert "workspace" in e3.value.detail

    # Gate 4 — enabled but over cap → 429 with the honest budget line.
    ws_on = SimpleNamespace(settings={"voice_live": {"enabled": True, "monthly_cap_minutes": 100}})
    monkeypatch.setattr(
        vr.voice_meter, "monthly_meter", lambda db, w: MeterReading(100, 0, 10)
    )
    with pytest.raises(HTTPException) as e4:
        _run_mint(db=_mint_db(workspace=ws_on))
    assert e4.value.status_code == 429
    assert "100/100" in e4.value.detail

    assert vendor_calls == []  # no refused gate ever reached Retell


def test_mint_requires_a_strictly_resolved_user(monkeypatch):
    from fastapi import HTTPException

    import api.voice_retell as vr
    from modules.voice.live_settings import RetellCredentials

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(
        vr.live_settings, "retell_credentials", lambda: RetellCredentials("k", "s", "a")
    )
    with pytest.raises(HTTPException) as exc:
        _run_mint(ctx=_mint_ctx(user_int=None))
    assert exc.value.status_code == 403
    assert "signed-in" in exc.value.detail


def test_mint_passes_dynamic_vars_and_inserts_minted_row(monkeypatch):
    import api.voice_retell as vr
    from core.models.voice_calls import VoiceCall
    from modules.voice.live_settings import RetellCredentials
    from modules.voice.retell_api import RetellWebCall
    from modules.voice.voice_meter import MeterReading

    ws_id = uuid.uuid4()
    chat_id = uuid.uuid4()
    seen = {}

    async def fake_vendor(api_key, payload):
        seen["api_key"] = api_key
        seen["payload"] = payload
        return RetellWebCall(call_id="call_abc", access_token="tok_xyz")

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(
        vr.live_settings, "retell_credentials", lambda: RetellCredentials("k", "s", "agent_9")
    )
    monkeypatch.setattr(vr.voice_meter, "monthly_meter", lambda db, w: MeterReading(0, 0, 10))
    monkeypatch.setattr(vr.retell_api, "create_web_call", fake_vendor)

    ws = SimpleNamespace(settings={"voice_live": {"enabled": True, "retell_voice_id": "v-1"}})
    my_chat = SimpleNamespace(id=chat_id, user_id=7, workspace_id=ws_id)
    db = _mint_db(workspace=ws, chat=my_chat)

    from api.voice_retell import MintWebCallRequest

    out = _run_mint(
        body=MintWebCallRequest(chat_id=str(chat_id), agent_id=3),
        ctx=_mint_ctx(ws_id=ws_id, user_int=7),
        db=db,
    )

    assert out == {"call_id": "call_abc", "access_token": "tok_xyz", "chat_id": str(chat_id)}
    dv = seen["payload"]["retell_llm_dynamic_variables"]
    assert dv == {
        "workspace_id": str(ws_id),
        "user_id": "7",
        "chat_id": str(chat_id),
        "agent_id": "3",
    }
    assert seen["payload"]["agent_override"]["agent"]["voice_id"] == "v-1"

    added = [a.args[0] for a in db.add.call_args_list if isinstance(a.args[0], VoiceCall)]
    assert len(added) == 1  # the row is BORN at mint
    row = added[0]
    assert row.call_id == "call_abc"
    assert row.status == "minted"
    assert row.user_id == 7
    assert row.chat_id == str(chat_id)
    assert db.commit.called


def test_mint_without_chat_creates_and_binds_thread(monkeypatch):
    """Gerard's first-call feedback: the spoken conversation must be the
    VISIBLE one. No chat on screen → mint creates the thread, binds it, and
    hands its id back so the screen can point at it."""
    import consumers.chatbot.service as chat_service_mod

    import api.voice_retell as vr
    from core.models.voice_calls import VoiceCall
    from modules.voice.live_settings import RetellCredentials
    from modules.voice.retell_api import RetellWebCall
    from modules.voice.voice_meter import MeterReading

    ws_id = uuid.uuid4()
    created_chat_id = uuid.uuid4()
    created = {}

    class FakeChatService:
        def __init__(self, db):
            pass

        def create_chat(self, *, user_id, title, workspace_id):
            created.update(user_id=user_id, title=title, workspace_id=workspace_id)
            return SimpleNamespace(id=created_chat_id)

    monkeypatch.setattr(chat_service_mod, "ChatService", FakeChatService)
    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(
        vr.live_settings, "retell_credentials", lambda: RetellCredentials("k", "s", "a")
    )
    monkeypatch.setattr(vr.voice_meter, "monthly_meter", lambda db, w: MeterReading(0, 0, 10))

    async def fake_vendor(api_key, payload):
        return RetellWebCall(call_id="call_fresh", access_token="tok")

    monkeypatch.setattr(vr.retell_api, "create_web_call", fake_vendor)

    ws = SimpleNamespace(settings={"voice_live": {"enabled": True}})
    db = _mint_db(workspace=ws)

    out = _run_mint(ctx=_mint_ctx(ws_id=ws_id, user_int=7), db=db)

    assert out["chat_id"] == str(created_chat_id)  # the screen can follow it
    # Title is time-stamped (unique_user_title makes fixed titles a 500
    # on the second call — 2026-07-18 incident).
    assert created["user_id"] == 7 and created["title"].startswith("Voice call — ")
    rows = [a.args[0] for a in db.add.call_args_list if isinstance(a.args[0], VoiceCall)]
    assert rows[0].chat_id == str(created_chat_id)  # mint-proven binding → webhook writes HERE


def test_mint_reuses_the_voice_thread(monkeypatch):
    """chats enforces unique_user_title: the SECOND welcome-screen call must
    REUSE the caller's 'Voice call' thread (the Auto-thread pattern), never
    insert-and-500 — the 'Failed to fetch' every call after the first."""
    import consumers.chatbot.service as chat_service_mod

    import api.voice_retell as vr
    from modules.voice.live_settings import RetellCredentials
    from modules.voice.retell_api import RetellWebCall
    from modules.voice.voice_meter import MeterReading

    ws_id = uuid.uuid4()
    existing_id = uuid.uuid4()
    existing = SimpleNamespace(id=existing_id, user_id=7, workspace_id=ws_id, title="Voice call")

    class NeverCreate:
        def __init__(self, db):
            pass

        def create_chat(self, **kwargs):
            raise AssertionError("must reuse the existing voice thread, not create")

    monkeypatch.setattr(chat_service_mod, "ChatService", NeverCreate)
    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(
        vr.live_settings, "retell_credentials", lambda: RetellCredentials("k", "s", "a")
    )
    monkeypatch.setattr(vr.voice_meter, "monthly_meter", lambda db, w: MeterReading(0, 0, 10))

    async def fake_vendor(api_key, payload):
        return RetellWebCall(call_id="call_again", access_token="tok")

    monkeypatch.setattr(vr.retell_api, "create_web_call", fake_vendor)

    ws = SimpleNamespace(settings={"voice_live": {"enabled": True}})
    out = _run_mint(
        ctx=_mint_ctx(ws_id=ws_id, user_int=7),
        db=_mint_db(workspace=ws, chat=existing),
    )
    assert out["chat_id"] == str(existing_id)  # same thread, conversation continues


def test_mint_supersedes_prior_active_calls(monkeypatch):
    """One live call per user: a fresh mint marks the caller's prior active
    (minted/started) calls 'superseded' so only the newest speaks — the guard
    against 'Auto talking over herself' from overlapping web calls."""
    import api.voice_retell as vr
    from core.models.core import Chat as ChatModel
    from core.models.voice_calls import VoiceCall
    from core.models.workspaces import Workspace as WorkspaceModel
    from modules.voice.live_settings import RetellCredentials
    from modules.voice.retell_api import RetellWebCall
    from modules.voice.voice_meter import MeterReading

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(
        vr.live_settings, "retell_credentials", lambda: RetellCredentials("k", "s", "a")
    )
    monkeypatch.setattr(vr.voice_meter, "monthly_meter", lambda db, w: MeterReading(0, 0, 10))

    async def fake_vendor(api_key, payload):
        return RetellWebCall(call_id="call_new", access_token="tok")

    monkeypatch.setattr(vr.retell_api, "create_web_call", fake_vendor)

    ws_id = uuid.uuid4()
    ws = SimpleNamespace(settings={"voice_live": {"enabled": True}})
    existing = SimpleNamespace(id=uuid.uuid4(), user_id=7, workspace_id=ws_id, title="Voice call")
    updates = []

    db = MagicMock()

    def dispatch(model):
        q = MagicMock()
        if model is WorkspaceModel:
            q.filter.return_value.first.return_value = ws
        elif model is ChatModel:
            q.filter.return_value.first.return_value = existing
            q.filter.return_value.order_by.return_value.first.return_value = existing
        elif model is VoiceCall:
            def _upd(vals, **k):
                updates.append(vals)
                return 2

            q.filter.return_value.update.side_effect = _upd
        else:
            q.filter.return_value.first.return_value = None
        return q

    db.query.side_effect = dispatch

    out = _run_mint(ctx=_mint_ctx(ws_id=ws_id, user_int=7), db=db)

    assert out["call_id"] == "call_new"
    assert updates, "mint did not supersede prior calls"
    assert "superseded" in list(updates[0].values())  # status -> superseded


def test_voice_turn_has_the_full_brain(monkeypatch):
    """ONE Auto: the spoken turn keeps the SAME BRAIN as the typed turn —
    memory retrieval + core tools + the caller's REAL privilege tier, and
    NEVER force_text_only (the flag that skips memory). Two measured
    latency trims that do NOT touch the brain: history is trimmed (rhythm),
    and Composio's third-party EXECUTION surface is skipped (the 20-90s
    root cause — 58 tools/137 actions/24-36k tokens per call)."""
    import consumers.chatbot as chatbot_pkg

    import api.voice_retell as vr
    from config import config as app_config
    from modules.voice.call_binding import CallBinding

    captured = {}
    ws_id = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())

    class FakeChatService:
        def __init__(self, db):
            pass

        def save_message(self, **kwargs):
            return SimpleNamespace(id=uuid.uuid4())

        def get_messages_by_chat_id(self, cid):
            return [
                SimpleNamespace(role="user", parts=[{"type": "text", "text": f"m{i}"}])
                for i in range(40)  # a long archive — the call must not replay it
            ]

    class FakeStreaming:
        def __init__(self, db, workspace_id=None):
            pass

        def stream_response_with_agent(self, **kwargs):
            captured.update(kwargs)

            async def gen():
                yield '0:"hi"'

            return gen()

    monkeypatch.setattr(chatbot_pkg, "ChatService", FakeChatService)
    monkeypatch.setattr(chatbot_pkg, "StreamingChatService", FakeStreaming)

    import modules.voice.call_binding as cb

    monkeypatch.setattr(
        cb, "resolve_call_binding",
        lambda db, **k: CallBinding(
            chat_id=chat_id, user_id=7, workspace_id=ws_id, bound=True, is_super_admin=True
        ),
    )
    monkeypatch.setattr(cb, "stamp_assistant_voice_source", lambda db, **k: 0)

    import api.chat as chat_api

    monkeypatch.setattr(chat_api, "get_default_agent_id", lambda db, w: 1)

    import services.board_events as be

    monkeypatch.setattr(be, "notify_chat_event", lambda db, **k: None)

    import modules.voice.telemetry as tel

    monkeypatch.setattr(tel, "record_voice_turn", lambda **k: None)

    from contextlib import contextmanager

    @contextmanager
    def fake_session():
        yield MagicMock()

    monkeypatch.setattr(vr, "get_db_session", fake_session)

    from modules.voice.providers.retell import RetellLLMRequest

    req = RetellLLMRequest(
        response_id=1, user_text="hey", interaction_type="response_required",
        workspace_id=ws_id, agent_id=None, call_id="call_fast",
    )

    async def run():
        return [f async for f in vr._agent_retell_stream(req)]

    frames = asyncio.run(run())
    assert any(f.get("content") for f in frames)  # the turn streamed

    # the LOBOTOMY flag stays gone — memory + tools live (force_text_only
    # skips memory entirely; it must never be set on a conversational turn)
    assert captured.get("force_text_only", False) is False
    assert captured["is_super_admin"] is True  # mint-captured tier rides the binding
    # Composio EXECUTION is skipped for latency (memory/knowledge/core tools
    # stay) — the measured 20-90s fix, a config dial, NOT the memory lobotomy
    assert captured.get("skip_composio") is True
    # the other voice-specific trim: recent conversation, not the archive
    assert len(captured["messages"]) <= int(app_config.VOICE_LIVE_TURN_HISTORY_MESSAGES)


def test_mint_refuses_binding_to_someone_elses_chat(monkeypatch):
    from fastapi import HTTPException

    import api.voice_retell as vr
    from modules.voice.live_settings import RetellCredentials
    from modules.voice.voice_meter import MeterReading

    ws_id = uuid.uuid4()
    chat_id = uuid.uuid4()
    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(
        vr.live_settings, "retell_credentials", lambda: RetellCredentials("k", "s", "a")
    )
    monkeypatch.setattr(vr.voice_meter, "monthly_meter", lambda db, w: MeterReading(0, 0, 10))

    ws = SimpleNamespace(settings={"voice_live": {"enabled": True}})
    other_users_chat = SimpleNamespace(id=chat_id, user_id=99, workspace_id=ws_id)

    from api.voice_retell import MintWebCallRequest

    with pytest.raises(HTTPException) as exc:
        _run_mint(
            body=MintWebCallRequest(chat_id=str(chat_id)),
            ctx=_mint_ctx(ws_id=ws_id, user_int=7),
            db=_mint_db(workspace=ws, chat=other_users_chat),
        )
    assert exc.value.status_code == 403
    assert "your own chat" in exc.value.detail


# ---------------------------------------------------------------------------
# S2 · parse — the webhook reads the binding vars
# ---------------------------------------------------------------------------

def test_parse_llm_request_extracts_binding_vars():
    from modules.voice.providers.retell import parse_llm_request

    req = parse_llm_request(
        {
            "response_id": 3,
            "interaction_type": "response_required",
            "transcript": [{"role": "user", "content": "hey auto"}],
            "call": {
                "call_id": "call_1",
                "retell_llm_dynamic_variables": {
                    "workspace_id": "ws-1",
                    "user_id": "7",
                    "chat_id": "chat-9",
                    "agent_id": "2",
                },
            },
        }
    )
    assert (req.user_id, req.chat_id) == ("7", "chat-9")
    assert req.workspace_id == "ws-1" and req.call_id == "call_1"


def test_parse_llm_request_binding_vars_default_none():
    from modules.voice.providers.retell import parse_llm_request

    req = parse_llm_request({"interaction_type": "reminder_required", "response_id": 0})
    assert req.user_id is None and req.chat_id is None


# ---------------------------------------------------------------------------
# S2 · the trust boundary (DB seam — PRD-205 idiom: skip without Postgres)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text

    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT call_id, fallback_chat_id FROM voice_calls LIMIT 1"))
            c.execute(text("SELECT source FROM messages LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"voice-live suite needs Postgres with the 207 schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def new_session(engine):
    from sqlalchemy.orm import sessionmaker

    maker = sessionmaker(bind=engine)
    sessions = []

    def _make():
        s = maker()
        sessions.append(s)
        return s

    yield _make
    for s in sessions:
        try:
            s.rollback()
            s.close()
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture
def voice_workspace(engine, new_session):
    """A workspace + two real users (member owns it) + the owner's chat."""
    from sqlalchemy import text

    s = new_session()
    ws_id = str(uuid.uuid4())
    marker = uuid.uuid4().hex[:10]
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": f"voice-ws-{marker}"},
    )
    uids = []
    for i in range(2):
        uname = f"voice_{marker}_{i}"
        row = s.execute(
            text(
                "INSERT INTO users (username, email, clerk_user_id) "
                "VALUES (:un, :em, :cid) RETURNING id"
            ),
            {"un": uname, "em": f"{uname}@test.local", "cid": f"user_{uname}"},
        ).fetchone()
        uids.append(int(row[0]))
    s.execute(
        text(
            "INSERT INTO workspace_members (workspace_id, user_id, role) "
            "VALUES (CAST(:ws AS uuid), :u, 'owner')"
        ),
        {"ws": ws_id, "u": uids[0]},
    )
    chat_id = str(uuid.uuid4())
    # chats.visibility is NOT NULL with a Python-side default only (no
    # server_default, unlike chats.kind) — a raw INSERT must set it explicitly.
    s.execute(
        text(
            "INSERT INTO chats (id, user_id, workspace_id, title, visibility) "
            "VALUES (CAST(:id AS uuid), :u, CAST(:ws AS uuid), 'my thread', 'private')"
        ),
        {"id": chat_id, "u": uids[0], "ws": ws_id},
    )
    s.commit()

    yield SimpleNamespace(ws_id=ws_id, owner=uids[0], other=uids[1], chat_id=chat_id)

    sweep = new_session()
    for stmt, params in (
        ("DELETE FROM voice_turns WHERE workspace_id = CAST(:ws AS uuid)", {"ws": ws_id}),
        ("DELETE FROM voice_calls WHERE workspace_id = CAST(:ws AS uuid)", {"ws": ws_id}),
        ("DELETE FROM messages WHERE workspace_id = CAST(:ws AS uuid)", {"ws": ws_id}),
        ("DELETE FROM chats WHERE workspace_id = CAST(:ws AS uuid)", {"ws": ws_id}),
        ("DELETE FROM workspace_members WHERE workspace_id = CAST(:ws AS uuid)", {"ws": ws_id}),
        ("DELETE FROM workspaces WHERE id = CAST(:ws AS uuid)", {"ws": ws_id}),
        ("DELETE FROM users WHERE id = ANY(:ids)", {"ids": uids}),
    ):
        sweep.execute(text(stmt), params)
    sweep.commit()


def _mint_row(session, *, call_id, ws_id, user_id, chat_id=None):
    from sqlalchemy import text

    session.execute(
        text(
            "INSERT INTO voice_calls (call_id, workspace_id, user_id, chat_id, status) "
            "VALUES (:c, CAST(:ws AS uuid), :u, :chat, 'minted')"
        ),
        {"c": call_id, "ws": ws_id, "u": user_id, "chat": chat_id},
    )
    session.commit()


def test_webhook_binds_to_existing_chat_and_user(voice_workspace, new_session):
    from modules.voice.call_binding import resolve_call_binding

    s = new_session()
    _mint_row(
        s, call_id="call_bind1", ws_id=voice_workspace.ws_id,
        user_id=voice_workspace.owner, chat_id=voice_workspace.chat_id,
    )
    b = resolve_call_binding(
        s,
        call_id="call_bind1",
        workspace_id=voice_workspace.ws_id,
        user_id_var=str(voice_workspace.owner),
        chat_id_var=voice_workspace.chat_id,
        first_text="hello",
    )
    assert b is not None and b.bound is True
    assert b.chat_id == voice_workspace.chat_id  # the on-screen thread IS the transcript
    assert b.user_id == voice_workspace.owner


def test_webhook_rejects_vars_mismatching_mint_row(voice_workspace, new_session):
    from modules.voice.call_binding import resolve_call_binding

    s = new_session()
    _mint_row(
        s, call_id="call_bind2", ws_id=voice_workspace.ws_id,
        user_id=voice_workspace.owner, chat_id=voice_workspace.chat_id,
    )
    # var claims a DIFFERENT user than the mint row proved → never the bound thread
    b = resolve_call_binding(
        s,
        call_id="call_bind2",
        workspace_id=voice_workspace.ws_id,
        user_id_var=str(voice_workspace.other),
        chat_id_var=voice_workspace.chat_id,
        first_text="spoof",
    )
    assert b is not None and b.bound is False
    assert b.chat_id != voice_workspace.chat_id  # fail-closed to the per-call chat
    assert b.user_id == voice_workspace.owner  # attributed to the MINT-proven user


def test_webhook_rejects_workspace_mismatch_outright(voice_workspace, new_session):
    from modules.voice.call_binding import resolve_call_binding

    s = new_session()
    _mint_row(
        s, call_id="call_bind3", ws_id=voice_workspace.ws_id,
        user_id=voice_workspace.owner, chat_id=voice_workspace.chat_id,
    )
    b = resolve_call_binding(
        s,
        call_id="call_bind3",
        workspace_id=str(uuid.uuid4()),  # not the minted workspace
        user_id_var=str(voice_workspace.owner),
        chat_id_var=voice_workspace.chat_id,
    )
    assert b is None  # no proven workspace → refuse the turn entirely


def test_webhook_binds_from_mint_row_when_vars_absent(voice_workspace, new_session):
    """First-live-contact regression: Retell omits dynamic vars unless the
    config frame asks — the mint row ALONE must authorise binding (it is the
    server-born source the vars merely echoed)."""
    from modules.voice.call_binding import resolve_call_binding

    s = new_session()
    _mint_row(
        s, call_id="call_novars", ws_id=voice_workspace.ws_id,
        user_id=voice_workspace.owner, chat_id=voice_workspace.chat_id,
    )
    b = resolve_call_binding(
        s,
        call_id="call_novars",
        workspace_id=None,  # no vars arrived at all
        user_id_var=None,
        chat_id_var=None,
        first_text="hello",
    )
    assert b is not None and b.bound is True
    assert b.chat_id == voice_workspace.chat_id
    assert b.user_id == voice_workspace.owner
    assert b.workspace_id == voice_workspace.ws_id  # row truth, not var


def test_webhook_fallback_chat_is_stable_across_turns(voice_workspace, new_session):
    """The per-turn-chat bug is dead: two turns of one call share ONE thread."""
    from modules.voice.call_binding import resolve_call_binding

    s = new_session()
    _mint_row(
        s, call_id="call_bind4", ws_id=voice_workspace.ws_id,
        user_id=voice_workspace.owner, chat_id=None,  # welcome-screen mint: no thread yet
    )
    kwargs = dict(
        call_id="call_bind4",
        workspace_id=voice_workspace.ws_id,
        user_id_var=str(voice_workspace.owner),
        chat_id_var=None,
        first_text="turn one",
    )
    b1 = resolve_call_binding(s, **kwargs)
    b2 = resolve_call_binding(s, **kwargs)
    assert b1 is not None and b2 is not None
    assert b1.chat_id == b2.chat_id  # remembered on voice_calls.fallback_chat_id
    assert b1.user_id == voice_workspace.owner


def test_phone_lane_attributes_to_workspace_steward(voice_workspace, new_session):
    """Unminted call (phone lane): loud orphan row + steward attribution —
    the old user_id=0 fallback violated the chats.user_id FK and could never
    have worked."""
    from sqlalchemy import text

    from modules.voice.call_binding import resolve_call_binding

    s = new_session()
    b = resolve_call_binding(
        s,
        call_id="call_phone1",
        workspace_id=voice_workspace.ws_id,
        user_id_var=None,
        chat_id_var=None,
        first_text="phone hello",
    )
    assert b is not None and b.bound is False
    assert b.user_id == voice_workspace.owner  # the earliest owner is the steward
    row = s.execute(
        text("SELECT status, fallback_chat_id FROM voice_calls WHERE call_id = 'call_phone1'")
    ).fetchone()
    assert row is not None and row[1] == b.chat_id  # orphan registered for reuse


# ---------------------------------------------------------------------------
# S7 · one-click arming — the card does the whole job
# ---------------------------------------------------------------------------

def _admin_ctx():
    return SimpleNamespace(
        workspace_id=uuid.uuid4(),
        user=SimpleNamespace(id=7, system_role="admin"),
        auth_type="clerk",
    )


def _arm_env(monkeypatch, *, creds, workspace=None):
    """Wire the arm endpoint's collaborators to recorders."""
    import api.voice_retell as vr

    # The endpoint calls SQLAlchemy's flag_modified to track the in-place JSONB
    # settings mutation; the fake workspace is a SimpleNamespace (no ORM state),
    # so neutralise that plumbing call — the test asserts the settings dict.
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda *a, **k: None)

    written = {}
    monkeypatch.setattr(
        vr.live_settings, "set_voice_setting", lambda db, k, v: written.__setitem__(k, v)
    )
    monkeypatch.setattr(vr.live_settings, "retell_credentials", lambda: creds)

    created = {}

    async def fake_create(api_key, **kwargs):
        created.update(kwargs, api_key=api_key)
        return "agent_new_1"

    monkeypatch.setattr(vr.retell_api, "create_custom_llm_agent", fake_create)

    # The re-tune (existing-agent arm) hits Retell too — stub it by default so
    # arm tests never make a real PATCH; a test that cares overrides this.
    async def fake_update(api_key, agent_id, settings):
        return None

    monkeypatch.setattr(vr.retell_api, "update_agent", fake_update)

    db = _mint_db(workspace=workspace)
    return vr, written, created, db


def test_arm_requires_admin(monkeypatch):
    from fastapi import HTTPException

    import api.voice_retell as vr
    from api.voice_retell import ArmVoiceRequest

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            vr.arm_voice_live(ArmVoiceRequest(api_key="key_x"), ctx=_mint_ctx(), db=MagicMock())
        )
    assert exc.value.status_code == 403


def test_arm_without_key_is_an_honest_400(monkeypatch):
    from fastapi import HTTPException

    from api.voice_retell import ArmVoiceRequest
    from modules.voice.live_settings import RetellCredentials

    vr, written, created, db = _arm_env(monkeypatch, creds=RetellCredentials("", "", ""))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(vr.arm_voice_live(ArmVoiceRequest(), ctx=_admin_ctx(), db=db))
    assert exc.value.status_code == 400
    assert "API key" in exc.value.detail
    assert written == {} and created == {}


def test_arm_creates_agent_stores_all_and_sweeps_misfiled_key(monkeypatch):
    from api.voice_retell import ArmVoiceRequest
    from modules.voice.live_settings import RetellCredentials

    stray_key = "key_" + "misfiled"
    ws = SimpleNamespace(
        id=uuid.uuid4(),
        settings={"voice_live": {"enabled": True, "retell_voice_id": stray_key}},
    )
    vr, written, created, db = _arm_env(
        monkeypatch, creds=RetellCredentials("", "", ""), workspace=ws
    )

    out = asyncio.run(
        vr.arm_voice_live(ArmVoiceRequest(api_key="key_real"), ctx=_admin_ctx(), db=db)
    )

    assert out == {"armed": True, "platform_enabled": True, "agent_id": "agent_new_1"}
    # the server built the transport URLs — nobody hand-copies wss strings
    assert created["llm_websocket_url"].startswith("wss://")
    assert created["llm_websocket_url"].endswith("/api/voice/retell/llm-websocket")
    assert created["webhook_url"].endswith("/api/voice/retell/events")
    # all four slots written; signing key defaults to the API key
    assert written["retell_api_key"] == "key_real"
    assert written["retell_webhook_secret"] == "key_real"
    assert written["retell_agent_id"] == "agent_new_1"
    assert written["live_enabled"] == "true"
    # the mis-filed key is swept out of the workspace voice field
    assert "retell_voice_id" not in ws.settings["voice_live"]
    assert db.commit.called


def test_arm_is_idempotent_when_agent_exists(monkeypatch):
    from api.voice_retell import ArmVoiceRequest
    from modules.voice.live_settings import RetellCredentials

    vr, written, created, db = _arm_env(
        monkeypatch, creds=RetellCredentials("key_old", "sec_old", "agent_existing")
    )
    out = asyncio.run(
        vr.arm_voice_live(ArmVoiceRequest(api_key="key_rotated"), ctx=_admin_ctx(), db=db)
    )
    assert created == {}  # no second agent — the existing one is kept
    assert out["agent_id"] == "agent_existing"
    assert written["retell_api_key"] == "key_rotated"
    # an explicitly-configured distinct signing secret is never overwritten
    assert "retell_webhook_secret" not in written


def test_agent_tuning_pins_language_and_denoises():
    """The STT/turn-taking tuning that keeps transcription honest in a noisy
    room: pinned language, denoising, accurate STT, sane interruption
    sensitivity, no backchannel.

    ``noise-and-background-speech-cancellation`` was WITHDRAWN after live
    measurement — the aggressive mode cancelled the speaker himself at normal
    volume (a whole call logged turns=0 until he shouted). Plain
    ``noise-cancellation`` filters the room and keeps the person.
    """
    from modules.voice.retell_api import build_agent_tuning

    t = build_agent_tuning()
    assert t["language"]  # pinned, never multilingual auto (hallucination source)
    assert t["denoising_mode"] == "noise-cancellation"
    assert t["stt_mode"] == "accurate"
    assert 0.2 < t["interruption_sensitivity"] <= 0.7  # barge-in works, noise doesn't
    assert t["enable_backchannel"] is False


def test_arm_retunes_the_existing_agent(monkeypatch):
    """A one-click re-arm re-applies the tuning to the ALREADY-armed agent —
    the fix path for a live agent producing confident-nonsense transcripts."""
    from api.voice_retell import ArmVoiceRequest
    from modules.voice.live_settings import RetellCredentials

    vr, written, created, db = _arm_env(
        monkeypatch, creds=RetellCredentials("key_k", "sec_k", "agent_existing")
    )
    tuned = {}

    async def capture(api_key, agent_id, settings):
        tuned.update(agent_id=agent_id, **settings)

    monkeypatch.setattr(vr.retell_api, "update_agent", capture)

    asyncio.run(vr.arm_voice_live(ArmVoiceRequest(), ctx=_admin_ctx(), db=db))

    assert created == {}  # existing agent kept, not recreated
    assert tuned["agent_id"] == "agent_existing"
    assert tuned["denoising_mode"] == "noise-and-background-speech-cancellation"
    assert tuned["language"]


def test_disarm_flips_toggle_only(monkeypatch):
    from api.voice_retell import ArmVoiceRequest
    from modules.voice.live_settings import RetellCredentials

    vr, written, created, db = _arm_env(
        monkeypatch, creds=RetellCredentials("key_k", "key_k", "agent_1")
    )
    out = asyncio.run(
        vr.arm_voice_live(ArmVoiceRequest(enabled=False), ctx=_admin_ctx(), db=db)
    )
    assert written == {"live_enabled": "false"}  # creds untouched — instant re-arm
    assert created == {}
    assert out["platform_enabled"] is False


# ---------------------------------------------------------------------------
# S2 · the custom-LLM WebSocket (Retell's actual transport)
# ---------------------------------------------------------------------------

class _FakeWebSocket:
    """Scripted Retell-side socket. receive_json yields a loop tick first so
    a just-created respond() task gets to start before the next message."""

    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent = []
        self.accepted = False
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000):
        self.close_code = code

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_json(self):
        from fastapi import WebSocketDisconnect

        await asyncio.sleep(0)
        if not self.incoming:
            raise WebSocketDisconnect(1000)
        return self.incoming.pop(0)


def _ws_db(minted: bool):
    from contextlib import contextmanager

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = (
        (1,) if minted else None
    )

    @contextmanager
    def fake_session():
        yield db

    return fake_session


def test_ws_wrap_shape():
    from modules.voice.providers.retell import wrap_ws_response

    assert wrap_ws_response({"response_id": 3, "content": "hi", "content_complete": False}) == {
        "response_type": "response",
        "response_id": 3,
        "content": "hi",
        "content_complete": False,
    }


def test_ws_closes_4403_when_platform_off(monkeypatch):
    import api.voice_retell as vr

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: False)
    ws = _FakeWebSocket([])
    asyncio.run(vr.retell_llm_websocket(ws, "call_x"))
    assert ws.close_code == 4403
    assert ws.accepted is False  # the kill-switch refuses before accept


def test_ws_closes_4401_for_unminted_call(monkeypatch):
    import api.voice_retell as vr

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(vr, "get_db_session", _ws_db(minted=False))
    ws = _FakeWebSocket([])
    asyncio.run(vr.retell_llm_websocket(ws, "call_forged"))
    assert ws.close_code == 4401  # the minted call_id IS the credential
    assert ws.accepted is False


def test_ws_superseded_call_closes_without_speaking(monkeypatch):
    """One live call per user: a call a newer mint marked 'superseded' must
    NOT speak on its next turn — it closes instead. Several superseded calls
    on one mic each streaming a reply is the 'Auto talking over herself' bug."""
    import api.voice_retell as vr

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)

    from contextlib import contextmanager

    db = MagicMock()
    # open-time minted check (.first()) passes; the per-turn status check
    # (.scalar()) reports this call was superseded by a newer mint.
    db.query.return_value.filter.return_value.first.return_value = (1,)
    db.query.return_value.filter.return_value.scalar.return_value = "superseded"

    @contextmanager
    def fake_session():
        yield db

    monkeypatch.setattr(vr, "get_db_session", fake_session)

    async def must_not_stream(req):
        raise AssertionError("a superseded call must never reach the brain")
        yield  # pragma: no cover

    monkeypatch.setattr(vr, "_agent_retell_stream", must_not_stream)

    ws = _FakeWebSocket(
        [
            {"interaction_type": "call_details",
             "call": {"retell_llm_dynamic_variables": {"workspace_id": "ws-9"}}},
            {"interaction_type": "response_required", "response_id": 1,
             "transcript": [{"role": "user", "content": "hello"}]},
        ]
    )
    asyncio.run(vr.retell_llm_websocket(ws, "call_superseded"))

    assert ws.close_code == 1000  # closed cleanly, did not speak
    # no turn response frame was ever sent (only the config + begin handshake)
    assert not any(
        m.get("response_type") == "response" and m.get("response_id") == 1
        for m in ws.sent
    )


def test_ws_rejects_superseded_call_at_open(monkeypatch):
    """A superseded call must be refused BEFORE accept — so an auto_reconnect
    of a call a newer mint replaced can't loop back and start a second voice."""
    import api.voice_retell as vr

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)

    from contextlib import contextmanager

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = ("superseded",)

    @contextmanager
    def fake_session():
        yield db

    monkeypatch.setattr(vr, "get_db_session", fake_session)

    ws = _FakeWebSocket([])
    asyncio.run(vr.retell_llm_websocket(ws, "call_superseded_open"))
    assert ws.close_code == 4409
    assert ws.accepted is False


def test_ws_user_speaks_first_then_answers_with_call_details_vars(monkeypatch):
    import api.voice_retell as vr

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(vr, "get_db_session", _ws_db(minted=True))

    seen_reqs = []

    async def fake_stream(req):
        seen_reqs.append(req)
        yield {"response_id": req.response_id, "content": "hello there", "content_complete": False}
        yield {"response_id": req.response_id, "content": "", "content_complete": True}

    monkeypatch.setattr(vr, "_agent_retell_stream", fake_stream)

    ws = _FakeWebSocket(
        [
            {"interaction_type": "ping_pong", "timestamp": 123},
            {
                "interaction_type": "call_details",
                "call": {
                    "call_id": "call_ws1",
                    "retell_llm_dynamic_variables": {
                        "workspace_id": "ws-9", "user_id": "7", "chat_id": "c-1",
                    },
                },
            },
            {
                "interaction_type": "response_required",
                "response_id": 1,
                "transcript": [{"role": "user", "content": "hey auto"}],
            },
        ]
    )
    asyncio.run(vr.retell_llm_websocket(ws, "call_ws1"))

    # handshake per Retell's demo: config FIRST; the empty floor-yielding
    # begin goes out only AFTER call_details arrives (not at accept)
    assert ws.sent[0] == {
        "response_type": "config", "config": {"auto_reconnect": True, "call_details": True},
    }
    begin = {"response_type": "response", "response_id": 0, "content": "", "content_complete": True}
    assert begin in ws.sent
    ping_reply_idx = next(i for i, m in enumerate(ws.sent) if m.get("response_type") == "ping_pong")
    assert ws.sent.index(begin) > ping_reply_idx  # begin followed call_details, not accept
    # ping echoed
    assert {"response_type": "ping_pong", "timestamp": 123} in ws.sent
    # the turn used the call_details vars + path call_id, and frames are wrapped
    assert len(seen_reqs) == 1
    req = seen_reqs[0]
    assert (req.workspace_id, req.user_id, req.chat_id, req.call_id) == ("ws-9", "7", "c-1", "call_ws1")
    assert req.user_text == "hey auto"
    assert {"response_type": "response", "response_id": 1, "content": "hello there",
            "content_complete": False} in ws.sent


def test_ws_new_turn_supersedes_streaming_one(monkeypatch):
    import api.voice_retell as vr

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(vr, "get_db_session", _ws_db(minted=True))

    events = []

    async def fake_stream(req):
        events.append(f"start:{req.response_id}")
        try:
            if req.response_id == 1:
                yield {"response_id": 1, "content": "long answer…", "content_complete": False}
                await asyncio.Event().wait()  # streams forever until cancelled
            else:
                yield {"response_id": 2, "content": "fresh", "content_complete": True}
        finally:
            if req.response_id == 1:
                events.append("cancelled:1")

    monkeypatch.setattr(vr, "_agent_retell_stream", fake_stream)

    turn = lambda rid: {
        "interaction_type": "response_required",
        "response_id": rid,
        "transcript": [{"role": "user", "content": f"turn {rid}"}],
    }
    ws = _FakeWebSocket(
        [
            {"interaction_type": "call_details",
             "call": {"retell_llm_dynamic_variables": {"workspace_id": "ws-9"}}},
            turn(1),
            turn(2),
        ]
    )
    asyncio.run(vr.retell_llm_websocket(ws, "call_ws2"))

    assert "start:1" in events and "start:2" in events
    assert "cancelled:1" in events  # barge-in: the superseded stream was stopped
    assert {"response_type": "response", "response_id": 2, "content": "fresh",
            "content_complete": True} in ws.sent


# ---------------------------------------------------------------------------
# Guard · the live path never touches the 120s-TTS pod client
# ---------------------------------------------------------------------------

def test_live_path_never_imports_pod_voice_client():
    """§6 grep-guard: the Retell lane must never import modules/voice/client
    (the blocking self-hosted pod path this PRD replaces for live mode).
    Word-boundary aware — the PRD-200 substring-scan lesson."""
    import re

    live_files = (
        "api/voice_retell.py",
        "modules/voice/retell_api.py",
        "modules/voice/call_binding.py",
        "modules/voice/live_settings.py",
        "modules/voice/voice_meter.py",
    )
    pattern = re.compile(
        r"from\s+modules\.voice\.client\s+import|from\s+modules\.voice\s+import\s+([\w,\s]*\b)?client\b|import\s+modules\.voice\.client\b"
    )
    offenders = [
        rel
        for rel in live_files
        if pattern.search((_orchestrator_root / rel).read_text(encoding="utf-8"))
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# S3 · lifecycle events — idempotent updates, loud orphans, HMAC fail-closed
# ---------------------------------------------------------------------------

def test_call_lifecycle_updates_minted_row_idempotently(voice_workspace, new_session):
    from sqlalchemy import text

    from api.voice_retell import _apply_call_event

    s = new_session()
    _mint_row(
        s, call_id="call_life1", ws_id=voice_workspace.ws_id,
        user_id=voice_workspace.owner, chat_id=voice_workspace.chat_id,
    )
    start_ms, end_ms = 1_700_000_000_000, 1_700_000_090_000  # 90s call

    _apply_call_event(s, "call_started", {"call_id": "call_life1", "start_timestamp": start_ms})
    for _ in range(2):  # Retell retries → replays must be harmless
        _apply_call_event(
            s,
            "call_ended",
            {
                "call_id": "call_life1",
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
                "disconnection_reason": "user_hangup",
            },
        )

    row = s.execute(
        text(
            "SELECT status, duration_seconds, disconnect_reason FROM voice_calls "
            "WHERE call_id = 'call_life1'"
        )
    ).fetchone()
    assert row[0] == "ended"
    assert row[1] == 90
    assert row[2] == "user_hangup"


def test_call_ended_without_start_marks_failed(voice_workspace, new_session):
    from sqlalchemy import text

    from api.voice_retell import _apply_call_event

    s = new_session()
    _mint_row(
        s, call_id="call_life2", ws_id=voice_workspace.ws_id,
        user_id=voice_workspace.owner,
    )
    _apply_call_event(
        s, "call_ended", {"call_id": "call_life2", "disconnection_reason": "dial_failed"}
    )
    row = s.execute(
        text("SELECT status FROM voice_calls WHERE call_id = 'call_life2'")
    ).fetchone()
    assert row[0] == "failed"  # never connected — not billable minutes


def test_unknown_call_id_is_loud_orphan(voice_workspace, new_session):
    from sqlalchemy import text

    from api.voice_retell import _apply_call_event

    s = new_session()
    disposition = _apply_call_event(
        s,
        "call_started",
        {"call_id": "call_never_minted", "start_timestamp": 1_700_000_000_000},
    )
    assert disposition == "orphan_created"
    row = s.execute(
        text("SELECT workspace_id, status FROM voice_calls WHERE call_id = 'call_never_minted'")
    ).fetchone()
    assert row is not None  # stored, not dropped
    assert row[0] is None  # unattributed — visibly an orphan
    s.execute(text("DELETE FROM voice_calls WHERE call_id = 'call_never_minted'"))
    s.commit()


class _FakeRequest:
    def __init__(self, body: bytes, signature=None):
        self._body = body
        self.headers = {"x-retell-signature": signature} if signature else {}

    async def body(self):
        return self._body


def test_events_webhook_hmac_fail_closed(monkeypatch):
    import hashlib
    import hmac as hmac_mod
    import json as json_mod
    from contextlib import contextmanager

    from fastapi import HTTPException

    import api.voice_retell as vr
    from modules.voice.live_settings import RetellCredentials

    secret = "whsec_" + "test"  # concat: never a literal secret-shaped token
    monkeypatch.setattr(
        vr.live_settings, "retell_credentials",
        lambda: RetellCredentials("k", secret, "a"),
    )
    applied = []
    monkeypatch.setattr(vr, "_apply_call_event", lambda db, e, c: applied.append(e) or "updated")

    @contextmanager
    def fake_session():
        yield MagicMock()

    monkeypatch.setattr(vr, "get_db_session", fake_session)

    body = json_mod.dumps({"event": "call_started", "call": {"call_id": "c1"}}).encode()

    # wrong signature → 401, nothing applied
    with pytest.raises(HTTPException) as exc:
        asyncio.run(vr.retell_events_webhook(_FakeRequest(body, "deadbeef")))
    assert exc.value.status_code == 401
    assert applied == []

    # correct signature (Retell's v={ts},d={hmac(body+ts)} scheme) → applied
    import time as time_mod

    ts = str(int(time_mod.time() * 1000))
    digest = hmac_mod.new(secret.encode(), body + ts.encode(), hashlib.sha256).hexdigest()
    out = asyncio.run(vr.retell_events_webhook(_FakeRequest(body, f"v={ts},d={digest}")))
    assert out == {"ok": True}
    assert applied == ["call_started"]


def test_monthly_minutes_rollup(voice_workspace, new_session):
    from datetime import datetime

    from sqlalchemy import text

    from modules.voice.voice_meter import monthly_meter

    s = new_session()
    now = datetime.utcnow()
    for i, dur in enumerate((90, 90)):  # 3 ended minutes this month
        s.execute(
            text(
                "INSERT INTO voice_calls (call_id, workspace_id, user_id, status, "
                " started_at, ended_at, duration_seconds) "
                "VALUES (:c, CAST(:ws AS uuid), :u, 'ended', :st, :en, :d)"
            ),
            {
                "c": f"call_meter{i}", "ws": voice_workspace.ws_id,
                "u": voice_workspace.owner, "st": now, "en": now, "d": dur,
            },
        )
    s.execute(  # one live call → reserves
        text(
            "INSERT INTO voice_calls (call_id, workspace_id, user_id, status, started_at) "
            "VALUES ('call_meter_live', CAST(:ws AS uuid), :u, 'started', :st)"
        ),
        {"ws": voice_workspace.ws_id, "u": voice_workspace.owner, "st": now},
    )
    s.commit()

    reading = monthly_meter(s, uuid.UUID(voice_workspace.ws_id))
    assert reading.ended_minutes == 3
    assert reading.active_calls == 1
    assert reading.reserved_minutes == reading.reserve_minutes_per_call


def test_voice_live_put_refuses_malformed_with_honest_reason():
    from fastapi import HTTPException

    from api.workspaces import save_voice_live_settings

    ws = SimpleNamespace(id=uuid.uuid4(), settings={"voice_live": {"enabled": False}})
    db = MagicMock()
    db.query.return_value.get.return_value = ws

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            save_voice_live_settings(
                payload={"voice_live": {"enabled": "yes"}}, ctx=_mint_ctx(), db=db
            )
        )
    assert exc.value.status_code == 400
    assert "boolean" in exc.value.detail
    assert ws.settings["voice_live"]["enabled"] is False  # nothing written


def test_voice_live_put_merges_not_replaces(monkeypatch):
    from api.workspaces import save_voice_live_settings

    # flag_modified is SQLAlchemy plumbing for the in-place JSONB mutation;
    # the fake workspace has no ORM state, so neutralise it — the test asserts
    # the merged settings dict, not the change-tracking call.
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda *a, **k: None)

    ws = SimpleNamespace(
        id=uuid.uuid4(), settings={"voice_live": {"enabled": True, "retell_voice_id": "keep"}}
    )
    db = MagicMock()
    db.query.return_value.get.return_value = ws

    out = asyncio.run(
        save_voice_live_settings(
            payload={"voice_live": {"monthly_cap_minutes": 200}}, ctx=_mint_ctx(), db=db
        )
    )
    assert out["voice_live"] == {
        "enabled": True,
        "retell_voice_id": "keep",
        "monthly_cap_minutes": 200,
    }
    assert db.commit.called


def test_live_status_payload_never_leaks_credentials(monkeypatch):
    import api.voice_retell as vr
    from modules.voice.live_settings import RetellCredentials
    from modules.voice.voice_meter import MeterReading

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(
        vr.live_settings, "retell_credentials",
        lambda: RetellCredentials("k_secret", "s_secret", "agent_1"),
    )
    monkeypatch.setattr(vr.voice_meter, "monthly_meter", lambda db, w: MeterReading(12, 1, 10))

    ws = SimpleNamespace(settings={"voice_live": {"enabled": True, "monthly_cap_minutes": 100}})
    out = asyncio.run(vr.voice_live_status(ctx=_mint_ctx(), db=_mint_db(workspace=ws)))

    assert out["platform_enabled"] is True and out["armed"] is True
    assert out["used_minutes"] == 12 and out["cap_minutes"] == 100
    blob = str(out)
    assert "k_secret" not in blob and "s_secret" not in blob  # presence only, never values


def test_spoken_sentence_grows_one_message(voice_workspace, new_session):
    """First live use: pausing mid-sentence stacked growing prefixes as
    separate messages. A continued utterance must UPDATE the message it grew
    from; a genuinely new utterance appends."""
    from sqlalchemy import text as sql

    from modules.voice.call_binding import upsert_voice_user_message

    s = new_session()
    grow = [
        "Is the chat, like, a single chat",
        "Is the chat, like, a single chat voice chat?",
        "Is the chat, like, a single chat voice chat, or is it mixed?",
    ]
    for t in grow:
        upsert_voice_user_message(
            s, chat_id=voice_workspace.chat_id, workspace_id=voice_workspace.ws_id, text=t
        )
    upsert_voice_user_message(
        s, chat_id=voice_workspace.chat_id, workspace_id=voice_workspace.ws_id,
        text="Different topic entirely.",
    )

    rows = s.execute(
        sql(
            "SELECT parts FROM messages WHERE chat_id = CAST(:c AS uuid) "
            "AND role = 'user' ORDER BY created_at"
        ),
        {"c": voice_workspace.chat_id},
    ).fetchall()
    texts = [r[0][0]["text"] for r in rows]
    assert texts == [grow[-1], "Different topic entirely."]  # one grown + one new


def test_assistant_stamp_bounded_to_turn_window(voice_workspace, new_session):
    from datetime import datetime, timedelta

    from sqlalchemy import text

    from modules.voice.call_binding import stamp_assistant_voice_source

    s = new_session()
    turn_start = datetime.utcnow()
    old = str(uuid.uuid4())
    fresh = str(uuid.uuid4())
    s.execute(
        text(
            "INSERT INTO messages (id, chat_id, workspace_id, role, parts, created_at) VALUES "
            "(CAST(:old AS uuid), CAST(:chat AS uuid), CAST(:ws AS uuid), 'assistant', '[]', :before), "
            "(CAST(:fresh AS uuid), CAST(:chat AS uuid), CAST(:ws AS uuid), 'assistant', '[]', :after)"
        ),
        {
            "old": old, "fresh": fresh, "chat": voice_workspace.chat_id,
            "ws": voice_workspace.ws_id,
            "before": turn_start - timedelta(minutes=5),
            "after": turn_start + timedelta(seconds=1),
        },
    )
    s.commit()

    stamped = stamp_assistant_voice_source(
        s, chat_id=voice_workspace.chat_id, turn_started_at=turn_start
    )
    assert stamped == 1
    rows = {
        str(r[0]): r[1]
        for r in s.execute(
            text("SELECT id, source FROM messages WHERE chat_id = CAST(:c AS uuid)"),
            {"c": voice_workspace.chat_id},
        ).fetchall()
    }
    assert rows[old] is None  # the pre-turn text reply keeps no voice badge
    assert rows[fresh] == {"origin": "voice", "label": "Auto · voice"}


# ---------------------------------------------------------------------------
# S2 · WS liveness — the brain must never starve the socket's event loop
# (first armed morning: sync I/O inside the turn froze the loop, Retell's 2s
# ping_pong went unanswered ~5s, the socket died 1006 mid-generation with
# frames=0 — every brained turn was killed by our own starved loop).
# ---------------------------------------------------------------------------

class _TailWaitWebSocket(_FakeWebSocket):
    """Keeps the socket open briefly after the script drains so in-flight
    respond()/watchdog tasks land their sends before the disconnect."""

    def __init__(self, incoming, tail_wait: float):
        super().__init__(incoming)
        self.tail_wait = tail_wait

    async def receive_json(self):
        from fastapi import WebSocketDisconnect

        await asyncio.sleep(0)
        if not self.incoming:
            await asyncio.sleep(self.tail_wait)
            raise WebSocketDisconnect(1000)
        return self.incoming.pop(0)


class _DyingWebSocket(_FakeWebSocket):
    """send_json severs once the turn has sent `fail_after` response frames."""

    def __init__(self, incoming, fail_after: int):
        super().__init__(incoming)
        self.fail_after = fail_after
        self._turn_sends = 0

    async def send_json(self, data):
        if data.get("response_type") == "response" and data.get("response_id") == 1:
            self._turn_sends += 1
            if self._turn_sends > self.fail_after:
                raise RuntimeError("socket severed")
        self.sent.append(data)


def _turn_msg(rid: int, text: str = "hey auto") -> dict:
    return {
        "interaction_type": "response_required",
        "response_id": rid,
        "transcript": [{"role": "user", "content": text}],
    }


_CALL_DETAILS_MSG = {
    "interaction_type": "call_details",
    "call": {"retell_llm_dynamic_variables": {"workspace_id": "ws-9"}},
}


def test_stream_bridge_isolates_the_brain_on_a_worker_thread(monkeypatch):
    """The REAL _agent_retell_stream runs the turn on a dedicated thread —
    the WS loop stays free to echo Retell's pings while tools grind."""
    import threading as _threading

    import api.voice_retell as vr
    from modules.voice.providers.retell import RetellLLMRequest

    seen = {}

    async def fake_inner(req):
        seen["thread"] = _threading.get_ident()
        yield {"response_id": req.response_id, "content": "hi", "content_complete": False}
        yield {"response_id": req.response_id, "content": "", "content_complete": True}

    monkeypatch.setattr(vr, "_agent_retell_stream_inner", fake_inner)
    req = RetellLLMRequest(
        response_id=1, user_text="hey", interaction_type="response_required",
        workspace_id="ws-9", agent_id=None, call_id="call_bridge1",
    )

    async def run():
        return [f async for f in vr._agent_retell_stream(req)]

    frames = asyncio.run(run())
    assert [f["content"] for f in frames] == ["hi", ""]
    assert seen["thread"] != _threading.get_ident()  # off the caller's thread


def test_stream_bridge_propagates_inner_errors(monkeypatch):
    import api.voice_retell as vr
    from modules.voice.providers.retell import RetellLLMRequest

    async def fake_inner(req):
        yield {"response_id": 1, "content": "a", "content_complete": False}
        raise RuntimeError("brain fault")

    monkeypatch.setattr(vr, "_agent_retell_stream_inner", fake_inner)
    req = RetellLLMRequest(
        response_id=1, user_text="hey", interaction_type="response_required",
        workspace_id="ws-9", agent_id=None, call_id="call_bridge2",
    )

    async def run():
        return [f async for f in vr._agent_retell_stream(req)]

    with pytest.raises(RuntimeError, match="brain fault"):
        asyncio.run(run())


def test_stream_bridge_close_cancels_the_inner_turn(monkeypatch):
    """Closing the bridge (barge-in cancel) reaches the worker: the inner
    generator's finally runs, so sessions and telemetry never leak."""
    import threading as _threading

    import api.voice_retell as vr
    from modules.voice.providers.retell import RetellLLMRequest

    inner_finally = _threading.Event()

    async def fake_inner(req):
        try:
            yield {"response_id": 1, "content": "first", "content_complete": False}
            await asyncio.Event().wait()  # parks until cancelled
        finally:
            inner_finally.set()

    monkeypatch.setattr(vr, "_agent_retell_stream_inner", fake_inner)
    req = RetellLLMRequest(
        response_id=1, user_text="hey", interaction_type="response_required",
        workspace_id="ws-9", agent_id=None, call_id="call_bridge3",
    )

    async def run():
        agen = vr._agent_retell_stream(req)
        first = await agen.__anext__()
        await agen.aclose()
        return first

    first = asyncio.run(run())
    assert first["content"] == "first"
    assert inner_finally.wait(timeout=3.0)


def test_ws_slow_first_frame_speaks_an_honest_ack(monkeypatch):
    """No first frame by the deadline → ONE short spoken acknowledgment
    (content_complete=False, same rid) — then the real answer streams."""
    import api.voice_retell as vr
    from config import config as app_config

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(vr, "get_db_session", _ws_db(minted=True))
    monkeypatch.setattr(app_config, "VOICE_LIVE_FIRST_FRAME_ACK_SECONDS", 0.05)
    monkeypatch.setattr(app_config, "VOICE_LIVE_FIRST_FRAME_ACK_TEXT", "One moment.")

    async def slow_stream(req):
        await asyncio.sleep(0.25)
        yield {"response_id": req.response_id, "content": "the answer", "content_complete": True}

    monkeypatch.setattr(vr, "_agent_retell_stream", slow_stream)

    ws = _TailWaitWebSocket([_CALL_DETAILS_MSG, _turn_msg(1)], tail_wait=0.6)
    asyncio.run(vr.retell_llm_websocket(ws, "call_slow"))

    turn_frames = [
        m for m in ws.sent
        if m.get("response_type") == "response" and m.get("response_id") == 1
    ]
    assert turn_frames, "the turn sent nothing at all"
    assert turn_frames[0]["content"].startswith("One moment.")
    assert turn_frames[0]["content_complete"] is False
    assert any(m.get("content") == "the answer" for m in turn_frames)
    acks = [m for m in turn_frames if "One moment." in str(m.get("content"))]
    assert len(acks) == 1  # the watchdog speaks once, never nags


def test_ws_fast_turn_sends_no_ack(monkeypatch):
    import api.voice_retell as vr
    from config import config as app_config

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(vr, "get_db_session", _ws_db(minted=True))
    monkeypatch.setattr(app_config, "VOICE_LIVE_FIRST_FRAME_ACK_SECONDS", 0.3)
    monkeypatch.setattr(app_config, "VOICE_LIVE_FIRST_FRAME_ACK_TEXT", "One moment.")

    async def fast_stream(req):
        yield {"response_id": req.response_id, "content": "instant", "content_complete": True}

    monkeypatch.setattr(vr, "_agent_retell_stream", fast_stream)

    ws = _TailWaitWebSocket([_CALL_DETAILS_MSG, _turn_msg(1)], tail_wait=0.05)
    asyncio.run(vr.retell_llm_websocket(ws, "call_fast_noack"))

    assert not any("One moment." in str(m.get("content")) for m in ws.sent)
    assert any(m.get("content") == "instant" for m in ws.sent)


def test_ws_dead_socket_mid_turn_still_drains_the_brain(monkeypatch):
    """A severed socket mid-reply must NOT kill the generation: the turn
    drains to its natural end (the streaming service persists the reply into
    the thread; only the TTS leg is lost) and later frames are simply not
    sent."""
    import api.voice_retell as vr
    from config import config as app_config

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(vr, "get_db_session", _ws_db(minted=True))
    monkeypatch.setattr(app_config, "VOICE_LIVE_FIRST_FRAME_ACK_SECONDS", 0)

    drained = []

    async def stream(req):
        yield {"response_id": 1, "content": "a", "content_complete": False}
        yield {"response_id": 1, "content": "b", "content_complete": False}
        yield {"response_id": 1, "content": "", "content_complete": True}
        drained.append(True)  # reached ONLY if the async-for pulled past every frame

    monkeypatch.setattr(vr, "_agent_retell_stream", stream)

    ws = _DyingWebSocket([_CALL_DETAILS_MSG, _turn_msg(1)], fail_after=1)
    asyncio.run(vr.retell_llm_websocket(ws, "call_severed"))

    assert drained == [True]
    turn_frames = [
        m for m in ws.sent
        if m.get("response_type") == "response" and m.get("response_id") == 1
    ]
    assert [m["content"] for m in turn_frames] == ["a"]  # sends stopped, drain did not
