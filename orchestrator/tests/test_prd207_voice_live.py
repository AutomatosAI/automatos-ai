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
            q.filter.return_value.first.return_value = None if chat == "unqueried" else chat
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

    assert out == {"call_id": "call_abc", "access_token": "tok_xyz"}
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
    s.execute(
        text(
            "INSERT INTO chats (id, user_id, workspace_id, title) "
            "VALUES (CAST(:id AS uuid), :u, CAST(:ws AS uuid), 'my thread')"
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
