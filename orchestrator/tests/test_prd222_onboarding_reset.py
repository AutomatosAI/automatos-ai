"""PRD-222 W1·S10 (US-016, decision D9) — the dev onboarding RESET.

Makes onboarding re-runnable in ONE workspace so the operator tests with a
single alias account instead of provisioning/deleting a workspace per attempt.

Two coverage tiers, mirroring the rest of the PRD-222 suite:

* **Pure / FakeDB tests (no Postgres)** — the local gate. The rebuild-don't-mutate
  contract, the resets counter, trial preservation/regrant, and the *scoping +
  survivor* SQL are all proven against fake sessions that record the statements
  the service issues. A ``DELETE`` that is correctly scoped ``WHERE workspace_id
  = :wid`` with ``wid`` = the current workspace is, by construction, unable to
  touch a second workspace's rows (including the platform-key workspace).
* **``@integration`` tests (real Postgres)** — row-level survivor / cross-workspace
  / platform-credential / trial-regrant behavior. These skip cleanly when no DB
  is reachable; ``test.yml`` runs them on each push.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from pathlib import Path

import pytest
from fastapi import HTTPException

from config import config
from services import onboarding_state
from services.onboarding_state import (
    ALL_STAGES,
    INITIAL_STAGE,
    InvalidStageTransition,
    advance_onboarding_stage,
    reset_onboarding,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeWorkspace:
    """Minimal stand-in carrying just what ``reset_onboarding`` reads/writes."""

    def __init__(self, onboarding=None, *, ws_id=None, owner_id=7):
        self.onboarding = onboarding
        self.id = ws_id or uuid.uuid4()
        self.owner_id = owner_id


class _Result:
    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        row = self.fetchone()
        return row[0] if row else None


class _Savepoint:
    def commit(self):  # noqa: D401
        pass

    def rollback(self):
        pass


class _FakeDB:
    """Session stand-in for the wipe path.

    Answers the two ``information_schema`` discovery queries with canned rows and
    RECORDS every ``DELETE`` (table + params) so a test can assert exactly which
    tables were touched, with what predicate, and with which ``workspace_id``.
    """

    def __init__(self, scoped_tables):
        self._scoped = [(t,) for t in scoped_tables]
        self.deletes = []            # list[(table, sql, params)]
        self.flushed = False
        self.committed = False
        self.refreshed = False

    def execute(self, statement, params=None):
        sql = str(statement)
        if "referential_constraints" in sql:
            # pre-cascade external-ref discovery — no external refs in the fake
            return _Result(rows=[])
        if "information_schema.columns" in sql:
            return _Result(rows=self._scoped)
        if "DELETE FROM" in sql:
            table = sql.split('DELETE FROM "', 1)[1].split('"', 1)[0]
            self.deletes.append((table, sql, params or {}))
            return _Result(rowcount=1)
        return _Result()

    def begin_nested(self):
        return _Savepoint()

    def add(self, _obj):
        pass

    def flush(self):
        self.flushed = True

    def commit(self):
        self.committed = True

    def refresh(self, _obj):
        self.refreshed = True

    # convenience for assertions
    def deleted_tables(self):
        return {d[0] for d in self.deletes}


# A representative mix: survivors + built artifacts.
_CANNED_SCOPED = [
    "agent_reports",
    "agents",
    "chats",
    "credentials",          # survivor (credential store)
    "documents",
    "messages",
    "missions",
    "orchestration_tasks",
    "user_api_keys",        # survivor (credential store)
    "workspace_graphs",
    "workspace_members",    # survivor (access/identity)
]


# =========================================================================== #
# Pure tests — the rebuild contract, counter, stage coverage, validator lock.
# =========================================================================== #


def test_reset_onboarding_signature_has_three_flags():
    params = inspect.signature(reset_onboarding).parameters
    assert {"reset_trial", "wipe_built", "wipe_credentials"} <= set(params)
    for name in ("reset_trial", "wipe_built", "wipe_credentials"):
        assert params[name].default is False


def test_reset_rebuilds_document_never_mutates_original():
    original = {
        "stage": "boom",
        "stages": {"questions": "t0", "boom": "t1"},
        "segment": {"business": "bakery"},
        "started_at": "t0",
    }
    ws = _FakeWorkspace(onboarding=original)

    report = reset_onboarding(None, ws)

    # A brand-new object was assigned — never the same dict edited in place.
    assert ws.onboarding is not original
    assert original["stage"] == "boom"          # the prior doc is untouched
    assert original["segment"] == {"business": "bakery"}
    # Fresh not_started, cleared funnel + segment.
    assert ws.onboarding["stage"] == INITIAL_STAGE
    assert ws.onboarding["stages"] == {}
    assert ws.onboarding["segment"] == {}
    assert report["stage"] == INITIAL_STAGE


def test_resets_counter_increments_and_stamps_last_reset_at():
    ws = _FakeWorkspace(onboarding={"stage": "questions"})

    r1 = reset_onboarding(None, ws)
    assert r1["resets"] == 1
    assert ws.onboarding["resets"] == 1
    assert ws.onboarding.get("last_reset_at")

    r2 = reset_onboarding(None, ws)
    assert r2["resets"] == 2
    assert ws.onboarding["resets"] == 2
    assert ws.onboarding.get("last_reset_at")


def test_trial_preserved_by_default():
    trial = {"granted_usd": 5.0, "spent_usd": 3.4, "state": "warned"}
    ws = _FakeWorkspace(onboarding={"stage": "powerup", "trial": trial})

    report = reset_onboarding(None, ws)

    assert ws.onboarding["trial"] == trial
    assert ws.onboarding["trial"] is not trial   # deep-copied, not aliased
    assert report["trial"] == trial
    assert report["trial_note"] == "preserved"


@pytest.mark.parametrize("stage", sorted(ALL_STAGES))
def test_reset_from_every_stage_including_terminal(stage):
    """Reset must work from ANY stage — including the terminal completed/skipped
    that ``advance_onboarding_stage`` refuses to leave. That is the whole point
    of a backward writer."""
    ws = _FakeWorkspace(onboarding={"stage": stage})
    report = reset_onboarding(None, ws)
    assert ws.onboarding["stage"] == INITIAL_STAGE
    assert report["stage"] == INITIAL_STAGE


def test_advance_validator_stays_strict_no_backward_or_terminal_escape():
    """Lock the monotonic/terminal validator (US-016 must not loosen it)."""
    # backward move rejected
    ws = _FakeWorkspace(onboarding={"stage": "questions"})
    with pytest.raises(InvalidStageTransition):
        advance_onboarding_stage(None, ws, INITIAL_STAGE)
    # from-terminal move rejected
    ws2 = _FakeWorkspace(onboarding={"stage": "completed"})
    with pytest.raises(InvalidStageTransition):
        advance_onboarding_stage(None, ws2, "questions")
    ws3 = _FakeWorkspace(onboarding={"stage": "skipped"})
    with pytest.raises(InvalidStageTransition):
        advance_onboarding_stage(None, ws3, "questions")


def test_reset_trial_regrants_via_provisioning_grant(monkeypatch):
    """reset_trial re-grants through grant_trial_at_provisioning (REUSE, not a
    second grant path) and threads its result into the report."""
    calls = {}
    fresh = {"granted_usd": 5.0, "spent_usd": 0, "state": "active"}

    def _fake_grant(db, workspace_id, *, owner_id):
        calls["args"] = (db, workspace_id, owner_id)
        return fresh

    monkeypatch.setattr(
        "services.trial_ledger.grant_trial_at_provisioning", _fake_grant
    )
    ws = _FakeWorkspace(
        onboarding={"stage": "powerup", "trial": {"granted_usd": 5.0, "spent_usd": 5.0, "state": "exhausted"}},
        owner_id=42,
    )

    report = reset_onboarding(None, ws, reset_trial=True)

    assert calls["args"][1] == ws.id and calls["args"][2] == 42   # reuse call-site
    assert report["trial"] == fresh
    assert report["trial_note"] == "granted"
    assert "trial" not in ws.onboarding   # doc was rebuilt trial-less (grant writes the row)


def test_reset_trial_decline_is_a_reported_pause_not_an_error(monkeypatch):
    monkeypatch.setattr(
        "services.trial_ledger.grant_trial_at_provisioning",
        lambda db, workspace_id, *, owner_id: None,   # kill switch / cap / already held
    )
    ws = _FakeWorkspace(onboarding={"stage": "boom", "trial": {"state": "active"}})

    report = reset_onboarding(None, ws, reset_trial=True)   # must NOT raise

    assert report["trial"] is None
    assert report["trial_note"].startswith("paused")


# =========================================================================== #
# FakeDB tests — scoping + survivor SQL (no Postgres).
# =========================================================================== #


def test_wipe_built_spares_survivors_and_scopes_to_workspace():
    ws = _FakeWorkspace(onboarding={"stage": "boom"})
    db = _FakeDB(_CANNED_SCOPED)

    reset_onboarding(db, ws, wipe_built=True)

    deleted = db.deleted_tables()
    # Built artifacts wiped.
    assert {"agents", "missions", "documents", "agent_reports", "workspace_graphs", "orchestration_tasks"} <= deleted
    # Survivors never touched.
    assert "workspace_members" not in deleted   # access/identity
    assert "user_api_keys" not in deleted        # credentials (wipe_credentials owns these)
    assert "credentials" not in deleted
    # The workspace + users rows are NEVER deleted on this path.
    assert "workspaces" not in deleted and "users" not in deleted
    # Every delete is scoped to THIS workspace (cross-workspace isolation).
    assert all(d[2].get("wid") == str(ws.id) for d in db.deletes)
    # The agents delete carries the system/onboarding survivor predicate.
    agents_sql = next(d[1] for d in db.deletes if d[0] == "agents")
    assert "is_system_agent" in agents_sql and "required_role" in agents_sql
    # Discovery was reused (information_schema), and the doc was rebuilt + committed.
    assert db.committed and ws.onboarding["stage"] == INITIAL_STAGE


def test_wipe_credentials_only_touches_credential_tables_scoped():
    ws = _FakeWorkspace(onboarding={"stage": "boom"})
    other_ws = uuid.uuid4()
    db = _FakeDB(_CANNED_SCOPED)

    reset_onboarding(db, ws, wipe_credentials=True)

    deleted = db.deleted_tables()
    assert deleted == {"user_api_keys", "credentials"}   # ONLY the credential stores
    assert "agents" not in deleted and "workspace_members" not in deleted
    # Scoped to this workspace — a second workspace's rows are out of range.
    assert all(d[2].get("wid") == str(ws.id) for d in db.deletes)
    assert all(d[2].get("wid") != str(other_ws) for d in db.deletes)


def test_wipe_built_reuses_purge_machinery_no_hand_list():
    """reset_onboarding drives services.workspace_purge (dynamic discovery),
    never a duplicated table list."""
    src = inspect.getsource(onboarding_state)
    assert "from services.workspace_purge import purge_built_artifacts" in src
    assert "from services.workspace_purge import purge_workspace_credentials" in src
    # purge_built_artifacts itself must derive its table set from discovery.
    from services import workspace_purge

    pb = inspect.getsource(workspace_purge.purge_built_artifacts)
    assert "_delete_rows" in pb                      # reuses the FK-safe deleter
    pc = inspect.getsource(workspace_purge.purge_workspace_credentials)
    assert "_delete_rows" in pc


# =========================================================================== #
# Endpoint tests — 404 disabled / 403 non-admin / admin success (no app boot).
# =========================================================================== #


def _run(coro):
    return asyncio.run(coro)


class _Ctx:
    def __init__(self, role):
        self.workspace_id = uuid.uuid4()
        self.user = type("U", (), {"system_role": role, "id": "user_x"})()


def test_endpoint_404_when_reset_disabled(monkeypatch):
    from api.workspaces import reset_current_onboarding, OnboardingResetRequest

    monkeypatch.setattr(config, "ONBOARDING_RESET_ENABLED", False)
    with pytest.raises(HTTPException) as exc:
        _run(reset_current_onboarding(OnboardingResetRequest(), _Ctx("admin"), object()))
    assert exc.value.status_code == 404   # unadvertised, NOT 403


def test_endpoint_403_for_non_admin_when_enabled(monkeypatch):
    from api.workspaces import reset_current_onboarding, OnboardingResetRequest

    monkeypatch.setattr(config, "ONBOARDING_RESET_ENABLED", True)
    with pytest.raises(HTTPException) as exc:
        _run(reset_current_onboarding(OnboardingResetRequest(), _Ctx("user"), object()))
    assert exc.value.status_code == 403


def test_endpoint_admin_success_returns_report(monkeypatch):
    from api import workspaces as wsmod
    from api.workspaces import reset_current_onboarding, OnboardingResetRequest

    monkeypatch.setattr(config, "ONBOARDING_RESET_ENABLED", True)

    fake_ws = _FakeWorkspace(onboarding={"stage": "boom"})

    class _DB:
        def query(self, _model):
            return self

        def get(self, _id):
            return fake_ws

    canned = {"stage": "not_started", "resets": 3}
    monkeypatch.setattr(
        "services.onboarding_state.reset_onboarding",
        lambda db, ws, **kw: canned,
    )

    out = _run(reset_current_onboarding(
        OnboardingResetRequest(wipe_built=True), _Ctx("super_admin"), _DB()
    ))
    assert out == canned


def test_reset_flag_lives_only_in_config_py():
    """ONBOARDING_RESET_ENABLED is read via os.getenv in config.py alone."""
    assert isinstance(config.ONBOARDING_RESET_ENABLED, bool)
    orch_root = Path(__file__).resolve().parent.parent
    # Built dynamically so this guard file does not contain the literal it hunts.
    env_key = "ONBOARDING_" + "RESET_ENABLED"
    needles = (f'getenv("{env_key}', f"getenv('{env_key}")
    offenders = []
    for py in orch_root.rglob("*.py"):
        # Production modules only — config.py owns the read; tests may mention it.
        if "__pycache__" in py.parts or "tests" in py.parts or py.name == "config.py":
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        if any(n in text for n in needles):
            offenders.append(str(py.relative_to(orch_root)))
    assert not offenders, f"{env_key} read via os.getenv outside config.py: {offenders}"


# =========================================================================== #
# @integration — real Postgres row-level behavior (skips cleanly without a DB).
# =========================================================================== #


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text

    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"onboarding-reset integration suite needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


def _mk_ws(session, ws_id, name, *, owner_id=None, onboarding=None):
    from sqlalchemy import text

    session.execute(
        text(
            "INSERT INTO workspaces (id, name, owner_id, onboarding) "
            "VALUES (CAST(:i AS uuid), :n, :o, CAST(:d AS jsonb))"
        ),
        {"i": ws_id, "n": name, "o": owner_id, "d": json.dumps(onboarding or {"stage": "boom"})},
    )


def _mk_agent(session, ws_id, name, *, is_system, role):
    from sqlalchemy import text

    return session.execute(
        text(
            "INSERT INTO agents (name, agent_type, workspace_id, is_system_agent, required_role) "
            "VALUES (:n, 'custom', CAST(:w AS uuid), :sys, :role) RETURNING id"
        ),
        {"n": name, "w": ws_id, "sys": is_system, "role": role},
    ).fetchone()[0]


def _mk_key(session, ws_id):
    from sqlalchemy import text

    session.execute(
        text(
            "INSERT INTO user_api_keys (workspace_id, provider, encrypted_key) "
            "VALUES (CAST(:w AS uuid), 'openrouter', 'ENC-FAKE')"
        ),
        {"w": ws_id},
    )


@pytest.mark.integration
def test_reset_wipes_built_spares_survivors_and_other_workspace(engine, new_session):
    from sqlalchemy import text

    from core.models.workspaces import Workspace

    ws1, ws2 = str(uuid.uuid4()), str(uuid.uuid4())
    s = new_session()
    _mk_ws(s, ws1, "reset-ws1")
    _mk_ws(s, ws2, "reset-ws2")
    a_reg1 = _mk_agent(s, ws1, "reg1", is_system=False, role=None)
    a_sys1 = _mk_agent(s, ws1, "sys1", is_system=True, role=None)
    a_onb1 = _mk_agent(s, ws1, "onb1", is_system=True, role="onboarding")
    a_reg2 = _mk_agent(s, ws2, "reg2", is_system=False, role=None)
    _mk_key(s, ws1)
    _mk_key(s, ws2)
    s.commit()

    ws1_obj = s.query(Workspace).get(ws1)
    reset_onboarding(s, ws1_obj, wipe_built=True, wipe_credentials=True)

    v = new_session()

    def agent_exists(aid):
        return v.execute(text("SELECT 1 FROM agents WHERE id = :i"), {"i": aid}).fetchone() is not None

    def key_count(ws):
        return v.execute(
            text("SELECT count(*) FROM user_api_keys WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws}
        ).scalar()

    try:
        assert not agent_exists(a_reg1)     # built → wiped
        assert agent_exists(a_sys1)         # system agent → survives
        assert agent_exists(a_onb1)         # onboarding agent → survives
        assert agent_exists(a_reg2)         # OTHER workspace → untouched
        assert v.execute(
            text("SELECT 1 FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": ws1}
        ).fetchone() is not None            # workspace row survives
        assert key_count(ws1) == 0          # this workspace's credentials wiped
        assert key_count(ws2) == 1          # other workspace's credentials survive
        ob = v.execute(
            text("SELECT onboarding FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": ws1}
        ).fetchone()[0]
        assert ob["stage"] == INITIAL_STAGE and ob["resets"] == 1
    finally:
        sw = new_session.sweep()
        for w in (ws1, ws2):
            sw.execute(text("DELETE FROM user_api_keys WHERE workspace_id = CAST(:w AS uuid)"), {"w": w})
            sw.execute(text("DELETE FROM agents WHERE workspace_id = CAST(:w AS uuid)"), {"w": w})
            sw.execute(text("DELETE FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": w})
        sw.commit()


@pytest.mark.integration
def test_wipe_credentials_spares_platform_key_workspace(engine, new_session, monkeypatch):
    from sqlalchemy import text

    from core.models.workspaces import Workspace

    plat, ws1 = str(uuid.uuid4()), str(uuid.uuid4())
    monkeypatch.setattr(config, "PLATFORM_KEY_WORKSPACE_ID", plat)

    s = new_session()
    _mk_ws(s, plat, "platform-key-ws")
    _mk_ws(s, ws1, "reset-ws")
    _mk_key(s, plat)
    _mk_key(s, ws1)
    s.commit()

    ws1_obj = s.query(Workspace).get(ws1)
    reset_onboarding(s, ws1_obj, wipe_credentials=True)

    v = new_session()

    def key_count(ws):
        return v.execute(
            text("SELECT count(*) FROM user_api_keys WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws}
        ).scalar()

    try:
        assert key_count(ws1) == 0          # reset workspace's credentials gone
        assert key_count(plat) == 1         # PLATFORM_KEY_WORKSPACE_ID survives
    finally:
        sw = new_session.sweep()
        for w in (plat, ws1):
            sw.execute(text("DELETE FROM user_api_keys WHERE workspace_id = CAST(:w AS uuid)"), {"w": w})
            sw.execute(text("DELETE FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": w})
        sw.commit()


@pytest.mark.integration
def test_reset_trial_regrants_fresh_active_trial(engine, new_session, monkeypatch):
    from sqlalchemy import text

    from core.models.workspaces import Workspace

    monkeypatch.setattr(config, "TRIAL_ENABLED", True)
    monkeypatch.setattr(config, "TRIAL_CREDIT_USD", 5.0)
    monkeypatch.setattr(config, "TRIAL_GLOBAL_DAILY_USD", 1_000_000.0)  # cap never blocks the test

    s = new_session()
    uid = s.execute(
        text("INSERT INTO users (email) VALUES (:e) RETURNING id"),
        {"e": f"reset-{uuid.uuid4()}@example.test"},
    ).fetchone()[0]
    ws1 = str(uuid.uuid4())
    _mk_ws(
        s, ws1, "reset-trial-ws", owner_id=uid,
        onboarding={"stage": "boom", "trial": {"granted_usd": 5.0, "spent_usd": 5.0, "state": "exhausted"}},
    )
    s.commit()

    ws1_obj = s.query(Workspace).get(ws1)
    report = reset_onboarding(s, ws1_obj, reset_trial=True)

    v = new_session()
    try:
        ob = v.execute(
            text("SELECT onboarding FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": ws1}
        ).fetchone()[0]
        assert ob["trial"]["state"] == "active"
        assert ob["trial"]["spent_usd"] == 0
        assert ob["trial"]["granted_usd"] == 5.0
        assert report["trial_note"] == "granted"
    finally:
        sw = new_session.sweep()
        sw.execute(text("DELETE FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": ws1})
        sw.execute(text("DELETE FROM users WHERE id = :u"), {"u": uid})
        sw.commit()
