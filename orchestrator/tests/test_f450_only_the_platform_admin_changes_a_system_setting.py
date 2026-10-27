"""F450 — only the platform admin changes a platform-wide setting, whoever clicked yes.

``platform_update_system_setting`` writes a ``system_settings`` row, every tenant's,
and a change of the orchestrator_llm route restamps the Auto agents of every workspace
(``handlers_workspace._restamp_auto_models``). In SaaS a workspace owner's yes on the
card must not be enough. The gate that holds it is the PRD-143 super-admin gate
(``PlatformActionExecutor.clear``, ``super_admin_only=True`` on the action since F147),
which runs before the confirmation gate and the owner's click: a grant on the exact
call does not get a non-admin past it. The local edition's operator is the super admin
(core/auth/hybrid.py), and the chat writes that role into the caller context
(consumers/chatbot/service.py build_tool_caller_context), so Auto still changes the
system model there on a click.

These pin it end to end with the REAL registry and the REAL handler (the F147 test
mocks the handler): the owner's yes changes no row and restamps nothing; the platform
admin's yes on the same call writes the row and restamps. The fake session is
tests/test_p2w2_tool_grant_consume.py's (it evaluates the services' equality filters),
plus the commit the handler makes.
"""
from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.llm import auto_model_sync  # noqa: E402
from core.models.system_settings import SystemSetting  # noqa: E402
from core.services.approval_grants import grant_grant  # noqa: E402
from modules.tools.discovery.platform_executor import PlatformActionExecutor  # noqa: E402
from modules.tools.execution import tool_grants  # noqa: E402

pytestmark = pytest.mark.asyncio

ACTION = "platform_update_system_setting"
OLD_MODEL, NEW_MODEL = "claude-haiku-5-5", "claude-sonnet-5-5"
PARAMS = {"category": auto_model_sync.ORCHESTRATOR_CATEGORY, "key": "model", "value": NEW_MODEL}
# A SaaS workspace owner's turn, as the stored grant replays it; no super-admin role.
OWNER = {"user_id": "user_clerk_owner"}
# The platform admin: the chat writes only the literal super_admin role.
PLATFORM_ADMIN = {"user_id": "user_clerk_platform", "system_role": "super_admin"}
MOVED = {"moved": 4, "by_route": {}}


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *conds):
        rows = self._rows
        for cond in conds:
            key = cond.left.key
            value = getattr(cond.right, "value", None)
            rows = [r for r in rows if str(getattr(r, key, None)) == str(value)]
        return _Query(rows)

    def order_by(self, *args):
        return _Query(list(reversed(self._rows)))

    def with_for_update(self, **_kw):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self):
        self.rows = []
        self.commits = 0

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.rows) + 1
        self.rows.append(obj)

    def flush(self):
        pass

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def query(self, model):
        return _Query([r for r in self.rows if isinstance(r, model)])


@pytest.fixture(autouse=True)
def _mute_notifications(monkeypatch):
    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(tool_grants, "_dispatch_approval_pending", _noop)


@pytest.fixture
def restamps(monkeypatch):
    calls = []
    monkeypatch.setattr(auto_model_sync, "restamp_after_settings_change",
                        lambda db, changed: calls.append([(s.key, old) for s, old in changed]) or MOVED)
    return calls


@pytest.fixture
def platform():
    """A workspace's executor over a session holding the platform's model setting, and
    a yes (GRANTED) on exactly this call, as the card's click leaves it."""
    db = _FakeSession()
    setting = SystemSetting(category=PARAMS["category"], key="model", value=OLD_MODEL,
                            is_sensitive=False, is_required=True, description=None)
    db.add(setting)
    executor = PlatformActionExecutor(db, uuid.uuid4())
    grant = tool_grants.issue_tool_grant(db, executor.workspace_id, action=ACTION, params=dict(PARAMS),
                                         permission_level="write", description="probe", caller_context=dict(OWNER))
    assert grant is not None, "seed failed: grant issuance broken"
    grant_grant(grant, granted_by="user:1")
    return db, executor, setting, grant


async def _run(executor, caller_context):
    with patch.object(PlatformActionExecutor, "_full_autonomy", return_value=False), \
            patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return await executor.execute(ACTION, dict(PARAMS), dict(caller_context))


async def test_a_workspace_owners_yes_changes_no_platform_setting(platform, restamps):
    db, executor, setting, _grant = platform
    reply = await _run(executor, OWNER)
    assert reply["success"] is False and reply.get("permission_denied") is True
    assert "super admin" in reply["error"]
    assert setting.value == OLD_MODEL           # the platform's row is untouched
    assert db.commits == 0
    assert restamps == []                        # no workspace's Auto was moved


async def test_the_platform_admins_yes_writes_the_setting_and_moves_the_autos(platform, restamps):
    _db, executor, setting, grant = platform
    reply = await _run(executor, PLATFORM_ADMIN)
    assert reply["success"] is True, reply
    assert reply["auto_models"] == MOVED
    assert reply.get("approved_via_grant_id") == grant.id
    assert setting.value == NEW_MODEL
    assert restamps == [[("model", OLD_MODEL)]]
