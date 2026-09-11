"""Team page — per-member activity from data the workspace already records.

``GET /api/workspaces/{id}/team/members`` gained ``last_active_at``,
``last_sign_in_at``, ``tool_runs_30d``, ``chats_30d`` and ``invited_by_email``
(2026-09-11). Field additions on the same route; the permission gate
(``members:read``) is untouched and covered by the authz sweep, so these tests
call the handler directly (the ``test_p2w2_authz_authority`` idiom) and pin the
LOGIC:

  * batched — one users query for members AND inviters, one grouped query per
    activity table, never one per member (the old handler was an N+1);
  * scoped — the activity queries filter on THIS workspace, so a member's work
    in another workspace never shows here;
  * "last active" is the max of last chat here / last tool run here — the
    platform-wide ``users.last_sign_in`` is deliberately NOT folded in, because
    a per-workspace page must not reveal when a shared user was active in some
    OTHER workspace; the latest is all-time while the counts are windowed;
  * timestamps carry an explicit UTC offset;
  * ``name`` passes through as-is (NULL for every Clerk user today — the page
    derives a label, the API does not invent one);
  * the activity fields are shown only to ``audit:view`` holders (owner/admin);
    everyone else gets ``None`` — not ``0`` — and no aggregate query is issued.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

WS = str(uuid4())
OTHER_WS = str(uuid4())


@pytest.fixture(autouse=True)
def _caller_can_see_activity(monkeypatch):
    """Default every test to an owner/admin caller; the gate has its own tests."""
    import api.team as team

    monkeypatch.setattr(team, "workspace_permission_granted", lambda db, ctx, perm: True)


class _Q:
    """A chainable query stand-in that records its filters and returns fixtures."""

    def __init__(self, rows, log):
        self._rows = rows
        self.filters = []
        log.append(self)

    def filter(self, *args):
        self.filters.extend(args)
        return self

    def group_by(self, *args):
        return self

    def all(self):
        return self._rows


def _owner(arg):
    """The mapped class a query's first argument belongs to."""
    return getattr(arg, "class_", arg)


def _fake_db(members, users, chats=(), tools=()):
    from core.models.composio_cache import ToolExecutionLog
    from core.models.core import Chat, User
    from core.workspaces.models import WorkspaceMember

    log: list = []
    fixtures = {WorkspaceMember: members, User: users, Chat: list(chats), ToolExecutionLog: list(tools)}

    class _DB:
        def query(self, *args):
            owner = _owner(args[0])
            assert owner in fixtures, f"unexpected query root {owner!r}"
            q = _Q(fixtures[owner], log)
            q.owner = owner
            return q

    db = _DB()
    db.log = log
    return db


def _member(mid, uid, role="admin", invited_by=None):
    return SimpleNamespace(id=mid, user_id=uid, role=role, joined_at=datetime(2026, 4, 28), invited_by=invited_by)


def _user(uid, email, last_sign_in=None, name=None):
    return SimpleNamespace(id=uid, email=email, name=name, last_sign_in=last_sign_in)


def _ctx():
    from core.auth.dependencies import RequestContext, UserContext

    return RequestContext(
        workspace_id=uuid4(),
        user=UserContext(id="u", email="owner@example.com", role="owner", system_role="user"),
        auth_type="clerk",
    )


def _call(db, workspace_id=WS):
    from api.team import list_team_members

    return asyncio.run(list_team_members(workspace_id=workspace_id, ctx=_ctx(), db=db))


def _queries(db, cls):
    return [q for q in db.log if q.owner is cls]


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #


def test_one_users_query_covers_members_and_inviters():
    from core.models.core import User

    members = [_member(1, 47, "owner"), _member(2, 2455, invited_by=47), _member(3, 2550, invited_by=47)]
    users = [_user(47, "owner@x.com"), _user(2455, "ar@x.com"), _user(2550, "d@x.com")]
    db = _fake_db(members, users)
    out = _call(db)
    assert len(out) == 3
    assert len(_queries(db, User)) == 1, "the old handler queried users once PER member"


def test_no_members_means_no_further_queries():
    from core.models.core import Chat, User

    db = _fake_db([], [])
    assert _call(db) == []
    assert _queries(db, User) == [] and _queries(db, Chat) == []


# --------------------------------------------------------------------------- #
# last active = max(last chat here, last tool run here) — workspace-scoped
# --------------------------------------------------------------------------- #


def test_last_active_takes_the_latest_of_chat_and_tool_run_here():
    chat = datetime(2026, 9, 11, 11, 30)   # the winner for user 1
    tool = datetime(2026, 9, 10, 8, 0)
    members = [_member(1, 1), _member(2, 2), _member(3, 3)]
    users = [_user(1, "a@x.com"), _user(2, "b@x.com"), _user(3, "c@x.com")]
    db = _fake_db(members, users, chats=[(1, 4, chat)], tools=[(1, 131, tool), (2, 2, tool)])
    by_uid = {m.user_id: m for m in _call(db)}

    assert by_uid[1].last_active_at == "2026-09-11T11:30:00+00:00"   # chat beat the tool run
    assert by_uid[2].last_active_at == "2026-09-10T08:00:00+00:00"   # only a tool run
    assert by_uid[3].last_active_at is None                          # nothing here
    assert by_uid[3].tool_runs_30d == 0 and by_uid[3].chats_30d == 0


def test_platform_wide_sign_in_never_leaks_into_a_workspace_page():
    """A user active in ANOTHER workspace five minutes ago must not read as
    active here. ``users.last_sign_in`` is global, so the team page ignores it —
    and there is no field carrying it either."""
    fresh_sign_in_elsewhere = datetime(2026, 9, 11, 11, 55)
    members = [_member(1, 1)]
    users = [_user(1, "shared@x.com", last_sign_in=fresh_sign_in_elsewhere)]
    (m,) = _call(_fake_db(members, users, chats=[(1, 0, datetime(2026, 8, 1))]))
    assert m.last_active_at == "2026-08-01T00:00:00+00:00"
    assert not hasattr(m, "last_sign_in_at")


def test_counts_pass_through_with_explicit_utc():
    members = [_member(1, 1)]
    users = [_user(1, "a@x.com")]
    db = _fake_db(members, users, chats=[(1, 4, datetime(2026, 9, 1))], tools=[(1, 102, datetime(2026, 9, 2))])
    (m,) = _call(db)
    assert m.chats_30d == 4 and m.tool_runs_30d == 102
    assert m.last_active_at.endswith("+00:00"), "an offset-less string is read as LOCAL by browsers"


# --------------------------------------------------------------------------- #
# scoping — this workspace only
# --------------------------------------------------------------------------- #


def _filters_on(q, key):
    return [f for f in q.filters if getattr(getattr(f, "left", None), "key", None) == key]


def test_activity_queries_are_scoped_to_the_requested_workspace():
    from core.models.composio_cache import ToolExecutionLog
    from core.models.core import Chat

    db = _fake_db([_member(1, 1)], [_user(1, "a@x.com")])
    _call(db, workspace_id=WS)
    for cls in (Chat, ToolExecutionLog):
        (q,) = _queries(db, cls)
        scoped = _filters_on(q, "workspace_id")
        assert scoped, f"{cls.__name__} activity query carries no workspace_id filter"
        assert scoped[0].right.value == WS


# --------------------------------------------------------------------------- #
# inviter, name, and a member whose users row is gone
# --------------------------------------------------------------------------- #


def test_invited_by_resolves_to_the_inviters_email():
    members = [_member(1, 47, "owner"), _member(2, 2455, invited_by=47)]
    users = [_user(47, "owner@x.com"), _user(2455, "ar@x.com")]
    by_uid = {m.user_id: m for m in _call(_fake_db(members, users))}
    assert by_uid[2455].invited_by_email == "owner@x.com"
    assert by_uid[47].invited_by_email is None


def test_name_is_passed_through_untouched_even_when_null():
    (m,) = _call(_fake_db([_member(1, 1)], [_user(1, "daniel@x.com", name=None)]))
    assert m.name is None  # the page derives a label; the API invents nothing


def test_member_without_a_users_row_is_skipped():
    out = _call(_fake_db([_member(1, 1), _member(2, 999)], [_user(1, "a@x.com")]))
    assert [m.user_id for m in out] == [1]


# --------------------------------------------------------------------------- #
# the audit:view gate — roster for everyone, telemetry for owner/admin
# --------------------------------------------------------------------------- #


def test_activity_is_hidden_and_not_even_queried_without_audit_view(monkeypatch):
    import api.team as team
    from core.models.composio_cache import ToolExecutionLog
    from core.models.core import Chat

    asked = []

    def _granted(db, ctx, perm):
        asked.append(perm)
        return False

    monkeypatch.setattr(team, "workspace_permission_granted", _granted)
    db = _fake_db([_member(1, 1, invited_by=47), _member(2, 47, "owner")],
                  [_user(1, "v@x.com"), _user(47, "owner@x.com")],
                  chats=[(1, 9, datetime(2026, 9, 1))])
    by_uid = {m.user_id: m for m in _call(db)}

    assert asked == ["audit:view"], "the gate must be the reused audit permission, asked once"
    assert by_uid[1].tool_runs_30d is None and by_uid[1].chats_30d is None
    assert by_uid[1].last_active_at is None
    # None, not 0: the page renders "—", never a misleading "0 tool runs".
    assert by_uid[1].invited_by_email == "owner@x.com"  # roster metadata still flows
    assert _queries(db, Chat) == [] and _queries(db, ToolExecutionLog) == []


def test_visible_but_idle_member_reads_zero_not_none():
    (m,) = _call(_fake_db([_member(1, 1)], [_user(1, "a@x.com")]))
    assert m.tool_runs_30d == 0 and m.chats_30d == 0 and m.last_active_at is None


# --------------------------------------------------------------------------- #
# the aggregate helper's contract
# --------------------------------------------------------------------------- #


def test_activity_by_user_maps_rows_and_coerces_null_counts():
    from api.team import _activity_by_user
    from core.models.core import Chat

    log: list = []

    class _DB:
        def query(self, *args):
            return _Q([(1, None, datetime(2026, 9, 1)), (2, 7, None)], log)

    out = _activity_by_user(_DB(), Chat.user_id, Chat.created_at, Chat.workspace_id, WS, {1, 2}, datetime(2026, 8, 12))
    assert out == {1: (0, datetime(2026, 9, 1)), 2: (7, None)}
