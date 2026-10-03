"""F258 — a pending connection learns how Composio's attempt ended.

3 Oct 2026: FAL_AI and KIEAI sat ``pending`` with no connection id for good, though
Composio had marked both attempts EXPIRED ("Authorization was started but not
completed within 10 minutes"). The listings looked only for an ACTIVE or INITIATED
account, so an ended attempt read as "not yet", and two of them counted INITIATED,
which means the user hasn't finished, as connected.
``core.composio.pending_connections.settle_pending`` now settles a pending row from
the app's attempts, and the three listings (the Tools page's ``/connected``,
``/refresh-connections`` and ``/api/composio/connections``) all use it.
"""
from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from core.composio.pending_connections import Settled, settle_pending  # noqa: E402

C1 = "00000000-0000-0000-0000-0000000000c1"
ENTITY = {"id": 11, "composio_entity_id": C1}
EXPIRED_REASON = "Authorization was started but not completed within 10 minutes"


def _account(account_id, status, created_at, reason=None):
    return SimpleNamespace(id=account_id, status=status, created_at=created_at, status_reason=reason)


class _Accounts:
    def __init__(self, items):
        self.items = list(items)
        self.queries = []

    def list(self, **query):
        self.queries.append(query)
        return SimpleNamespace(items=list(self.items))


class _Rows:
    """The EntityManager surface the settling reads and writes."""

    def __init__(self, auth_config_id=None):
        self.meta = {"auth_config_id": auth_config_id} if auth_config_id else {}
        self.writes = []

    def get_connection_metadata(self, entity_id, app_name):
        return self.meta

    def update_connection_status(self, **kwargs):
        self.writes.append(kwargs)
        return True


def _client(*accounts):
    return SimpleNamespace(composio=SimpleNamespace(connected_accounts=_Accounts(accounts)))


def test_an_expired_sign_in_is_not_connected_and_offers_connect_again():
    client = _client(_account("ca__M1K7R2dpPrg", "EXPIRED", "2026-10-03T12:33:19.831Z", EXPIRED_REASON))
    rows = _Rows()

    assert settle_pending(rows, client, ENTITY, "FAL_AI", "ac_0jpN6E_qMSmK") == Settled("added", None)
    assert rows.writes == [{"entity_id": 11, "app_name": "FAL_AI", "status": "added", "connection_id": None}]


@pytest.mark.parametrize("ended", ["FAILED", "INACTIVE"])
def test_every_ended_attempt_offers_connect_again(ended):
    rows = _Rows()
    assert settle_pending(rows, _client(_account("ca_1", ended, "2026-10-03T12:33:00Z")), ENTITY, "KIEAI", None) == Settled("added", None)
    assert [w["status"] for w in rows.writes] == ["added"]


@pytest.mark.parametrize("under_way", ["INITIATED", "INITIALIZING"])
def test_a_sign_in_still_under_way_stays_pending(under_way):
    rows = _Rows()
    client = _client(_account("ca_new", under_way, "2026-10-03T12:35:00Z"))

    assert settle_pending(rows, client, ENTITY, "KIEAI", "ac_eO3gDGkZVvdG") == Settled("pending", None)
    assert rows.writes == [], "INITIATED means the user hasn't finished: never connected"


def test_an_active_account_is_connected():
    rows = _Rows()
    client = _client(_account("ca_CWr5OWfrnj1w", "ACTIVE", "2026-10-03T12:53:00.928Z"))

    assert settle_pending(rows, client, ENTITY, "INSTAGRAM", "ac_q0kjwICTW0-Z") == Settled("active", "ca_CWr5OWfrnj1w")
    assert rows.writes == [{"entity_id": 11, "app_name": "INSTAGRAM", "status": "active", "connection_id": "ca_CWr5OWfrnj1w"}]


def test_a_working_account_outranks_a_newer_abandoned_attempt():
    client = _client(
        _account("ca_new", "EXPIRED", "2026-10-03T12:33:19Z", EXPIRED_REASON),
        _account("ca_old_ok", "ACTIVE", "2026-10-01T09:00:00Z"),
    )
    assert settle_pending(_Rows(), client, ENTITY, "INSTAGRAM", "ac_q0") == Settled("active", "ca_old_ok")


def test_the_newest_attempt_decides_whatever_order_composio_lists_them_in():
    rows = _Rows()
    client = _client(
        _account("ca_old", "EXPIRED", "2026-10-03T12:00:00Z", EXPIRED_REASON),
        _account("ca_new", "INITIATED", "2026-10-03T12:40:00Z"),
    )
    assert settle_pending(rows, client, ENTITY, "FAL_AI", None) == Settled("pending", None)
    assert rows.writes == []


def test_no_attempt_yet_stays_pending():
    rows = _Rows()
    assert settle_pending(rows, _client(), ENTITY, "FAL_AI", None) == Settled("pending", None)
    assert rows.writes == []


def test_the_lookup_asks_for_this_workspaces_app_under_the_stored_config():
    client = _client()
    settle_pending(_Rows(), client, ENTITY, "HIGGSFIELD_MCP", "ac_dcr")
    assert client.composio.connected_accounts.queries == [{
        "user_ids": [C1],
        "toolkit_slugs": ["higgsfield_mcp"],
        "order_by": "created_at",
        "order_direction": "desc",
        "auth_config_ids": ["ac_dcr"],
    }]

    no_config = _client()
    settle_pending(_Rows(), no_config, ENTITY, "FAL_AI", None)
    assert "auth_config_ids" not in no_config.composio.connected_accounts.queries[0], "no stored config: every account for the toolkit"


# --- the three listings settle through it -------------------------------------------------


@pytest.fixture()
def routes():
    import tests.conftest as _conftest

    _conftest._restore_real_app_modules()
    from api import composio as composio_api
    from api import tools as tools_api

    return SimpleNamespace(tools=tools_api, composio=composio_api)


def test_the_tools_page_shows_an_expired_sign_in_as_not_connected(routes):
    rows = _Rows(auth_config_id="ac_0jpN6E_qMSmK")
    client = _client(_account("ca__M1K7R2dpPrg", "EXPIRED", "2026-10-03T12:33:19Z", EXPIRED_REASON))
    row = {"app_name": "FAL_AI", "status": "pending", "connection_id": None}

    shown = routes.tools._settle_row(rows, client, ENTITY, row, "ws-c1")

    assert shown == {"app_name": "FAL_AI", "status": "added", "connection_id": None}
    assert row["status"] == "pending", "the listing builds a new row; the one it was given is unchanged"
    assert client.composio.connected_accounts.queries[0]["auth_config_ids"] == ["ac_0jpN6E_qMSmK"]


def test_shopify_going_active_starts_its_catalog_sync(routes, monkeypatch):
    started = []
    monkeypatch.setattr(routes.tools, "_start_shopify_autosync", started.append)
    client = _client(_account("ca_shop", "ACTIVE", "2026-10-03T13:00:00Z"))
    row = {"app_name": "SHOPIFY", "status": "pending", "connection_id": None}

    shown = routes.tools._settle_row(_Rows(), client, ENTITY, row, "ws-shop")

    assert shown["status"] == "active" and shown["connection_id"] == "ca_shop"
    assert started == ["ws-shop"]


def test_a_refresh_stamps_a_row_still_under_way_and_counts_what_settled(routes):
    rows = _Rows()
    under_way = routes.tools._refresh_row(rows, _client(_account("ca_new", "INITIATED", "2026-10-03T12:40:00Z")), ENTITY, {"app_name": "KIEAI"})
    ended = routes.tools._refresh_row(rows, _client(_account("ca_old", "EXPIRED", "2026-10-03T12:33:31Z")), ENTITY, {"app_name": "KIEAI"})

    assert (under_way, ended) == ("pending", "added")
    assert rows.writes == [
        {"entity_id": 11, "app_name": "KIEAI", "status": "pending"},  # stamped as checked
        {"entity_id": 11, "app_name": "KIEAI", "status": "added", "connection_id": None},
    ]


def test_the_composio_connections_listing_settles_a_pending_row(routes):
    rows = _Rows(auth_config_id="ac_eO3gDGkZVvdG")
    client = _client(_account("ca_FFqsVHPWHhLs", "EXPIRED", "2026-10-03T12:33:32Z", EXPIRED_REASON))
    row = {"app_name": "KIEAI", "status": "pending", "connection_id": None, "connected_at": None}

    shown = routes.composio._connection_response(rows, client, ENTITY, row)

    assert (shown.app_name, shown.status, shown.connection_id) == ("KIEAI", "added", None)
