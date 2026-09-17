"""A pending Composio connection reconciles to active on the next listing.

2026-09-17 (local edition): Gmail and Google Calendar completed OAuth, Composio
held both accounts ACTIVE, and the Tools page kept showing "Connect" until the
process restarted. The auth-config resolver had cached a miss with no expiry —
the declared one-hour TTL was never enforced — and creating the config seconds
later did not update it, so every pending→active check short-circuited before
calling Composio. These tests pin the repaired shape: a miss expires quickly, a
created config is remembered at once, and the status lookup uses the config the
connection was initiated under, falling back to a toolkit match.
"""
from __future__ import annotations

from types import SimpleNamespace

import core.composio.client as cc


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _AuthConfigs:
    def __init__(self, items):
        self.items = list(items)
        self.list_calls = 0
        self.created = []

    def list(self):
        self.list_calls += 1
        return _Obj(items=list(self.items))

    def create(self, toolkit, options):
        self.created.append((toolkit, options))
        return _Obj(
            id=f"ac_new_{len(self.created)}",
            toolkit=_Obj(slug=toolkit),
            status="ENABLED",
            auth_scheme=options.get("authScheme", "OAUTH2"),
        )


class _ConnectedAccounts:
    def __init__(self, router):
        self._router = router
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return _Obj(items=list(self._router(kwargs)))


class _FakeSDK:
    def __init__(self, configs=(), accounts=lambda kwargs: []):
        self.auth_configs = _AuthConfigs(configs)
        self.connected_accounts = _ConnectedAccounts(accounts)
        self.toolkits = SimpleNamespace(get=lambda: [_Obj(slug="gmail", auth_schemes=["OAUTH2"])])


def _client(sdk) -> cc.ComposioClient:
    client = cc.ComposioClient.__new__(cc.ComposioClient)
    client.composio = sdk
    client._auth_config_cache = {}
    client._auth_config_cache_ttl = 3600
    client._auth_config_miss_ttl = 30
    return client


def _config(cid, slug="gmail", status="ENABLED", scheme="OAUTH2"):
    return _Obj(id=cid, toolkit=_Obj(slug=slug), status=status, auth_scheme=scheme)


def _account(aid, slug, status="ACTIVE"):
    return _Obj(id=aid, status=status, toolkit=_Obj(slug=slug), created_at="2026-09-17T09:25:09Z")


def test_a_miss_is_cached_only_briefly(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(cc, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    sdk = _FakeSDK(configs=[])
    client = _client(sdk)

    assert client._resolve_auth_config_id("gmail") is None
    assert client._resolve_auth_config_id("gmail") is None
    assert sdk.auth_configs.list_calls == 1, "a fresh miss is served from cache"

    sdk.auth_configs.items.append(_config("ac_gmail"))
    clock[0] += 31  # past the miss TTL
    assert client._resolve_auth_config_id("gmail") == "ac_gmail"
    assert sdk.auth_configs.list_calls == 2


def test_a_resolved_id_is_cached_for_an_hour_then_refreshed(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(cc, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    sdk = _FakeSDK(configs=[_config("ac_gmail")])
    client = _client(sdk)

    assert client._resolve_auth_config_id("gmail") == "ac_gmail"
    clock[0] += 3599
    assert client._resolve_auth_config_id("gmail") == "ac_gmail"
    assert sdk.auth_configs.list_calls == 1
    clock[0] += 2
    assert client._resolve_auth_config_id("gmail") == "ac_gmail"
    assert sdk.auth_configs.list_calls == 2


def test_a_created_config_is_remembered_without_relisting():
    sdk = _FakeSDK(configs=[])  # Composio's list has not caught up yet
    client = _client(sdk)

    created = client._ensure_auth_config_id("gmail", preferred_scheme="OAUTH2")
    assert created == "ac_new_1"
    calls_after_create = sdk.auth_configs.list_calls

    assert client._resolve_auth_config_id("gmail") == "ac_new_1"
    assert client._resolve_auth_config_id("gmail", preferred_scheme="OAUTH2") == "ac_new_1"
    assert sdk.auth_configs.list_calls == calls_after_create, "no lookup depends on the list catching up"


def test_status_lookup_uses_the_stored_config_first():
    def accounts(kwargs):
        if kwargs.get("auth_config_ids") == ["ac_stored"]:
            return [_account("ca_1", "gmail")]
        return []

    sdk = _FakeSDK(configs=[_config("ac_other")], accounts=accounts)
    client = _client(sdk)

    found = client.get_connection_status("entity-1", "GMAIL", auth_config_id="ac_stored")
    assert found == {"id": "ca_1", "status": "ACTIVE", "created_at": "2026-09-17T09:25:09Z"}
    assert sdk.connected_accounts.calls[0] == {"user_ids": ["entity-1"], "auth_config_ids": ["ac_stored"]}
    assert sdk.auth_configs.list_calls == 0, "a stored config is the answer — no resolution"


def test_status_lookup_falls_back_to_a_toolkit_match():
    def accounts(kwargs):
        if kwargs.get("auth_config_ids"):
            return []  # the account lives under another config for the toolkit
        return [_account("ca_slack", "slack"), _account("ca_gmail", "gmail")]

    sdk = _FakeSDK(configs=[_config("ac_gmail")], accounts=accounts)
    client = _client(sdk)

    found = client.get_connection_status("entity-1", "GMAIL")
    assert found is not None and found["id"] == "ca_gmail"
    assert [c.get("auth_config_ids") for c in sdk.connected_accounts.calls] == [["ac_gmail"], None]


def test_status_lookup_without_any_config_still_matches_the_toolkit():
    sdk = _FakeSDK(configs=[], accounts=lambda kw: [_account("ca_gmail", "gmail")])
    client = _client(sdk)

    found = client.get_connection_status("entity-1", "gmail")
    assert found is not None and found["id"] == "ca_gmail"
    assert sdk.connected_accounts.calls == [{"user_ids": ["entity-1"]}]


def test_an_initialising_account_is_not_connected():
    sdk = _FakeSDK(configs=[_config("ac_gmail")], accounts=lambda kw: [_account("ca_gmail", "gmail", status="INITIALIZING")])
    client = _client(sdk)
    assert client.get_connection_status("entity-1", "GMAIL") is None


def test_stored_auth_config_id_helper_reads_the_connect_metadata():
    from api.tools import _stored_auth_config_id

    class _Manager:
        def __init__(self, meta):
            self.meta = meta
            self.asked = []

        def get_connection_metadata(self, entity_id, app_name):
            self.asked.append((entity_id, app_name))
            return self.meta

    m = _Manager({"auth_scheme": "OAUTH2", "auth_config_id": "ac_9z"})
    assert _stored_auth_config_id(m, 1, "gmail") == "ac_9z"
    assert m.asked == [(1, "GMAIL")]
    assert _stored_auth_config_id(_Manager({}), 1, "gmail") is None
    assert _stored_auth_config_id(_Manager(None), 1, "gmail") is None
