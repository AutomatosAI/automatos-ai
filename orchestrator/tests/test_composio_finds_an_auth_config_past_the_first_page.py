"""A toolkit's auth config is found wherever it sits in Composio's list (3 Oct 2026).

Local build, Tools → X: "Failed to start connection flow: Failed to initiate OAuth
connection". The log: "Managed auth unavailable for TWITTER, falling back to custom
OAUTH2", then Composio's 400 "Missing required field "Client id"". The project held
62 auth configs, 20 to a page, newest first, and the resolver read only the first
page. X's own-app config ("Automatos-X", custom OAUTH2, May) was on page 3, so the
platform tried to make one: Composio has had no managed X credentials since
February 2026, and a custom config needs the app's client id. 23 toolkits had no
config on page 1; a first connect to one Composio can manage made a duplicate
(github had four), and disconnect looked only under the newest of them.
"""
from __future__ import annotations

from types import SimpleNamespace

import core.composio.client as cc

PAGE = 20  # Composio's page on 3 Oct, whatever the limit asked for


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _config(cid, slug, created, *, scheme="OAUTH2", managed=True, status="ENABLED"):
    return _Obj(id=cid, toolkit=_Obj(slug=slug), created_at=created, auth_scheme=scheme,
                is_composio_managed=managed, status=status)


def _others(n, newer_than="2026-06"):
    """``n`` newer configs of other toolkits, which filled the project's first pages."""
    return [_config(f"ac_other_{i}", f"toolkit{i % 17}", f"{newer_than}-{10 + i // 30:02d}T{i % 24:02d}:00:00Z")
            for i in range(n)]


class _AuthConfigs:
    """Composio's auth-config list: newest first, ``toolkit_slug`` filters, ``cursor``
    turns the page. A config for a toolkit Composio can't manage can't be made
    without the app's credentials."""

    def __init__(self, configs):
        self.configs = sorted(configs, key=lambda c: c.created_at, reverse=True)
        self.calls, self.created = [], []

    def list(self, **query):
        self.calls.append(query)
        rows = [c for c in self.configs if c.toolkit.slug == query.get("toolkit_slug", c.toolkit.slug)]
        start = int(query.get("cursor") or 0)
        end = start + min(query.get("limit") or PAGE, PAGE)
        return _Obj(items=rows[start:end], next_cursor=str(end) if end < len(rows) else None)

    def create(self, toolkit, options):
        self.created.append((toolkit, options))
        raise RuntimeError("Error code: 400 - Missing required field \"Client id\" for auth scheme \"OAUTH2\"")

    def get(self, config_id):
        return next(c for c in self.configs if c.id == config_id)


class _Accounts:
    def __init__(self, accounts):
        self.accounts, self.deleted = accounts, []

    def list(self, **query):
        rows = [a for a in self.accounts if a.user_id in query.get("user_ids", [a.user_id])]
        if "auth_config_ids" in query:
            rows = [a for a in rows if a.auth_config_id in query["auth_config_ids"]]
        if "toolkit_slugs" in query:
            rows = [a for a in rows if a.toolkit.slug in query["toolkit_slugs"]]
        return _Obj(items=rows)

    def delete(self, nanoid):
        self.deleted.append(nanoid)

    def link(self, user_id, auth_config_id, callback_url):
        return _Obj(redirect_url=f"https://connect.composio.example/{auth_config_id}")


class _SDK:
    def __init__(self, configs, accounts=()):
        self.auth_configs = _AuthConfigs(configs)
        self.connected_accounts = _Accounts(list(accounts))
        self.toolkits = SimpleNamespace(get=lambda: [_Obj(slug="twitter", auth_schemes=["OAUTH2"]),
                                                     _Obj(slug="github", auth_schemes=["OAUTH2"])])


def _client(sdk) -> cc.ComposioClient:
    client = cc.ComposioClient.__new__(cc.ComposioClient)
    client.api_key = "test-key-not-a-secret"
    client._composio = sdk  # the lazy ``composio`` property returns this once set
    client._auth_config_cache = {}
    client._auth_config_cache_ttl = 3600
    client._auth_config_miss_ttl = 30
    return client


def _the_project():
    """3 Oct's project: X's two configs behind 60 newer ones."""
    return _others(60) + [
        _config("ac_automatos_x", "twitter", "2026-05-06T10:00:00Z", managed=False),   # the owner's own X app
        _config("ac_twitter_managed", "twitter", "2026-01-26T10:00:00Z"),              # Composio's, gone since February
    ]


def test_x_connects_through_its_own_app_config_past_the_first_page():
    sdk = _SDK(_the_project())
    client = _client(sdk)

    link = client.initiate_connection(entity_id="ws-c1", app="TWITTER", callback_url="http://localhost:3000/cb")

    assert link["auth_config_id"] == "ac_automatos_x" and link["redirect_url"].endswith("ac_automatos_x")
    assert sdk.auth_configs.created == []                       # no config made, so no Composio 400
    assert sdk.auth_configs.calls[0]["toolkit_slug"] == "twitter"


def test_every_page_is_read_and_the_newest_enabled_config_of_the_scheme_wins():
    configs = [_config(f"ac_hub_{i}", "hubspot", f"2026-09-{1 + i % 28:02d}T{i % 24:02d}:00:00Z", scheme="API_KEY")
               for i in range(44)]
    configs += [_config("ac_hub_off", "hubspot", "2026-03-02T00:00:00Z", status="DISABLED"),
                _config("ac_hub_oauth", "hubspot", "2026-03-01T00:00:00Z")]   # the oldest: page 3
    sdk = _SDK(configs)

    found = _client(sdk)._resolve_auth_config_id("hubspot", preferred_scheme="OAUTH2")

    assert found == "ac_hub_oauth"
    assert [call.get("cursor") for call in sdk.auth_configs.calls] == [None, "20", "40"]


def test_a_first_connect_never_duplicates_a_config_past_the_first_page():
    sdk = _SDK(_others(60) + [_config("ac_github", "github", "2026-01-21T10:00:00Z")])

    assert _client(sdk)._ensure_auth_config_id("GITHUB") == "ac_github"
    assert sdk.auth_configs.created == []                       # before: github's fourth config


def _github_with_a_duplicate():
    configs = [_config("ac_github_dup", "github", "2026-09-30T10:00:00Z"),
               _config("ac_github", "github", "2026-01-21T10:00:00Z")]
    account = _Obj(id="ca_github_1", user_id="ws-c1", auth_config_id="ac_github", toolkit=_Obj(slug="github"),
                   status="ACTIVE", access_token="token-under-the-older-config")
    return _SDK(configs, [account])


def test_disconnect_finds_the_account_under_an_older_config_of_the_toolkit():
    sdk = _github_with_a_duplicate()

    assert _client(sdk).disconnect_app("ws-c1", "GITHUB") is True
    assert sdk.connected_accounts.deleted == ["ca_github_1"]


def test_the_access_token_is_found_under_an_older_config_of_the_toolkit():
    sdk = _github_with_a_duplicate()

    assert _client(sdk).get_app_access_token("ws-c1", "GITHUB") == "token-under-the-older-config"
