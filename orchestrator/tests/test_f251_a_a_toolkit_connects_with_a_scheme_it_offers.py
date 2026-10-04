"""F251 — connecting a toolkit Composio can't manage uses an auth scheme it offers.

3 Oct 2026, local build from main: Higgsfield never connected, from the Tools page
("Failed to initiate OAuth connection") or from Socials ("Could not start the
Higgsfield connection"). Composio has no managed auth for HIGGSFIELD_MCP, and the
platform's fallback created a custom OAUTH2 config, which Composio refused: 400
Auth_Config_AuthSchemeNotFound, 'Available auth schemes: DCR_OAUTH'. The fallback
now takes a scheme from the toolkit's own list (``custom_auth_fallback_scheme``).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.composio.client as cc
from core.composio.auth_schemes import custom_auth_fallback_scheme

HIGGSFIELD = "HIGGSFIELD_MCP"


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _AuthConfigs:
    """Composio's auth-config API as F251 met it: no managed auth for any toolkit
    here, and a 400 for a custom scheme the toolkit doesn't offer."""

    def __init__(self, offered):
        self.offered = offered
        self.created = []

    def list(self, **query):  # the SDK's signature: toolkit_slug, limit, cursor
        return _Obj(items=[])

    def create(self, toolkit, options):
        if options["type"] == "use_composio_managed_auth":
            raise RuntimeError("Error code: 400 - {'error': {'slug': 'DefaultAuthConfigNotFound'}}")
        scheme = options["authScheme"]
        if scheme not in self.offered[toolkit]:
            raise RuntimeError(
                f"Error code: 400 - Auth scheme \"{scheme}\" not found for toolkit "
                f"\"{toolkit.lower()}\" (Auth_Config_AuthSchemeNotFound)"
            )
        self.created.append((toolkit, options))
        return _Obj(id=f"ac_{toolkit.lower()}", auth_scheme=scheme)

    def get(self, config_id):
        return next(_Obj(auth_scheme=opts["authScheme"]) for slug, opts in self.created
                    if f"ac_{slug.lower()}" == config_id)


class _FakeSDK:
    def __init__(self, offered):
        self.auth_configs = _AuthConfigs(offered)
        self.toolkits = SimpleNamespace(
            get=lambda: [_Obj(slug=slug.lower(), auth_schemes=schemes) for slug, schemes in offered.items()]
        )
        self.links = []
        self.connected_accounts = SimpleNamespace(link=self._link)

    def _link(self, user_id, auth_config_id, callback_url):
        self.links.append((user_id, auth_config_id))
        return _Obj(redirect_url=f"https://connect.composio.example/{auth_config_id}")


def _client(sdk) -> cc.ComposioClient:
    client = cc.ComposioClient.__new__(cc.ComposioClient)
    client.api_key = "test-key-not-a-secret"
    client._composio = sdk  # the lazy ``composio`` property returns this once set
    client._auth_config_cache = {}
    client._auth_config_cache_ttl = 3600
    client._auth_config_miss_ttl = 30
    return client


@pytest.mark.parametrize(
    ("offered", "chosen"),
    [
        (["DCR_OAUTH"], "DCR_OAUTH"),                # HIGGSFIELD_MCP, night of 3 Oct
        (["dcr_oauth"], "DCR_OAUTH"),                # case never decides it
        (["OAUTH2"], "OAUTH2"),
        (["OAUTH2", "DCR_OAUTH"], "OAUTH2"),         # a toolkit that offers OAUTH2 is set up as before
        (["BEARER_TOKEN", "DCR_OAUTH"], "DCR_OAUTH"),
        (["BEARER_TOKEN"], "BEARER_TOKEN"),          # never a scheme the toolkit doesn't list
        ([], "OAUTH2"),                              # schemes unknown: what was always asked for
    ],
)
def test_the_fallback_scheme_is_one_the_toolkit_offers(offered, chosen):
    assert custom_auth_fallback_scheme(offered) == chosen


def test_higgsfield_gets_a_dcr_oauth_config_and_a_sign_in_link():
    sdk = _FakeSDK({HIGGSFIELD: ["DCR_OAUTH"]})
    client = _client(sdk)

    link = client.initiate_connection(entity_id="ws-c1", app=HIGGSFIELD, callback_url="http://localhost:3000/cb")

    assert sdk.auth_configs.created == [(HIGGSFIELD, {"type": "use_custom_auth", "authScheme": "DCR_OAUTH"})]
    assert link == {
        "redirect_url": "https://connect.composio.example/ac_higgsfield_mcp",
        "auth_config_id": "ac_higgsfield_mcp",
        "auth_scheme": "DCR_OAUTH",  # stored on the connection, so a reconnect reuses this config
    }
    assert sdk.links == [("ws-c1", "ac_higgsfield_mcp")]


def test_a_reconnect_finds_the_dcr_config_without_creating_another():
    sdk = _FakeSDK({HIGGSFIELD: ["DCR_OAUTH"]})
    client = _client(sdk)
    first = client._ensure_auth_config_id(HIGGSFIELD)

    again = client._ensure_auth_config_id(HIGGSFIELD, preferred_scheme="DCR_OAUTH")

    assert again == first == "ac_higgsfield_mcp"
    assert len(sdk.auth_configs.created) == 1


def test_a_toolkit_that_offers_oauth2_still_gets_oauth2():
    sdk = _FakeSDK({"INSTAGRAM": ["OAUTH2"]})
    client = _client(sdk)

    assert client._ensure_auth_config_id("INSTAGRAM") == "ac_instagram"
    assert sdk.auth_configs.created == [("INSTAGRAM", {"type": "use_custom_auth", "authScheme": "OAUTH2"})]
