"""PRD-277 US-401 (FR-9, FR-10): which hosts sign-in may reach, and how it calls them.

``core.auth.oidc_http`` is the one way sign-in calls the identity provider. Proven here without a
network (the resolver seam ``_getaddrinfo`` answers instead of DNS):

* only ``https`` with a host: ``http``, no scheme, a user in the address and junk are refused
  before anything is resolved;
* the configured issuer's own host may resolve to a private or loopback address (an identity
  provider inside the customer's network); any other host that resolves to one is refused, even
  when only one of its answers is private;
* a public host is reached at exactly the address that was checked (the pinned target);
* an unresolvable host is refused;
* the client follows no redirect, ignores proxy settings, has its timeouts, and trusts
  ``OIDC_CA_BUNDLE_FILE`` on top of the system's authorities when it is set (httpx's own default
  authorities when it is not);
* a failure names no address, and is raised with neither the HTTP client's exception as its
  cause nor as its context.
"""
from __future__ import annotations

import logging
import os
import socket
import ssl

import httpx
import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.auth import oidc_http  # noqa: E402
from core.auth.oidc_http import ProviderUnreachable  # noqa: E402

ISSUER = "https://idp.corp.example.test/realms/prd277"
PUBLIC_IP = "93.184.216.34"
PRIVATE_IP = "10.20.30.40"


def _answers(table):
    """A resolver that answers ``table[host]`` (a list of addresses), and nothing for other hosts."""
    def resolve(host, port, proto=0):
        if host not in table:
            raise socket.gaierror("no such host")
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, proto, "", (ip, port))
                for ip in table[host]]
    return resolve


@pytest.fixture
def resolver(monkeypatch):
    monkeypatch.setattr(config, "OIDC_ISSUER", ISSUER)
    monkeypatch.setattr(config, "OIDC_CA_BUNDLE_FILE", "")
    table = {}
    monkeypatch.setattr(oidc_http, "_getaddrinfo", _answers(table))
    return table


@pytest.mark.parametrize("address", [
    "http://idp.corp.example.test/token", "idp.corp.example.test/token", "https:///token",
    "https://user@idp.corp.example.test/token", "https://idp.corp.example.test:notaport/token", "", None, 42,
])
def test_only_an_https_address_with_a_host_is_ever_resolved(resolver, monkeypatch, address):
    monkeypatch.setattr(oidc_http, "_getaddrinfo", lambda *a, **k: pytest.fail("resolved"))
    with pytest.raises(ProviderUnreachable):
        oidc_http.pin(address, "token")


@pytest.mark.parametrize("ip", [PRIVATE_IP, "127.0.0.1", "172.16.0.9", "192.168.1.2", "169.254.169.254", "::1"])
def test_the_issuers_own_host_may_be_private(resolver, ip):
    resolver["idp.corp.example.test"] = [ip]

    target = oidc_http.pin("https://idp.corp.example.test/realms/prd277/keys", "keys")

    assert (target.host, target.ip, target.port, target.scheme) == ("idp.corp.example.test", ip, 443, "https")


@pytest.mark.parametrize("ip", [PRIVATE_IP, "127.0.0.1", "169.254.169.254", "100.64.0.1", "::1", "fd00::1"])
def test_no_other_host_may_be_private(resolver, ip):
    resolver["tokens.elsewhere.example.test"] = [ip]

    with pytest.raises(ProviderUnreachable):
        oidc_http.pin("https://tokens.elsewhere.example.test/token", "token")


def test_one_private_answer_among_public_ones_is_enough_to_refuse(resolver):
    resolver["tokens.elsewhere.example.test"] = [PUBLIC_IP, PRIVATE_IP]

    with pytest.raises(ProviderUnreachable):
        oidc_http.pin("https://tokens.elsewhere.example.test/token", "token")


def test_a_public_host_is_reached_at_the_address_that_was_checked(resolver):
    resolver["oauth2.example.test"] = [PUBLIC_IP]

    target = oidc_http.pin("https://oauth2.example.test:8443/token", "token")

    assert (target.ok, target.host, target.ip, target.port) == (True, "oauth2.example.test", PUBLIC_IP, 8443)


def test_an_unresolvable_host_is_refused(resolver):
    with pytest.raises(ProviderUnreachable):
        oidc_http.pin("https://nowhere.example.test/token", "token")


def test_the_issuer_host_is_compared_exactly_not_as_a_suffix(resolver):
    resolver["evil-idp.corp.example.test.attacker.example"] = [PRIVATE_IP]

    with pytest.raises(ProviderUnreachable):
        oidc_http.pin("https://evil-idp.corp.example.test.attacker.example/token", "token")


def test_the_client_follows_no_redirect_ignores_proxies_and_has_timeouts(resolver):
    client = oidc_http._client()
    try:
        assert client.follow_redirects is False
        assert client.trust_env is False
        assert client.timeout.connect == oidc_http.CONNECT_TIMEOUT_SECONDS
        assert client.timeout.read == oidc_http.READ_TIMEOUT_SECONDS
    finally:
        client.close()


def test_without_a_ca_bundle_httpx_verifies_with_its_default_authorities(resolver):
    assert oidc_http._tls_verify() is True


def test_a_ca_bundle_is_trusted_on_top_of_the_system_authorities(resolver, monkeypatch, tmp_path):
    from tests.helpers_fake_oidc_provider import _self_signed

    cert_file, _key = _self_signed(tmp_path)
    monkeypatch.setattr(config, "OIDC_CA_BUNDLE_FILE", str(cert_file))
    system = ssl.create_default_context().cert_store_stats()["x509_ca"]

    context = oidc_http._tls_verify()

    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True
    assert context.cert_store_stats()["x509_ca"] == system + 1


def test_a_failed_call_names_no_address_and_carries_no_client_exception(resolver, monkeypatch, caplog):
    resolver["oauth2.example.test"] = [PUBLIC_IP]
    secret_path = "/token?code=prd277-" + "w4-code-marker"

    def refused(*args, **kwargs):
        raise httpx.ConnectError(f"cannot reach https://oauth2.example.test{secret_path}")

    monkeypatch.setattr(oidc_http, "_send", refused)
    with caplog.at_level(logging.DEBUG), pytest.raises(ProviderUnreachable) as failed:
        oidc_http.call("POST", "https://oauth2.example.test/token", "token", data={"code": "x"})

    assert str(failed.value) == oidc_http.UNREACHABLE_MESSAGE
    assert failed.value.__cause__ is None and failed.value.__context__ is None
    assert "oauth2.example.test" not in caplog.text and "code-marker" not in caplog.text
    assert "ConnectError" in caplog.text


def test_an_answer_that_is_not_200_json_is_refused_by_get_json(resolver, monkeypatch):
    monkeypatch.setattr(oidc_http, "_send", lambda *a, **k: (302, b""))
    with pytest.raises(ProviderUnreachable):
        oidc_http.get_json("https://idp.corp.example.test/x", "metadata")
    monkeypatch.setattr(oidc_http, "_send", lambda *a, **k: (200, b"<html>"))
    with pytest.raises(ProviderUnreachable):
        oidc_http.get_json("https://idp.corp.example.test/x", "metadata")
    monkeypatch.setattr(oidc_http, "_send", lambda *a, **k: (200, None))
    with pytest.raises(ProviderUnreachable):
        oidc_http.get_json("https://idp.corp.example.test/x", "metadata")
