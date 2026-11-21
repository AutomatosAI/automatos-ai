"""PRD-277 US-401 (FR-9, FR-10): which hosts sign-in may reach, and how it calls them.

``core.auth.oidc_http`` is the one way sign-in calls the identity provider. Proven here without a
network (the resolver seam ``_getaddrinfo`` answers instead of DNS):

* only ``https`` with a host: ``http``, no scheme, a user in the address, a space or control
  character (tab, CR, LF, NUL, DEL) and junk are refused before anything is resolved, and an
  address the HTTP client itself cannot parse is a refusal, not an error;
* the configured issuer's own host and port may resolve to a private or loopback address (an
  identity provider inside the customer's network); any other host, or another port of that host,
  that resolves to one is refused, even when only one of its answers is private, and also when an
  IPv6 answer carries the private address (mapped, 6to4 or NAT64);
* the reserved ranges (``192.0.0.0/24``, ``198.18.0.0/15``, multicast and ``fec0::/10``) are
  refused on every host, the issuer's included;
* an answer longer than ``MAX_RESPONSE_BYTES`` is refused;
* over a pinned address, TLS is still checked against the host name: a certificate for that name
  is accepted and one for another name refused (the fake provider on loopback, resolution patched);
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
from tests.helpers_fake_oidc_provider import ISSUER_PATH, FakeIdentityProvider  # noqa: E402

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
    "https://idp.corp.example.test/to\tken", "https://idp.corp.example.test/to\rken", "https://idp.corp.example.test/to\nken",
    "https://idp.corp.\nexample.test/token", "https://idp.corp.example.test/to\x00ken", "https://idp.corp.example.test/to\x7fken",
    "https://idp.corp.example.test/to ken",
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
    assert context.minimum_version == ssl.TLSVersion.TLSv1_2
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


@pytest.mark.parametrize("ip", ["192.0.0.8", "198.18.0.1", "198.19.255.1", "224.0.0.1", "239.255.0.1",
                                "ff02::1", "fec0::1", "::ffff:198.18.0.1"])
def test_the_reserved_ranges_are_refused_even_for_the_issuers_host(resolver, ip):
    resolver["idp.corp.example.test"] = [ip]

    with pytest.raises(ProviderUnreachable):
        oidc_http.pin("https://idp.corp.example.test/realms/prd277/keys", "keys")


@pytest.mark.parametrize("ip", ["::ffff:10.20.30.40", "2002:a14:1e28::1", "64:ff9b::a14:1e28"])
def test_a_private_address_inside_an_ipv6_answer_is_still_private(resolver, ip):
    resolver["tokens.elsewhere.example.test"] = [ip]

    with pytest.raises(ProviderUnreachable):
        oidc_http.pin("https://tokens.elsewhere.example.test/token", "token")


def test_only_the_issuers_own_port_may_be_private(resolver):
    resolver["idp.corp.example.test"] = [PRIVATE_IP]

    assert oidc_http.pin("https://idp.corp.example.test:443/realms/prd277/keys", "keys").ip == PRIVATE_IP
    with pytest.raises(ProviderUnreachable):
        oidc_http.pin("https://idp.corp.example.test:8443/admin", "keys")


def test_an_address_the_http_client_cannot_parse_is_a_refusal(resolver, monkeypatch):
    resolver["oauth2.example.test"] = [PUBLIC_IP]

    def unparsable(*args, **kwargs):
        raise httpx.InvalidURL("Invalid non-printable ASCII character in URL")

    monkeypatch.setattr(oidc_http, "build_pinned_request", unparsable)
    with pytest.raises(ProviderUnreachable) as refused:
        oidc_http.call("POST", "https://oauth2.example.test/token", "token")
    assert refused.value.__cause__ is None and refused.value.__context__ is None


def _client_answering(body: bytes):
    return lambda: httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))


def test_an_answer_over_the_size_cap_is_refused(resolver, monkeypatch):
    resolver["oauth2.example.test"] = [PUBLIC_IP]
    monkeypatch.setattr(oidc_http, "_client", _client_answering(b"{}" + b" " * oidc_http.MAX_RESPONSE_BYTES))

    with pytest.raises(ProviderUnreachable):
        oidc_http.call("GET", "https://oauth2.example.test/keys", "keys")

    monkeypatch.setattr(oidc_http, "_client", _client_answering(b'{"keys": []}'))
    assert oidc_http.call("GET", "https://oauth2.example.test/keys", "keys").body == {"keys": []}


@pytest.fixture
def named_provider(tmp_path, monkeypatch):
    """The fake provider with a certificate for ``idp.prd277.test`` only, that name resolved to loopback."""
    (tmp_path / "named").mkdir()
    provider = FakeIdentityProvider(tmp_path / "named", dns_names=("idp.prd277.test",))
    monkeypatch.setattr(config, "OIDC_CA_BUNDLE_FILE", str(provider.ca_file))
    monkeypatch.setattr(oidc_http, "_getaddrinfo", _answers({"idp.prd277.test": ["127.0.0.1"],
                                                              "other.prd277.test": ["127.0.0.1"]}))
    yield provider
    provider.close()


@pytest.mark.parametrize("host", ["idp.prd277.test", "other.prd277.test"])
def test_tls_is_checked_against_the_host_name_over_the_pinned_address(named_provider, monkeypatch, host):
    issuer = f"https://{host}:{named_provider.port}{ISSUER_PATH}"
    monkeypatch.setattr(config, "OIDC_ISSUER", issuer)            # its host and port may be loopback
    address = issuer + "/.well-known/openid-configuration"

    if host == "idp.prd277.test":
        assert oidc_http.get_json(address, "metadata")["issuer"] == named_provider.issuer
    else:
        with pytest.raises(ProviderUnreachable):
            oidc_http.get_json(address, "metadata")
