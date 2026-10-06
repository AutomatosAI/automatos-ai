"""F375: a document's http(s) fetch goes to the address it checked, and checks every redirect.

The WeasyPrint URL fetcher (``url_fetcher._safe_url_fetcher``) checked that a host
resolved only to public addresses, then called WeasyPrint's own fetcher, which
resolved the host again (DNS rebinding) and followed redirects; the Word renderer's
image fetch resolved again too. Now both go through ``pinned_fetch.fetch_public``:
one resolution per hop, the request sent to that very address with the hostname as
``Host``, and each redirect hop resolved and checked again.

Pure: a fake resolver (``web_access._getaddrinfo``, the seam the webhook tests use)
and an httpx MockTransport. Nothing leaves the process.
"""
from __future__ import annotations

import socket
from typing import Callable, Dict, List

import httpx
import pytest

import core.security.web_access as wa
from modules.documents import pinned_fetch, url_fetcher
from modules.documents.blocks.docx_renderer import _safe_image_bytes
from modules.documents.pinned_fetch import FetchRefused, fetch_public

PUBLIC_IP, OTHER_PUBLIC_IP, PRIVATE_IP = "93.184.216.34", "151.101.1.69", "10.0.0.5"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class _Resolver:
    """Fake DNS: each host's answers in turn (the last one repeats); counts lookups."""

    def __init__(self, answers: Dict[str, List[str]]) -> None:
        self.answers, self.asked = answers, []

    def __call__(self, host, port, proto=None):
        self.asked.append(host)
        queue = self.answers.get(host)
        if queue is None:
            ip = host  # a literal address resolves to itself
            try:
                socket.inet_aton(ip)
            except OSError:
                raise socket.gaierror(f"no fake DNS for {host}") from None
        else:
            ip = queue.pop(0) if len(queue) > 1 else queue[0]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]


class _Body(httpx.SyncByteStream):
    """A body that streams as a network response does (``content=`` is read at once)."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self):
        for start in range(0, len(self.data), 16):
            yield self.data[start:start + 16]


def _streamed(status: int, data: bytes, **headers: str) -> httpx.Response:
    return httpx.Response(status, stream=_Body(data), headers=headers)


def _transport(handler: Callable[[httpx.Request], httpx.Response], sent: List[httpx.Request]):
    def record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return handler(request)

    return lambda: httpx.Client(transport=httpx.MockTransport(record), follow_redirects=False)


@pytest.fixture
def dns(monkeypatch):
    def install(answers: Dict[str, List[str]]) -> _Resolver:
        resolver = _Resolver({host: list(ips) for host, ips in answers.items()})
        monkeypatch.setattr(wa, "_getaddrinfo", resolver)
        return resolver
    return install


def test_the_request_goes_to_the_address_that_was_checked(dns):
    resolver = dns({"cdn.example": [PUBLIC_IP, PRIVATE_IP]})  # a second lookup would rebind
    sent: List[httpx.Request] = []
    fetched = fetch_public("https://cdn.example/logo.png", max_bytes=1024, client_factory=_transport(
        lambda r: _streamed(200, PNG, **{"content-type": "image/png; q=1"}), sent))

    (request,) = sent
    assert request.url.host == PUBLIC_IP and request.headers["host"] == "cdn.example"
    assert request.extensions.get("sni_hostname") == "cdn.example"
    assert resolver.asked == ["cdn.example"]  # resolved once: no second answer is ever used
    assert fetched.data == PNG and fetched.mime_type == "image/png"


def test_a_redirect_to_loopback_is_refused_and_never_requested(dns):
    dns({"cdn.example": [PUBLIC_IP]})
    sent: List[httpx.Request] = []
    with pytest.raises(FetchRefused):
        fetch_public("http://cdn.example/a.png", max_bytes=1024, client_factory=_transport(
            lambda r: httpx.Response(302, headers={"location": "http://127.0.0.1:8000/admin"}), sent))
    assert [r.url.host for r in sent] == [PUBLIC_IP]


def test_a_redirect_whose_host_now_resolves_privately_is_refused(dns):
    dns({"cdn.example": [PUBLIC_IP, PRIVATE_IP]})  # the second resolution rebinds to 10.0.0.5
    sent: List[httpx.Request] = []
    with pytest.raises(FetchRefused):
        fetch_public("http://cdn.example/a.png", max_bytes=1024, client_factory=_transport(
            lambda r: httpx.Response(302, headers={"location": "/b.png"}), sent))
    assert [r.url.host for r in sent] == [PUBLIC_IP]


def test_a_redirect_to_a_public_host_is_checked_pinned_and_followed(dns):
    resolver = dns({"cdn.example": [PUBLIC_IP], "img.example": [OTHER_PUBLIC_IP]})
    sent: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["host"] == "cdn.example":
            return httpx.Response(301, headers={"location": "https://img.example/final.png"})
        return _streamed(200, PNG, **{"content-type": "image/png"})

    fetched = fetch_public("https://cdn.example/a.png", max_bytes=1024, client_factory=_transport(handler, sent))
    assert [(r.url.host, r.headers["host"]) for r in sent] == [(PUBLIC_IP, "cdn.example"), (OTHER_PUBLIC_IP, "img.example")]
    assert resolver.asked == ["cdn.example", "img.example"] and fetched.url == "https://img.example/final.png"


def test_too_many_redirects_and_too_many_bytes_are_refused(dns):
    dns({"cdn.example": [PUBLIC_IP]})
    loop: List[httpx.Request] = []
    with pytest.raises(FetchRefused, match="redirects"):
        fetch_public("http://cdn.example/a", max_bytes=1024, client_factory=_transport(
            lambda r: httpx.Response(302, headers={"location": "/a"}), loop))
    assert len(loop) == pinned_fetch.MAX_REDIRECTS + 1
    with pytest.raises(FetchRefused, match="larger"):
        fetch_public("http://cdn.example/big", max_bytes=8, client_factory=_transport(
            lambda r: _streamed(200, b"x" * 64), []))
    with pytest.raises(FetchRefused, match="larger"):  # a body a transport already loaded: capped too
        fetch_public("http://cdn.example/big", max_bytes=8, client_factory=_transport(
            lambda r: httpx.Response(200, content=b"x" * 64), []))


def test_the_weasyprint_fetcher_hands_back_what_the_pinned_fetch_read(dns, monkeypatch):
    dns({"cdn.example": [PUBLIC_IP]})
    sent: List[httpx.Request] = []
    monkeypatch.setattr(pinned_fetch, "_client", _transport(
        lambda r: _streamed(200, PNG, **{"content-type": "image/png"}), sent))
    got = url_fetcher._safe_url_fetcher("https://cdn.example/logo.png")
    assert got == {"string": PNG, "mime_type": "image/png", "redirected_url": "https://cdn.example/logo.png"}
    assert sent[0].url.host == PUBLIC_IP


def test_the_weasyprint_and_word_fetchers_refuse_a_redirect_into_the_metadata_address(dns, monkeypatch):
    dns({"cdn.example": [PUBLIC_IP]})
    sent: List[httpx.Request] = []
    monkeypatch.setattr(pinned_fetch, "_client", _transport(
        lambda r: httpx.Response(307, headers={"location": "http://169.254.169.254/latest/meta-data/"}), sent))
    with pytest.raises(ValueError):
        url_fetcher._safe_url_fetcher("https://cdn.example/logo.png")
    assert _safe_image_bytes("https://cdn.example/logo.png") is None
    assert {r.url.host for r in sent} == {PUBLIC_IP}  # the metadata address was never requested


def test_a_compressed_body_is_never_decoded_past_the_cap(dns):
    """A gzip body could decode to far more than the cap in one chunk: only an unencoded body is read."""
    import gzip

    dns({"cdn.example": [PUBLIC_IP]})
    bomb = gzip.compress(b"\x00" * 10_000_000)  # 10 MB of zeros, about 10 KB on the wire
    sent: List[httpx.Request] = []
    with pytest.raises(FetchRefused, match="encoded"):
        fetch_public("http://cdn.example/bomb", max_bytes=1024, client_factory=_transport(
            lambda r: httpx.Response(200, content=bomb, headers={"content-encoding": "gzip"}), sent))
    assert sent[0].headers["accept-encoding"] == "identity"


def test_a_declared_size_over_the_cap_is_refused_before_reading(dns):
    dns({"cdn.example": [PUBLIC_IP]})
    with pytest.raises(FetchRefused, match="larger"):
        fetch_public("http://cdn.example/big", max_bytes=8, client_factory=_transport(
            lambda r: httpx.Response(200, content=b"x" * 4, headers={"content-length": "999999999"}), []))


def test_the_whole_fetch_has_a_deadline(dns, monkeypatch):
    dns({"cdn.example": [PUBLIC_IP]})
    monkeypatch.setattr(pinned_fetch, "FETCH_DEADLINE_SECONDS", -1.0)  # already past
    with pytest.raises(FetchRefused, match="longer than"):
        fetch_public("http://cdn.example/slow", max_bytes=1024, client_factory=_transport(
            lambda r: httpx.Response(200, content=PNG), []))


def test_a_body_a_transport_already_loaded_is_read_within_the_cap(dns):
    dns({"cdn.example": [PUBLIC_IP]})
    fetched = fetch_public("http://cdn.example/logo.png", max_bytes=1024, client_factory=_transport(
        lambda r: httpx.Response(200, content=PNG, headers={"content-type": "image/png"}), []))
    assert fetched.data == PNG
