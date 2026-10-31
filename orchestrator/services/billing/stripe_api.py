"""A thin Stripe REST client on httpx (PRD-265 W2).

httpx is already a dependency, so Stripe costs no new package: Stripe's API is
form-encoded POSTs and GETs with a bearer key. Every request pins
``config.STRIPE_API_VERSION`` (``Stripe-Version``), so the shapes the webhook
handlers read never change when Stripe ships a new version. A POST carries an
idempotency key when the caller gives one, and a request is retried on a network
error, a 409 (a lock or idempotency conflict), a 429 or a 5xx (Stripe's advice; a
retried POST is safe only with a key, so a POST without one is never retried).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Optional
from urllib.parse import urlencode, urlsplit

import httpx

logger = logging.getLogger(__name__)

STRIPE_API_BASE = "https://api.stripe.com"
FORM_CONTENT_TYPE = "application/x-www-form-urlencoded"
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.5
_RETRY_STATUSES = frozenset({409, 429, 500, 502, 503, 504})
# An invoice PDF Stripe hands out (``invoice.invoice_pdf``) lives on a Stripe host.
STRIPE_FILE_HOST_SUFFIX = ".stripe.com"


class StripeNotConfigured(Exception):
    """Billing has no Stripe key (or the price it needs) in this environment."""


class StripeError(Exception):
    """Stripe refused a request. ``message`` is Stripe's own text, for the server
    log; callers show the user a plain sentence instead."""

    def __init__(self, status: int, message: str, code: Optional[str] = None):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(f"Stripe {status} {code or ''}: {message}".strip())


def _flatten(prefix: str, value: Any) -> Iterable[tuple[str, str]]:
    if value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(f"{prefix}[{key}]" if prefix else str(key), item)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _flatten(f"{prefix}[{index}]", item)
    elif isinstance(value, bool):
        yield prefix, "true" if value else "false"
    else:
        yield prefix, str(value)


def encode_form(params: Optional[dict]) -> list[tuple[str, str]]:
    """Stripe's bracket form encoding: ``{"a": {"b": [{"c": 1}]}}`` becomes
    ``a[b][0][c]=1``; booleans are ``true``/``false``; None is left out."""
    return list(_flatten("", params or {}))


def _error_of(response: httpx.Response) -> StripeError:
    try:
        error = (response.json() or {}).get("error") or {}
    except ValueError:
        error = {}
    return StripeError(response.status_code, str(error.get("message") or response.text[:200]),
                       error.get("code") or error.get("type"))


class StripeApi:
    """Stripe's REST API for one secret key. ``transport`` is for tests."""

    def __init__(self, secret_key: str, *, api_version: str, timeout: float,
                 transport: Optional[httpx.BaseTransport] = None, sleep=time.sleep):
        if not secret_key:
            raise StripeNotConfigured("STRIPE_SECRET_KEY is not set")
        self._headers = {"Authorization": f"Bearer {secret_key}", "Stripe-Version": api_version}
        self._timeout = timeout
        self._transport = transport
        self._sleep = sleep

    def post(self, path: str, params: Optional[dict] = None, *, idempotency_key: Optional[str] = None) -> dict:
        headers = {"Content-Type": FORM_CONTENT_TYPE}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return self._request("POST", path, content=urlencode(encode_form(params)), headers=headers,
                             retry=bool(idempotency_key))

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        return self._request("GET", path, params=encode_form(params), retry=True)

    def fetch_file(self, url: str, *, max_bytes: int) -> bytes:
        """A file Stripe links to (an invoice PDF), so our server can hand it to the
        browser. Only Stripe's https hosts, before and after redirects; the secret key
        is never sent to them (the link is signed); at most ``max_bytes``."""
        if not _is_stripe_file_url(url):
            raise StripeError(0, "not a Stripe file link")
        with httpx.Client(timeout=self._timeout, transport=self._transport, follow_redirects=True) as client:
            response = client.get(url)
        if not _is_stripe_file_url(str(response.url)):
            raise StripeError(0, "the file link left Stripe")
        if response.status_code >= 400:
            raise StripeError(response.status_code, "the file could not be fetched")
        if len(response.content) > max_bytes:
            raise StripeError(0, "the file is larger than expected")
        return response.content

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        extra = kwargs.pop("headers", None) or {}
        with httpx.Client(base_url=STRIPE_API_BASE, timeout=self._timeout, transport=self._transport) as client:
            return client.request(method, path, headers={**self._headers, **extra}, **kwargs)

    def _request(self, method: str, path: str, *, retry: bool, **kwargs: Any) -> dict:
        attempts = MAX_ATTEMPTS if retry else 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._send(method, path, **kwargs)
            except httpx.TransportError:
                if attempt == attempts:
                    logger.exception("[billing] Stripe %s %s unreachable", method, path)
                    raise
                self._sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            if response.status_code < 400:
                return response.json()
            if attempt == attempts or response.status_code not in _RETRY_STATUSES:
                raise _error_of(response)
            self._sleep(RETRY_BACKOFF_SECONDS * attempt)
        raise AssertionError("unreachable")  # the loop always returns or raises


def _is_stripe_file_url(url: str) -> bool:
    parts = urlsplit(url or "")
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and host.endswith(STRIPE_FILE_HOST_SUFFIX)


def stripe_api(transport: Optional[httpx.BaseTransport] = None) -> StripeApi:
    """The client for this environment's key; :class:`StripeNotConfigured` without one."""
    from config import config

    return StripeApi(config.STRIPE_SECRET_KEY, api_version=config.STRIPE_API_VERSION,
                     timeout=float(config.STRIPE_TIMEOUT_SECONDS), transport=transport)
