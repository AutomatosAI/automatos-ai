"""PRD-277 US-402 (§5.2, FR-5): the one-time code exchanged, once, for the ID token.

``exchange_code`` posts the code to the metadata's ``token_endpoint`` through ``core.auth.oidc_http``
(https, pinned, no redirect), with:

* ``grant_type=authorization_code``, the ``code``, the exact ``redirect_uri`` the sign-in started
  with, and the PKCE ``code_verifier`` kept for it (RFC 7636 §4.5);
* the client's secret as the provider asks for it: HTTP Basic (``client_secret_basic``, RFC 6749
  §2.3.1: the id and the secret each form-encoded first), else in the form (``client_secret_post``).

The answer must be 200 with a JSON object holding an ``id_token`` string; only that is read. The
access token and anything else in it are never kept or logged. Anything else is
``CodeExchangeRefused``: the log names only the status, never the provider's error text, which may
repeat what it was sent.

The code is exchanged at most once by this server: the sign-in's state is claimed, and so ended,
before this is called (``core.auth.oidc_pending.claim``), so a replayed callback never reaches here.
"""
from __future__ import annotations

import base64
import logging
from typing import Dict, Tuple
from urllib.parse import quote

from config import config
from core.auth import oidc_http
from core.auth.oidc_provider import AUTH_BASIC, ProviderMetadata

logger = logging.getLogger(__name__)

GRANT_TYPE = "authorization_code"
REFUSED_MESSAGE = "The identity provider did not accept the sign-in's one-time code."


class CodeExchangeRefused(Exception):
    """The token endpoint did not answer with an ID token. Its text never holds a value."""


def _form_encoded(value: str) -> str:
    return quote(value, safe="")


def _client_authentication(method: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """``(form fields, headers)`` that authenticate this client by ``method``."""
    client_id, secret = config.OIDC_CLIENT_ID, config.OIDC_CLIENT_SECRET
    if method == AUTH_BASIC:
        pair = f"{_form_encoded(client_id)}:{_form_encoded(secret)}".encode("utf-8")
        return {}, {"Authorization": "Basic " + base64.b64encode(pair).decode("ascii")}
    return {"client_id": client_id, "client_secret": secret}, {}


def exchange_code(metadata: ProviderMetadata, *, code: str, code_verifier: str, redirect_uri: str) -> str:
    """The ID token the provider gives for ``code``; ``CodeExchangeRefused`` or ``ProviderUnreachable``."""
    fields, headers = _client_authentication(metadata.token_auth_method)
    form = {"grant_type": GRANT_TYPE, "code": code, "redirect_uri": redirect_uri,
            "code_verifier": code_verifier, **fields}
    answer = oidc_http.call("POST", metadata.token_endpoint, "token", data=form, headers=headers)
    if answer.status != oidc_http.HTTP_OK:
        logger.warning("Identity provider token call answered %s", answer.status)
        raise CodeExchangeRefused(REFUSED_MESSAGE)
    id_token = answer.body.get("id_token") if isinstance(answer.body, dict) else None
    if not isinstance(id_token, str) or not id_token:
        logger.warning("Identity provider token call answered without an ID token")
        raise CodeExchangeRefused(REFUSED_MESSAGE)
    return id_token


__all__ = ["GRANT_TYPE", "CodeExchangeRefused", "exchange_code"]
