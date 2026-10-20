"""The Clerk ``azp`` (authorized party) check on session tokens.

Clerk sets a session token's ``azp`` claim to the ``Origin`` of the request that
minted it: the app's own URL for a browser sign-in, ``https://ui.automatos.app`` in
production. After the production cutover every sub-domain of automatos.app shares
one Clerk instance (studio.automatos.app already loads it for the waitlist), so a
token minted on another origin still carries a valid signature for this API. The
check accepts a token only when its ``azp`` is one of the frontend origins that
really call the API (``config.CLERK_AUTHORIZED_PARTIES``, plus ``FRONTEND_URL``).

What happens to a token:

- ``azp`` present and allowed: accepted.
- ``azp`` present and not allowed: refused. The callers of
  ``ClerkAuth.verify_token`` answer 401.
- ``azp`` missing or empty, with the native marker: accepted, whatever
  ``CLERK_AZP_REQUIRED`` says. Clerk omits ``azp`` when the minting request carried
  no ``Origin`` (its session-token docs: "could be omitted if ... Origin is empty or
  null"), and a native app's requests carry none. Automatos Go asks for its token
  from the ``mobile`` JWT template, whose static claim ``client: "automatos-go"`` is
  the marker (``core/auth/clerk_mobile_template.py``, PRD-266 US-020). It is logged
  at debug only, so the warning below keeps meaning "an unknown azp-less client".
- ``azp`` missing or empty, without the marker (or with any other ``client``
  value): tokens minted through the Backend API, or another native client. No flow
  in this repo mints one today, but that has not been checked against live traffic.
  So by default such a token is accepted and a warning is logged.
  ``CLERK_AZP_REQUIRED=true`` refuses it, which is what ``@clerk/backend`` does when
  ``authorizedParties`` is set (since clerk/javascript#8877, July 2026).

The marker never stands in for a wrong ``azp``: a token that has one is judged by
the allow-list alone, so a browser on another origin that asks for the ``mobile``
template is still refused. The marker is not a credential; what it does and does not
keep out is in ``core/auth/clerk_mobile_template.py``.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, FrozenSet, Mapping
from urllib.parse import urlsplit

from config import config
from core.auth.clerk_mobile_template import NATIVE_CLIENT_CLAIM, NATIVE_CLIENT_MARKER

logger = logging.getLogger(__name__)


def _origin(value: str) -> str:
    """``scheme://host[:port]`` of a configured URL, lower-cased; "" if it has none."""
    parts = urlsplit(value.strip())
    if not (parts.scheme and parts.netloc):
        return ""
    return f"{parts.scheme}://{parts.netloc}".lower()


@lru_cache(maxsize=8)
def parse_authorized_parties(raw: str, frontend_url: str = "") -> FrozenSet[str]:
    """The allowed origins from a comma-separated list, plus ``frontend_url``'s origin.

    Each entry is reduced to its origin, so a trailing slash or a path in the
    configuration still matches the browser's ``Origin``. An entry without a scheme
    and host can never match a real ``azp``; it is dropped with a warning rather than
    left to 401 every sign-in silently.
    """
    allowed = set()
    for entry in [*(raw or "").split(","), frontend_url or ""]:
        if not entry.strip():
            continue
        origin = _origin(entry)
        if origin:
            allowed.add(origin)
        else:
            logger.warning("Clerk azp allow-list: ignoring %r (not a scheme://host origin)", entry.strip())
    return frozenset(allowed)


def authorized_parties() -> FrozenSet[str]:
    """The origins whose Clerk session tokens this API accepts, from config."""
    return parse_authorized_parties(config.CLERK_AUTHORIZED_PARTIES or "", config.FRONTEND_URL or "")


def carries_native_marker(claims: Mapping[str, Any]) -> bool:
    """Whether a verified token carries Automatos Go's marker: the ``mobile`` JWT
    template's static ``client`` claim, compared exactly."""
    return claims.get(NATIVE_CLIENT_CLAIM) == NATIVE_CLIENT_MARKER


def _missing_azp_allowed(claims: Mapping[str, Any], subject: Any) -> bool:
    """The rule for a token with no ``azp``: the native marker, else ``CLERK_AZP_REQUIRED``."""
    if carries_native_marker(claims):
        logger.debug("Clerk token accepted as Automatos Go: no azp, native marker present, sub=%s", subject)
        return True
    client = claims.get(NATIVE_CLIENT_CLAIM)
    if config.CLERK_AZP_REQUIRED:
        logger.warning(
            "Clerk token refused: no azp claim (CLERK_AZP_REQUIRED=true), sub=%s, client=%r", subject, client,
        )
        return False
    logger.warning(
        "Clerk token has no azp claim; accepted (CLERK_AZP_REQUIRED=false), sub=%s, client=%r", subject, client,
    )
    return True


def azp_allowed(claims: Mapping[str, Any]) -> bool:
    """Whether a verified token's ``azp`` claim passes the check (rules in the module docstring)."""
    azp = claims.get("azp")
    subject = claims.get("sub")
    if azp is None or azp == "":
        return _missing_azp_allowed(claims, subject)
    if isinstance(azp, str) and azp.strip().rstrip("/").lower() in authorized_parties():
        return True
    logger.warning("Clerk token refused: azp %r is not in CLERK_AUTHORIZED_PARTIES, sub=%s", azp, subject)
    return False
