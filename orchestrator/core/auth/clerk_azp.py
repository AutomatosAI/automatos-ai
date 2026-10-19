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
- ``azp`` missing or empty: Clerk omits it when the minting request carried no
  ``Origin`` (its session-token docs: "could be omitted if ... Origin is empty or
  null"). That covers native clients, such as the dormant automatos-mobile Expo app,
  and tokens minted through the Backend API. No flow in this repo mints one today,
  but that has not been checked against live traffic. So by default such a token
  is accepted and a warning is logged. ``CLERK_AZP_REQUIRED=true`` refuses it, which
  is what ``@clerk/backend`` does when ``authorizedParties`` is set (since
  clerk/javascript#8877, July 2026).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, FrozenSet, Mapping
from urllib.parse import urlsplit

from config import config

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


def azp_allowed(claims: Mapping[str, Any]) -> bool:
    """Whether a verified token's ``azp`` claim passes the check (rules in the module docstring)."""
    azp = claims.get("azp")
    subject = claims.get("sub")
    if azp is None or azp == "":
        if config.CLERK_AZP_REQUIRED:
            logger.warning("Clerk token refused: no azp claim (CLERK_AZP_REQUIRED=true), sub=%s", subject)
            return False
        logger.warning("Clerk token has no azp claim; accepted (CLERK_AZP_REQUIRED=false), sub=%s", subject)
        return True
    if isinstance(azp, str) and azp.strip().rstrip("/").lower() in authorized_parties():
        return True
    logger.warning("Clerk token refused: azp %r is not in CLERK_AUTHORIZED_PARTIES, sub=%s", azp, subject)
    return False
