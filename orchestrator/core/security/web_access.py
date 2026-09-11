"""PRD-240 — the ONE place that decides whether an agent may reach a URL.

Two layers, deliberately separate:

* **Always refused, not configurable** — private, loopback, link-local and
  metadata ranges, resolved through DNS (``url_validator.validate_webhook_url``).
  This is what keeps an agent out of ``postgres:5432``, ``minio:9000``,
  ``host.docker.internal`` and ``169.254.169.254`` on the compose network.
* **The operator's denylist** — ``WEB_ACCESS_DENY``, suffix-matched hosts. Adds
  to the first layer; can never widen it.

``WEB_ACCESS`` is the switch: on by default in the local edition, off in saas
unless set. Both the platform actions (``handlers_web``) and the provider-side
search tool (``openai_compatible_client``) ask this module, so the two never
disagree about what "web access" means.
"""

from __future__ import annotations

from typing import Optional, Tuple
from urllib.parse import urlparse

from config import config
from core.security.url_validator import validate_webhook_url

WEB_ACCESS_OFF_REASON = (
    "Web access is off on this server (WEB_ACCESS=off). Set WEB_ACCESS=on in "
    ".env and restart the backend to let agents read and search the web."
)
_ALLOWED_SCHEMES = ("http", "https")


def web_access_enabled() -> bool:
    return bool(config.WEB_ACCESS)


def denied_hosts() -> Tuple[str, ...]:
    return tuple(config.WEB_ACCESS_DENY or ())


def host_denied(host: Optional[str]) -> bool:
    """True when ``host`` is, or is under, a denied host (``example.com`` also
    covers ``www.example.com``; ``notexample.com`` is untouched)."""
    if not host:
        return False
    h = host.lower().rstrip(".")
    for denied in denied_hosts():
        if h == denied or h.endswith("." + denied):
            return True
    return False


def validate_outbound_url(url: str) -> Tuple[bool, str, str]:
    """(ok, reason, host) for a URL an agent wants to fetch.

    Order matters: the switch, then the scheme, then the denylist, then the
    private-range/DNS check — so a denied host is named as denied rather than
    surfacing a DNS failure, and a switched-off server never resolves anything.
    """
    if not web_access_enabled():
        return False, WEB_ACCESS_OFF_REASON, ""
    try:
        parsed = urlparse((url or "").strip())
    except Exception:  # noqa: BLE001 — a malformed URL is a refusal, not a crash
        return False, "Malformed URL", ""
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"Only http and https URLs can be fetched (got '{parsed.scheme or 'none'}')", host
    if not host:
        return False, "No hostname in URL", host
    if host_denied(host):
        return False, f"'{host}' is on this server's WEB_ACCESS_DENY list", host
    ok, reason = validate_webhook_url(url)
    if not ok:
        return False, f"'{host}' is not reachable from agents: {reason}", host
    return True, "OK", host
