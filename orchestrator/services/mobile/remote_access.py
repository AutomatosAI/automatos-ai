"""PRD-266 §4.3 — remote access on a local install: the boot check and the anonymous lane.

Pure: values in, a decision out. Nothing here reads config, a database or a request.
``config.validate_auth_edition`` hands it the edition, ``LOCAL_REMOTE_ACCESS`` and the two
addresses compose publishes the API and the web app on (``API_PUBLISH_ADDRESS``,
``WEB_PUBLISH_ADDRESS``: docker-compose.yml passes both into the API container), and
``core/auth/bearer_registry.py`` asks it whether a request with no credential may still be
the operator.

* ``off`` (the default): the anonymous lane is open, so the install must stay on this
  machine. A publish address beyond loopback refuses boot, naming the fix.
* ``paired``: every caller proves who it is with a credential (a paired device's token, the
  API key, …); a request with none is ``401 {"code": "device_pairing_required"}``.
* ``open``: today's behaviour on a published install, for an operator who accepts the risk:
  a WARNING at every boot while either address reaches beyond this machine.
* An unknown mode refuses boot. The saas edition is never checked: Clerk is its boundary,
  and these answer "allowed" / "open" for it whatever the other values are.

Loopback is 127.0.0.0/8, ``::1`` and ``localhost``; every other address, ``0.0.0.0`` and
``::`` among them, reaches beyond this machine. The address is the one compose PUBLISHES
on, never a request's source: identity is a credential, never an address (inside the
container a caller on this machine and one on the LAN can both arrive from the Docker
gateway).
"""
from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

LOCAL_EDITION = "local"
MODE_OFF = "off"
MODE_PAIRED = "paired"
MODE_OPEN = "open"
MODES = (MODE_OFF, MODE_PAIRED, MODE_OPEN)
# The modes in which a request with no credential is still the operator.
ANONYMOUS_MODES = frozenset({MODE_OFF, MODE_OPEN})
LOOPBACK_NAMES = frozenset({"localhost"})
# The three ways out a refused boot names.
FIX_HINT = (
    "Set LOCAL_REMOTE_ACCESS=paired (every phone and browser then pairs with a one-time "
    "code: docker compose exec backend python -m scripts.mint_pairing_code), or "
    "LOCAL_REMOTE_ACCESS=open to accept the risk, or publish on 127.0.0.1 "
    "(API_PUBLISH_ADDRESS, WEB_PUBLISH_ADDRESS, BIND_ADDRESS)."
)


class RemoteAccessRefused(RuntimeError):
    """Boot refused: the remote-access mode and the publish addresses contradict each other."""


@dataclass(frozen=True)
class BootDecision:
    """What boot does: refuse with ``refusal``, start and log ``warning``, or just start."""

    refusal: Optional[str] = None
    warning: Optional[str] = None


def is_loopback(address: Optional[str]) -> bool:
    """True for 127.0.0.0/8, ``::1`` (bracketed or not) and ``localhost``; False for anything else."""
    value = (address or "").strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if value.lower() in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def published_beyond_loopback(addresses: Mapping[str, str]) -> Tuple[str, ...]:
    """``NAME=address`` for each publish address that reaches beyond this machine, in name order."""
    return tuple(f"{name}={addresses[name]}" for name in sorted(addresses) if not is_loopback(addresses[name]))


def boot_decision(edition: str, mode: str, addresses: Mapping[str, str]) -> BootDecision:
    """Refuse, warn or start, for this edition, ``LOCAL_REMOTE_ACCESS`` and publish addresses."""
    if edition != LOCAL_EDITION:
        return BootDecision()
    if mode not in MODES:
        return BootDecision(refusal=f"LOCAL_REMOTE_ACCESS={mode!r} is not one of {', '.join(MODES)}.")
    exposed = published_beyond_loopback(addresses)
    if not exposed or mode == MODE_PAIRED:
        return BootDecision()
    where = ", ".join(exposed)
    if mode == MODE_OPEN:
        return BootDecision(warning=(
            f"LOCAL_REMOTE_ACCESS=open with {where}: this install is published beyond this "
            "machine and a request with no credential is its operator, so whoever reaches it "
            "owns it. Set LOCAL_REMOTE_ACCESS=paired to require a paired device."
        ))
    return BootDecision(refusal=(
        f"LOCAL_REMOTE_ACCESS=off, but {where} publishes this install beyond this machine, "
        "where a request with no credential would be its operator. " + FIX_HINT
    ))


def enforce_boot_decision(edition: str, mode: str, addresses: Mapping[str, str]) -> None:
    """Raise ``RemoteAccessRefused`` for a refused boot; log the warning of an ``open`` one."""
    decision = boot_decision(edition, mode, addresses)
    if decision.refusal:
        raise RemoteAccessRefused(decision.refusal)
    if decision.warning:
        logger.warning(decision.warning)


def anonymous_lane_open(edition: str, mode: str) -> bool:
    """Whether remote access leaves a request with no credential its anonymous lane.

    False only on a local install that is not ``off`` or ``open`` (``paired``, and an unknown
    mode, which boot refuses anyway: fail closed). On saas it is always True: remote access
    never applies there, and ``REQUIRE_AUTH`` still governs that lane.
    """
    return edition != LOCAL_EDITION or mode in ANONYMOUS_MODES


__all__ = [
    "BootDecision",
    "MODES",
    "MODE_OFF",
    "MODE_OPEN",
    "MODE_PAIRED",
    "RemoteAccessRefused",
    "anonymous_lane_open",
    "boot_decision",
    "enforce_boot_decision",
    "is_loopback",
    "published_beyond_loopback",
]
