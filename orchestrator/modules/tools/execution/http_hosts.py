"""PRD-277 D-12 (#1605): the hosts an agent's ``http_request`` tool may call, per edition.

The tool may call the platform's own API and web app: the internal hosts ``config`` names and the
loopback names, and, where the edition's agents may reach them, Automatos's public API and web
hosts. In ``enterprise`` the platform's API is the install's own, so the list holds only the
install's hosts (``EditionCapabilities.agent_http_own_hosts_only``). ``local`` and ``saas`` keep
the list, and the tool's description, exactly as they were.

``exec_shell._get_allowed_http_domains`` checks a call against ``allowed_http_hosts``; the tool
registry gives the model a description naming the same hosts (``fitted_to_edition``). Every
redirect the call follows is checked against the same list (``refuse_hosts_off_the_list``), and
outside ``local`` the server's certificate is checked (``verifies_tls``, #1656).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Tuple

from core import editions

AUTOMATOS_PUBLIC_HOSTS: Tuple[str, ...] = ("api.automatos.app", "ui.automatos.app")
LOOPBACK_HOSTS: Tuple[str, ...] = ("localhost", "127.0.0.1")
# How the registry's description names the two public hosts, between the internal hosts and localhost.
_PUBLIC_HOSTS_NAMED = "".join(f"{host}, " for host in AUTOMATOS_PUBLIC_HOSTS)


def automatos_public_hosts() -> Tuple[str, ...]:
    """Automatos's public API and web hosts where the edition's agents may call them; none in ``enterprise``."""
    return () if editions.current().agent_http_own_hosts_only else AUTOMATOS_PUBLIC_HOSTS


def allowed_http_hosts() -> Tuple[str, ...]:
    """Every host the tool may call, in the order its description and metadata name them."""
    from config import config

    return (
        config.INTERNAL_API_HOSTNAME,
        config.INTERNAL_FRONTEND_HOSTNAME,
        *automatos_public_hosts(),
        *LOOPBACK_HOSTS,
    )


class HostNotAllowed(Exception):
    """A request the tool was about to send, a redirect's included, is to a host off the list."""

    def __init__(self, host: str) -> None:
        super().__init__(f"Redirect to '{host}' refused: that host is not whitelisted")
        self.host = host


async def refuse_hosts_off_the_list(request: Any) -> None:
    """An httpx ``request`` event hook: raise ``HostNotAllowed`` unless ``request``'s host is on the list.

    httpx runs a client's request hooks before every request it sends, each redirect it follows
    included, so a redirect from an allowed host to any other is refused before it is sent (#1656).
    """
    host = (request.url.host or "").lower()
    if host not in allowed_http_hosts():
        raise HostNotAllowed(host)


def verifies_tls() -> bool:
    """Whether the tool checks an ``https`` server's certificate: everywhere but ``local`` (#1656)."""
    return editions.current().agent_http_verifies_tls


def fitted_to_edition(spec: Any) -> Any:
    """The ``http_request`` tool spec as the registry wrote it, or, in ``enterprise``, a copy naming only the install's hosts.

    The registry's spec names both public hosts in its description and its ``allowed_domains``
    metadata. Where the edition's agents may call them it is returned unchanged; otherwise a new
    spec is returned with both left out of each, and nothing else changed.
    """
    if automatos_public_hosts():
        return spec
    return replace(
        spec,
        description=spec.description.replace(_PUBLIC_HOSTS_NAMED, ""),
        metadata={**spec.metadata, "allowed_domains": list(allowed_http_hosts())},
    )


__all__ = [
    "AUTOMATOS_PUBLIC_HOSTS",
    "LOOPBACK_HOSTS",
    "HostNotAllowed",
    "allowed_http_hosts",
    "automatos_public_hosts",
    "fitted_to_edition",
    "refuse_hosts_off_the_list",
    "verifies_tls",
]
