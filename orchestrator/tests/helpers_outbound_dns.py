"""Fake DNS for tests that send to a saved destination through the outbound guard.

On an edition where several people share the server, a webhook or a credential test
resolves its host through ``core.security.web_access._getaddrinfo`` before it sends
anything (``web_access.operator_destination_async``). A test that only cares what
was sent names its hosts here, so nothing depends on the network's own DNS.
"""
from __future__ import annotations

import ipaddress
import socket
from typing import Mapping

import core.security.web_access as web_access

#: A public address (example.com's): outside every range the guard refuses.
PUBLIC_IP = "93.184.216.34"


def fake_dns(monkeypatch, answers: Mapping[str, str]) -> None:
    """Resolve each host in ``answers`` to its address, and an IP literal to itself;
    any other name fails, as an unknown name does."""

    def getaddrinfo(host, port, proto=None):
        ip = answers.get(host)
        if ip is None:
            try:
                ip = str(ipaddress.ip_address(host))
            except ValueError:
                raise socket.gaierror(f"no fake DNS for {host}") from None
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr(web_access, "_getaddrinfo", getaddrinfo)


def public_dns(monkeypatch, *hosts: str) -> None:
    """Resolve each of ``hosts`` to :data:`PUBLIC_IP`."""
    fake_dns(monkeypatch, {host: PUBLIC_IP for host in hosts})
