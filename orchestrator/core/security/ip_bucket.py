"""The per-client rate-limit bucket rule for the public routes (PRD-264 EUAI-10, PRD-266 K13).

A route that takes no account can only count its callers by address. Two rules keep
that count honest and keep Redis keys clean:

* An IPv6 subscriber is usually handed a whole /64, so per-address buckets would let
  them rotate addresses inside it; the bucket is the /64 instead. An IPv4-mapped IPv6
  address counts as its IPv4 address.
* A client-IP value that is not an IP address (a junk ``X-Forwarded-For``) shares one
  ``<prefix>:unparsable`` bucket, so a caller can never write an arbitrary key.

The address itself comes from ``api.credentials.get_client_ip``; this module only turns
it into a bucket name, so it never trusts or reads a header.
"""
from __future__ import annotations

import ipaddress

IPV6_BUCKET_PREFIX_LENGTH = 64
UNPARSABLE_BUCKET_SUFFIX = "unparsable"


def ip_rate_bucket(client_ip: str, prefix: str) -> str:
    """The bucket for ``client_ip`` under ``prefix``: one IPv4 address or one IPv6 /64."""
    try:
        address = ipaddress.ip_address(client_ip)
    except ValueError:
        return f"{prefix}:{UNPARSABLE_BUCKET_SUFFIX}"
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    if address.version == 6:
        network = ipaddress.IPv6Network((int(address), IPV6_BUCKET_PREFIX_LENGTH), strict=False)
        return f"{prefix}:{network}"
    return f"{prefix}:{address}"
