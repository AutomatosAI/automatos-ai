"""The auth scheme the platform's own Composio auth config is created with (F251).

When Composio has no managed auth for a toolkit, the platform creates a custom auth
config itself, and Composio refuses a scheme the toolkit doesn't offer (400
``Auth_Config_AuthSchemeNotFound``). On 2026-10-03 every Higgsfield connect failed
that way, from the Tools page and from Socials: the fallback was hardcoded to OAUTH2,
and HIGGSFIELD_MCP offers only DCR_OAUTH.
"""
from __future__ import annotations

from typing import Iterable

# The only fallback before F251: it stays first, so a toolkit that offers it is
# configured as before, and it is the answer when a toolkit's schemes are unknown.
DEFAULT_CUSTOM_AUTH_SCHEME = "OAUTH2"

# The OAuth schemes the platform prefers, in order, when a toolkit offers several.
# DCR_OAUTH registers its client at sign-in, so it needs no credentials from us.
PREFERRED_CUSTOM_AUTH_SCHEMES = (DEFAULT_CUSTOM_AUTH_SCHEME, "DCR_OAUTH")


def custom_auth_fallback_scheme(offered: Iterable[str]) -> str:
    """The scheme for a custom auth config when Composio can't manage a toolkit's auth.

    The first of ``PREFERRED_CUSTOM_AUTH_SCHEMES`` the toolkit offers, otherwise the
    first scheme it offers at all. OAUTH2 when its schemes are unknown (an empty list:
    the toolkit lookup failed), which is what the platform always asked for before.
    """
    schemes = [str(scheme).upper() for scheme in offered if scheme]
    for preferred in PREFERRED_CUSTOM_AUTH_SCHEMES:
        if preferred in schemes:
            return preferred
    return schemes[0] if schemes else DEFAULT_CUSTOM_AUTH_SCHEME
