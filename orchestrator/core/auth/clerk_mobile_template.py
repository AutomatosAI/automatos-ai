"""The Clerk ``mobile`` JWT template: the token Automatos Go signs in with (PRD-266 US-020).

A native request carries no ``Origin``, so Clerk mints the phone's token without an
``azp`` claim (Clerk's session-token docs: ``azp`` "could be omitted if ... ``Origin``
is empty or null"). The app therefore asks for its token with
``getToken({template: "mobile"})``. The ``mobile`` JWT template, created by hand in
the Clerk dashboard, adds the static claim ``client: "automatos-go"``, the native
marker. ``core/auth/clerk_azp.py`` accepts an azp-less token that carries the marker,
also when ``CLERK_AZP_REQUIRED`` is true. A token that has an ``azp`` is judged by the
allow-list, marker or not.

The marker says which template minted a token; it is not a credential, and its name
and value are public (the app is open source). Any signed-in user can ask Clerk for a
``mobile`` token:

- A page in a browser sends its ``Origin`` with the request to Clerk's Frontend API,
  so its ``mobile`` token has an ``azp``, and a page on another origin is refused.
- A client outside a browser (an app, curl) that holds a live session gets an
  azp-less marked token. It could equally send an allowed ``Origin`` and get an
  allowed ``azp``: the azp check never bound such clients.
- The case the marker reopens under ``CLERK_AZP_REQUIRED=true``: a browser request
  that reaches Clerk with an empty or ``null`` ``Origin`` (a sandboxed frame, for
  one), from a context that still holds the user's Clerk session, asking for the
  ``mobile`` template.

A template token is not a session token. Clerk adds only ``azp``, ``exp``, ``iat``,
``iss``, ``jti``, ``nbf`` and ``sub`` to it, and cannot add the session-bound claims
(``sid``, ``v``, ``pla``, ``fea``; Clerk's JWT-template docs). Every other claim the
API reads has to come from the template, or a phone would be verified, named and
given a role differently from the same person on the web:

- ``MOBILE_JWT_TEMPLATE_CLAIMS`` is the template, the JSON to paste into the
  dashboard. The shortcodes are Clerk's.
- ``CLAIMS_NOT_IN_TEMPLATE`` names each claim the API reads that the template leaves
  out on purpose, with the reason.

``tests/test_clerk_azp_check.py`` reads ``core/auth/clerk.py`` and
``core/auth/clerk_azp.py`` and fails when the claims they read stop matching these
two constants, so a new claim read there cannot ship without a decision here.

Not pinned here, because no constant can hold them:

- ``aud``: when ``CLERK_AUDIENCE`` is set, ``ClerkAuth.verify_token`` requires it, so
  the template must also emit ``"aud"`` with that value, as the session token does.
- ``fva``: Clerk documents no template shortcode for it, so a phone's token fails
  every ``require_step_up`` route on SaaS (``core/auth/step_up.py``).
"""
from __future__ import annotations

from types import MappingProxyType
from typing import FrozenSet, Mapping

# The template's name in the Clerk dashboard: the app calls getToken({template: "mobile"}).
MOBILE_JWT_TEMPLATE_NAME = "mobile"

# The native marker: a static claim only the mobile template emits.
NATIVE_CLIENT_CLAIM = "client"
NATIVE_CLIENT_MARKER = "automatos-go"

# The template's claims, as entered in the Clerk dashboard. An org shortcode is null
# when the session has no active organization, which clerk.py reads as absent.
MOBILE_JWT_TEMPLATE_CLAIMS: Mapping[str, str] = MappingProxyType({
    NATIVE_CLIENT_CLAIM: NATIVE_CLIENT_MARKER,
    "email": "{{user.primary_email_address}}",
    "metadata": "{{user.public_metadata}}",
    "name": "{{user.full_name}}",
    "first_name": "{{user.first_name}}",
    "last_name": "{{user.last_name}}",
    "image_url": "{{user.image_url}}",
    "org_id": "{{org.id}}",
    "org_slug": "{{org.slug}}",
})

# Clerk puts these in every template token itself; a template cannot set them.
CLERK_TEMPLATE_DEFAULT_CLAIMS: FrozenSet[str] = frozenset({"azp", "exp", "iat", "iss", "jti", "nbf", "sub"})

# Claims core/auth/clerk.py reads that the template leaves out, and why.
CLAIMS_NOT_IN_TEMPLATE: Mapping[str, str] = MappingProxyType({
    "primary_email": "a fallback for email, which the template sets",
    "email_address": "a fallback for email, which the template sets",
    "emailAddress": "a fallback for email, which the template sets",
    "email_addresses": "a fallback for email, which the template sets",
    "primary_email_address_id": "a fallback for email, which the template sets",
    "public_metadata": "the legacy name of metadata, which the template sets",
    "full_name": "a fallback for name, which the template sets",
    "profile_image_url": "a fallback for image_url, which the template sets",
    "org_role": (
        "extract_user_info falls back to it for the system role when metadata has no role; "
        "left out so a phone's system role comes from metadata alone, never from an org role"
    ),
    "org_permissions": "Clerk documents no template shortcode for it, and nothing reads it after extract_user_info",
})

