"""Credential types whose seeded row differs from ``credential_types_seed.json``.

PRD-262 US-031: the LinkedIn Community Management schema used to be a direct ``UPDATE``
in ``main.py``'s lifespan, run on every boot after the credential-type upsert had written
the seed file's version, so the row was rewritten twice per boot. The override now lives
here, under ``core/seeds/``, where the seed fingerprint covers it, and the upsert
(``core.database.load_seed_data``) applies it before writing: the row is written once,
in its final shape, and left alone while it stays that way.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Mapping

LINKEDIN_COMMUNITY_MANAGEMENT = "linkedInCommunityManagementOAuth2Api"

_LINKEDIN_SCHEMA = (
    {"displayName": "Client ID", "name": "client_id", "type": "string", "required": True,
     "description": "OAuth2 Client ID from LinkedIn Developer Portal"},
    {"displayName": "Client Secret", "name": "client_secret", "type": "password", "required": True,
     "description": "OAuth2 Client Secret"},
    {"displayName": "Access Token", "name": "access_token", "type": "password", "required": True,
     "description": "OAuth2 access token"},
    {"displayName": "Refresh Token", "name": "refresh_token", "type": "password", "required": False,
     "description": "OAuth2 refresh token for automatic renewal"},
    {"displayName": "Organization URN", "name": "organization_urn", "type": "string", "required": True,
     "description": "e.g. urn:li:organization:108072660"},
)

# The fields each overridden type's row carries instead of the seed file's.
OVERRIDES: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    LINKEDIN_COMMUNITY_MANAGEMENT: MappingProxyType({
        "schema_definition": _LINKEDIN_SCHEMA,
        "category": "social",
        "description": "LinkedIn Community Management API — posts images to organization pages",
    }),
})


def with_overrides(cred: Mapping[str, Any]) -> Dict[str, Any]:
    """A seed file entry as its row is written: a new dict, with any override applied."""
    override = OVERRIDES.get(cred.get("name"), {})
    merged = {**cred, **override}
    if "schema_definition" in override:
        merged["schema_definition"] = [dict(field) for field in override["schema_definition"]]
    return merged
