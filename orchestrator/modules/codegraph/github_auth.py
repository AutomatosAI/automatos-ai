"""GitHub auth-token resolution (PRD-165 S4 / Q36).

Prefers a GitHub App *installation token* when a GitHub App is configured;
otherwise falls back to the personal access token (``GITHUB``). The App path is
the seam: it activates the moment ``GITHUB_APP_ID`` + private key + installation
id are set — no code change needed. Every failure degrades to the PAT, so
codegraph indexing never breaks because token minting did.

This keeps the working PAT path live today (no GitHub App credentials are
configured yet) while making the App upgrade a config-only switch.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# installation_id -> (token, expiry_epoch). Installation tokens last ~1h; we
# cache with a safety margin and re-mint on miss.
_TOKEN_CACHE: Dict[str, Tuple[str, float]] = {}


def _pat() -> Optional[str]:
    from config import Config
    return Config().GITHUB_PAT or None


async def resolve_github_token() -> Optional[str]:
    """A usable GitHub token: a GitHub App installation token when the App is
    configured, else the PAT. Never raises."""
    from config import Config
    cfg = Config()
    app_id = getattr(cfg, "GITHUB_APP_ID", "") or ""
    private_key = getattr(cfg, "GITHUB_APP_PRIVATE_KEY", "") or ""
    installation_id = getattr(cfg, "GITHUB_APP_INSTALLATION_ID", "") or ""

    if not (app_id and private_key and installation_id):
        return _pat()

    try:
        token = await _installation_token(app_id, private_key, installation_id)
        return token or _pat()
    except Exception as exc:  # noqa: BLE001 — any failure falls back to PAT
        logger.warning("GitHub App token mint failed (%s) — using PAT", exc)
        return _pat()


async def _installation_token(
    app_id: str, private_key: str, installation_id: str
) -> Optional[str]:
    cached = _TOKEN_CACHE.get(installation_id)
    if cached and cached[1] - 60 > time.time():
        return cached[0]

    import jwt  # PyJWT
    import httpx

    now = int(time.time())
    app_jwt = jwt.encode(
        {"iat": now - 60, "exp": now + 540, "iss": app_id},
        private_key,
        algorithm="RS256",
    )
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"https://api.github.com/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    token = data.get("token")
    if token:
        # Cache ~50 minutes regardless of the (later) GitHub expiry.
        _TOKEN_CACHE[installation_id] = (token, time.time() + 3000)
    return token
