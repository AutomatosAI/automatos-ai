"""
LinkedIn Direct Image Post
==========================
Composio cannot upload images to LinkedIn (May 2026 — issues #3094, #3113, #3231).
This module bypasses Composio and calls LinkedIn's Community Management API directly
for image posts, using the same flow as Postiz (github.com/gitroomhq/postiz-app).

Credentials are loaded from the platform's credential store (PRD-18) — the same
system used for PostgreSQL, OpenAI, and MCP server credentials. Each workspace
adds its own "LinkedIn Community Management OAuth2 API" credential.

Workspace-scoped (PRD-251 S0.4, owner 2026-09-23): the credential, its
organisation URN and the access token are ALWAYS the calling workspace's own.
Credentials and tokens are cached per workspace; a workspace with no active
credential of its own gets a clear error and no LinkedIn call is made — it never
falls back to another workspace's credential.

The caches never outlive the credential (P251-RVW-4). Every load re-reads the
workspace's active credential: its id, ``updated_at`` and a digest of its
encrypted data. A different credential, an edited one, or none at all drops
the workspace's cached credential and token (``clear_workspace_cache``). This
holds for a change made by another worker as well, because it is read from the
database and not pushed by an in-process hook. A deactivated or deleted
credential stops the next post before any LinkedIn call, and a replaced or
edited one is used from the next post on.

An image URL is an agent's tool argument, so fetching it goes through the
platform's one SSRF decision, ``core/security/web_access.py`` (P251-RVW-7).
The host is checked against the operator's denylist, and every address it
resolves to against the blocked ranges. The request then connects to the
address that was checked, so a DNS answer that changes in between cannot
redirect it. A refused URL sends no request, and its reason reaches the
result. The ``WEB_ACCESS`` switch does not apply here
(``enforce_switch=False``, as for the heartbeat webhook): posting an image the
workspace asked for is not agent web browsing. Redirects are not followed.

Text-only posts still go through Composio. This module only activates when
the agent passes image file references (media_urls, images, etc.).

The hooks in tool_executor.py and recipe_executor.py intercept
LINKEDIN_CREATE_LINKED_IN_POST calls with image params and route them here.
The function signature matches what both hooks expect.

REMOVAL CHECKLIST (when Composio ships a working image post action):
  1. Delete this file
  2. Remove the hook in tool_executor.py  (search: linkedin_image_workaround)
  3. Remove the hook in recipe_executor.py (search: linkedin_image_workaround)
  4. Remove the smoke-test route in api/composio.py (search: linkedin_image_workaround)
  5. Update SKILL.md to use the native Composio action
"""

import asyncio
import base64
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import httpx

from core.security.web_access import build_pinned_request, resolve_outbound_async
from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)

LINKEDIN_API = "https://api.linkedin.com"
LINKEDIN_VERSION = "202601"
CREDENTIAL_TYPE_NAME = "linkedInCommunityManagementOAuth2Api"
MAX_IMAGE_BYTES = 5 * 1024 * 1024

_RESTLI_HEADERS = {
    "LinkedIn-Version": LINKEDIN_VERSION,
    "X-Restli-Protocol-Version": "2.0.0",
}

# The Composio action this module stands in for (image posts): the deny list
# (core/composio/deny_list.py) is consulted with this slug.
IMAGE_POST_ACTION = "LINKEDIN_CREATE_LINKED_IN_POST"

# How long a stored access token is trusted before a refresh is attempted.
STORED_TOKEN_TTL_SECONDS = 86400
REFRESHED_TOKEN_DEFAULT_TTL_SECONDS = 3600
REFRESH_SAFETY_MARGIN_SECONDS = 60

# Per-workspace caches, keyed by the workspace id (str). There is no
# process-wide credential or token: one workspace's never serves another.
_creds_by_workspace: Dict[str, Dict[str, Any]] = {}
_token_by_workspace: Dict[str, Tuple[str, float]] = {}
_lock_by_workspace: Dict[str, asyncio.Lock] = {}
# The credential the cached credential and token came from (_credential_version).
_version_by_workspace: Dict[str, Tuple[int, Optional[str], str]] = {}


class LinkedInCredentialError(ValueError):
    """The workspace has no usable LinkedIn credential of its own."""


def _workspace_key(workspace_id: Any) -> str:
    if workspace_id is None:
        raise LinkedInCredentialError("LinkedIn image posts need a workspace")
    try:
        return str(UUID(str(workspace_id)))
    except ValueError as exc:
        raise LinkedInCredentialError(f"LinkedIn image posts need a workspace id, got {workspace_id!r}") from exc


def clear_workspace_cache(workspace_id: Any) -> None:
    """Forget a workspace's cached credential and token. The loader calls this
    when the active credential is not the one they came from."""
    key = _workspace_key(workspace_id)
    _creds_by_workspace.pop(key, None)
    _token_by_workspace.pop(key, None)
    _version_by_workspace.pop(key, None)


def _credential_version(cred: Any) -> Tuple[int, Optional[str], str]:
    """Identifies the credential as it is now: its id, when it was last updated,
    and a digest of its encrypted data (which catches an edit that skipped
    updated_at)."""
    updated_at = cred.updated_at.isoformat() if cred.updated_at else None
    digest = hashlib.sha256((cred.encrypted_data or "").encode("utf-8")).hexdigest()
    return cred.id, updated_at, digest


# ---------------------------------------------------------------------------
# Credential resolution via platform credential store
# ---------------------------------------------------------------------------

def _load_linkedin_credentials(workspace_id: Any) -> Dict[str, Any]:
    """Load THIS workspace's LinkedIn credential from the credential store.

    Returns dict with keys: client_id, client_secret, access_token,
    refresh_token (optional), organization_urn. Raises
    :class:`LinkedInCredentialError` when the workspace has no active
    credential of its own — never another workspace's.

    The active credential is re-read on every call. The cached copy is returned
    only while it is still that credential, unchanged. Otherwise the
    workspace's cached credential and token are dropped first, so a token is
    never paired with a credential it did not come from.
    """
    key = _workspace_key(workspace_id)

    from core.database.database import SessionLocal
    from core.credentials.service import CredentialStore
    from core.models.credentials import CredentialType, Credential

    db = SessionLocal()
    try:
        cred_type = db.query(CredentialType).filter(
            CredentialType.name == CREDENTIAL_TYPE_NAME
        ).first()
        if not cred_type:
            raise LinkedInCredentialError(
                f"Credential type '{CREDENTIAL_TYPE_NAME}' not found. "
                "Run seed_credential_types to add it."
            )

        cred = (
            db.query(Credential)
            .filter(
                Credential.credential_type_id == cred_type.id,
                Credential.workspace_id == UUID(key),
                Credential.is_active == True,  # noqa: E712
            )
            .order_by(Credential.id.desc())
            .first()
        )
        if not cred:
            clear_workspace_cache(key)
            raise LinkedInCredentialError(
                "This workspace has no active LinkedIn Community Management credential. "
                "Add one in Settings > Credentials."
            )

        version = _credential_version(cred)
        if _version_by_workspace.get(key, version) != version:
            clear_workspace_cache(key)  # replaced or edited since it was cached
        cached = _creds_by_workspace.get(key)
        if cached and _version_by_workspace.get(key) == version:
            return cached

        store = CredentialStore(db)
        data = store.get_decrypted_credential(
            cred.id,
            service_name="linkedin_image_post",
        )
        if not data.get("access_token"):
            raise LinkedInCredentialError("LinkedIn credential missing access_token field")
        if not data.get("organization_urn"):
            raise LinkedInCredentialError("LinkedIn credential missing organization_urn field")

        _creds_by_workspace[key] = data
        _version_by_workspace[key] = version
        return data
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

async def _get_access_token(http: httpx.AsyncClient, workspace_id: Any) -> str:
    """Return a valid LinkedIn access token for THIS workspace, refreshing if needed."""
    key = _workspace_key(workspace_id)
    lock = _lock_by_workspace.setdefault(key, asyncio.Lock())

    async with lock:
        # First: a replaced, edited or removed credential drops the cached token.
        creds = _load_linkedin_credentials(key)
        cached = _token_by_workspace.get(key)
        if cached and time.time() < cached[1]:
            return cached[0]

        if not cached:
            token = creds["access_token"]
            _token_by_workspace[key] = (token, time.time() + STORED_TOKEN_TTL_SECONDS)
            return token

        refresh_token = creds.get("refresh_token")
        if not refresh_token:
            raise LinkedInCredentialError(
                "LinkedIn access token may be expired and no refresh_token is set. "
                "Update the credential in Settings > Credentials."
            )

        resp = await http.post(
            "https://www.linkedin.com/oauth/v2/accessToken",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"],
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            logger.debug("[LinkedIn] Token refresh error body: %s", resp.text[:300])
            raise ValueError(f"LinkedIn token refresh failed with status {resp.status_code}")

        data = resp.json()
        token = data["access_token"]
        expires_in = data.get("expires_in", REFRESHED_TOKEN_DEFAULT_TTL_SECONDS)
        _token_by_workspace[key] = (token, time.time() + expires_in - REFRESH_SAFETY_MARGIN_SECONDS)
        _creds_by_workspace.pop(key, None)
        logger.info("[LinkedIn] Access token refreshed for workspace %s, expires in %ds", key, expires_in)
        return token


def _auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        **_RESTLI_HEADERS,
    }


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

def has_image_params(params: Dict[str, Any]) -> bool:
    """Return True if params contain file references that need image upload."""
    for key in ("media_urls", "images", "media", "media_files", "image_urls"):
        val = params.get(key)
        if not val:
            continue
        if isinstance(val, list) and len(val) > 0:
            return True
        if isinstance(val, str) and "/" in val and not val.startswith("urn:"):
            return True
    return False


def _normalize_path(v) -> str:
    """Extract a usable file path from a string or dict (workspace file ref)."""
    if isinstance(v, dict):
        return v.get("s3key") or v.get("path") or v.get("name") or ""
    return str(v)


def _extract_image_paths(params: Dict[str, Any]) -> List[str]:
    """Pull image paths/URLs from whichever param name the agent used."""
    for key in ("media_urls", "images", "media", "media_files", "image_urls"):
        val = params.get(key)
        if isinstance(val, list) and len(val) > 0:
            return [p for p in (_normalize_path(v) for v in val) if p]
        if isinstance(val, str) and "/" in val and not val.startswith("urn:"):
            return [val]
    return []


def _extract_text(params: Dict[str, Any]) -> str:
    """Pull post text from whichever param name the agent used."""
    for key in ("text", "commentary", "content", "message", "body"):
        val = params.get(key)
        if val and isinstance(val, str):
            return val
    return ""


def _extract_author(params: Dict[str, Any], creds: Dict[str, Any]) -> Optional[str]:
    """Pull author URN, defaulting to the workspace credential's organization_urn."""
    return params.get("author") or params.get("owner") or creds.get("organization_urn")


# ---------------------------------------------------------------------------
# Image download (a URL, or a workspace file)
# ---------------------------------------------------------------------------

async def _fetch_image_url(url: str, http: httpx.AsyncClient) -> Tuple[Optional[bytes], Optional[str]]:
    """Image bytes from ``url``, or ``(None, why not)``. The URL is checked and
    pinned by ``core/security/web_access.py``: a refused URL sends no request."""
    target = await resolve_outbound_async(url, enforce_switch=False)
    if not target.ok:
        logger.warning("[LinkedIn] Refused image URL %s: %s", url[:80], target.reason)
        return None, f"refused: {target.reason}"
    resp = await http.send(build_pinned_request(http, "GET", url, target))
    if resp.status_code != 200:
        logger.warning("[LinkedIn] Failed to fetch URL %s: %s", url[:80], resp.status_code)
        return None, f"the URL answered HTTP {resp.status_code}"
    ct = resp.headers.get("content-type", "")
    if ct and not ct.startswith("image/"):
        logger.warning("[LinkedIn] URL %s returned non-image content-type: %s", url[:80], ct)
        return None, f"the URL is not an image ({ct})"
    return resp.content, None


async def _download_image(
    img_path: str,
    ws_client: WorkspaceClient,
    http: httpx.AsyncClient,
) -> Tuple[Optional[bytes], Optional[str]]:
    """Image bytes from a URL or a workspace path, or ``(None, why not)``."""
    if img_path.startswith(("http://", "https://")):
        return await _fetch_image_url(img_path, http)

    dl = await ws_client.download_file(img_path)
    if not dl.get("success"):
        logger.info("[LinkedIn] download_file failed, trying read_file: %s", img_path)
        dl = await ws_client.read_file(img_path)
    if not dl.get("success"):
        # The worker's error stays in the server log, not the agent's result.
        logger.warning("[LinkedIn] All download methods failed for %s: %s", img_path, dl.get("error"))
        return None, "could not read it from the workspace"

    content = dl.get("content") or dl.get("data")
    if isinstance(content, str):
        try:
            return base64.b64decode(content), None
        except Exception:
            return content.encode("utf-8"), None
    if not content:
        return None, "the workspace file is empty"
    return content, None


# ---------------------------------------------------------------------------
# LinkedIn API calls (following Postiz flow)
# ---------------------------------------------------------------------------

async def _initialize_image_upload(
    http: httpx.AsyncClient,
    token: str,
    owner_urn: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Step 1: Initialize image upload. Returns (upload_url, image_urn)."""
    resp = await http.post(
        f"{LINKEDIN_API}/rest/images?action=initializeUpload",
        headers=_auth_headers(token),
        json={"initializeUploadRequest": {"owner": owner_urn}},
    )
    if resp.status_code not in (200, 201):
        logger.error("[LinkedIn] initializeUpload failed: %s", resp.status_code)
        return None, None

    value = resp.json().get("value", {})
    return value.get("uploadUrl"), value.get("image")


async def _upload_image_bytes(
    http: httpx.AsyncClient,
    token: str,
    upload_url: str,
    image_bytes: bytes,
) -> bool:
    """Step 2: PUT binary bytes to LinkedIn's upload URL."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
        **_RESTLI_HEADERS,
    }
    resp = await http.put(upload_url, content=image_bytes, headers=headers)
    if resp.status_code not in (200, 201):
        logger.error("[LinkedIn] PUT upload failed: %s", resp.status_code)
        return False
    return True


async def _create_post(
    http: httpx.AsyncClient,
    token: str,
    author: str,
    text: str,
    image_urns: List[str],
) -> Tuple[bool, str, Optional[str]]:
    """Step 3: Create the LinkedIn post with image URNs."""
    if len(image_urns) == 1:
        content_block = {"media": {"id": image_urns[0]}}
    else:
        content_block = {
            "multiImage": {
                "images": [{"id": urn} for urn in image_urns]
            }
        }

    body = {
        "author": author,
        "commentary": text,
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "content": content_block,
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }

    resp = await http.post(
        f"{LINKEDIN_API}/rest/posts",
        headers=_auth_headers(token),
        json=body,
    )

    if resp.status_code in (200, 201):
        return True, resp.headers.get("x-restli-id", ""), None
    else:
        logger.debug("[LinkedIn] createPost error body: %s", resp.text[:300])
        return False, "", f"LinkedIn API returned {resp.status_code}"


# ---------------------------------------------------------------------------
# Main entry point — same signature so hooks don't change
# ---------------------------------------------------------------------------

async def execute_linkedin_image_post(
    params: Dict[str, Any],
    workspace_id: UUID,
    entity_id: str,
    composio_client,
) -> Dict[str, Any]:
    """Post to LinkedIn with images via direct API calls.

    Bypasses Composio entirely for image posts. Uses LinkedIn's Community
    Management API directly: initializeUpload -> PUT binary -> createPost.

    The credential, organisation and token are the calling workspace's own
    (``workspace_id``). A workspace without a LinkedIn credential gets an error
    and no LinkedIn request is made.

    The composio_client and entity_id params are accepted but unused —
    kept for interface compatibility with the hooks.
    """
    image_paths = _extract_image_paths(params)
    text = _extract_text(params)

    if not image_paths:
        return {"success": False, "data": None, "error": "No image paths found in params"}
    if not text:
        return {"success": False, "data": None, "error": "No post text found in params"}

    try:
        creds = _load_linkedin_credentials(workspace_id)
    except LinkedInCredentialError as exc:
        return {"success": False, "data": None, "error": str(exc)}

    author = _extract_author(params, creds)
    if not author:
        return {"success": False, "data": None, "error": "No LinkedIn author URN configured for this workspace"}

    ws_client = WorkspaceClient(workspace_id)
    image_urns: List[str] = []
    # One {"image", "reason"} per image that was not uploaded, so the caller
    # learns why (a refused URL names its refusal).
    failed: List[Dict[str, str]] = []

    async with httpx.AsyncClient(timeout=60) as http:
        try:
            token = await _get_access_token(http, workspace_id)
        except LinkedInCredentialError as exc:
            return {"success": False, "data": None, "error": str(exc)}

        for i, img_path in enumerate(image_paths):
            label = f"image[{i}]"

            image_bytes, why_not = await _download_image(img_path, ws_client, http)
            if not image_bytes:
                failed.append({"image": label, "reason": why_not or "no image data"})
                continue

            if len(image_bytes) > MAX_IMAGE_BYTES:
                logger.warning("[LinkedIn] %s is %d bytes, exceeds 5MB limit", label, len(image_bytes))
                too_big = f"{len(image_bytes)} bytes is over the {MAX_IMAGE_BYTES}-byte limit"
                failed.append({"image": label, "reason": too_big})
                continue

            logger.info("[LinkedIn] Initializing upload for %s (%d bytes)", label, len(image_bytes))
            upload_url, image_urn = await _initialize_image_upload(http, token, author)
            if not upload_url or not image_urn:
                failed.append({"image": label, "reason": "LinkedIn did not start the upload"})
                continue

            logger.info("[LinkedIn] Uploading %s -> %s", label, image_urn)
            ok = await _upload_image_bytes(http, token, upload_url, image_bytes)
            if not ok:
                failed.append({"image": label, "reason": "LinkedIn refused the upload"})
                continue

            image_urns.append(image_urn)

        if not image_urns:
            reasons = "; ".join(f"{f['image']}: {f['reason']}" for f in failed)
            return {
                "success": False,
                "data": None,
                "error": f"All image uploads failed ({len(failed)} failures): {reasons}",
            }

        logger.info("[LinkedIn] Creating post with %d images", len(image_urns))
        ok, post_id, err = await _create_post(http, token, author, text, image_urns)

        if ok:
            logger.info("[LinkedIn] Post created: %s", post_id)
            return {
                "success": True,
                "data": {
                    "successful": True,
                    "post_id": post_id,
                    "images_uploaded": len(image_urns),
                    "images_failed": len(failed),
                    "failures": failed,
                    "image_urns": image_urns,
                },
                "error": None,
            }
        else:
            logger.error("[LinkedIn] Create post failed: %s", err)
            return {
                "success": False,
                "data": None,
                "error": f"LinkedIn create post failed: {err}",
            }
