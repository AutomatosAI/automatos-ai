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
import logging
import time
from ipaddress import ip_address
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from uuid import UUID

import httpx

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
    """Forget a workspace's cached credential and token (e.g. after an edit)."""
    key = _workspace_key(workspace_id)
    _creds_by_workspace.pop(key, None)
    _token_by_workspace.pop(key, None)


# ---------------------------------------------------------------------------
# Credential resolution via platform credential store
# ---------------------------------------------------------------------------

def _load_linkedin_credentials(workspace_id: Any) -> Dict[str, Any]:
    """Load THIS workspace's LinkedIn credential from the credential store.

    Returns dict with keys: client_id, client_secret, access_token,
    refresh_token (optional), organization_urn. Raises
    :class:`LinkedInCredentialError` when the workspace has no active
    credential of its own — never another workspace's.
    """
    key = _workspace_key(workspace_id)
    cached = _creds_by_workspace.get(key)
    if cached:
        return cached

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
            raise LinkedInCredentialError(
                "This workspace has no active LinkedIn Community Management credential. "
                "Add one in Settings > Credentials."
            )

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
        cached = _token_by_workspace.get(key)
        if cached and time.time() < cached[1]:
            return cached[0]

        creds = _load_linkedin_credentials(key)

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
# URL safety
# ---------------------------------------------------------------------------

def _is_safe_url(url: str) -> bool:
    """Reject URLs pointing to private/link-local/loopback addresses."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if not host:
            return False
        addr = ip_address(host)
        return addr.is_global
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Image download from workspace
# ---------------------------------------------------------------------------

async def _download_image(
    img_path: str,
    ws_client: WorkspaceClient,
    http: httpx.AsyncClient,
) -> Optional[bytes]:
    """Download image bytes from a URL or workspace path."""
    if img_path.startswith(("http://", "https://")):
        if not _is_safe_url(img_path):
            logger.warning("[LinkedIn] Blocked fetch to non-public URL: %s", img_path[:80])
            return None
        resp = await http.get(img_path)
        if resp.status_code != 200:
            logger.warning("[LinkedIn] Failed to fetch URL %s: %s", img_path[:80], resp.status_code)
            return None
        ct = resp.headers.get("content-type", "")
        if ct and not ct.startswith("image/"):
            logger.warning("[LinkedIn] URL %s returned non-image content-type: %s", img_path[:80], ct)
            return None
        return resp.content

    dl = await ws_client.download_file(img_path)
    if not dl.get("success"):
        logger.info("[LinkedIn] download_file failed, trying read_file: %s", img_path)
        dl = await ws_client.read_file(img_path)
    if not dl.get("success"):
        logger.warning("[LinkedIn] All download methods failed for %s: %s", img_path, dl.get("error"))
        return None

    content = dl.get("content") or dl.get("data")
    if isinstance(content, str):
        try:
            return base64.b64decode(content)
        except Exception:
            return content.encode("utf-8")
    return content or None


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
    failed: List[str] = []

    async with httpx.AsyncClient(timeout=60) as http:
        token = await _get_access_token(http, workspace_id)

        for i, img_path in enumerate(image_paths):
            label = f"image[{i}]"

            image_bytes = await _download_image(img_path, ws_client, http)
            if not image_bytes:
                failed.append(img_path)
                continue

            if len(image_bytes) > MAX_IMAGE_BYTES:
                logger.warning("[LinkedIn] %s is %d bytes, exceeds 5MB limit", label, len(image_bytes))
                failed.append(img_path)
                continue

            logger.info("[LinkedIn] Initializing upload for %s (%d bytes)", label, len(image_bytes))
            upload_url, image_urn = await _initialize_image_upload(http, token, author)
            if not upload_url or not image_urn:
                failed.append(img_path)
                continue

            logger.info("[LinkedIn] Uploading %s -> %s", label, image_urn)
            ok = await _upload_image_bytes(http, token, upload_url, image_bytes)
            if not ok:
                failed.append(img_path)
                continue

            image_urns.append(image_urn)

        if not image_urns:
            return {
                "success": False,
                "data": None,
                "error": f"All image uploads failed ({len(failed)} failures)",
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
