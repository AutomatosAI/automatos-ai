"""
LinkedIn Direct Image Post
==========================
Composio cannot upload images to LinkedIn (May 2026 — issues #3094, #3113, #3231).
This module bypasses Composio and calls LinkedIn's Community Management API directly
for image posts, using the same flow as Postiz (github.com/gitroomhq/postiz-app).

Text-only posts still go through Composio. This module only activates when
the agent passes image file references (media_urls, images, etc.).

The hooks in tool_executor.py and recipe_executor.py intercept
LINKEDIN_CREATE_LINKED_IN_POST calls with image params and route them here.
The function signature matches what both hooks expect.

REMOVAL CHECKLIST (when Composio ships a working image post action):
  1. Delete this file
  2. Remove the hook in tool_executor.py  (search: linkedin_image_workaround)
  3. Remove the hook in recipe_executor.py (search: linkedin_image_workaround)
  4. Remove LINKEDIN_* env vars from config.py
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

from config import config
from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)

LINKEDIN_API = "https://api.linkedin.com"
LINKEDIN_VERSION = "202601"
MAX_IMAGE_BYTES = 5 * 1024 * 1024

_RESTLI_HEADERS = {
    "LinkedIn-Version": LINKEDIN_VERSION,
    "X-Restli-Protocol-Version": "2.0.0",
}

_cached_token: Optional[str] = None
_cached_token_expires: float = 0
_token_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

async def _get_access_token(http: httpx.AsyncClient) -> str:
    """Return a valid LinkedIn access token, refreshing if needed."""
    global _cached_token, _cached_token_expires

    async with _token_lock:
        if _cached_token and time.time() < _cached_token_expires:
            return _cached_token

        if config.LINKEDIN_ACCESS_TOKEN and not _cached_token:
            _cached_token = config.LINKEDIN_ACCESS_TOKEN
            _cached_token_expires = time.time() + 86400
            return _cached_token

        if not config.LINKEDIN_REFRESH_TOKEN:
            raise ValueError(
                "LINKEDIN_REFRESH_TOKEN not set — cannot authenticate with LinkedIn."
            )

        resp = await http.post(
            "https://www.linkedin.com/oauth/v2/accessToken",
            data={
                "grant_type": "refresh_token",
                "refresh_token": config.LINKEDIN_REFRESH_TOKEN,
                "client_id": config.LINKEDIN_CLIENT_ID,
                "client_secret": config.LINKEDIN_CLIENT_SECRET,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            logger.debug("[LinkedIn] Token refresh error body: %s", resp.text[:300])
            raise ValueError(f"LinkedIn token refresh failed with status {resp.status_code}")

        data = resp.json()
        _cached_token = data["access_token"]
        _cached_token_expires = time.time() + data.get("expires_in", 3600) - 60
        logger.info("[LinkedIn] Access token refreshed, expires in %ds", data.get("expires_in", 0))
        return _cached_token


def _auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        **_RESTLI_HEADERS,
    }


# ---------------------------------------------------------------------------
# Parameter extraction (reused from original workaround)
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


def _extract_author(params: Dict[str, Any]) -> Optional[str]:
    """Pull author URN, defaulting to configured org page."""
    return params.get("author") or params.get("owner") or config.LINKEDIN_ORG_URN or None


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
    upload_url = value.get("uploadUrl")
    image_urn = value.get("image")
    return upload_url, image_urn


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
    """Step 3: Create the LinkedIn post with image URNs.
    Returns (success, post_id_or_error, safe_error_msg).
    """
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
        post_id = resp.headers.get("x-restli-id", "")
        return True, post_id, None
    else:
        logger.debug("[LinkedIn] createPost error body: %s", resp.text[:300])
        return False, "", f"LinkedIn API returned {resp.status_code}"


# ---------------------------------------------------------------------------
# Main entry point — same signature as before so hooks don't change
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

    The composio_client and entity_id params are accepted but unused —
    kept for interface compatibility with the hooks in tool_executor.py
    and recipe_executor.py.

    Returns a result dict matching ComposioClient.execute_action() shape.
    """
    image_paths = _extract_image_paths(params)
    text = _extract_text(params)
    author = _extract_author(params)

    if not image_paths:
        return {"success": False, "data": None, "error": "No image paths found in params"}
    if not text:
        return {"success": False, "data": None, "error": "No post text found in params"}
    if not author:
        return {"success": False, "data": None, "error": "No LinkedIn author URN configured"}

    ws_client = WorkspaceClient(workspace_id)
    image_urns: List[str] = []
    failed: List[str] = []

    async with httpx.AsyncClient(timeout=60) as http:
        token = await _get_access_token(http)

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
