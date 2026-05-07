"""
LinkedIn Image Post — TEMPORARY WORKAROUND
===========================================
Composio's LinkedIn integration cannot upload images (May 2026).
LINKEDIN_CREATE_LINKED_IN_POST is text-only. LINKEDIN_INITIALIZE_IMAGE_UPLOAD
returns an upload URL but no action to PUT the binary bytes.
Composio also redacts OAuth tokens and mangles the LinkedIn-Version header
(appends "01" to YYYYMM format, causing 426 errors via proxy).

Solution: use Composio execute_action for the JSON API calls (version
header handled correctly internally), and PUT image bytes directly to
LinkedIn's pre-signed upload URL (no auth needed).

See Composio issues: #3094, #3113, #3231.

REMOVAL CHECKLIST (when Composio ships a working image post action):
  1. Delete this file
  2. Remove the hook in tool_executor.py  (search: linkedin_image_workaround)
  3. Remove the hook in recipe_executor.py (search: linkedin_image_workaround)
  4. Update SKILL.md to use the native Composio action
"""

import logging
import httpx
from typing import Any, Dict, List
from uuid import UUID

from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)

ORG_URN = "urn:li:organization:108072660"


def has_image_params(params: Dict[str, Any]) -> bool:
    """Return True if params contain file references that need image upload."""
    for key in ("media_urls", "images", "media", "media_files", "image_urls"):
        val = params.get(key)
        if not val:
            continue
        if isinstance(val, list) and len(val) > 0:
            return True
        if isinstance(val, str) and "/" in val:
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
        if isinstance(val, str) and "/" in val:
            return [val]
    return []


def _extract_text(params: Dict[str, Any]) -> str:
    """Pull post text from whichever param name the agent used."""
    for key in ("text", "commentary", "content", "message", "body"):
        val = params.get(key)
        if val and isinstance(val, str):
            return val
    return ""


def _extract_author(params: Dict[str, Any]) -> str:
    """Pull author URN, defaulting to the Automatos org page."""
    return params.get("author") or params.get("owner") or ORG_URN


async def execute_linkedin_image_post(
    params: Dict[str, Any],
    workspace_id: UUID,
    entity_id: str,
    composio_client,
) -> Dict[str, Any]:
    """Post to LinkedIn with images via Composio actions + direct binary upload.

    Steps:
      1. LINKEDIN_INITIALIZE_IMAGE_UPLOAD via Composio (handles auth+version)
      2. PUT binary bytes to pre-signed upload URL (no auth needed)
      3. Create post via Composio proxy with image URNs

    Returns a result dict matching ComposioClient.execute_action() shape.
    """
    image_paths = _extract_image_paths(params)
    text = _extract_text(params)
    author = _extract_author(params)

    if not image_paths:
        return {"success": False, "data": None,
                "error": "No image paths found in params"}
    if not text:
        return {"success": False, "data": None,
                "error": "No post text found in params"}

    ws_client = WorkspaceClient(workspace_id)
    image_urns: List[str] = []
    failed_images: List[str] = []

    async with httpx.AsyncClient(timeout=60) as http:
        for i, img_path in enumerate(image_paths):
            label = f"image[{i}]"

            # --- 1. Download image from workspace ---
            if img_path.startswith(("http://", "https://")):
                logger.info("[LinkedInWorkaround] Downloading URL %s", img_path[:100])
                resp = await http.get(img_path)
                if resp.status_code != 200:
                    logger.warning("[LinkedInWorkaround] Failed to fetch %s: %s",
                                   img_path[:100], resp.status_code)
                    failed_images.append(img_path)
                    continue
                image_bytes = resp.content
            else:
                logger.info("[LinkedInWorkaround] Downloading workspace file: %s", img_path)
                dl = await ws_client.download_file(img_path)
                if not dl.get("success"):
                    logger.info("[LinkedInWorkaround] download_file failed, trying read_file: %s", img_path)
                    dl = await ws_client.read_file(img_path)
                if not dl.get("success"):
                    logger.warning("[LinkedInWorkaround] All download methods failed for %s: %s",
                                   img_path, dl.get("error"))
                    failed_images.append(img_path)
                    continue
                image_bytes = dl.get("content") or dl.get("data", b"")

            if not image_bytes:
                logger.warning("[LinkedInWorkaround] Empty file: %s", img_path)
                failed_images.append(img_path)
                continue

            # --- 2. Initialize upload via Composio action ---
            logger.info("[LinkedInWorkaround] Initializing upload for %s (%d bytes)", label, len(image_bytes))
            init_result = composio_client.execute_action(
                "LINKEDIN_INITIALIZE_IMAGE_UPLOAD",
                {"owner": author},
                entity_id,
            )
            logger.info("[LinkedInWorkaround] initializeUpload result: %s", str(init_result)[:500])

            if not init_result.get("success"):
                logger.error("[LinkedInWorkaround] initializeUpload failed: %s", init_result.get("error"))
                failed_images.append(img_path)
                continue

            # Extract uploadUrl and image URN from response
            init_data = init_result.get("data", {})
            # Composio wraps responses differently — dig for the values
            upload_url = None
            image_urn = None
            if isinstance(init_data, dict):
                # Try nested paths
                value = init_data.get("value", init_data)
                if isinstance(value, dict):
                    upload_url = value.get("uploadUrl") or value.get("upload_url")
                    image_urn = value.get("image")
                # Try flat
                if not upload_url:
                    upload_url = init_data.get("uploadUrl") or init_data.get("upload_url")
                if not image_urn:
                    image_urn = init_data.get("image")
                # Try response_data wrapper
                rd = init_data.get("response_data", {})
                if isinstance(rd, dict) and not upload_url:
                    value = rd.get("value", rd)
                    upload_url = value.get("uploadUrl") or value.get("upload_url")
                    image_urn = image_urn or value.get("image")

            if not upload_url or not image_urn:
                logger.error("[LinkedInWorkaround] Missing uploadUrl/image in init response: %s",
                             str(init_data)[:500])
                failed_images.append(img_path)
                continue

            # --- 3. PUT binary to pre-signed upload URL (no auth needed) ---
            logger.info("[LinkedInWorkaround] Uploading %d bytes to LinkedIn for %s", len(image_bytes), label)
            put_resp = await http.put(upload_url, content=image_bytes)
            if put_resp.status_code not in (200, 201):
                logger.error("[LinkedInWorkaround] PUT upload failed: %s %s",
                             put_resp.status_code, put_resp.text[:300])
                failed_images.append(img_path)
                continue

            logger.info("[LinkedInWorkaround] Uploaded %s -> %s", label, image_urn)
            image_urns.append(image_urn)

        if not image_urns:
            return {"success": False, "data": None,
                    "error": f"All image uploads failed ({len(failed_images)} failures)"}

        # --- 4. Create the post with image URNs ---
        if len(image_urns) == 1:
            content_block = {
                "media": {
                    "id": image_urns[0],
                    "altText": params.get("alt_text", ""),
                }
            }
        else:
            content_block = {
                "multiImage": {
                    "images": [
                        {"id": urn, "altText": params.get("alt_text", "")}
                        for urn in image_urns
                    ]
                }
            }

        post_params = {
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

        logger.info("[LinkedInWorkaround] Creating post with %d images", len(image_urns))
        post_result = composio_client.execute_action(
            "LINKEDIN_CREATE_LINKED_IN_POST",
            post_params,
            entity_id,
        )
        logger.info("[LinkedInWorkaround] createPost result: %s", str(post_result)[:500])

        if post_result.get("success"):
            post_data = post_result.get("data", {})
            post_id = ""
            if isinstance(post_data, dict):
                post_id = post_data.get("id", post_data.get("post_id", ""))
            logger.info("[LinkedInWorkaround] Post created: %s", post_id)
            return {
                "success": True,
                "data": {
                    "successful": True,
                    "post_id": post_id,
                    "images_uploaded": len(image_urns),
                    "images_failed": len(failed_images),
                    "image_urns": image_urns,
                    "workaround": "composio_execute_action",
                },
                "error": None,
            }
        else:
            logger.error("[LinkedInWorkaround] Create post failed: %s", post_result.get("error"))
            return {
                "success": False,
                "data": post_result.get("data"),
                "error": f"LinkedIn create post failed: {post_result.get('error')}",
            }
