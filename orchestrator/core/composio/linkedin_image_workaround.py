"""
LinkedIn Image Post — TEMPORARY WORKAROUND
===========================================
Composio's LinkedIn integration cannot upload images (May 2026).
LINKEDIN_CREATE_LINKED_IN_POST is text-only. LINKEDIN_INITIALIZE_IMAGE_UPLOAD
returns an upload URL but no action exists to PUT the binary bytes.

See Composio issues: #3094 (SDK returns 4/22 actions), #3113 (426 errors),
#3231 (version header fix — text posts only).

This module calls the LinkedIn REST API directly, using Composio only for
OAuth token resolution via ComposioClient.get_app_access_token().

REMOVAL CHECKLIST (when Composio ships a working image post action):
  1. Delete this file
  2. Remove the hook in tool_executor.py  (search: linkedin_image_workaround)
  3. Remove the hook in recipe_executor.py (search: linkedin_image_workaround)
  4. Update SKILL.md to use the native Composio action
"""

import logging
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)

LINKEDIN_API = "https://api.linkedin.com"
LINKEDIN_VERSION = "202405"
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


def _extract_image_paths(params: Dict[str, Any]) -> List[str]:
    """Pull image paths/URLs from whichever param name the agent used."""
    for key in ("media_urls", "images", "media", "media_files", "image_urls"):
        val = params.get(key)
        if isinstance(val, list) and len(val) > 0:
            return [str(v) for v in val]
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
    """Post to LinkedIn with images, bypassing Composio's broken actions.

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

    # --- 1. Get OAuth token from Composio connection ---
    token = composio_client.get_app_access_token(entity_id, "LINKEDIN")
    if not token:
        return {"success": False, "data": None,
                "error": "No LinkedIn OAuth token found. Is LinkedIn connected?"}

    headers = {
        "Authorization": f"Bearer {token}",
        "LinkedIn-Version": LINKEDIN_VERSION,
        "X-Restli-Protocol-Version": "2.0.0",
    }

    ws_client = WorkspaceClient(workspace_id)
    image_urns: List[str] = []
    failed_images: List[str] = []

    async with httpx.AsyncClient(timeout=60) as http:
        for i, img_path in enumerate(image_paths):
            label = f"image[{i}]"

            # --- 2a. Download image from workspace ---
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
                logger.info("[LinkedInWorkaround] Downloading workspace file %s", img_path)
                dl = await ws_client.download_file(img_path)
                if not dl.get("success"):
                    logger.warning("[LinkedInWorkaround] Workspace download failed for %s: %s",
                                   img_path, dl.get("error"))
                    failed_images.append(img_path)
                    continue
                image_bytes = dl["content"]

            if not image_bytes:
                logger.warning("[LinkedInWorkaround] Empty file: %s", img_path)
                failed_images.append(img_path)
                continue

            # --- 2b. Initialize upload with LinkedIn ---
            init_body = {
                "initializeUploadRequest": {
                    "owner": author,
                }
            }
            logger.info("[LinkedInWorkaround] Initializing upload for %s (%d bytes)",
                        label, len(image_bytes))
            init_resp = await http.post(
                f"{LINKEDIN_API}/rest/images?action=initializeUpload",
                headers={**headers, "Content-Type": "application/json"},
                json=init_body,
            )
            if init_resp.status_code not in (200, 201):
                logger.error("[LinkedInWorkaround] initializeUpload failed: %s %s",
                             init_resp.status_code, init_resp.text[:300])
                failed_images.append(img_path)
                continue

            init_data = init_resp.json().get("value", init_resp.json())
            upload_url = init_data.get("uploadUrl")
            image_urn = init_data.get("image")

            if not upload_url or not image_urn:
                logger.error("[LinkedInWorkaround] Missing uploadUrl/image in response: %s",
                             str(init_data)[:300])
                failed_images.append(img_path)
                continue

            # --- 2c. PUT binary image bytes to LinkedIn's upload URL ---
            logger.info("[LinkedInWorkaround] Uploading %d bytes to LinkedIn for %s",
                        len(image_bytes), label)
            put_resp = await http.put(
                upload_url,
                headers={"Authorization": f"Bearer {token}"},
                content=image_bytes,
            )
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

        # --- 3. Create the post with image URNs ---
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

        post_body = {
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
        post_resp = await http.post(
            f"{LINKEDIN_API}/rest/posts",
            headers={**headers, "Content-Type": "application/json"},
            json=post_body,
        )

        if post_resp.status_code in (200, 201):
            post_id = post_resp.headers.get("x-restli-id", "")
            logger.info("[LinkedInWorkaround] Post created: %s", post_id)
            return {
                "success": True,
                "data": {
                    "successful": True,
                    "post_id": post_id,
                    "post_urn": post_id,
                    "images_uploaded": len(image_urns),
                    "images_failed": len(failed_images),
                    "image_urns": image_urns,
                    "workaround": "direct_linkedin_api",
                },
                "error": None,
            }
        else:
            logger.error("[LinkedInWorkaround] Create post failed: %s %s",
                         post_resp.status_code, post_resp.text[:500])
            return {
                "success": False,
                "data": {"status_code": post_resp.status_code,
                         "response": post_resp.text[:500]},
                "error": f"LinkedIn create post failed: {post_resp.status_code} — {post_resp.text[:200]}",
            }
