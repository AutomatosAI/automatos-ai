"""
Cloud File Downloader (PRD-42)
==============================

Downloads files from cloud storage providers via the Composio REST API.

Bypasses the Composio SDK entirely — the SDK mangles binary responses
(saves to disk, returns paths instead of content). The REST API returns
file content inline as expected.

For Google Drive downloads that already work through the SDK/executor,
we keep that path. For all other providers, we call the REST API directly.
"""

import base64
import logging
import os
import tempfile
from typing import Any, Dict, Optional
from uuid import UUID

import httpx
from sqlalchemy.orm import Session

from core.composio.tool_executor import ComposioToolExecutor

logger = logging.getLogger(__name__)

COMPOSIO_API_BASE = "https://backend.composio.dev/api/v2"

# Composio action names for downloading files per cloud provider
_DOWNLOAD_ACTIONS = {
    "GOOGLEDRIVE": "GOOGLEDRIVE_DOWNLOAD_FILE",
    "DROPBOX": "DROPBOX_READ_FILE",
    "ONEDRIVE": "ONEDRIVE_DOWNLOAD_FILE",
    "BOX": "BOX_DOWNLOAD_FILE",
}

# Known content keys across providers (priority order)
_CONTENT_KEYS = [
    "file_content_bytes",       # Dropbox
    "downloaded_file_content",  # Google Drive
    "content",                  # Generic
    "file_content",             # OneDrive
    "body",                     # Some APIs
    "raw",                      # Some APIs
]

# Known URL keys (checked BEFORE content keys for Google Drive,
# because Composio returns full file at s3url but truncated content inline)
_URL_KEYS = [
    "s3url", "s3Url",                      # Composio R2 presigned URL (full content)
    "downloadUrl", "download_url", "url",
    "webContentLink", "web_content_link",
    "temporary_link", "link",
]


class CloudFileDownloader:
    """
    Downloads files from cloud storage.

    Uses Composio REST API directly (bypasses SDK) for reliable content
    extraction across all providers.
    """

    def __init__(self, db: Session):
        self.db = db
        self.executor = ComposioToolExecutor(db)

    async def download_file(
        self,
        app_name: str,
        external_file_id: str,
        workspace_id: UUID,
        file_name: Optional[str] = None,
    ) -> str:
        """
        Download a file from cloud storage and save to a temp file.

        Returns:
            Path to the temporary file. Caller must delete when done.
        """
        app_upper = app_name.upper()
        action = _DOWNLOAD_ACTIONS.get(app_upper)
        if not action:
            raise ValueError(
                f"Unsupported cloud provider: {app_name}. "
                f"Supported: {', '.join(_DOWNLOAD_ACTIONS.keys())}"
            )

        params = self._build_params(app_upper, external_file_id)

        # Call Composio REST API directly (bypasses SDK response mangling)
        data = await self._execute_via_rest_api(action, app_upper, params, workspace_id)

        logger.info(
            f"Composio {app_upper} response keys: "
            f"{list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
        )
        # Log URL keys and content sizes for debugging truncation
        if isinstance(data, dict):
            for k in _URL_KEYS:
                if k in data:
                    logger.info(f"  Found URL key '{k}': {str(data[k])[:120]}...")
            for k in _CONTENT_KEYS:
                if k in data:
                    val = data[k]
                    size = len(val) if isinstance(val, (str, bytes)) else "N/A"
                    logger.info(f"  Found content key '{k}': size={size}")

        # Extract file content
        content = self._extract_content(data)

        if content is None:
            raise RuntimeError(
                f"Download returned no file content for {external_file_id}. "
                f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
            )

        binary = self._to_bytes(content)

        # Google Drive REST API truncates content to ~500 bytes.
        # Detect and fall back to Composio SDK (which saves full file to disk).
        MIN_EXPECTED_SIZE = 2048  # text files should be > 2KB
        if app_upper == "GOOGLEDRIVE" and len(binary) < MIN_EXPECTED_SIZE:
            logger.warning(
                f"Composio REST API returned only {len(binary)} bytes for "
                f"{external_file_id} — likely truncated. "
                f"Falling back to Composio SDK download."
            )
            try:
                sdk_binary = await self._download_via_sdk(
                    action, app_upper, external_file_id, workspace_id
                )
                if sdk_binary and len(sdk_binary) > len(binary):
                    logger.info(
                        f"Composio SDK download: {len(sdk_binary):,} bytes "
                        f"(vs {len(binary)} from REST API)"
                    )
                    binary = sdk_binary
                else:
                    logger.warning(
                        "SDK download returned same or less content, "
                        "keeping REST API result"
                    )
            except Exception as e:
                logger.warning(
                    f"Composio SDK download failed, "
                    f"using REST API result ({len(binary)} bytes): {e}",
                    exc_info=True
                )

        # Write to temp file
        suffix = ""
        if file_name and "." in file_name:
            suffix = file_name[file_name.rfind("."):]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb')
        try:
            tmp.write(binary)
        finally:
            tmp.close()

        file_size = os.path.getsize(tmp.name)
        logger.info(
            f"Downloaded {app_upper}/{external_file_id} → {tmp.name} "
            f"({file_size:,} bytes)"
        )
        return tmp.name

    # ------------------------------------------------------------------
    # Direct REST API call (bypasses SDK)
    # ------------------------------------------------------------------

    async def _execute_via_rest_api(
        self,
        action: str,
        app_name: str,
        params: dict,
        workspace_id: UUID,
    ) -> Dict[str, Any]:
        """
        Call Composio REST API directly instead of going through the SDK.

        The SDK transforms binary responses (saves to disk, returns paths).
        The REST API returns content inline as JSON.
        """
        # Get API key
        api_key = os.getenv("COMPOSIO_API_KEY") or os.getenv("COMPOSIO_KEY")
        if not api_key:
            raise RuntimeError("COMPOSIO_API_KEY/COMPOSIO_KEY not set")

        # Get entity ID for this workspace
        entity = self.executor.get_entity_for_workspace(workspace_id)
        entity_id = entity.get("composio_entity_id")
        if not entity_id:
            raise RuntimeError(f"No Composio entity for workspace {workspace_id}")

        url = f"{COMPOSIO_API_BASE}/actions/{action}/execute"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "entityId": entity_id,
                    "appName": app_name,
                    "input": params,
                },
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Composio API error {response.status_code}: {response.text[:500]}"
            )

        result = response.json()

        # Check for API-level failure
        if not result.get("successful", result.get("success", True)):
            error = result.get("error") or result.get("message") or "Unknown error"
            raise RuntimeError(f"Composio action {action} failed: {error}")

        # Log full response structure to find s3url / download URLs
        logger.info(
            f"Composio full response keys: {list(result.keys())}"
        )
        # Check for metadata section (Composio sometimes puts s3url here)
        metadata = result.get("metadata", {})
        if metadata:
            logger.info(f"Composio metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else type(metadata).__name__}")

        # Return the data dict, merging in any metadata that has URLs
        data = result.get("data", result)
        if isinstance(data, dict) and isinstance(metadata, dict):
            # Merge s3url and other URL keys from metadata into data
            for key in _URL_KEYS:
                if key in metadata and key not in data:
                    data[key] = metadata[key]
            # Also check top-level result for s3url
            for key in _URL_KEYS:
                if key in result and key not in data:
                    data[key] = result[key]
        return data

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    @classmethod
    def _extract_content(cls, data: Dict[str, Any]) -> Optional[Any]:
        """Extract file content from Composio response — provider-agnostic.

        Priority:
        1. URL keys (s3url etc.) — Composio hosts full file on R2/S3; the
           inline ``downloaded_file_content`` is often truncated to ~500 bytes.
        2. Content keys — inline content (works for Dropbox, small files).
        3. Deep-search fallback.
        """
        if not isinstance(data, dict):
            return data if data else None

        # 1. Check URL keys FIRST — full file content lives here
        for key in _URL_KEYS:
            url = data.get(key)
            if url and isinstance(url, str) and url.startswith("http"):
                logger.info(f"Found download URL in response key '{key}'")
                return cls._download_from_url(url)

        # 2. Check known inline content keys
        for key in _CONTENT_KEYS:
            val = data.get(key)
            if val is not None and val != "":
                return val

        # 3. Deep-search: any large string value is likely content
        for key, val in data.items():
            if key in ("successful", "success", "error", "message", "metadata",
                       "file_name", "name", "id", "rev", "path_display",
                       "path_lower", "client_modified", "server_modified",
                       "logId", "successfull"):
                continue
            if isinstance(val, bytes):
                return val
            if isinstance(val, str) and len(val) > 200:
                return val

        return None

    @staticmethod
    def _download_from_url(url: str) -> bytes:
        """Download file content from a URL."""
        import requests
        logger.info(f"Downloading from URL: {url[:100]}...")
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download from URL: {response.status_code}")
        logger.info(f"Downloaded {len(response.content):,} bytes from URL")
        return response.content

    @staticmethod
    def _to_bytes(content: Any) -> bytes:
        """Convert content (bytes, str, file path, base64) to raw bytes."""
        if isinstance(content, bytes):
            return content
        if isinstance(content, str):
            # File path on disk (Composio sometimes saves files locally)
            if os.path.isfile(content):
                logger.info(f"Content is a file path, reading from: {content}")
                with open(content, 'rb') as f:
                    return f.read()
            # Try base64 (common for binary files)
            try:
                decoded = base64.b64decode(content, validate=True)
                if len(decoded) > 0:
                    return decoded
            except Exception:
                pass
            return content.encode("utf-8")
        return str(content).encode("utf-8")

    # ------------------------------------------------------------------
    # SDK-based download (handles binary files properly)
    # ------------------------------------------------------------------

    async def _download_via_sdk(
        self,
        action: str,
        app_name: str,
        file_id: str,
        workspace_id: UUID,
    ) -> bytes:
        """
        Download file via Composio Python SDK instead of REST API.

        The SDK handles binary responses differently — it may save files to
        disk and return a path, or return an s3url to the full content.
        The REST API truncates inline content to ~500 bytes.
        """
        from core.composio.client import get_composio_client

        client = get_composio_client()
        if not client or not client.composio:
            raise RuntimeError("Composio SDK client not available")

        entity = self.executor.get_entity_for_workspace(workspace_id)
        entity_id = entity.get("composio_entity_id")
        if not entity_id:
            raise RuntimeError(f"No Composio entity for workspace {workspace_id}")

        params = self._build_params(app_name, file_id)

        # Execute via SDK — this may download the file properly
        result = client.execute_action(
            action=action,
            params=params,
            entity_id=entity_id,
        )

        logger.info(
            f"Composio SDK response: success={result.get('success')}, "
            f"data type={type(result.get('data')).__name__}"
        )

        sdk_data = result.get("data", {})

        # The SDK response may contain the full data at various levels
        # Log all keys for debugging
        if isinstance(sdk_data, dict):
            logger.info(f"SDK data keys: {list(sdk_data.keys())}")
            # Check for s3url or download URLs at any level
            for key in _URL_KEYS:
                val = sdk_data.get(key)
                if val and isinstance(val, str) and val.startswith("http"):
                    logger.info(f"SDK returned URL in '{key}', downloading...")
                    return self._download_from_url(val)

            # Check nested 'data' dict (SDK sometimes double-wraps)
            nested = sdk_data.get("data", {})
            if isinstance(nested, dict):
                logger.info(f"SDK nested data keys: {list(nested.keys())}")
                for key in _URL_KEYS:
                    val = nested.get(key)
                    if val and isinstance(val, str) and val.startswith("http"):
                        logger.info(f"SDK nested URL in '{key}', downloading...")
                        return self._download_from_url(val)

            # Check for file path on disk (SDK saves files locally)
            for key in ("file_path", "path", "local_path", "file"):
                path = sdk_data.get(key) or (nested.get(key) if isinstance(nested, dict) else None)
                if path and isinstance(path, str) and os.path.isfile(path):
                    logger.info(f"SDK saved file to disk: {path}")
                    with open(path, "rb") as f:
                        return f.read()

        # Try extracting content from SDK response
        content = self._extract_content(sdk_data)
        if content is not None:
            return self._to_bytes(content)

        raise RuntimeError(
            f"SDK response had no extractable content. "
            f"Keys: {list(sdk_data.keys()) if isinstance(sdk_data, dict) else 'N/A'}"
        )

    @staticmethod
    def _build_params(app_name: str, external_file_id: str) -> dict:
        """Build Composio action params based on cloud provider."""
        if app_name == "GOOGLEDRIVE":
            return {"fileId": external_file_id}
        if app_name in ("DROPBOX", "ONEDRIVE"):
            return {"path": external_file_id}
        return {"id": external_file_id}
