"""
Cloud File Downloader (PRD-42)
==============================

Downloads files from cloud storage providers via the Composio API.

Strategy per provider:
- **Dropbox, OneDrive, Box**: Composio v3 REST API returns full content.
- **Google Drive**: Composio v3 API truncates inline content to ~500 bytes.
  Fallback: SDK (which saves full file to disk on the container).
"""

import base64
import logging
import os
import tempfile
from typing import Any, Dict, Optional
from uuid import UUID

import httpx
from sqlalchemy.orm import Session

from config import config
from core.composio.tool_executor import ComposioToolExecutor

logger = logging.getLogger(__name__)

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

# Known URL keys (checked BEFORE content keys — Composio hosts full
# file at s3url but truncates inline content)
_URL_KEYS = [
    "s3url", "s3Url",                      # Composio R2 presigned URL (full content)
    "downloadUrl", "download_url", "url",
    "webContentLink", "web_content_link",
    "temporary_link", "link",
]

# Minimum expected file size for text documents (below = likely truncated)
_MIN_EXPECTED_SIZE = 2048


class CloudFileDownloader:
    """
    Downloads files from cloud storage via Composio.

    Uses v3 REST API as primary, with SDK and long-running operation
    fallbacks for Google Drive truncation.
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

        # ---- Layer 1: Composio v3 REST API ----
        data = await self._execute_via_rest_api(action, app_upper, params, workspace_id)
        binary = self._extract_binary(data, label="v3 REST")

        # ---- Layer 2: SDK fallback (Google Drive only) ----
        # Composio v3 API truncates Google Drive inline content to ~500 bytes.
        # The SDK saves the full file to disk on the container.
        if app_upper == "GOOGLEDRIVE" and (binary is None or len(binary) < _MIN_EXPECTED_SIZE):
            truncated_size = len(binary) if binary else 0
            logger.warning(
                f"v3 REST returned {truncated_size} bytes for "
                f"{external_file_id} — likely truncated. Trying SDK..."
            )
            try:
                sdk_binary = await self._download_via_sdk(
                    action, app_upper, external_file_id, workspace_id
                )
                if sdk_binary and len(sdk_binary) > truncated_size:
                    logger.info(
                        f"SDK download: {len(sdk_binary):,} bytes "
                        f"(vs {truncated_size} from REST)"
                    )
                    binary = sdk_binary
            except Exception as e:
                logger.warning(f"SDK fallback failed: {e}", exc_info=True)

        if binary is None or len(binary) == 0:
            raise RuntimeError(
                f"All download methods failed for {external_file_id}. "
                f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
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
    # Helpers
    # ------------------------------------------------------------------

    def _get_api_key(self) -> str:
        """Get Composio API key from config."""
        from config import config
        api_key = config.COMPOSIO_API_KEY
        if not api_key:
            raise RuntimeError("COMPOSIO_API_KEY not set in config")
        return api_key

    def _get_entity_id(self, workspace_id: UUID) -> str:
        """Get Composio entity ID for a workspace."""
        entity = self.executor.get_entity_for_workspace(workspace_id)
        entity_id = entity.get("composio_entity_id")
        if not entity_id:
            raise RuntimeError(f"No Composio entity for workspace {workspace_id}")
        return entity_id

    def _extract_binary(self, data: Dict[str, Any], label: str = "") -> Optional[bytes]:
        """Extract file content from API response and convert to bytes."""
        if isinstance(data, dict):
            self._log_response_keys(data)
        content = self._extract_content(data)
        if content is None:
            logger.warning(f"[{label}] No content found in response")
            return None
        binary = self._to_bytes(content)
        logger.info(f"[{label}] Extracted {len(binary):,} bytes")
        return binary

    @staticmethod
    def _log_response_keys(data: Dict[str, Any]) -> None:
        """Log response structure for debugging."""
        logger.info(f"Response keys: {list(data.keys())}")
        for k in _URL_KEYS:
            if k in data:
                logger.info(f"  URL key '{k}': {str(data[k])[:120]}...")
        for k in _CONTENT_KEYS:
            if k in data:
                val = data[k]
                size = len(val) if isinstance(val, (str, bytes)) else "N/A"
                logger.info(f"  Content key '{k}': size={size}")

    # ------------------------------------------------------------------
    # Layer 1: Composio v3 REST API
    # ------------------------------------------------------------------

    async def _execute_via_rest_api(
        self,
        action: str,
        app_name: str,
        params: dict,
        workspace_id: UUID,
    ) -> Dict[str, Any]:
        """
        Call Composio REST API directly.

        Endpoint: POST {COMPOSIO_API_BASE_URL}/tools/execute/{action}
        Defaults to v3.1 (latest toolkit version served automatically) via the
        canonical config var. Uses entity_id (snake_case) — v3+ convention.
        """
        api_key = self._get_api_key()
        entity_id = self._get_entity_id(workspace_id)

        url = f"{config.COMPOSIO_API_BASE_URL}/tools/execute/{action}"
        logger.info(f"Calling Composio: {url}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "entity_id": entity_id,
                    "arguments": params,
                },
            )

        if response.status_code != 200:
            logger.error(
                f"Composio v3 API error {response.status_code}: "
                f"{response.text[:500]}"
            )
            raise RuntimeError(
                f"Composio API error {response.status_code}: {response.text[:500]}"
            )

        result = response.json()
        logger.info(f"Composio v3 full response keys: {list(result.keys())}")

        # Check for API-level failure
        if not result.get("successful", result.get("success", True)):
            error = result.get("error") or result.get("message") or "Unknown error"
            raise RuntimeError(f"Composio action {action} failed: {error}")

        # Extract data dict, merging URL keys from metadata/top-level
        data = result.get("data", result)
        metadata = result.get("metadata", {})

        if metadata and isinstance(metadata, dict):
            logger.info(f"Composio metadata keys: {list(metadata.keys())}")

        if isinstance(data, dict):
            # Merge URL keys from metadata and top-level into data
            for source in (metadata if isinstance(metadata, dict) else {}, result):
                for key in _URL_KEYS:
                    if key in source and key not in data:
                        data[key] = source[key]

        return data

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    @classmethod
    def _extract_content(cls, data: Dict[str, Any]) -> Optional[Any]:
        """Extract file content from Composio response — provider-agnostic.

        Priority:
        1. URL keys (s3url etc.) — Composio hosts full file on R2/S3.
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
        _skip = {
            "successful", "success", "error", "message", "metadata",
            "file_name", "name", "id", "rev", "path_display",
            "path_lower", "client_modified", "server_modified",
            "logId", "successfull",
        }
        for key, val in data.items():
            if key in _skip:
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
    # Layer 2: SDK-based download (Google Drive fallback)
    # ------------------------------------------------------------------

    async def _download_via_sdk(
        self,
        action: str,
        app_name: str,
        file_id: str,
        workspace_id: UUID,
    ) -> Optional[bytes]:
        """
        Download file via Composio Python SDK.

        The SDK (composio.tools.execute) may handle binary responses
        differently — saving to disk, returning s3url, etc.
        """
        from core.composio.client import get_composio_client

        client = get_composio_client()
        if not client or not client.composio:
            raise RuntimeError("Composio SDK client not available")

        entity_id = self._get_entity_id(workspace_id)
        params = self._build_params(app_name, file_id)

        result = client.execute_action(
            action=action,
            params=params,
            entity_id=entity_id,
        )

        logger.info(
            f"SDK response: success={result.get('success')}, "
            f"data type={type(result.get('data')).__name__}"
        )

        sdk_data = result.get("data", {})
        if isinstance(sdk_data, dict):
            logger.info(f"SDK data keys: {list(sdk_data.keys())}")

            # Check for download URLs at top level
            for key in _URL_KEYS:
                val = sdk_data.get(key)
                if val and isinstance(val, str) and val.startswith("http"):
                    logger.info(f"SDK URL in '{key}', downloading...")
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

            # Check for file path on disk
            for key in ("file_path", "path", "local_path", "file"):
                path = sdk_data.get(key) or (
                    nested.get(key) if isinstance(nested, dict) else None
                )
                if path and isinstance(path, str) and os.path.isfile(path):
                    logger.info(f"SDK saved file to disk: {path}")
                    with open(path, "rb") as f:
                        return f.read()

        # Try extracting inline content — check nested data FIRST
        # (SDK wraps: {data: {data: {downloaded_file_content: ...}}})
        if isinstance(sdk_data, dict):
            nested = sdk_data.get("data", {})
            if isinstance(nested, dict):
                content = self._extract_content(nested)
                if content is not None:
                    return self._to_bytes(content)

        # Then try outer level
        content = self._extract_content(sdk_data)
        if content is not None:
            return self._to_bytes(content)

        logger.warning(
            f"SDK response had no extractable content. "
            f"Keys: {list(sdk_data.keys()) if isinstance(sdk_data, dict) else 'N/A'}"
        )
        return None

    # ------------------------------------------------------------------
    # Build params
    # ------------------------------------------------------------------

    @staticmethod
    def _build_params(app_name: str, external_file_id: str) -> dict:
        """Build Composio action params based on cloud provider."""
        if app_name == "GOOGLEDRIVE":
            return {"fileId": external_file_id}
        if app_name in ("DROPBOX", "ONEDRIVE"):
            return {"path": external_file_id}
        return {"id": external_file_id}
