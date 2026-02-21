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

# Known URL keys
_URL_KEYS = [
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

        # Extract file content
        content = self._extract_content(data)

        if content is None:
            raise RuntimeError(
                f"Download returned no file content for {external_file_id}. "
                f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
            )

        # Write to temp file
        suffix = ""
        if file_name and "." in file_name:
            suffix = file_name[file_name.rfind("."):]

        binary = self._to_bytes(content)
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

        # Return the data dict (top-level "data" key from API response)
        return result.get("data", result)

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    @classmethod
    def _extract_content(cls, data: Dict[str, Any]) -> Optional[Any]:
        """Extract file content from Composio response — provider-agnostic."""
        if not isinstance(data, dict):
            return data if data else None

        # 1. Check known content keys
        for key in _CONTENT_KEYS:
            val = data.get(key)
            if val is not None and val != "":
                return val

        # 2. Check known URL keys → download
        for key in _URL_KEYS:
            url = data.get(key)
            if url and isinstance(url, str) and url.startswith("http"):
                return cls._download_from_url(url)

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

    @staticmethod
    def _build_params(app_name: str, external_file_id: str) -> dict:
        """Build Composio action params based on cloud provider."""
        if app_name == "GOOGLEDRIVE":
            return {"fileId": external_file_id}
        if app_name in ("DROPBOX", "ONEDRIVE"):
            return {"path": external_file_id}
        return {"id": external_file_id}
