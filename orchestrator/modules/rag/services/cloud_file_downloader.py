"""
Cloud File Downloader (PRD-42)
==============================

Downloads files from cloud storage providers using the existing
ComposioToolExecutor. Thin wrapper that maps app names to Composio
download actions and saves results to temp files.

Provider-agnostic: instead of hardcoding per-provider response keys,
we search the Composio response for anything that looks like file content
(binary data, base64, URL, or file path). Works for Google Drive,
Dropbox, OneDrive, Box, and any future provider without code changes.
"""

import base64
import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.composio.tool_executor import ComposioToolExecutor

logger = logging.getLogger(__name__)

# Composio action names for downloading files per cloud provider
_DOWNLOAD_ACTIONS = {
    "GOOGLEDRIVE": "GOOGLEDRIVE_DOWNLOAD_FILE",
    "DROPBOX": "DROPBOX_READ_FILE",
    "ONEDRIVE": "ONEDRIVE_DOWNLOAD_FILE",
    "BOX": "BOX_DOWNLOAD_FILE",
}

# Keys that likely contain file content (checked in priority order)
_CONTENT_KEYS = [
    "file_content_bytes",       # Dropbox
    "downloaded_file_content",  # Google Drive
    "content",                  # Generic
    "file_content",             # OneDrive
    "body",                     # Some APIs
    "raw",                      # Some APIs
]

# Keys that likely contain a download URL
_URL_KEYS = [
    "downloadUrl", "download_url", "url",
    "webContentLink", "web_content_link",
    "temporary_link", "link",
]


class CloudFileDownloader:
    """
    Downloads files from cloud storage via the existing ComposioToolExecutor.

    Provider-agnostic content extraction: scans the Composio response for
    file content or download URLs regardless of provider-specific key names.
    """

    def __init__(self, db: Session):
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

        result = await self.executor.execute(
            action=action,
            params=params,
            agent_id=0,
            workspace_id=workspace_id,
            app_name=app_upper,
            skip_validation=True,
        )

        if not result.get("success"):
            error = result.get("error", "Unknown download error")
            raise RuntimeError(
                f"Failed to download {external_file_id} from {app_upper}: {error}"
            )

        # Unwrap nested Composio response: {data: {data: {...actual content...}}}
        data = self._unwrap_response(result)

        logger.info(
            f"Composio {app_upper} response keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
        )

        # Extract file content — provider-agnostic
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
    # Provider-agnostic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_response(result: Dict[str, Any]) -> Dict[str, Any]:
        """Unwrap nested Composio response to get the actual data dict."""
        outer = result.get("data", {})
        if not isinstance(outer, dict):
            return outer

        # Composio often nests: {data: {data: {actual_content}}}
        inner = outer.get("data")
        if isinstance(inner, dict):
            return inner
        return outer

    @classmethod
    def _extract_content(cls, data: Dict[str, Any]) -> Optional[Any]:
        """
        Extract file content from a Composio response — provider-agnostic.

        Strategy:
        1. Check known content keys
        2. Check known URL keys (download from URL if found)
        3. Deep-search: scan all values for anything that looks like file data
        """
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

        # 3. Deep-search: look for any string value that's large enough to be
        #    file content (>100 chars) or looks like base64/binary, or any
        #    bytes value. Skip metadata-like short strings.
        for key, val in data.items():
            if key in ("successful", "success", "error", "message", "metadata",
                       "file_name", "name", "id", "rev", "path_display",
                       "path_lower", "client_modified", "server_modified"):
                continue
            if isinstance(val, bytes):
                return val
            if isinstance(val, str):
                # URL?
                if val.startswith("http") and ("download" in val.lower() or "content" in val.lower()):
                    return cls._download_from_url(val)
                # File path on disk?
                if os.path.exists(val):
                    logger.info(f"Content key '{key}' is a file path: {val}")
                    with open(val, 'rb') as f:
                        return f.read()
                # Large string = likely content or base64
                if len(val) > 200:
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
            # Check if it's a file path (Composio saves large files to disk
            # and returns the path instead of inline content)
            if os.path.isfile(content):
                logger.info(f"Content is a file path, reading from: {content}")
                with open(content, 'rb') as f:
                    return f.read()
            # Try base64 (common for binary files from APIs)
            try:
                decoded = base64.b64decode(content, validate=True)
                if len(decoded) > 0:
                    return decoded
            except Exception:
                pass
            return content.encode("utf-8")
        # Fallback: try to convert
        return str(content).encode("utf-8")

    @staticmethod
    def _build_params(app_name: str, external_file_id: str) -> dict:
        """Build Composio action params based on cloud provider."""
        if app_name == "GOOGLEDRIVE":
            return {"fileId": external_file_id}
        if app_name in ("DROPBOX", "ONEDRIVE"):
            return {"path": external_file_id}
        # BOX and others use generic "id"
        return {"id": external_file_id}
