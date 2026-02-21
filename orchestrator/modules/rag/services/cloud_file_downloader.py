"""
Cloud File Downloader (PRD-42)
==============================

Downloads files from cloud storage providers using the existing
ComposioToolExecutor. Thin wrapper that maps app names to Composio
download actions and saves results to temp files.
"""

import logging
import tempfile
from typing import Optional
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


class CloudFileDownloader:
    """
    Downloads files from cloud storage via the existing ComposioToolExecutor.

    Usage::

        downloader = CloudFileDownloader(db)
        tmp_path = await downloader.download_file(
            app_name="GOOGLEDRIVE",
            external_file_id="1a2b3c",
            workspace_id=workspace_uuid,
        )
        # ... process tmp_path ...
        os.unlink(tmp_path)  # caller is responsible for cleanup
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

        Args:
            app_name: Cloud provider (GOOGLEDRIVE, DROPBOX, ONEDRIVE, BOX).
            external_file_id: Provider-specific file identifier.
            workspace_id: Workspace UUID for Composio entity resolution.
            file_name: Optional original file name (used for temp suffix).

        Returns:
            Path to the temporary file. Caller must delete when done.

        Raises:
            ValueError: If app_name is unsupported.
            RuntimeError: If the Composio download action fails.
        """
        app_upper = app_name.upper()
        action = _DOWNLOAD_ACTIONS.get(app_upper)
        if not action:
            raise ValueError(
                f"Unsupported cloud provider: {app_name}. "
                f"Supported: {', '.join(_DOWNLOAD_ACTIONS.keys())}"
            )

        # Build provider-specific params
        params = self._build_params(app_upper, external_file_id)

        # Execute download via existing ToolExecutor (skip agent validation —
        # this is a system-level service call, not an agent action).
        result = await self.executor.execute(
            action=action,
            params=params,
            agent_id=0,  # system-level; no real agent
            workspace_id=workspace_id,
            app_name=app_upper,
            skip_validation=True,
        )

        if not result.get("success"):
            error = result.get("error", "Unknown download error")
            raise RuntimeError(
                f"Failed to download {external_file_id} from {app_upper}: {error}"
            )

        # Extract file content from response
        # Composio wraps response in nested "data" key: {data: {data: {downloaded_file_content: ...}}}
        outer_data = result.get("data", {})
        data = outer_data.get("data", {}) if isinstance(outer_data, dict) and "data" in outer_data else outer_data

        # DEBUG: Log the actual response structure
        logger.warning(f"DEBUG - Composio response keys: {list(data.keys())}")
        logger.warning(f"DEBUG - Full response data: {str(data)[:1000]}")

        # Check if response contains a download URL instead of content
        download_url = data.get("downloadUrl") or data.get("download_url") or data.get("url") or data.get("webContentLink")

        if download_url:
            logger.info(f"Downloading file from URL: {download_url[:100]}...")
            import requests
            response = requests.get(download_url)
            if response.status_code == 200:
                content = response.content
                logger.info(f"Downloaded {len(content)} bytes from URL")
            else:
                raise RuntimeError(f"Failed to download from URL: {response.status_code}")
        else:
            # Try multiple possible keys for file content
            content = (
                data.get("downloaded_file_content") or
                data.get("content") or
                data.get("file_content") or
                data.get("data")
            )

            if content is None:
                raise RuntimeError(
                    f"Download returned no file content or URL for {external_file_id}. "
                    f"Response structure: {list(data.keys())}"
                )

        # Determine temp file suffix from original file name
        suffix = ""
        if file_name and "." in file_name:
            suffix = file_name[file_name.rfind("."):]

        # Check if content is a file path (Composio saves large files to disk)
        import os
        if isinstance(content, str) and os.path.exists(content):
            logger.info(f"Content is a file path, reading from: {content}")
            # Read the actual file from the Composio output directory
            with open(content, 'rb') as source_file:
                file_content = source_file.read()
            # Write to our temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb')
            try:
                tmp.write(file_content)
            finally:
                tmp.close()
        else:
            # Write content directly to temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb')
            try:
                if isinstance(content, str):
                    # Check if it's base64-encoded (common for binary files from APIs)
                    import base64
                    try:
                        # Try to decode as base64
                        binary_content = base64.b64decode(content)
                        tmp.write(binary_content)
                        logger.debug(f"Decoded base64 content for {file_name}")
                    except Exception:
                        # Not base64, write as UTF-8
                        tmp.write(content.encode("utf-8"))
                else:
                    tmp.write(content)
            finally:
                tmp.close()

        # Log file size for debugging
        import os
        file_size = os.path.getsize(tmp.name)
        logger.info(
            f"Downloaded {app_upper}/{external_file_id} → {tmp.name} "
            f"(size: {file_size} bytes)"
        )
        return tmp.name

    @staticmethod
    def _build_params(app_name: str, external_file_id: str) -> dict:
        """Build Composio action params based on cloud provider."""
        if app_name == "GOOGLEDRIVE":
            return {"fileId": external_file_id}
        if app_name in ("DROPBOX", "ONEDRIVE"):
            return {"path": external_file_id}
        # BOX and others use generic "id"
        return {"id": external_file_id}
