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
    "DROPBOX": "DROPBOX_DOWNLOAD_FILE",
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
        data = result.get("data", {})
        content = data.get("content") or data.get("file_content") or data.get("data")
        if content is None:
            raise RuntimeError(
                f"Download returned no file content for {external_file_id}"
            )

        # Determine temp file suffix from original file name
        suffix = ""
        if file_name and "." in file_name:
            suffix = file_name[file_name.rfind("."):]

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            if isinstance(content, str):
                tmp.write(content.encode("utf-8"))
            else:
                tmp.write(content)
        finally:
            tmp.close()

        logger.info(
            f"Downloaded {app_upper}/{external_file_id} → {tmp.name} "
            f"({tmp.name})"
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
