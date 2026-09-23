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
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

import httpx
from sqlalchemy.orm import Session

from config import config
from core.composio.deny_list import composio_action_denial_async
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

# How deep to look inside a nested response for the file. F072: Composio now
# nests the download — Drive under downloaded_file_content, Dropbox under its
# content key, the SDK one level deeper again — as {mimetype, name, s3url}.
_MAX_NEST_DEPTH = 3

# A presigned-URL signature in what would be stored as document TEXT means we
# are about to ingest a download receipt, not a file (F072).
_SIGNED_URL_MARKERS = (b"X-Amz-Signature=", b"X-Goog-Signature=", b"X-Amz-Credential=")
_RECEIPT_MAX_BYTES = 4096

# Hops a download link may redirect through; each one is checked again.
_MAX_REDIRECTS = 3


class CloudContentError(RuntimeError):
    """The provider answered with something that is not file content — a
    descriptor, a receipt, a signed link — and it must not be ingested."""


class CloudDownloadError(RuntimeError):
    """The file could not be fetched. The message is written for the
    workspace (it lands in cloud_documents.sync_error) — never an upstream
    error body."""


def _is_bare_url(value: str) -> bool:
    """A value that is nothing but one http(s) URL."""
    v = value.strip()
    return v.startswith(("http://", "https://")) and not any(c.isspace() for c in v)


def _sdk_download_dir() -> Optional[Path]:
    """Where the Composio SDK writes the files it downloads for a tool call."""
    try:
        from composio.core.models._files import LOCAL_OUTPUT_FILE_DIRECTORY
        return Path(LOCAL_OUTPUT_FILE_DIRECTORY).resolve()
    except Exception:  # noqa: BLE001 — no known directory: read no local files
        return None


def _is_sdk_download(value: str) -> bool:
    """A path to a file the Composio SDK downloaded — inside its own output
    directory. A path anywhere else is a synced file's TEXT naming a server
    file (``/proc/self/environ``), and is never opened."""
    if len(value) > 4096 or "\x00" in value:
        return False
    outdir = _sdk_download_dir()
    if outdir is None:
        return False
    try:
        path = Path(value).resolve()
        return path.is_relative_to(outdir) and path.is_file()
    except (OSError, ValueError, RuntimeError):
        return False


def _looks_like_a_receipt(binary: bytes) -> bool:
    """A short blob carrying a presigned-URL signature is a receipt, not a file."""
    if len(binary) > _RECEIPT_MAX_BYTES:
        return False
    return any(marker in binary for marker in _SIGNED_URL_MARKERS)


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
            raise CloudDownloadError(
                f"All download methods failed for {external_file_id}. "
                f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
            )
        if _looks_like_a_receipt(binary):
            # The last line of defence: whatever path produced these bytes, a
            # short blob carrying a presigned-URL signature is a download
            # RECEIPT. Ingesting it would put a live signed link into the
            # knowledge base as text (F072). Refuse, so the sync item fails
            # loudly instead of reporting "synced".
            raise CloudContentError(
                f"{app_upper} returned a download receipt for {external_file_id}, "
                "not the file — nothing was ingested"
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
        try:
            binary = self._to_bytes(content)
        except CloudContentError as e:
            logger.warning(f"[{label}] {e}")
            return None
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
        # PRD-251 S0.6 (D16): the platform deny list, before any network call.
        denial = await composio_action_denial_async(action)
        if denial:
            raise RuntimeError(denial)

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
        1. A URL key (s3url etc.) at ANY depth — Composio hosts the full file
           on R2/S3 and truncates inline content; it now nests the link inside
           a {mimetype, name, s3url} descriptor (F072).
        2. Content keys — inline content (small files), searched through
           nested descriptors. A dict is never content.
        3. Deep-search fallback — strings and bytes only, never a structure
           and never a link.
        """
        if not isinstance(data, dict):
            return data if data else None

        # 1. A download URL anywhere in the response — full file content
        url = cls._find_download_url(data)
        if url:
            return cls._download_from_url(url)

        # 2. Known inline content keys, through nested descriptors
        val = cls._find_inline_content(data)
        if val is not None:
            return val

        # 3. Deep-search: any large string value is likely content — but a
        #    link under an unknown key is not, and is never stored as text,
        #    and neither is Composio's own envelope (its execution message
        #    runs past 200 characters)
        _skip = {
            "successful", "success", "error", "message", "metadata",
            "file_name", "name", "id", "rev", "path_display",
            "path_lower", "client_modified", "server_modified",
            "logId", "successfull",
            "composio_execution_message", "display_url", "link_label",
            "mimeType", "mimetype", "kind",
        }
        for key, val in data.items():
            if key in _skip:
                continue
            if isinstance(val, bytes):
                return val
            if isinstance(val, str) and len(val) > 200 and not _is_bare_url(val):
                return val

        return None

    @staticmethod
    def _find_download_url(data: Dict[str, Any]) -> Optional[str]:
        """The download URL at any depth of a response: URL keys in priority
        order (s3url first), the shallowest occurrence of each."""
        by_depth = []
        frontier = [data]
        for _ in range(_MAX_NEST_DEPTH + 1):
            by_depth.extend(frontier)
            frontier = [v for d in frontier for v in d.values() if isinstance(v, dict)]
        for key in _URL_KEYS:
            for level in by_depth:
                url = level.get(key)
                if url and isinstance(url, str) and url.startswith("http"):
                    logger.info(f"Found download URL in response key '{key}'")
                    return url
        return None

    @classmethod
    def _find_inline_content(cls, data: Dict[str, Any], _depth: int = 0) -> Optional[Any]:
        """The first non-empty content-key value, looking inside nested
        descriptors; a descriptor dict itself is never returned."""
        for key in _CONTENT_KEYS:
            val = data.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, dict):
                if _depth < _MAX_NEST_DEPTH:
                    nested = cls._find_inline_content(val, _depth + 1)
                    if nested is not None:
                        return nested
                continue
            return val
        return None

    @staticmethod
    def _download_from_url(url: str) -> bytes:
        """Download file content from a URL — a public address only, pinned.

        The link is looked for at any depth of the provider's response, and a
        synced JSON file's own content can sit there. So every hop goes through
        the platform's one outbound check (resolve-and-pin, PRD-240): a link
        into the compose network or a metadata address is refused, not fetched.
        """
        from core.security.web_access import build_pinned_request, resolve_outbound

        logger.info(f"Downloading from URL: {url[:100]}...")
        with httpx.Client(timeout=60.0, follow_redirects=False) as client:
            for _ in range(_MAX_REDIRECTS + 1):
                target = resolve_outbound(url, enforce_switch=False)
                if not target.ok:
                    # The reason names what a host resolved to — server log
                    # only, or the sync listing becomes an internal-DNS oracle
                    logger.warning(f"Refused download link {url[:100]}: {target.reason}")
                    raise CloudContentError(
                        "Refused the download link: not an address this server may fetch"
                    )
                response = client.send(build_pinned_request(client, "GET", url, target))
                location = response.headers.get("location")
                if not (response.is_redirect and location):
                    break
                url = str(httpx.URL(url).join(location))
            else:
                raise CloudDownloadError(f"Failed to download from URL: more than {_MAX_REDIRECTS} redirects")
        if response.status_code != 200:
            raise CloudDownloadError(f"Failed to download from URL: {response.status_code}")
        logger.info(f"Downloaded {len(response.content):,} bytes from URL")
        return response.content

    @staticmethod
    def _to_bytes(content: Any) -> bytes:
        """Convert content (bytes, str, file path, base64) to raw bytes.

        Never a structure: a dict or list here is a provider's descriptor or
        receipt, and str() of one is what F072 stored as "document text" —
        mimetype, name and a presigned R2 link — for every Drive and Dropbox
        file synced.
        """
        if isinstance(content, (dict, list, tuple)):
            raise CloudContentError(
                "the provider returned a file descriptor, not the file's content "
                f"(keys: {list(content.keys())[:6] if isinstance(content, dict) else type(content).__name__})"
            )
        if isinstance(content, bytes):
            return content
        if isinstance(content, str):
            # A file the Composio SDK saved for us — only inside its own
            # download directory; any other path is text, never opened
            if _is_sdk_download(content):
                logger.info(f"Content is an SDK-downloaded file, reading from: {content}")
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
        raise CloudContentError(f"unexpected content type {type(content).__name__} — not ingested")

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

            # A download URL at any depth (the SDK double-wraps: {data: {data: …}})
            url = self._find_download_url(sdk_data)
            if url:
                return self._download_from_url(url)

            nested = sdk_data.get("data", {})

            # Check for file path on disk
            for key in ("file_path", "path", "local_path", "file"):
                path = sdk_data.get(key) or (
                    nested.get(key) if isinstance(nested, dict) else None
                )
                if path and isinstance(path, str) and _is_sdk_download(path):
                    logger.info(f"SDK saved file to disk: {path}")
                    with open(path, "rb") as f:
                        return f.read()

        # Try extracting inline content — check nested data FIRST
        # (SDK wraps: {data: {data: {downloaded_file_content: ...}}})
        if isinstance(sdk_data, dict):
            nested = sdk_data.get("data", {})
            if isinstance(nested, dict):
                binary = self._extract_binary(nested, label="SDK nested")
                if binary:
                    return binary

        # Then try outer level
        binary = self._extract_binary(sdk_data, label="SDK")
        if binary:
            return binary

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
