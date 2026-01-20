"""
Adapter Admin API client for tool metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from config import Config


class AdapterClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout_seconds: float = 20,
    ) -> None:
        config = Config()
        self.base_url = (base_url or config.ADAPTER_ADMIN_BASE_URL or "").rstrip("/")
        self.token = token or config.ADAPTER_ADMIN_TOKEN
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def is_configured(self) -> bool:
        return bool(self.base_url)

    async def list_tools(self) -> List[Dict[str, Any]]:
        if not self.base_url:
            raise RuntimeError("Adapter base URL is not configured")
        url = f"{self.base_url}/admin/tools"
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.json()

    async def get_tool(self, tool_id: int) -> Dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("Adapter base URL is not configured")
        url = f"{self.base_url}/admin/tools/{tool_id}"
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.json()
