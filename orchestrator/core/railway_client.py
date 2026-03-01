"""
Railway API Client — GraphQL proxy for deployment log retrieval
================================================================

Fetches deployment logs from Railway's public GraphQL API (v2).
Used by platform_get_logs action so agents can snapshot server-side
logs during test runs or incident investigation.

Requires:
  - RAILWAY_API_TOKEN  (Railway API token with read access)
  - RAILWAY_PROJECT_ID (project UUID)
  - RAILWAY_ENVIRONMENT_ID (environment UUID, optional — defaults to production)
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from config import config

logger = logging.getLogger(__name__)

RAILWAY_GQL_URL = config.RAILWAY_GQL_URL


class RailwayClient:
    """Async client for Railway's GraphQL API."""

    def __init__(self):
        self.token = config.RAILWAY_API_TOKEN
        self.project_id = config.RAILWAY_PROJECT_ID
        self.environment_id = config.RAILWAY_ENVIRONMENT_ID

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.project_id)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def _gql(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a GraphQL query against Railway's API."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                RAILWAY_GQL_URL,
                json={"query": query, "variables": variables},
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                errors = data["errors"]
                msg = errors[0].get("message", str(errors)) if errors else "Unknown GQL error"
                raise RuntimeError(f"Railway API error: {msg}")
            return data.get("data", {})

    # ── Service discovery ─────────────────────────────────────────────

    async def list_services(self) -> List[Dict[str, str]]:
        """List all services in the project. Returns [{id, name}, ...]."""
        query = """
        query ($projectId: String!) {
            project(id: $projectId) {
                services {
                    edges {
                        node {
                            id
                            name
                        }
                    }
                }
            }
        }
        """
        data = await self._gql(query, {"projectId": self.project_id})
        edges = data.get("project", {}).get("services", {}).get("edges", [])
        return [{"id": e["node"]["id"], "name": e["node"]["name"]} for e in edges]

    async def resolve_service_id(self, service_name: str) -> Optional[str]:
        """Resolve a service name (e.g. 'automatos-api') to its Railway service ID."""
        services = await self.list_services()
        name_lower = service_name.lower()
        for svc in services:
            if svc["name"].lower() == name_lower:
                return svc["id"]
        # Fuzzy: partial match
        for svc in services:
            if name_lower in svc["name"].lower() or svc["name"].lower() in name_lower:
                return svc["id"]
        return None

    # ── Deployments ───────────────────────────────────────────────────

    async def get_latest_deployment_id(
        self,
        service_id: str,
        environment_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get the most recent deployment ID for a service."""
        env_id = environment_id or self.environment_id
        query = """
        query ($input: DeploymentListInput!) {
            deployments(input: $input, first: 1) {
                edges {
                    node {
                        id
                        status
                        createdAt
                    }
                }
            }
        }
        """
        variables: Dict[str, Any] = {
            "input": {
                "projectId": self.project_id,
                "serviceId": service_id,
            }
        }
        if env_id:
            variables["input"]["environmentId"] = env_id

        data = await self._gql(query, variables)
        edges = data.get("deployments", {}).get("edges", [])
        if edges:
            return edges[0]["node"]["id"]
        return None

    # ── Logs ──────────────────────────────────────────────────────────

    async def get_deployment_logs(
        self,
        deployment_id: str,
        limit: int = 200,
        filter_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch deploy logs for a specific deployment.

        Returns list of {timestamp, message, severity} dicts.
        """
        query = """
        query ($deploymentId: String!, $limit: Int, $filter: String) {
            deploymentLogs(deploymentId: $deploymentId, limit: $limit, filter: $filter) {
                ... on Log {
                    timestamp
                    message
                    severity
                }
            }
        }
        """
        variables: Dict[str, Any] = {
            "deploymentId": deployment_id,
            "limit": min(limit, 1000),
        }
        if filter_text:
            variables["filter"] = filter_text

        data = await self._gql(query, variables)
        logs = data.get("deploymentLogs", [])
        # Normalize — Railway may return list of dicts or nested structure
        if isinstance(logs, list):
            return [
                {
                    "timestamp": entry.get("timestamp", ""),
                    "message": entry.get("message", ""),
                    "severity": entry.get("severity", ""),
                }
                for entry in logs
                if isinstance(entry, dict)
            ]
        return []

    # ── High-level: fetch logs by service name ────────────────────────

    async def fetch_service_logs(
        self,
        service_name: str,
        lines: int = 200,
        filter_text: Optional[str] = None,
        environment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        High-level: resolve service name → latest deployment → fetch logs.

        Returns:
            {
                "success": True/False,
                "service": "automatos-api",
                "deployment_id": "...",
                "log_count": N,
                "logs": [{timestamp, message, severity}, ...],
                "truncated": bool,
            }
        """
        if not self.is_configured:
            return {
                "success": False,
                "error": "Railway API not configured. Set RAILWAY_API_TOKEN and RAILWAY_PROJECT_ID.",
            }

        try:
            # 1. Resolve service name → ID
            service_id = await self.resolve_service_id(service_name)
            if not service_id:
                available = await self.list_services()
                names = [s["name"] for s in available]
                return {
                    "success": False,
                    "error": f"Service '{service_name}' not found. Available: {names}",
                }

            # 2. Get latest deployment
            deployment_id = await self.get_latest_deployment_id(
                service_id, environment_id or self.environment_id
            )
            if not deployment_id:
                return {
                    "success": False,
                    "error": f"No deployments found for service '{service_name}'",
                }

            # 3. Fetch logs
            logs = await self.get_deployment_logs(
                deployment_id=deployment_id,
                limit=lines,
                filter_text=filter_text,
            )

            return {
                "success": True,
                "service": service_name,
                "service_id": service_id,
                "deployment_id": deployment_id,
                "log_count": len(logs),
                "logs": logs,
                "truncated": len(logs) >= lines,
            }

        except httpx.HTTPStatusError as exc:
            logger.error("Railway API HTTP error: %s", exc, exc_info=True)
            return {"success": False, "error": f"Railway API HTTP {exc.response.status_code}"}
        except Exception as exc:
            logger.error("Railway API error: %s", exc, exc_info=True)
            return {"success": False, "error": f"Railway API error: {exc}"}
