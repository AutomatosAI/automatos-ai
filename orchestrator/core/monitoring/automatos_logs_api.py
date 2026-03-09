"""
Automatos Loki Query API — Log Search for SENTINEL

Proxies LogQL queries to Loki for agent-based log investigation.
Provides both raw LogQL pass-through and convenience parameters
that auto-build queries from workspace_id, service, level, etc.

Usage:
    from core.monitoring.automatos_logs_api import create_logs_router
    app.include_router(create_logs_router(), prefix="/api")
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Header, HTTPException, Query

logger = logging.getLogger(__name__)

LOKI_URL = os.environ.get("LOKI_QUERY_URL", "http://loki.railway.internal:3100")
ALERT_INGEST_TOKEN = os.environ.get("ALERT_INGEST_TOKEN", "")


def _build_logql(
    query: Optional[str],
    service: Optional[str],
    level: Optional[str],
    workspace_id: Optional[str],
    error_fingerprint: Optional[str],
    request_id: Optional[str],
    module: Optional[str],
) -> str:
    """Build LogQL query from convenience parameters or pass through raw query."""
    if query:
        return query

    # Build label matchers
    labels = []
    if service:
        labels.append(f'service="{service}"')
    if level:
        labels.append(f'level="{level}"')
    if module:
        labels.append(f'module="{module}"')

    label_selector = "{" + ", ".join(labels) + "}" if labels else '{service=~".+"}'

    # Build pipeline stages
    pipeline = []

    # Parse as JSON for structured log lines
    pipeline.append("json")

    if workspace_id:
        pipeline.append(f'ctx_ws="{workspace_id}"')
    if error_fingerprint:
        pipeline.append(f'err_fp="{error_fingerprint}"')
    if request_id:
        pipeline.append(f'ctx_rid="{request_id}"')

    if pipeline:
        return f"{label_selector} | " + " | ".join(pipeline)
    return label_selector


def _parse_iso_or_relative(value: Optional[str], default_delta: timedelta) -> str:
    """Parse ISO8601 timestamp or return default (now - delta) as nanoseconds."""
    if not value:
        dt = datetime.now(timezone.utc) - default_delta
        return str(int(dt.timestamp() * 1e9))
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return str(int(dt.timestamp() * 1e9))
    except (ValueError, AttributeError):
        # Try as relative duration like "1h", "30m"
        return value


def create_logs_router() -> APIRouter:
    """Create the logs query router."""
    router = APIRouter(tags=["logs"])

    @router.get("/logs/query")
    async def query_logs(
        query: Optional[str] = Query(None, description="Raw LogQL query"),
        service: Optional[str] = Query(None, description="Filter by service label"),
        level: Optional[str] = Query(None, description="Filter by log level"),
        workspace_id: Optional[str] = Query(None, description="Filter by workspace ID"),
        error_fingerprint: Optional[str] = Query(None, description="Filter by error fingerprint"),
        request_id: Optional[str] = Query(None, description="Filter by request ID"),
        module: Optional[str] = Query(None, description="Filter by module"),
        start: Optional[str] = Query(None, description="Start time (ISO8601 or relative)"),
        end: Optional[str] = Query(None, description="End time (ISO8601 or relative)"),
        limit: int = Query(100, ge=1, le=1000, description="Max entries to return"),
        direction: str = Query("backward", description="Sort direction: backward or forward"),
        authorization: Optional[str] = Header(None),
    ):
        """Query logs from Loki.

        Supports raw LogQL or convenience parameters that auto-build the query.
        Requires Bearer token authentication (same as alert ingest).
        """
        # Auth
        if ALERT_INGEST_TOKEN:
            expected = f"Bearer {ALERT_INGEST_TOKEN}"
            if authorization != expected:
                raise HTTPException(status_code=401, detail="Invalid token")

        # Build LogQL
        logql = _build_logql(query, service, level, workspace_id, error_fingerprint, request_id, module)

        # Time range
        start_ns = _parse_iso_or_relative(start, timedelta(hours=1))
        end_ns = _parse_iso_or_relative(end, timedelta(seconds=0))
        if not end:
            end_ns = str(int(datetime.now(timezone.utc).timestamp() * 1e9))

        # Query Loki
        params = {
            "query": logql,
            "start": start_ns,
            "end": end_ns,
            "limit": str(limit),
            "direction": direction,
        }

        import asyncio
        from urllib.request import Request, urlopen
        from urllib.error import URLError

        loki_url = f"{LOKI_URL}/loki/api/v1/query_range?{urlencode(params)}"

        try:
            req = Request(loki_url, method="GET")
            loop = asyncio.get_event_loop()
            resp_body = await loop.run_in_executor(
                None,
                lambda: urlopen(req, timeout=10).read().decode("utf-8"),
            )
            loki_response = json.loads(resp_body)
        except URLError as e:
            logger.error(f"Loki query failed: {e}")
            raise HTTPException(status_code=502, detail=f"Loki query failed: {e}")
        except Exception as e:
            logger.error(f"Loki query error: {e}")
            raise HTTPException(status_code=500, detail=f"Loki query error: {e}")

        # Parse Loki response into friendlier format
        results = []
        data = loki_response.get("data", {})
        for stream in data.get("result", []):
            labels = stream.get("stream", {})
            for ts_ns, line in stream.get("values", []):
                entry = {
                    "timestamp": datetime.fromtimestamp(
                        int(ts_ns) / 1e9, tz=timezone.utc
                    ).isoformat(),
                    "labels": labels,
                    "line": line,
                }
                # Try to parse structured JSON log line
                try:
                    parsed = json.loads(line)
                    entry["parsed"] = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
                results.append(entry)

        return {
            "status": "ok",
            "query": logql,
            "results": results,
            "stats": {
                "entries_returned": len(results),
                "loki_status": loki_response.get("status", "unknown"),
            },
        }

    @router.get("/logs/labels")
    async def list_labels(
        authorization: Optional[str] = Header(None),
    ):
        """List available Loki label names."""
        if ALERT_INGEST_TOKEN:
            expected = f"Bearer {ALERT_INGEST_TOKEN}"
            if authorization != expected:
                raise HTTPException(status_code=401, detail="Invalid token")

        import asyncio
        from urllib.request import Request, urlopen

        try:
            req = Request(f"{LOKI_URL}/loki/api/v1/labels", method="GET")
            loop = asyncio.get_event_loop()
            resp_body = await loop.run_in_executor(
                None,
                lambda: urlopen(req, timeout=5).read().decode("utf-8"),
            )
            return json.loads(resp_body)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Loki labels query failed: {e}")

    @router.get("/logs/label/{label_name}/values")
    async def label_values(
        label_name: str,
        authorization: Optional[str] = Header(None),
    ):
        """List values for a specific Loki label."""
        if ALERT_INGEST_TOKEN:
            expected = f"Bearer {ALERT_INGEST_TOKEN}"
            if authorization != expected:
                raise HTTPException(status_code=401, detail="Invalid token")

        import asyncio
        from urllib.request import Request, urlopen

        try:
            req = Request(f"{LOKI_URL}/loki/api/v1/label/{label_name}/values", method="GET")
            loop = asyncio.get_event_loop()
            resp_body = await loop.run_in_executor(
                None,
                lambda: urlopen(req, timeout=5).read().decode("utf-8"),
            )
            return json.loads(resp_body)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Loki label values query failed: {e}")

    return router
