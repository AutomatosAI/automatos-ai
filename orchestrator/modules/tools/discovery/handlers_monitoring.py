"""Monitoring handlers for PlatformActionExecutor — Loki, Prometheus, alerts, system health, service logs."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def get_logs(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch deployment logs from a Railway service."""
    from core.railway_client import RailwayClient

    client = RailwayClient()
    if not client.is_configured:
        return {
            "success": False,
            "error": "Railway API not configured. Set RAILWAY_API_TOKEN and RAILWAY_PROJECT_ID env vars.",
        }

    service_name = params.get("service", "")
    if not service_name:
        return {"success": False, "error": "service parameter is required"}

    # Special case: "list" returns available services
    if service_name.lower() == "list":
        return await list_services(db, workspace_id, params)

    lines = min(params.get("lines", 200), 1000)
    filter_text = params.get("filter")

    result = await client.fetch_service_logs(
        service_name=service_name,
        lines=lines,
        filter_text=filter_text,
    )

    if not result.get("success"):
        return result

    # Format logs for LLM consumption -- compact text format
    logs = result.get("logs", [])
    log_lines = []
    for entry in logs:
        ts = entry.get("timestamp", "")
        sev = entry.get("severity", "")
        msg = entry.get("message", "")
        prefix = f"[{sev}]" if sev else ""
        log_lines.append(f"{ts} {prefix} {msg}".strip())

    result["formatted_logs"] = "\n".join(log_lines)
    # Truncate formatted output for LLM context (keep under 8K chars)
    if len(result["formatted_logs"]) > 8000:
        result["formatted_logs"] = result["formatted_logs"][:8000] + "\n... (truncated)"
        result["truncated"] = True

    return result


async def list_services(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List all Railway services in the project."""
    from core.railway_client import RailwayClient

    client = RailwayClient()
    if not client.is_configured:
        return {
            "success": False,
            "error": "Railway API not configured. Set RAILWAY_API_TOKEN and RAILWAY_PROJECT_ID env vars.",
        }

    try:
        services = await client.list_services()
        return {
            "success": True,
            "services": services,
            "count": len(services),
        }
    except Exception as exc:
        logger.error("[PlatformExecutor] list_services failed: %s", exc, exc_info=True)
        return {"success": False, "error": f"Failed to list services: {exc}"}


async def query_loki_logs(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Query application logs from Loki (via Grafana datasource proxy, fallback to direct Loki)."""
    import httpx
    from config import config

    minutes = min(params.get("minutes", 60), 10080)
    limit = min(params.get("limit", 100), 500)
    service = params.get("service")
    level = params.get("level")
    search = params.get("search")

    # Build LogQL query — Loki rejects empty '{}' selectors
    label_parts = []
    if service:
        label_parts.append(f'service="{service}"')
    if level:
        label_parts.append(f'level="{level}"')
    if not label_parts:
        # Default to the API service when no labels specified —
        # Loki rejects empty '{}' selectors with 400
        label_parts.append('service="automatos-backend"')
    label_selector = "{" + ", ".join(label_parts) + "}"

    line_filter = ""
    if search:
        line_filter = f' |= `{search}`'

    logql = f"{label_selector}{line_filter}"

    import time as _time
    end_ns = int(_time.time() * 1e9)
    start_ns = int((_time.time() - minutes * 60) * 1e9)

    query_params = {
        "query": logql,
        "start": str(start_ns),
        "end": str(end_ns),
        "limit": str(limit),
        "direction": "backward",
    }

    data = None
    source = "unknown"

    # Try Grafana datasource proxy first (works from outside Railway network)
    grafana_url = getattr(config, "GRAFANA_URL", "") or ""
    grafana_token = getattr(config, "GRAFANA_SERVICE_ACCOUNT_TOKEN", "") or ""
    loki_ds_uid = getattr(config, "GRAFANA_LOKI_DATASOURCE_UID", "loki") or "loki"

    if grafana_url and grafana_token:
        try:
            proxy_url = f"{grafana_url.rstrip('/')}/api/datasources/proxy/uid/{loki_ds_uid}/loki/api/v1/query_range"
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    proxy_url,
                    params=query_params,
                    headers={"Authorization": f"Bearer {grafana_token}"},
                )
                resp.raise_for_status()
                data = resp.json()
                source = "grafana"
        except Exception as exc:
            logger.warning("[PlatformExecutor] Grafana Loki proxy failed, falling back to direct: %s", exc)

    # Fallback: direct Loki (internal network only)
    if data is None:
        loki_url = getattr(config, "LOKI_URL", None) or "http://loki.railway.internal:3100"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{loki_url}/loki/api/v1/query_range",
                    params=query_params,
                )
                resp.raise_for_status()
                data = resp.json()
                source = "loki-direct"
        except httpx.ConnectError:
            return {
                "success": False,
                "error": (
                    f"Cannot reach Loki at {loki_url}. "
                    "Loki is only accessible within the Railway internal network. "
                    "Set GRAFANA_URL and GRAFANA_SERVICE_ACCOUNT_TOKEN for external access."
                ),
            }
        except Exception as exc:
            logger.error("[PlatformExecutor] Loki query failed: %s", exc, exc_info=True)
            return {"success": False, "error": f"Loki query failed: {exc}"}

    # Parse results
    results = data.get("data", {}).get("result", [])
    log_lines = []
    for stream in results:
        labels = stream.get("stream", {})
        svc = labels.get("service", "unknown")
        lvl = labels.get("level", "")
        for ts_ns, msg in stream.get("values", []):
            ts_sec = int(ts_ns) / 1e9
            ts_str = datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%H:%M:%S")
            log_lines.append(f"[{ts_str}] [{svc}] [{lvl.upper()}] {msg}")

    formatted = "\n".join(log_lines[:limit])
    if len(formatted) > 8000:
        formatted = formatted[:8000] + "\n... (truncated)"

    return {
        "success": True,
        "source": source,
        "query": logql,
        "total_entries": len(log_lines),
        "time_range_minutes": minutes,
        "formatted_logs": formatted,
    }


async def query_prometheus(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Query Prometheus metrics with presets or raw PromQL."""
    import httpx
    from config import config

    prom_url = getattr(config, "PROMETHEUS_URL", None) or "http://prometheus.railway.internal:9090"
    query_input = params.get("query", "health")
    range_minutes = min(params.get("range_minutes", 15), 1440)

    # Preset queries for common health checks
    presets = {
        "health": [
            ("Service Health", "up"),
        ],
        "error_rate": [
            ("HTTP 5xx Rate (5m)", 'rate(automatos_http_requests_total{status_code=~"5.."}[5m])'),
            ("HTTP Total Rate (5m)", "rate(automatos_http_requests_total[5m])"),
        ],
        "latency": [
            ("p95 Response Time", "histogram_quantile(0.95, rate(automatos_http_request_duration_seconds_bucket[5m]))"),
            ("p50 Response Time", "histogram_quantile(0.50, rate(automatos_http_request_duration_seconds_bucket[5m]))"),
        ],
        "postgres": [
            ("DB Connections", "pg_stat_activity_count"),
            ("Cache Hit Ratio", "pg_stat_database_blks_hit / (pg_stat_database_blks_hit + pg_stat_database_blks_read)"),
            ("Dead Tuples", "pg_stat_user_tables_n_dead_tup"),
        ],
        "redis": [
            ("Redis Memory (MB)", "redis_memory_used_bytes / 1024 / 1024"),
            ("Redis Clients", "redis_connected_clients"),
            ("Redis Evicted Keys (5m)", "rate(redis_evicted_keys_total[5m])"),
            ("Redis Command Latency", "redis_commands_duration_seconds_total"),
        ],
        "all": [],  # filled below
    }
    # "all" = union of all presets
    for k, v in presets.items():
        if k != "all":
            presets["all"].extend(v)

    query_lower = query_input.lower().strip()
    queries_to_run = presets.get(query_lower, [(query_input, query_input)])

    try:
        results = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            for label, promql in queries_to_run:
                resp = await client.get(
                    f"{prom_url}/api/v1/query",
                    params={"query": promql},
                )
                resp.raise_for_status()
                data = resp.json()

                metric_results = data.get("data", {}).get("result", [])
                formatted_values = []
                for m in metric_results:
                    metric_labels = m.get("metric", {})
                    value = m.get("value", [None, None])
                    val = value[1] if len(value) > 1 else "N/A"

                    # Human-readable label
                    desc_parts = []
                    for k in ["job", "instance", "service", "datname", "status_code", "relname"]:
                        if k in metric_labels:
                            desc_parts.append(f"{k}={metric_labels[k]}")
                    desc = ", ".join(desc_parts) if desc_parts else "global"
                    formatted_values.append({"labels": desc, "value": val})

                results.append({
                    "metric": label,
                    "query": promql,
                    "values": formatted_values,
                })

        # Format for LLM consumption
        lines = []
        for r in results:
            lines.append(f"### {r['metric']}")
            if not r["values"]:
                lines.append("  No data")
            for v in r["values"]:
                lines.append(f"  {v['labels']}: {v['value']}")
            lines.append("")

        return {
            "success": True,
            "preset_used": query_lower if query_lower in presets else None,
            "results": results,
            "formatted": "\n".join(lines),
        }
    except httpx.ConnectError:
        return {
            "success": False,
            "error": (
                f"Cannot reach Prometheus at {prom_url}. "
                "Prometheus is only accessible within the Railway internal network."
            ),
        }
    except Exception as exc:
        logger.error("[PlatformExecutor] Prometheus query failed: %s", exc, exc_info=True)
        return {"success": False, "error": f"Prometheus query failed: {exc}"}


async def get_alerts(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get infrastructure alerts from the database."""
    from sqlalchemy import text

    status_filter = params.get("status", "all")
    severity_filter = params.get("severity")
    hours = min(params.get("hours", 24), 168)

    try:
        conditions = ["created_at > NOW() - INTERVAL ':hours hours'"]
        bind_params: Dict[str, Any] = {"hours": hours}

        if status_filter and status_filter != "all":
            conditions.append("status = :status")
            bind_params["status"] = status_filter
        if severity_filter:
            conditions.append("severity = :severity")
            bind_params["severity"] = severity_filter

        where_clause = " AND ".join(conditions)

        rows = db.execute(
            text(f"""
                SELECT alertname, severity, status, service,
                       annotations, agent_response, created_at, resolved_at
                FROM infrastructure_alerts
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT 50
            """),
            bind_params,
        ).fetchall()

        alerts = []
        for r in rows:
            annotations = r.annotations if isinstance(r.annotations, dict) else {}
            alerts.append({
                "alert": r.alertname,
                "severity": r.severity,
                "status": r.status,
                "service": r.service,
                "summary": annotations.get("summary", ""),
                "description": annotations.get("description", ""),
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
                "investigated": bool(r.agent_response),
            })

        # Summary
        firing = [a for a in alerts if a["status"] == "firing"]
        critical = [a for a in firing if a["severity"] == "critical"]

        formatted_lines = []
        if not alerts:
            formatted_lines.append(f"No alerts found in the last {hours} hours.")
        else:
            if critical:
                formatted_lines.append(f"🔴 {len(critical)} CRITICAL alert(s) firing!")
            if firing:
                formatted_lines.append(f"⚠️ {len(firing)} alert(s) currently firing")
            formatted_lines.append(f"Total: {len(alerts)} alert(s) in last {hours}h\n")

            for a in alerts[:20]:
                icon = "🔴" if a["severity"] == "critical" else "🟡" if a["severity"] == "warning" else "ℹ️"
                status_icon = "🔥" if a["status"] == "firing" else "✅"
                formatted_lines.append(
                    f"{icon}{status_icon} [{a['severity'].upper()}] {a['alert']} "
                    f"({a['service'] or 'unknown'}) — {a['summary']}"
                )

        return {
            "success": True,
            "total": len(alerts),
            "firing_count": len(firing),
            "critical_count": len(critical),
            "alerts": alerts,
            "formatted": "\n".join(formatted_lines),
        }
    except Exception as exc:
        # Recover the DB session so subsequent queries don't cascade-fail
        try:
            db.rollback()
        except Exception:
            pass
        # Table might not exist yet
        if "infrastructure_alerts" in str(exc) and ("does not exist" in str(exc) or "UndefinedTable" in str(exc)):
            return {
                "success": True,
                "total": 0,
                "firing_count": 0,
                "critical_count": 0,
                "alerts": [],
                "formatted": "No alerts table found -- monitoring alerts not yet configured.",
            }
        logger.error("[PlatformExecutor] get_alerts failed: %s", exc, exc_info=True)
        return {"success": False, "error": f"Alert query failed: {exc}"}


async def get_system_health(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """System health check -- database, Redis, Mem0, RAG, server metrics."""
    import time as _t
    components = {}

    # 1. Database
    try:
        from sqlalchemy import select as sa_select
        db.execute(sa_select(1))
        components["database"] = {"status": "healthy"}
    except Exception as e:
        components["database"] = {"status": "unhealthy", "error": str(e)[:100]}

    # 2. Redis — functional read/write test, not just ping
    try:
        from core.redis.client import get_redis_client
        rc = get_redis_client()
        if rc:
            r = rc.get_redis()
            # Write + read a probe key to verify the data path works
            probe_val = str(int(_t.time()))
            r.set("_health_probe", probe_val, ex=30)
            readback = r.get("_health_probe")
            if readback != probe_val:
                components["redis"] = {
                    "status": "unhealthy",
                    "error": f"write/read mismatch: wrote {probe_val!r}, got {readback!r}",
                }
            else:
                info = r.info(section="memory")
                components["redis"] = {
                    "status": "healthy",
                    "used_memory_human": info.get("used_memory_human", "?"),
                    "connected_clients": r.info(section="clients").get("connected_clients", "?"),
                    "keys": r.dbsize(),
                }
        else:
            components["redis"] = {"status": "unavailable"}
    except Exception as e:
        components["redis"] = {"status": "unhealthy", "error": str(e)[:100]}

    # 3. Mem0 (long-term memory service)
    try:
        from modules.memory.integrations.mem0_client import Mem0Client
        mem0_breaker = Mem0Client._get_breaker(str(workspace_id))
        if mem0_breaker.is_open:
            elapsed = _t.monotonic() - mem0_breaker.last_failure_time
            components["mem0"] = {
                "status": "unhealthy",
                "error": "circuit breaker open",
                "failures": mem0_breaker.failures,
                "cooldown_remaining_s": max(0, int(60 - elapsed)),
            }
        else:
            # Probe with a lightweight search (empty query, limit 1)
            from modules.memory.unified_memory_service import get_unified_memory_service
            svc = get_unified_memory_service()
            probe = await svc.search_long_term(
                workspace_id=str(workspace_id), query="health_probe", limit=1
            )
            components["mem0"] = {
                "status": "healthy",
                "circuit_breaker": "closed",
                "failures": mem0_breaker.failures,
            }
    except Exception as e:
        components["mem0"] = {"status": "unhealthy", "error": str(e)[:100]}

    # 4. RAG pipeline (renumbered)
    try:
        from core.models import Document
        doc_count = (
            db.query(func.count(Document.id))
            .filter(Document.workspace_id == workspace_id)
            .scalar()
        ) or 0
        completed = (
            db.query(func.count(Document.id))
            .filter(
                Document.workspace_id == workspace_id,
                Document.status == "completed",
            )
            .scalar()
        ) or 0
        components["rag"] = {
            "status": "healthy",
            "total_documents": doc_count,
            "processed": completed,
        }
    except Exception as e:
        components["rag"] = {"status": "unhealthy", "error": str(e)[:100]}

    # 5. Server metrics (psutil)
    try:
        import psutil
        components["server"] = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }
    except ImportError:
        components["server"] = {"status": "psutil not installed"}
    except Exception as e:
        components["server"] = {"status": "error", "error": str(e)[:100]}

    # Overall status
    unhealthy = [k for k, v in components.items() if v.get("status") == "unhealthy"]
    overall = "unhealthy" if unhealthy else "healthy"

    return {
        "success": True,
        "overall_status": overall,
        "components": components,
        "unhealthy": unhealthy or None,
    }
