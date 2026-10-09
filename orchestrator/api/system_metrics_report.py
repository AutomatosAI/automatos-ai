"""What GET /api/system/metrics reports: the host now, the analytics sections, and history.

Split out of ``api/system.py`` (#1100), where one 155-line handler built it all inline.
Nothing here writes: the ``system_metrics`` history is read only, and a chart with no
stored rows (or no table) shows the current reading as its one point.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import psutil
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

HOURS_BY_RANGE = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
DEFAULT_HOURS = 24
MAX_API_POINTS = 24
HISTORY_METRICS = ("cpu_usage", "memory_usage", "disk_usage")
# Migration 128a785a7681 dropped this table and no model recreates it, so on a migrated
# database it is absent: the history then falls back to the current reading.
HISTORY_TABLE = "system_metrics"

EMPTY_CONTEXT = {"tokens_saved": 0, "compression_ratio": 1.0, "total_optimizations": 0, "efficiency": 0.0}
EMPTY_LEARNING = {
    "total_memories": 0, "recent_memories": 0, "knowledge_nodes": 0,
    "active_collaborations": 0, "total_collaborations": 0,
    "knowledge_growth": 0, "memory_consolidations": 0, "avg_improvement": 0.0,
}


async def analytics_sections(db: Session) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(context_optimization, learning) from the analytics engine; zeros when it fails."""
    try:
        from core.services.analytics_engine import AnalyticsEngine

        engine = AnalyticsEngine(db)
        context = await engine._get_context_metrics()
        learning = await engine._get_learning_metrics()
    except Exception:
        logger.exception("system metrics: the analytics sections could not be read")
        return dict(EMPTY_CONTEXT), dict(EMPTY_LEARNING)
    return {
        "tokens_saved": context.get("tokensSaved", 0),
        "compression_ratio": context.get("avgCompressionRatio", 1.0),
        "total_optimizations": context.get("totalOptimizations", 0),
        "efficiency": context.get("efficiency", 0.0),
    }, {
        "total_memories": learning.get("totalMemoryItems", 0),
        "recent_memories": learning.get("recentMemoryItems", 0),
        "knowledge_nodes": learning.get("knowledgeNodes", 0),
        "active_collaborations": learning.get("activeCollaborations", 0),
        "total_collaborations": learning.get("totalCollaborations", 0),
        "knowledge_growth": learning.get("knowledgeGrowth", 0),
        "memory_consolidations": learning.get("memoryConsolidations", 0),
        "avg_improvement": learning.get("avgImprovement", 0.0),
    }


def host_snapshot(cpu_percent: List[float]) -> Dict[str, Any]:
    """The host right now. ``cpu_percent`` is the per-CPU reading the caller took."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage("/")
    disk_io = psutil.disk_io_counters()
    network = psutil.net_io_counters()
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": {"count": psutil.cpu_count(), "usage_percent": cpu_percent,
                "average_usage": sum(cpu_percent) / len(cpu_percent)},
        "memory": {"total": memory.total, "available": memory.available,
                   "used": memory.used, "percent": memory.percent},
        "swap": {"total": swap.total, "used": swap.used, "percent": swap.percent},
        "disk": {
            "total": disk.total, "used": disk.used, "free": disk.free,
            "percent": disk.percent, "usage_percent": disk.percent,
            "read_bytes": disk_io.read_bytes if disk_io else 0,
            "write_bytes": disk_io.write_bytes if disk_io else 0,
        },
        "network": {"bytes_sent": network.bytes_sent, "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent, "packets_recv": network.packets_recv},
    }


def _series(db: Session, metric: str, cutoff: datetime, current: float) -> List[Dict[str, Any]]:
    fallback = [{"time": datetime.utcnow().isoformat(), "value": round(current, 2)}]
    if not inspect(db.get_bind()).has_table(HISTORY_TABLE):
        return fallback
    rows = db.execute(
        text("SELECT recorded_at, metric_value FROM system_metrics "
             "WHERE metric_name = :name AND recorded_at >= :cutoff ORDER BY recorded_at ASC"),
        {"name": metric, "cutoff": cutoff},
    ).fetchall()
    return [{"time": row[0].isoformat(), "value": round(row[1], 2)} for row in rows] or fallback


def _api_calls(hours: int, points: int) -> Dict[str, Any]:
    """The request middleware's totals, spread evenly over the window (it keeps no timestamps)."""
    try:
        import main

        stats = list(main.api_call_stats.values())
        total = sum(s["call_count"] for s in stats)
        avg_time = sum(s["avg_time"] for s in stats) / len(stats) if stats else 0
    except Exception:
        logger.exception("system metrics: the API call stats could not be read")
        return {"total": 0, "avg_time": 0, "calls": [], "response_time": []}
    now = datetime.utcnow()
    times = [(now - timedelta(hours=hours - (i * hours / points))).isoformat() for i in range(points)]
    per_point = total / points if total > 0 and points else 0
    return {
        "total": total, "avg_time": avg_time,
        "calls": [{"time": t, "value": round(per_point, 0)} for t in times],
        "response_time": [{"time": t, "value": round(avg_time, 2)} for t in times],
    }


def _average(series: List[Dict[str, Any]]) -> float:
    return round(sum(point["value"] for point in series) / len(series), 2)


def history(db: Session, time_range: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """The time-series keys for ``time_range`` (1h, 24h, 7d, 30d; anything else is 24h)."""
    hours = HOURS_BY_RANGE.get(time_range, DEFAULT_HOURS)
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    current = {"cpu_usage": snapshot["cpu"]["average_usage"], "memory_usage": snapshot["memory"]["percent"],
               "disk_usage": snapshot["disk"]["percent"]}
    series = {metric: _series(db, metric, cutoff, current[metric]) for metric in HISTORY_METRICS}
    api = _api_calls(hours, min(len(series["cpu_usage"]), MAX_API_POINTS))
    return {
        **series,
        "api_calls": api["calls"],
        "response_time": api["response_time"],
        "aggregated": {
            "cpu_average": _average(series["cpu_usage"]),
            "memory_average": _average(series["memory_usage"]),
            "disk_average": _average(series["disk_usage"]),
            "api_calls_total": api["total"],
            "response_time_average": round(api["avg_time"], 2),
        },
    }


__all__ = ["analytics_sections", "history", "host_snapshot"]
