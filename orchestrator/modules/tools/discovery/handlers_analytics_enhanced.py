"""Enhanced analytics handlers — dashboard metrics & performance analytics for agents."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import func, and_, extract, cast, Date
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def get_success_rate(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Agent success rate with 7-day trend."""
    from core.models import WorkflowExecution

    try:
        total = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
        ).scalar() or 0

        successful = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status == "completed",
        ).scalar() or 0

        success_rate = round((successful / total * 100) if total > 0 else 0, 1)

        week_ago = datetime.now() - timedelta(days=7)
        week_total = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.started_at >= week_ago,
        ).scalar() or 0

        week_successful = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status == "completed",
            WorkflowExecution.started_at >= week_ago,
        ).scalar() or 0

        week_rate = round((week_successful / week_total * 100) if week_total > 0 else 0, 1)

        return {
            "success": True,
            "value": success_rate,
            "trend": round(success_rate - week_rate, 1),
            "total_executions": total,
            "successful_executions": successful,
        }
    except Exception as e:
        logger.error("get_success_rate failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def get_completion_time(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Average task completion time in minutes with 24h comparison."""
    from core.models import WorkflowExecution

    try:
        avg_seconds = db.query(
            func.avg(
                extract("epoch", WorkflowExecution.completed_at)
                - extract("epoch", WorkflowExecution.started_at)
            )
        ).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status == "completed",
            WorkflowExecution.completed_at.isnot(None),
            WorkflowExecution.started_at.isnot(None),
        ).scalar() or 0

        avg_minutes = round(float(avg_seconds) / 60, 1)

        day_ago = datetime.now() - timedelta(hours=24)
        daily_avg_seconds = db.query(
            func.avg(
                extract("epoch", WorkflowExecution.completed_at)
                - extract("epoch", WorkflowExecution.started_at)
            )
        ).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status == "completed",
            WorkflowExecution.completed_at.isnot(None),
            WorkflowExecution.started_at.isnot(None),
            WorkflowExecution.started_at >= day_ago,
        ).scalar() or 0

        daily_avg_minutes = round(float(daily_avg_seconds) / 60, 1)

        return {
            "success": True,
            "value": avg_minutes,
            "daily_average": daily_avg_minutes,
            "improvement": round(daily_avg_minutes - avg_minutes, 1),
            "unit": "minutes",
        }
    except Exception as e:
        logger.error("get_completion_time failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def get_error_rates(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Error rate breakdown by agent type."""
    from core.models import Agent, WorkflowExecution

    try:
        days = params.get("days", 30)
        since = datetime.now() - timedelta(days=days)

        rows = (
            db.query(
                Agent.agent_type,
                func.count(WorkflowExecution.id).label("total"),
                func.count(WorkflowExecution.id).filter(
                    WorkflowExecution.status == "failed"
                ).label("failed"),
            )
            .join(Agent, WorkflowExecution.agent_id == Agent.id)
            .filter(
                WorkflowExecution.workspace_id == workspace_id,
                WorkflowExecution.started_at >= since,
            )
            .group_by(Agent.agent_type)
            .all()
        )

        error_rates = {}
        for agent_type, total, failed in rows:
            rate = round((failed / total * 100) if total > 0 else 0, 1)
            error_rates[agent_type or "unknown"] = {
                "error_rate": rate,
                "total_executions": total,
                "failed_executions": failed,
                "status": "good" if rate < 5 else "warning" if rate < 10 else "critical",
            }

        return {"success": True, "period_days": days, "error_rates": error_rates}
    except Exception as e:
        logger.error("get_error_rates failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def get_queue_depth(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Real-time queue depth — pending + running tasks."""
    from core.models import WorkflowExecution

    try:
        pending = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status.in_(["pending", "queued", "running"]),
        ).scalar() or 0

        high_priority = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status.in_(["pending", "queued"]),
            WorkflowExecution.started_at >= datetime.now() - timedelta(hours=1),
        ).scalar() or 0

        return {
            "success": True,
            "total_pending": pending,
            "high_priority": high_priority,
            "normal_priority": max(pending - high_priority, 0),
            "trend": "stable",
        }
    except Exception as e:
        logger.error("get_queue_depth failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def get_efficiency_score(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Composite efficiency score (0-100) with grade."""
    from core.models import Agent, WorkflowExecution

    try:
        # Agent utilization
        total_agents = db.query(func.count(Agent.id)).filter(
            Agent.workspace_id == workspace_id,
        ).scalar() or 0
        active_agents = db.query(func.count(Agent.id)).filter(
            Agent.workspace_id == workspace_id, Agent.status == "active",
        ).scalar() or 0
        agent_efficiency = (active_agents / total_agents * 100) if total_agents > 0 else 0

        # Workflow completion efficiency (24h)
        day_ago = datetime.now() - timedelta(hours=24)
        recent = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.started_at >= day_ago,
        ).scalar() or 0
        completed = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status == "completed",
            WorkflowExecution.started_at >= day_ago,
        ).scalar() or 0
        workflow_efficiency = (completed / recent * 100) if recent > 0 else 0

        # Composite: agent utilization 60%, workflow completion 40%
        score = round(agent_efficiency * 0.6 + workflow_efficiency * 0.4, 0)

        if score >= 90:
            grade, color = "A", "green"
        elif score >= 80:
            grade, color = "B", "blue"
        elif score >= 70:
            grade, color = "C", "yellow"
        else:
            grade, color = "D", "red"

        return {
            "success": True,
            "score": int(score),
            "grade": grade,
            "color": color,
            "breakdown": {
                "agent_efficiency": round(agent_efficiency, 1),
                "workflow_efficiency": round(workflow_efficiency, 1),
            },
        }
    except Exception as e:
        logger.error("get_efficiency_score failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def get_cost_per_execution(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Average cost per execution with daily breakdown."""
    from core.models.core import LLMUsage

    try:
        days = params.get("days", 30)
        since = datetime.now() - timedelta(days=days)

        rows = (
            db.query(
                cast(LLMUsage.created_at, Date).label("date"),
                func.sum(LLMUsage.total_cost).label("daily_cost"),
                func.count(func.distinct(LLMUsage.execution_id)).label("unique_executions"),
                func.count(LLMUsage.id).label("total_requests"),
            )
            .filter(
                LLMUsage.workspace_id == workspace_id,
                LLMUsage.created_at >= since,
            )
            .group_by(cast(LLMUsage.created_at, Date))
            .order_by(cast(LLMUsage.created_at, Date))
            .all()
        )

        cost_data = []
        for row in rows:
            daily_cost = float(row.daily_cost or 0)
            execs = max(row.unique_executions or row.total_requests, 1)
            cost_data.append({
                "date": row.date.strftime("%Y-%m-%d") if row.date else "",
                "total_executions": execs,
                "total_cost": round(daily_cost, 4),
                "cost_per_execution": round(daily_cost / execs, 6),
            })

        if cost_data:
            avg_cost = round(sum(d["cost_per_execution"] for d in cost_data) / len(cost_data), 6)
            mid = len(cost_data) // 2
            first_half = sum(d["total_cost"] for d in cost_data[:mid]) if mid > 0 else 0
            second_half = sum(d["total_cost"] for d in cost_data[mid:]) if mid > 0 else 0
            trend = "decreasing" if second_half < first_half else "increasing" if second_half > first_half else "stable"
        else:
            avg_cost = 0
            trend = "stable"

        return {
            "success": True,
            "average_cost_per_execution": avg_cost,
            "monthly_data": cost_data,
            "cost_trend": trend,
            "period_days": days,
        }
    except Exception as e:
        logger.error("get_cost_per_execution failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def get_peak_hours(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Peak usage hours — 24h activity pattern."""
    from core.models.core import LLMUsage

    try:
        days = params.get("days", 30)
        since = datetime.now() - timedelta(days=days)

        rows = (
            db.query(
                extract("hour", LLMUsage.created_at).label("hour"),
                func.count(LLMUsage.id).label("api_calls"),
                func.count(func.distinct(LLMUsage.agent_id)).label("active_agents"),
            )
            .filter(
                LLMUsage.workspace_id == workspace_id,
                LLMUsage.created_at >= since,
            )
            .group_by(extract("hour", LLMUsage.created_at))
            .all()
        )

        hour_map = {int(r.hour): {"api_calls": r.api_calls, "active_agents": r.active_agents} for r in rows}
        max_calls = max((v["api_calls"] for v in hour_map.values()), default=1) or 1

        hourly_data = []
        for hour in range(24):
            data = hour_map.get(hour, {"api_calls": 0, "active_agents": 0})
            usage_pct = round(data["api_calls"] / max_calls * 100, 1) if max_calls else 0
            category = "peak" if usage_pct > 75 else "medium" if usage_pct > 40 else "low"
            hourly_data.append({
                "hour": hour,
                "usage_percent": usage_pct,
                "api_calls": data["api_calls"],
                "active_agents": data["active_agents"],
                "category": category,
            })

        peak_hours = [h["hour"] for h in hourly_data if h["category"] == "peak"]

        return {
            "success": True,
            "hourly_pattern": hourly_data,
            "peak_hours": peak_hours,
            "peak_period": f"{min(peak_hours)}-{max(peak_hours)}h" if peak_hours else "N/A",
            "period_days": days,
        }
    except Exception as e:
        logger.error("get_peak_hours failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def get_bottlenecks(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Detect resource bottlenecks with recommendations."""
    from core.models import WorkflowExecution

    try:
        bottlenecks = []

        # Check for high failure rate (DB-based, not psutil — runs on Railway)
        day_ago = datetime.now() - timedelta(hours=24)
        recent_total = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.started_at >= day_ago,
        ).scalar() or 0

        recent_failed = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status == "failed",
            WorkflowExecution.started_at >= day_ago,
        ).scalar() or 0

        if recent_total > 0:
            failure_rate = recent_failed / recent_total * 100
            if failure_rate > 15:
                bottlenecks.append({
                    "type": "execution_failures",
                    "severity": "high" if failure_rate > 30 else "medium",
                    "current_value": round(failure_rate, 1),
                    "threshold": 15,
                    "description": f"{round(failure_rate, 1)}% failure rate in last 24h",
                    "recommendation": "Review failing agents and their error logs",
                    "impact": "Tasks not completing, wasted compute spend",
                })

        # Check for queue buildup
        queued = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status.in_(["pending", "queued"]),
        ).scalar() or 0

        if queued > 10:
            bottlenecks.append({
                "type": "queue_buildup",
                "severity": "high" if queued > 50 else "medium",
                "current_value": queued,
                "threshold": 10,
                "description": f"{queued} tasks queued/pending",
                "recommendation": "Scale worker capacity or prioritize queue",
                "impact": "Delayed task execution",
            })

        # Check for slow executions
        from sqlalchemy import extract as sa_extract

        avg_sec = db.query(
            func.avg(
                sa_extract("epoch", WorkflowExecution.completed_at)
                - sa_extract("epoch", WorkflowExecution.started_at)
            )
        ).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status == "completed",
            WorkflowExecution.completed_at.isnot(None),
            WorkflowExecution.started_at.isnot(None),
            WorkflowExecution.started_at >= day_ago,
        ).scalar() or 0

        if float(avg_sec) > 300:  # > 5 min avg
            bottlenecks.append({
                "type": "slow_execution",
                "severity": "medium",
                "current_value": round(float(avg_sec), 1),
                "threshold": 300,
                "description": f"Average execution time {round(float(avg_sec)/60, 1)} min (last 24h)",
                "recommendation": "Check agent model configs and tool response times",
                "impact": "Reduced throughput",
            })

        health = "good" if len(bottlenecks) == 0 else "warning" if len(bottlenecks) < 3 else "critical"

        return {
            "success": True,
            "bottlenecks_detected": len(bottlenecks),
            "bottlenecks": bottlenecks,
            "overall_health": health,
            "last_check": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error("get_bottlenecks failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def get_predictive_alerts(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Predictive capacity alerts based on DB trends."""
    from core.models import Agent
    from core.models.core import LLMUsage

    try:
        threshold = params.get("threshold", 60)
        alerts = []

        # Agent capacity alert
        active_agents = db.query(func.count(Agent.id)).filter(
            Agent.workspace_id == workspace_id, Agent.status == "active",
        ).scalar() or 0
        total_agents = db.query(func.count(Agent.id)).filter(
            Agent.workspace_id == workspace_id,
        ).scalar() or 0
        utilization = (active_agents / total_agents * 100) if total_agents > 0 else 0

        if utilization > 85:
            alerts.append({
                "type": "agent_capacity",
                "severity": "warning",
                "prediction": "Agent capacity nearing maximum",
                "current_usage": round(utilization, 1),
                "recommended_action": "Deploy additional agents or deactivate unused ones",
                "confidence": 78,
            })

        # API rate trend
        last_hour = datetime.now() - timedelta(hours=1)
        hourly_requests = db.query(func.count(LLMUsage.id)).filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.created_at >= last_hour,
        ).scalar() or 0

        if hourly_requests > 1000:
            alerts.append({
                "type": "api_rate_limit",
                "severity": "medium",
                "prediction": "API rate limit may be exceeded during peak hours",
                "current_usage": hourly_requests,
                "recommended_action": "Implement request throttling or stagger agent schedules",
                "confidence": 65,
            })

        # Cost trend alert — compare last 7d vs prior 7d
        seven_days = datetime.now() - timedelta(days=7)
        fourteen_days = datetime.now() - timedelta(days=14)

        recent_cost = float(
            db.query(func.sum(LLMUsage.total_cost)).filter(
                LLMUsage.workspace_id == workspace_id,
                LLMUsage.created_at >= seven_days,
            ).scalar() or 0
        )
        prior_cost = float(
            db.query(func.sum(LLMUsage.total_cost)).filter(
                LLMUsage.workspace_id == workspace_id,
                LLMUsage.created_at >= fourteen_days,
                LLMUsage.created_at < seven_days,
            ).scalar() or 0
        )

        if prior_cost > 0 and recent_cost > prior_cost * 1.5:
            pct_increase = round((recent_cost - prior_cost) / prior_cost * 100, 0)
            alerts.append({
                "type": "cost_spike",
                "severity": "warning",
                "prediction": f"Costs up {pct_increase}% week-over-week (${round(recent_cost, 2)} vs ${round(prior_cost, 2)})",
                "current_usage": round(recent_cost, 2),
                "recommended_action": "Review top-spending agents and model choices",
                "confidence": 90,
            })

        # Filter by confidence threshold
        alerts = [a for a in alerts if a["confidence"] >= threshold]

        return {
            "success": True,
            "predictive_alerts": alerts,
            "alerts_count": len(alerts),
            "forecast_period": "7 days",
            "confidence_threshold": threshold,
        }
    except Exception as e:
        logger.error("get_predictive_alerts failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def get_agent_ranking(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Agent performance ranking — composite score."""
    from core.models import Agent, WorkflowExecution

    try:
        metric = params.get("metric", "performance_score")
        limit = params.get("limit", 20)

        agents = db.query(Agent).filter(
            Agent.workspace_id == workspace_id,
            Agent.status == "active",
        ).all()

        rankings = []
        for agent in agents:
            total = db.query(func.count(WorkflowExecution.id)).filter(
                WorkflowExecution.agent_id == agent.id,
                WorkflowExecution.workspace_id == workspace_id,
            ).scalar() or 0

            successful = db.query(func.count(WorkflowExecution.id)).filter(
                WorkflowExecution.agent_id == agent.id,
                WorkflowExecution.workspace_id == workspace_id,
                WorkflowExecution.status == "completed",
            ).scalar() or 0

            success_rate = round((successful / total * 100) if total > 0 else 0, 1)

            avg_sec = db.query(
                func.avg(
                    extract("epoch", WorkflowExecution.completed_at)
                    - extract("epoch", WorkflowExecution.started_at)
                )
            ).filter(
                WorkflowExecution.agent_id == agent.id,
                WorkflowExecution.workspace_id == workspace_id,
                WorkflowExecution.status == "completed",
                WorkflowExecution.completed_at.isnot(None),
                WorkflowExecution.started_at.isnot(None),
            ).scalar() or 0

            avg_response_time = round(float(avg_sec), 2)
            speed_score = max(0, 100 - avg_response_time * 10)
            volume_score = min(100, total / 5) if total > 0 else 0
            score = round(success_rate * 0.5 + speed_score * 0.3 + volume_score * 0.2, 1)

            rankings.append({
                "agent_id": agent.id,
                "name": agent.name,
                "agent_type": agent.agent_type,
                "performance_score": score,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "tasks_completed": total,
                "rank": 0,
            })

        rankings.sort(key=lambda x: x[metric] if metric in x else x["performance_score"], reverse=True)
        for i, entry in enumerate(rankings):
            entry["rank"] = i + 1

        return {
            "success": True,
            "agent_rankings": rankings[:limit],
            "total_agents": len(agents),
            "top_performer": rankings[0] if rankings else None,
            "average_score": round(
                sum(a["performance_score"] for a in rankings) / len(rankings), 1
            ) if rankings else 0,
            "ranked_by": metric,
        }
    except Exception as e:
        logger.error("get_agent_ranking failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def get_sla_compliance(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """SLA compliance — task completion + response time vs targets."""
    from core.models import WorkflowExecution

    try:
        days = params.get("days", 30)
        since = datetime.now() - timedelta(days=days)

        total = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.started_at >= since,
        ).scalar() or 0

        completed = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status == "completed",
            WorkflowExecution.started_at >= since,
        ).scalar() or 0

        failed = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == workspace_id,
            WorkflowExecution.status == "failed",
            WorkflowExecution.started_at >= since,
        ).scalar() or 0

        completion_rate = round((completed / total * 100) if total > 0 else 0, 1)

        avg_exec_sec = float(
            db.query(
                func.avg(
                    extract("epoch", WorkflowExecution.completed_at)
                    - extract("epoch", WorkflowExecution.started_at)
                )
            ).filter(
                WorkflowExecution.workspace_id == workspace_id,
                WorkflowExecution.status == "completed",
                WorkflowExecution.completed_at.isnot(None),
                WorkflowExecution.started_at.isnot(None),
                WorkflowExecution.started_at >= since,
            ).scalar() or 0
        )

        avg_exec_sec = round(avg_exec_sec, 2)

        sla_metrics = {
            "task_completion": {
                "sla_target": 95.0,
                "current_rate": completion_rate,
                "compliance_rate": round(min(completion_rate / 95.0 * 100, 100), 1) if completion_rate > 0 else 0,
                "status": "excellent" if completion_rate >= 95 else "good" if completion_rate >= 85 else "warning",
                "failed_tasks": failed,
            },
            "response_time": {
                "sla_target": 120.0,
                "current_average": avg_exec_sec,
                "compliance_rate": round(min(120.0 / max(avg_exec_sec, 0.01) * 100, 100), 1) if avg_exec_sec > 0 else 100,
                "status": "good" if avg_exec_sec <= 120 else "warning",
            },
        }

        real_rates = [
            m["compliance_rate"]
            for m in sla_metrics.values()
            if isinstance(m.get("compliance_rate"), (int, float))
        ]
        overall = round(sum(real_rates) / len(real_rates), 1) if real_rates else 0

        if overall >= 95:
            status, color = "excellent", "green"
        elif overall >= 85:
            status, color = "good", "blue"
        elif overall >= 75:
            status, color = "warning", "yellow"
        else:
            status, color = "critical", "red"

        return {
            "success": True,
            "overall_compliance": overall,
            "overall_status": status,
            "status_color": color,
            "sla_metrics": sla_metrics,
            "reporting_period": f"Last {days} days",
        }
    except Exception as e:
        logger.error("get_sla_compliance failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}
