
"""
Analytics API Endpoints - Real Data for Dashboard
Exposes real analytics data instead of mock data for the frontend dashboard.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from backend_missing.analytics.analytics_service import analytics_service


router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/performance")
async def get_performance_metrics(
    days_back: int = Query(30, description="Number of days to analyze", ge=1, le=365)
) -> Dict[str, Any]:
    """
    Get real performance metrics from telemetry data.
    
    Returns:
    - Agent success rate (%)
    - Average task completion time 
    - System load trend (24h)
    - Error rate by agent type
    - Queue depth for pending tasks
    - Resource utilization efficiency score
    """
    try:
        metrics = analytics_service.get_performance_metrics(days_back)
        return {
            "status": "success",
            "data": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.get("/cost")
async def get_cost_analytics(
    days_back: int = Query(30, description="Number of days to analyze", ge=1, le=365)
) -> Dict[str, Any]:
    """
    Get real cost analytics from usage data.
    
    Returns:
    - Cost per successful execution
    - Total cost breakdown by task type
    - Daily cost trends
    - Cost optimization opportunities
    """
    try:
        cost_data = analytics_service.get_cost_analytics(days_back)
        return {
            "status": "success",
            "data": cost_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cost analytics: {str(e)}")


@router.get("/usage")
async def get_usage_analytics(
    days_back: int = Query(30, description="Number of days to analyze", ge=1, le=365)
) -> Dict[str, Any]:
    """
    Get usage analytics from telemetry data.
    
    Returns:
    - Peak usage hours identification
    - Resource bottleneck detection  
    - Task type distribution
    - Agent utilization patterns
    """
    try:
        usage_data = analytics_service.get_usage_analytics(days_back)
        return {
            "status": "success",
            "data": usage_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get usage analytics: {str(e)}")


@router.get("/agents/ranking")
async def get_agent_performance_ranking(
    days_back: int = Query(30, description="Number of days to analyze", ge=1, le=365),
    limit: int = Query(50, description="Maximum number of agents to return", ge=1, le=200)
) -> Dict[str, Any]:
    """
    Get agent performance rankings.
    
    Returns:
    - Agent performance ranking by efficiency
    - Success rates, duration, cost per agent
    - Top and bottom performers
    """
    try:
        rankings = analytics_service.get_agent_rankings(days_back)
        return {
            "status": "success",
            "data": {
                "rankings": rankings[:limit],
                "total_agents": len(rankings),
                "top_performers": rankings[:5],
                "needs_improvement": rankings[-5:] if len(rankings) > 5 else []
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent rankings: {str(e)}")


@router.get("/sla-compliance")
async def get_sla_compliance(
    days_back: int = Query(30, description="Number of days to analyze", ge=1, le=365)
) -> Dict[str, Any]:
    """
    Get SLA compliance tracking.
    
    Returns:
    - Overall SLA compliance rate
    - Compliance by task type
    - SLA violation analysis
    """
    try:
        sla_data = analytics_service.get_sla_compliance(days_back)
        return {
            "status": "success",
            "data": sla_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get SLA compliance: {str(e)}")


@router.get("/overview")
async def get_analytics_overview(
    days_back: int = Query(7, description="Number of days to analyze", ge=1, le=90)
) -> Dict[str, Any]:
    """
    Get comprehensive analytics overview for dashboard.
    Combines key metrics from all analytics endpoints.
    """
    try:
        # Get data from all analytics services
        performance = analytics_service.get_performance_metrics(days_back)
        cost = analytics_service.get_cost_analytics(days_back)
        usage = analytics_service.get_usage_analytics(days_back)
        rankings = analytics_service.get_agent_rankings(days_back)
        sla = analytics_service.get_sla_compliance(days_back)
        
        # Combine into overview
        overview = {
            # Key performance indicators
            "kpis": {
                "success_rate": performance["success_rate"],
                "avg_task_time_seconds": performance["avg_task_completion_time_seconds"],
                "efficiency_score": performance["efficiency_score"],
                "total_cost": cost["total_cost"],
                "cost_per_success": cost["cost_per_successful_execution"],
                "sla_compliance": sla["overall_compliance_rate"],
                "active_agents": performance["active_agents"],
                "queue_depth": performance["queue_depth"]
            },
            
            # Trends and insights
            "trends": {
                "system_load": performance["system_load_trend"][-7:],  # Last 7 days
                "daily_costs": list(cost["daily_costs"].items())[-7:],
                "peak_hours": usage["peak_usage_hours"]
            },
            
            # Top performers
            "top_agents": rankings[:3],
            
            # Alerts and issues
            "alerts": {
                "bottlenecks": usage["resource_bottlenecks"],
                "high_error_rates": [
                    {"type": k, "rate": v} 
                    for k, v in performance["error_rate_by_agent_type"].items() 
                    if v > 10
                ],
                "sla_violations": [
                    {"task_type": k, "compliance": v["compliance_rate"]}
                    for k, v in sla["sla_by_task_type"].items()
                    if v["compliance_rate"] < 90
                ]
            },
            
            "period_days": days_back,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return {
            "status": "success",
            "data": overview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics overview: {str(e)}")


@router.get("/predictive-alerts")
async def get_predictive_alerts(
    days_back: int = Query(14, description="Days of historical data to analyze", ge=7, le=90)
) -> Dict[str, Any]:
    """
    Get predictive capacity alerts based on usage trends.
    
    Returns:
    - Capacity warnings
    - Performance degradation predictions
    - Resource scaling recommendations
    """
    try:
        usage = analytics_service.get_usage_analytics(days_back)
        performance = analytics_service.get_performance_metrics(days_back)
        
        alerts = []
        
        # Check for capacity issues
        current_queue = performance["queue_depth"]
        if current_queue > 10:
            alerts.append({
                "type": "capacity",
                "severity": "high" if current_queue > 50 else "medium",
                "message": f"High queue depth: {current_queue} pending tasks",
                "recommendation": "Consider scaling up agent capacity"
            })
        
        # Check for performance degradation
        efficiency = performance["efficiency_score"]
        if efficiency < 70:
            alerts.append({
                "type": "performance",
                "severity": "high" if efficiency < 50 else "medium", 
                "message": f"Low efficiency score: {efficiency}%",
                "recommendation": "Review failed tasks and optimize workflows"
            })
        
        # Check for bottlenecks
        bottlenecks = usage["resource_bottlenecks"]
        if bottlenecks:
            for bottleneck in bottlenecks[:2]:  # Top 2 bottlenecks
                alerts.append({
                    "type": "bottleneck",
                    "severity": bottleneck["severity"],
                    "message": f"Resource bottleneck during {bottleneck['time_period']}: {bottleneck['failure_rate']}% failure rate",
                    "recommendation": "Consider load balancing or capacity adjustments during peak hours"
                })
        
        return {
            "status": "success",
            "data": {
                "alerts": alerts,
                "alert_count": len(alerts),
                "generated_at": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get predictive alerts: {str(e)}")


@router.get("/health")
async def analytics_health_check() -> Dict[str, str]:
    """Health check endpoint for analytics service."""
    return {
        "status": "healthy",
        "service": "analytics",
        "timestamp": datetime.utcnow().isoformat()
    }


# Export router for inclusion in main app
analytics_router = router
