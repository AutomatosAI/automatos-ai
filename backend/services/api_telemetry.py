
"""
Telemetry API Endpoints - Pattern Mining and Run Tracking
Exposes telemetry data and pattern mining capabilities for the frontend.
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from backend_missing.telemetry.pattern_miner import pattern_miner, Pattern
from backend_missing.telemetry.telemetry_emitter import telemetry
from database import get_db_session
from models.run_telemetry import RunTrace, RunEvent


router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


@router.get("/health")
async def telemetry_health_check() -> Dict[str, str]:
    """Health check endpoint for telemetry service."""
    return {
        "status": "healthy",
        "service": "telemetry",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/runs")
async def get_runs(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    success: Optional[bool] = Query(None, description="Filter by success status"),
    days_back: int = Query(7, description="Number of days to look back", ge=1, le=90),
    limit: int = Query(100, description="Maximum number of runs to return", ge=1, le=1000),
    offset: int = Query(0, description="Number of runs to skip", ge=0)
) -> Dict[str, Any]:
    """
    Get agent run traces with optional filtering.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        with get_db_session() as db:
            query = db.query(RunTrace).filter(
                RunTrace.start_time >= cutoff_date
            )
            
            if agent_id:
                query = query.filter(RunTrace.agent_id == agent_id)
            if task_type:
                query = query.filter(RunTrace.task_type == task_type)
            if success is not None:
                query = query.filter(RunTrace.success == success)
            
            total_count = query.count()
            runs = query.order_by(RunTrace.start_time.desc()).offset(offset).limit(limit).all()
            
            return {
                "status": "success",
                "data": {
                    "runs": [run.to_dict() for run in runs],
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get runs: {str(e)}")


@router.get("/runs/{run_id}")
async def get_run_details(run_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific run including all events.
    """
    try:
        with get_db_session() as db:
            run = db.query(RunTrace).filter(RunTrace.run_id == run_id).first()
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            
            events = db.query(RunEvent).filter(
                RunEvent.run_id == run_id
            ).order_by(RunEvent.timestamp).all()
            
            return {
                "status": "success",
                "data": {
                    "run": run.to_dict(),
                    "events": [event.to_dict() for event in events]
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get run details: {str(e)}")


@router.get("/patterns/mine")
async def mine_patterns(
    task_type: Optional[str] = Query(None, description="Filter patterns by task type"),
    agent_id: Optional[str] = Query(None, description="Filter patterns by agent ID"),
    days_back: int = Query(30, description="Number of days to analyze", ge=1, le=365),
    min_runs: int = Query(5, description="Minimum runs required for pattern", ge=3, le=100)
) -> Dict[str, Any]:
    """
    Mine successful patterns from telemetry data.
    """
    try:
        # Update pattern miner configuration
        pattern_miner.min_runs_for_pattern = min_runs
        
        patterns = pattern_miner.mine_patterns(
            task_type=task_type,
            agent_id=agent_id,
            days_back=days_back
        )
        
        # Convert patterns to dict format
        pattern_data = []
        for pattern in patterns:
            pattern_data.append({
                "pattern_id": pattern.pattern_id,
                "task_type": pattern.task_type,
                "steps": pattern.steps,
                "success_rate": pattern.success_rate,
                "avg_duration_ms": pattern.avg_duration_ms,
                "total_runs": pattern.total_runs,
                "avg_cost": pattern.avg_cost,
                "context_requirements": pattern.context_requirements,
                "tools_used": pattern.tools_used,
                "confidence_score": pattern.confidence_score
            })
        
        return {
            "status": "success",
            "data": {
                "patterns": pattern_data,
                "total_patterns": len(pattern_data),
                "analysis_period_days": days_back,
                "min_runs_threshold": min_runs
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mine patterns: {str(e)}")


@router.post("/patterns/{pattern_id}/create-playbook")
async def create_playbook_from_pattern(pattern_id: str) -> Dict[str, Any]:
    """
    Create a playbook from a discovered pattern.
    """
    try:
        # First, mine current patterns to find the requested one
        patterns = pattern_miner.mine_patterns()
        target_pattern = next((p for p in patterns if p.pattern_id == pattern_id), None)
        
        if not target_pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        with get_db_session() as db:
            playbook = pattern_miner.create_playbook_from_pattern(target_pattern, db)
            
            return {
                "status": "success",
                "data": {
                    "playbook_id": playbook.id,
                    "name": playbook.name,
                    "description": playbook.description,
                    "created_from_pattern": pattern_id
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create playbook: {str(e)}")


@router.get("/stats")
async def get_telemetry_stats(
    days_back: int = Query(30, description="Number of days to analyze", ge=1, le=365)
) -> Dict[str, Any]:
    """
    Get overall telemetry statistics.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        with get_db_session() as db:
            from sqlalchemy import func, and_
            
            # Basic run statistics
            total_runs = db.query(RunTrace).filter(
                RunTrace.start_time >= cutoff_date
            ).count()
            
            successful_runs = db.query(RunTrace).filter(
                and_(
                    RunTrace.start_time >= cutoff_date,
                    RunTrace.success == True
                )
            ).count()
            
            # Agent activity
            active_agents = db.query(func.count(func.distinct(RunTrace.agent_id))).filter(
                RunTrace.start_time >= cutoff_date
            ).scalar() or 0
            
            # Task type distribution
            task_types = db.query(
                RunTrace.task_type,
                func.count(RunTrace.id).label('count')
            ).filter(
                RunTrace.start_time >= cutoff_date
            ).group_by(RunTrace.task_type).all()
            
            # Average metrics
            avg_duration = db.query(func.avg(RunTrace.duration_ms)).filter(
                and_(
                    RunTrace.start_time >= cutoff_date,
                    RunTrace.duration_ms.isnot(None)
                )
            ).scalar() or 0
            
            # Total events
            total_events = db.query(RunEvent).join(
                RunTrace, RunEvent.run_id == RunTrace.run_id
            ).filter(
                RunTrace.start_time >= cutoff_date
            ).count()
            
            return {
                "status": "success",
                "data": {
                    "total_runs": total_runs,
                    "successful_runs": successful_runs,
                    "success_rate": (successful_runs / total_runs * 100) if total_runs > 0 else 0,
                    "active_agents": active_agents,
                    "avg_duration_ms": int(avg_duration),
                    "total_events": total_events,
                    "task_type_distribution": {tt.task_type: tt.count for tt in task_types},
                    "period_days": days_back,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get telemetry stats: {str(e)}")


@router.get("/events")
async def get_events(
    run_id: Optional[str] = Query(None, description="Filter by run ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    step_name: Optional[str] = Query(None, description="Filter by step name"),
    limit: int = Query(100, description="Maximum number of events", ge=1, le=1000)
) -> Dict[str, Any]:
    """
    Get telemetry events with optional filtering.
    """
    try:
        with get_db_session() as db:
            query = db.query(RunEvent)
            
            if run_id:
                query = query.filter(RunEvent.run_id == run_id)
            if event_type:
                query = query.filter(RunEvent.event_type == event_type)
            if step_name:
                query = query.filter(RunEvent.step_name == step_name)
            
            events = query.order_by(RunEvent.timestamp.desc()).limit(limit).all()
            
            return {
                "status": "success",
                "data": {
                    "events": [event.to_dict() for event in events],
                    "count": len(events)
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get events: {str(e)}")


@router.post("/emit")
async def emit_telemetry_event(
    event_data: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """
    Manually emit a telemetry event (for testing or external integrations).
    """
    try:
        required_fields = ['run_id', 'event_type', 'event_data']
        for field in required_fields:
            if field not in event_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        telemetry.emit_event(
            run_id=event_data['run_id'],
            event_type=event_data['event_type'],
            event_data=event_data['event_data'],
            step_name=event_data.get('step_name')
        )
        
        return {
            "status": "success",
            "message": "Event emitted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to emit event: {str(e)}")


@router.get("/analysis/success-patterns")
async def analyze_success_patterns(
    task_type: Optional[str] = Query(None, description="Analyze specific task type"),
    days_back: int = Query(30, description="Days to analyze", ge=1, le=365)
) -> Dict[str, Any]:
    """
    Analyze patterns in successful runs vs failed runs.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        with get_db_session() as db:
            from sqlalchemy import and_, func
            
            # Get successful vs failed runs
            base_query = db.query(RunTrace).filter(
                RunTrace.start_time >= cutoff_date
            )
            
            if task_type:
                base_query = base_query.filter(RunTrace.task_type == task_type)
            
            successful_runs = base_query.filter(RunTrace.success == True).all()
            failed_runs = base_query.filter(RunTrace.success == False).all()
            
            # Analyze patterns
            analysis = {
                "successful_runs": len(successful_runs),
                "failed_runs": len(failed_runs),
                "success_rate": len(successful_runs) / (len(successful_runs) + len(failed_runs)) * 100 if (successful_runs or failed_runs) else 0,
                "avg_duration_success": sum(r.duration_ms for r in successful_runs if r.duration_ms) / len(successful_runs) if successful_runs else 0,
                "avg_duration_failed": sum(r.duration_ms for r in failed_runs if r.duration_ms) / len(failed_runs) if failed_runs else 0,
                "common_success_contexts": {},
                "common_failure_patterns": {}
            }
            
            # Analyze context patterns in successful runs
            success_contexts = {}
            for run in successful_runs[:50]:  # Limit for performance
                if run.context_data:
                    for key in run.context_data.keys():
                        success_contexts[key] = success_contexts.get(key, 0) + 1
            
            analysis["common_success_contexts"] = dict(sorted(
                success_contexts.items(), key=lambda x: x[1], reverse=True
            )[:10])
            
            # Analyze error patterns in failed runs
            failure_patterns = {}
            for run in failed_runs[:50]:  # Limit for performance
                if run.error_message:
                    # Extract first few words of error as pattern
                    error_pattern = ' '.join(run.error_message.split()[:5])
                    failure_patterns[error_pattern] = failure_patterns.get(error_pattern, 0) + 1
            
            analysis["common_failure_patterns"] = dict(sorted(
                failure_patterns.items(), key=lambda x: x[1], reverse=True
            )[:10])
            
            return {
                "status": "success",
                "data": analysis
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze success patterns: {str(e)}")


# Export router for inclusion in main app
telemetry_router = router
