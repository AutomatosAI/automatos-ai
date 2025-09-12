
"""
Analytics Service - Real Data Pipeline for Performance Metrics
Connects analytics dashboard to actual telemetry data instead of mock data.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from database import get_db_session
from models.run_telemetry import RunTrace, RunEvent
from models.agents import Agent
from models.documents import Document


class AnalyticsService:
    """Service for calculating real analytics metrics from telemetry data."""
    
    def get_performance_metrics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get real performance metrics from telemetry data."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        with get_db_session() as db:
            # Get run statistics
            total_runs = db.query(RunTrace).filter(
                RunTrace.start_time >= cutoff_date
            ).count()
            
            successful_runs = db.query(RunTrace).filter(
                and_(
                    RunTrace.start_time >= cutoff_date,
                    RunTrace.success == True
                )
            ).count()
            
            # Calculate success rate
            success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
            
            # Average task completion time
            avg_duration = db.query(func.avg(RunTrace.duration_ms)).filter(
                and_(
                    RunTrace.start_time >= cutoff_date,
                    RunTrace.duration_ms.isnot(None)
                )
            ).scalar() or 0
            
            # Get active agents count
            active_agents = db.query(func.count(func.distinct(RunTrace.agent_id))).filter(
                RunTrace.start_time >= cutoff_date
            ).scalar() or 0
            
            # Get error rate by agent type
            error_rates = self._calculate_error_rates_by_type(db, cutoff_date)
            
            # Get queue depth (current pending tasks)
            queue_depth = db.query(RunTrace).filter(
                RunTrace.status == 'running'
            ).count()
            
            # Calculate system load trend
            system_load_trend = self._calculate_system_load_trend(db, cutoff_date)
            
            # Resource utilization efficiency
            efficiency_score = self._calculate_efficiency_score(db, cutoff_date)
            
            return {
                'success_rate': round(success_rate, 1),
                'avg_task_completion_time_ms': int(avg_duration),
                'avg_task_completion_time_seconds': round(avg_duration / 1000, 1),
                'system_load_trend': system_load_trend,
                'error_rate_by_agent_type': error_rates,
                'queue_depth': queue_depth,
                'efficiency_score': round(efficiency_score, 1),
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'active_agents': active_agents,
                'period_days': days_back,
                'calculated_at': datetime.utcnow().isoformat()
            }
    
    def get_cost_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get real cost analytics from usage data."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        with get_db_session() as db:
            # Get cost data from run metadata and events
            total_cost = 0
            cost_breakdown = defaultdict(float)
            execution_costs = []
            
            runs = db.query(RunTrace).filter(
                RunTrace.start_time >= cutoff_date
            ).all()
            
            for run in runs:
                # Get cost from metadata
                run_cost = run.metadata.get('cost', 0) if run.metadata else 0
                
                # Get tool usage costs from events
                events = db.query(RunEvent).filter(
                    and_(
                        RunEvent.run_id == run.run_id,
                        RunEvent.event_type == 'tool_usage'
                    )
                ).all()
                
                tool_cost = sum(
                    event.event_data.get('cost', 0) 
                    for event in events 
                    if event.event_data
                )
                
                total_run_cost = run_cost + tool_cost
                total_cost += total_run_cost
                
                # Breakdown by task type
                cost_breakdown[run.task_type] += total_run_cost
                
                # Track per execution for trends
                execution_costs.append({
                    'date': run.start_time.date().isoformat(),
                    'cost': total_run_cost,
                    'task_type': run.task_type,
                    'success': run.success
                })
            
            # Calculate cost per successful execution
            successful_runs = len([r for r in runs if r.success])
            cost_per_success = (total_cost / successful_runs) if successful_runs > 0 else 0
            
            # Daily cost trend
            daily_costs = defaultdict(float)
            for exec_cost in execution_costs:
                daily_costs[exec_cost['date']] += exec_cost['cost']
            
            return {
                'total_cost': round(total_cost, 2),
                'cost_per_successful_execution': round(cost_per_success, 4),
                'cost_breakdown_by_task': dict(cost_breakdown),
                'daily_costs': dict(daily_costs),
                'execution_costs': execution_costs[:100],  # Limit for performance
                'avg_cost_per_run': round(total_cost / len(runs), 4) if runs else 0,
                'period_days': days_back,
                'calculated_at': datetime.utcnow().isoformat()
            }
    
    def get_usage_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get usage analytics from telemetry data."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        with get_db_session() as db:
            # Peak usage hours analysis
            hourly_usage = defaultdict(int)
            task_type_usage = defaultdict(int)
            agent_usage = defaultdict(int)
            
            runs = db.query(RunTrace).filter(
                RunTrace.start_time >= cutoff_date
            ).all()
            
            for run in runs:
                # Hour-of-day analysis
                hour = run.start_time.hour
                hourly_usage[hour] += 1
                
                # Task type analysis
                task_type_usage[run.task_type] += 1
                
                # Agent usage
                agent_usage[run.agent_id] += 1
            
            # Identify peak hours
            peak_hours = sorted(hourly_usage.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Resource bottlenecks (high failure rates at specific times)
            bottlenecks = self._identify_bottlenecks(db, cutoff_date)
            
            # Usage trends (daily aggregation)
            daily_usage = self._calculate_daily_usage_trends(db, cutoff_date)
            
            return {
                'peak_usage_hours': [
                    {'hour': hour, 'count': count} for hour, count in peak_hours
                ],
                'task_type_distribution': dict(task_type_usage),
                'agent_utilization': dict(sorted(agent_usage.items(), key=lambda x: x[1], reverse=True)[:20]),
                'hourly_distribution': dict(hourly_usage),
                'resource_bottlenecks': bottlenecks,
                'daily_usage_trend': daily_usage,
                'total_executions': len(runs),
                'period_days': days_back,
                'calculated_at': datetime.utcnow().isoformat()
            }
    
    def get_agent_rankings(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get agent performance rankings."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        with get_db_session() as db:
            # Query agent performance stats
            agent_stats = db.query(
                RunTrace.agent_id,
                func.count(RunTrace.id).label('total_runs'),
                func.sum(func.case([(RunTrace.success == True, 1)], else_=0)).label('successful_runs'),
                func.avg(RunTrace.duration_ms).label('avg_duration_ms'),
                func.sum(func.coalesce(RunTrace.metadata['cost'].astext.cast(db.Float), 0)).label('total_cost')
            ).filter(
                RunTrace.start_time >= cutoff_date
            ).group_by(RunTrace.agent_id).all()
            
            rankings = []
            for stat in agent_stats:
                success_rate = (stat.successful_runs / stat.total_runs * 100) if stat.total_runs > 0 else 0
                avg_cost = stat.total_cost / stat.total_runs if stat.total_runs > 0 else 0
                
                # Calculate efficiency score
                efficiency = self._calculate_agent_efficiency(
                    success_rate, stat.avg_duration_ms or 0, avg_cost
                )
                
                rankings.append({
                    'agent_id': stat.agent_id,
                    'total_runs': stat.total_runs,
                    'success_rate': round(success_rate, 1),
                    'avg_duration_ms': int(stat.avg_duration_ms or 0),
                    'total_cost': round(stat.total_cost, 2),
                    'avg_cost_per_run': round(avg_cost, 4),
                    'efficiency_score': round(efficiency, 1)
                })
            
            # Sort by efficiency score
            rankings.sort(key=lambda x: x['efficiency_score'], reverse=True)
            
            return rankings
    
    def get_sla_compliance(self, days_back: int = 30) -> Dict[str, Any]:
        """Get SLA compliance metrics."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # SLA thresholds (configurable)
        sla_thresholds = {
            'document_analysis': 30000,  # 30 seconds
            'text_generation': 15000,    # 15 seconds
            'classification': 5000,      # 5 seconds
            'default': 60000             # 1 minute
        }
        
        with get_db_session() as db:
            runs = db.query(RunTrace).filter(
                and_(
                    RunTrace.start_time >= cutoff_date,
                    RunTrace.duration_ms.isnot(None)
                )
            ).all()
            
            sla_compliance = {}
            overall_compliant = 0
            total_runs = len(runs)
            
            for run in runs:
                task_type = run.task_type
                threshold = sla_thresholds.get(task_type, sla_thresholds['default'])
                compliant = run.duration_ms <= threshold
                
                if task_type not in sla_compliance:
                    sla_compliance[task_type] = {
                        'total': 0,
                        'compliant': 0,
                        'threshold_ms': threshold
                    }
                
                sla_compliance[task_type]['total'] += 1
                if compliant:
                    sla_compliance[task_type]['compliant'] += 1
                    overall_compliant += 1
            
            # Calculate compliance percentages
            for task_type, stats in sla_compliance.items():
                stats['compliance_rate'] = (
                    stats['compliant'] / stats['total'] * 100
                ) if stats['total'] > 0 else 0
            
            overall_compliance_rate = (
                overall_compliant / total_runs * 100
            ) if total_runs > 0 else 0
            
            return {
                'overall_compliance_rate': round(overall_compliance_rate, 1),
                'sla_by_task_type': sla_compliance,
                'total_runs_analyzed': total_runs,
                'period_days': days_back,
                'calculated_at': datetime.utcnow().isoformat()
            }
    
    def _calculate_error_rates_by_type(self, db: Session, cutoff_date: datetime) -> Dict[str, float]:
        """Calculate error rates by agent type."""
        # This would need agent type information - simplified version
        task_types = db.query(
            RunTrace.task_type,
            func.count(RunTrace.id).label('total'),
            func.sum(func.case([(RunTrace.success == False, 1)], else_=0)).label('errors')
        ).filter(
            RunTrace.start_time >= cutoff_date
        ).group_by(RunTrace.task_type).all()
        
        error_rates = {}
        for task_type, total, errors in task_types:
            error_rates[task_type] = round((errors / total * 100), 1) if total > 0 else 0
        
        return error_rates
    
    def _calculate_system_load_trend(self, db: Session, cutoff_date: datetime) -> List[Dict[str, Any]]:
        """Calculate system load trend over time."""
        # Daily load calculation
        daily_loads = db.query(
            func.date(RunTrace.start_time).label('date'),
            func.count(RunTrace.id).label('runs'),
            func.avg(RunTrace.duration_ms).label('avg_duration')
        ).filter(
            RunTrace.start_time >= cutoff_date
        ).group_by(func.date(RunTrace.start_time)).all()
        
        trend = []
        for date, runs, avg_duration in daily_loads:
            # Simple load calculation (runs * avg_duration)
            load_score = runs * (avg_duration or 0) / 1000  # Convert to seconds
            trend.append({
                'date': date.isoformat(),
                'runs': runs,
                'load_score': round(load_score, 2)
            })
        
        return sorted(trend, key=lambda x: x['date'])
    
    def _calculate_efficiency_score(self, db: Session, cutoff_date: datetime) -> float:
        """Calculate overall system efficiency score."""
        # Get successful runs vs total runs, weighted by duration
        stats = db.query(
            func.count(RunTrace.id).label('total'),
            func.sum(func.case([(RunTrace.success == True, 1)], else_=0)).label('successful'),
            func.avg(RunTrace.duration_ms).label('avg_duration')
        ).filter(
            RunTrace.start_time >= cutoff_date
        ).first()
        
        if not stats.total:
            return 0.0
        
        success_rate = stats.successful / stats.total
        
        # Normalize duration (lower is better)
        # Assume 30 seconds is baseline good performance
        duration_efficiency = min(1.0, 30000 / (stats.avg_duration or 30000))
        
        # Combined efficiency score
        efficiency = (success_rate * 0.7 + duration_efficiency * 0.3) * 100
        
        return efficiency
    
    def _identify_bottlenecks(self, db: Session, cutoff_date: datetime) -> List[Dict[str, Any]]:
        """Identify resource bottlenecks."""
        # Find time periods with high failure rates
        hourly_failures = db.query(
            func.extract('hour', RunTrace.start_time).label('hour'),
            func.count(RunTrace.id).label('total'),
            func.sum(func.case([(RunTrace.success == False, 1)], else_=0)).label('failures')
        ).filter(
            RunTrace.start_time >= cutoff_date
        ).group_by(func.extract('hour', RunTrace.start_time)).all()
        
        bottlenecks = []
        for hour, total, failures in hourly_failures:
            if total > 5:  # Only consider hours with significant activity
                failure_rate = failures / total * 100
                if failure_rate > 20:  # Consider >20% failure rate a bottleneck
                    bottlenecks.append({
                        'time_period': f"{int(hour):02d}:00-{int(hour)+1:02d}:00",
                        'failure_rate': round(failure_rate, 1),
                        'total_runs': total,
                        'severity': 'high' if failure_rate > 40 else 'medium'
                    })
        
        return sorted(bottlenecks, key=lambda x: x['failure_rate'], reverse=True)
    
    def _calculate_daily_usage_trends(self, db: Session, cutoff_date: datetime) -> List[Dict[str, Any]]:
        """Calculate daily usage trends."""
        daily_stats = db.query(
            func.date(RunTrace.start_time).label('date'),
            func.count(RunTrace.id).label('total_runs'),
            func.sum(func.case([(RunTrace.success == True, 1)], else_=0)).label('successful_runs'),
            func.avg(RunTrace.duration_ms).label('avg_duration')
        ).filter(
            RunTrace.start_time >= cutoff_date
        ).group_by(func.date(RunTrace.start_time)).all()
        
        trends = []
        for date, total, successful, avg_duration in daily_stats:
            trends.append({
                'date': date.isoformat(),
                'total_runs': total,
                'successful_runs': successful,
                'success_rate': round((successful / total * 100), 1) if total > 0 else 0,
                'avg_duration_ms': int(avg_duration or 0)
            })
        
        return sorted(trends, key=lambda x: x['date'])
    
    def _calculate_agent_efficiency(self, success_rate: float, avg_duration_ms: float, avg_cost: float) -> float:
        """Calculate agent efficiency score."""
        # Normalize metrics (higher success rate, lower duration and cost = better)
        success_score = success_rate / 100  # 0-1 scale
        
        # Duration score (assume 30s is good baseline)
        duration_score = min(1.0, 30000 / max(avg_duration_ms, 1))
        
        # Cost score (assume $0.01 is good baseline)
        cost_score = min(1.0, 0.01 / max(avg_cost, 0.001))
        
        # Weighted efficiency score
        efficiency = (success_score * 0.5 + duration_score * 0.3 + cost_score * 0.2) * 100
        
        return efficiency


# Global analytics service instance
analytics_service = AnalyticsService()
