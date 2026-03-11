"""
Advanced Monitoring Service
============================

Provides comprehensive monitoring for the Automatos AI platform:
- Performance metrics
- Error tracking
- Agent activity monitoring
- Resource utilization
- Workflow analytics
"""

import logging
import time
import psutil
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Represents a single metric measurement"""
    timestamp: datetime
    name: str
    value: float
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "value": self.value,
            "tags": self.tags or {},
            "metadata": self.metadata or {}
        }


@dataclass
class Alert:
    """Represents a monitoring alert"""
    id: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    title: str
    message: str
    timestamp: datetime
    source: str
    resolved: bool = False
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Collects and stores metrics"""
    
    def __init__(self, max_points: int = 10000, retention_hours: int = 24):
        self.metrics = defaultdict(lambda: deque(maxlen=max_points))
        self.retention_hours = retention_hours
        self.lock = threading.Lock()
    
    def record(self, metric: MetricPoint):
        """Record a metric point"""
        with self.lock:
            key = f"{metric.name}:{json.dumps(metric.tags or {}, sort_keys=True)}"
            self.metrics[key].append(metric)
    
    def query(
        self, 
        name: str, 
        tags: Dict[str, str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """Query metrics by name and tags"""
        with self.lock:
            key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
            
            if key not in self.metrics:
                return []
            
            points = list(self.metrics[key])
            
            # Filter by time range
            if start_time:
                points = [p for p in points if p.timestamp >= start_time]
            if end_time:
                points = [p for p in points if p.timestamp <= end_time]
            
            return points
    
    def get_latest(self, name: str, tags: Dict[str, str] = None) -> Optional[MetricPoint]:
        """Get the latest metric point"""
        points = self.query(name, tags)
        return points[-1] if points else None
    
    def cleanup(self):
        """Remove old metrics beyond retention period"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self.lock:
            for key in list(self.metrics.keys()):
                # Remove old points
                while self.metrics[key] and self.metrics[key][0].timestamp < cutoff:
                    self.metrics[key].popleft()
                
                # Remove empty metric keys
                if not self.metrics[key]:
                    del self.metrics[key]


class AlertManager:
    """Manages monitoring alerts"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.lock = threading.Lock()
        self.alert_handlers = []
    
    def create_alert(
        self,
        severity: str,
        title: str,
        message: str,
        source: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create a new alert"""
        with self.lock:
            alert_id = f"{source}_{int(time.time() * 1000)}"
            
            alert = Alert(
                id=alert_id,
                severity=severity,
                title=title,
                message=message,
                timestamp=datetime.now(),
                source=source,
                metadata=metadata or {}
            )
            
            self.alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Trigger handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}", exc_info=True)
            
            logger.warning(f"Alert created: [{severity}] {title} - {message}")
            return alert_id
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self.lock:
            return [a for a in self.alerts.values() if not a.resolved]
    
    def add_handler(self, handler):
        """Add an alert handler callback"""
        self.alert_handlers.append(handler)


class MonitoringService:
    """
    Main monitoring service for the Automatos AI platform
    """
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self.monitoring_thread = None
        self.running = False
        
        # Performance thresholds
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "response_time_ms": 5000,
            "error_rate": 0.05,
            "token_usage_per_min": 10000
        }
    
    def start(self):
        """Start the monitoring service"""
        if not self.running:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Monitoring service started")
    
    def stop(self):
        """Stop the monitoring service"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Monitoring service stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check thresholds and create alerts
                self._check_thresholds()
                
                # Cleanup old metrics
                self.metrics.cleanup()
                
                # Sleep for monitoring interval
                time.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}", exc_info=True)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.record(MetricPoint(
            timestamp=timestamp,
            name="system.cpu.percent",
            value=cpu_percent
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics.record(MetricPoint(
            timestamp=timestamp,
            name="system.memory.percent",
            value=memory.percent
        ))
        self.metrics.record(MetricPoint(
            timestamp=timestamp,
            name="system.memory.used_gb",
            value=memory.used / (1024 ** 3)
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics.record(MetricPoint(
            timestamp=timestamp,
            name="system.disk.percent",
            value=disk.percent
        ))
        
        # Network metrics (if available)
        try:
            net_io = psutil.net_io_counters()
            self.metrics.record(MetricPoint(
                timestamp=timestamp,
                name="system.network.bytes_sent",
                value=net_io.bytes_sent
            ))
            self.metrics.record(MetricPoint(
                timestamp=timestamp,
                name="system.network.bytes_recv",
                value=net_io.bytes_recv
            ))
        except Exception:
            logger.error("Failed to collect network I/O metrics", exc_info=True)

    def _check_thresholds(self):
        """Check metrics against thresholds and create alerts"""
        
        # Check CPU usage
        cpu_metric = self.metrics.get_latest("system.cpu.percent")
        if cpu_metric and cpu_metric.value > self.thresholds["cpu_percent"]:
            self.alerts.create_alert(
                severity="warning",
                title="High CPU Usage",
                message=f"CPU usage is {cpu_metric.value:.1f}% (threshold: {self.thresholds['cpu_percent']}%)",
                source="system_monitor"
            )
        
        # Check memory usage
        memory_metric = self.metrics.get_latest("system.memory.percent")
        if memory_metric and memory_metric.value > self.thresholds["memory_percent"]:
            self.alerts.create_alert(
                severity="warning",
                title="High Memory Usage",
                message=f"Memory usage is {memory_metric.value:.1f}% (threshold: {self.thresholds['memory_percent']}%)",
                source="system_monitor"
            )
        
        # Check disk usage
        disk_metric = self.metrics.get_latest("system.disk.percent")
        if disk_metric and disk_metric.value > self.thresholds["disk_percent"]:
            self.alerts.create_alert(
                severity="error",
                title="High Disk Usage",
                message=f"Disk usage is {disk_metric.value:.1f}% (threshold: {self.thresholds['disk_percent']}%)",
                source="system_monitor"
            )
    
    def record_workflow_execution(
        self,
        workflow_id: int,
        execution_time_ms: float,
        tokens_used: int,
        success: bool,
        agent_count: int = 0,
        error_message: Optional[str] = None
    ):
        """Record workflow execution metrics"""
        timestamp = datetime.now()
        
        # Execution time
        self.metrics.record(MetricPoint(
            timestamp=timestamp,
            name="workflow.execution_time_ms",
            value=execution_time_ms,
            tags={"workflow_id": str(workflow_id), "success": str(success)}
        ))
        
        # Token usage
        self.metrics.record(MetricPoint(
            timestamp=timestamp,
            name="workflow.tokens_used",
            value=tokens_used,
            tags={"workflow_id": str(workflow_id)}
        ))
        
        # Agent count
        self.metrics.record(MetricPoint(
            timestamp=timestamp,
            name="workflow.agent_count",
            value=agent_count,
            tags={"workflow_id": str(workflow_id)}
        ))
        
        # Success/failure counter
        self.metrics.record(MetricPoint(
            timestamp=timestamp,
            name="workflow.executions",
            value=1,
            tags={"status": "success" if success else "failure"}
        ))
        
        # Check response time threshold
        if execution_time_ms > self.thresholds["response_time_ms"]:
            self.alerts.create_alert(
                severity="warning",
                title="Slow Workflow Execution",
                message=f"Workflow {workflow_id} took {execution_time_ms:.0f}ms (threshold: {self.thresholds['response_time_ms']}ms)",
                source="workflow_monitor",
                metadata={"workflow_id": workflow_id}
            )
        
        # Alert on failure
        if not success and error_message:
            self.alerts.create_alert(
                severity="error",
                title=f"Workflow {workflow_id} Failed",
                message=error_message,
                source="workflow_monitor",
                metadata={"workflow_id": workflow_id}
            )
    
    def record_agent_execution(
        self,
        agent_id: int,
        agent_name: str,
        task: str,
        execution_time_ms: float,
        tokens_used: int,
        success: bool
    ):
        """Record agent execution metrics"""
        timestamp = datetime.now()
        
        self.metrics.record(MetricPoint(
            timestamp=timestamp,
            name="agent.execution_time_ms",
            value=execution_time_ms,
            tags={"agent_id": str(agent_id), "agent_name": agent_name, "success": str(success)}
        ))
        
        self.metrics.record(MetricPoint(
            timestamp=timestamp,
            name="agent.tokens_used",
            value=tokens_used,
            tags={"agent_id": str(agent_id), "agent_name": agent_name}
        ))
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        
        # Calculate aggregates
        workflow_executions = self.metrics.query("workflow.executions", start_time=last_hour)
        success_count = sum(1 for m in workflow_executions if m.tags.get("status") == "success")
        failure_count = sum(1 for m in workflow_executions if m.tags.get("status") == "failure")
        
        workflow_times = self.metrics.query("workflow.execution_time_ms", start_time=last_hour)
        avg_execution_time = sum(m.value for m in workflow_times) / len(workflow_times) if workflow_times else 0
        
        token_metrics = self.metrics.query("workflow.tokens_used", start_time=last_hour)
        total_tokens = sum(m.value for m in token_metrics)
        
        # Get latest system metrics
        cpu_metric = self.metrics.get_latest("system.cpu.percent")
        memory_metric = self.metrics.get_latest("system.memory.percent")
        disk_metric = self.metrics.get_latest("system.disk.percent")
        
        return {
            "timestamp": now.isoformat(),
            "period": "last_hour",
            "workflow_metrics": {
                "total_executions": success_count + failure_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": success_count / max(success_count + failure_count, 1),
                "avg_execution_time_ms": avg_execution_time,
                "total_tokens_used": total_tokens
            },
            "system_metrics": {
                "cpu_percent": cpu_metric.value if cpu_metric else 0,
                "memory_percent": memory_metric.value if memory_metric else 0,
                "disk_percent": disk_metric.value if disk_metric else 0
            },
            "active_alerts": len(self.alerts.get_active_alerts()),
            "alerts": [a.to_dict() for a in self.alerts.get_active_alerts()[:10]]
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "alerts": [a.to_dict() for a in self.alerts.get_active_alerts()]
        }
        
        # Collect all metrics
        for key in self.metrics.metrics.keys():
            name = key.split(':')[0]
            if name not in metrics_data["metrics"]:
                metrics_data["metrics"][name] = []
            
            points = list(self.metrics.metrics[key])
            for point in points[-100:]:  # Last 100 points
                metrics_data["metrics"][name].append(point.to_dict())
        
        if format == "json":
            return json.dumps(metrics_data, indent=2)
        else:
            return str(metrics_data)


# Singleton instance
_monitoring_service = None

def get_monitoring_service() -> MonitoringService:
    """Get or create the monitoring service singleton"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
        _monitoring_service.start()
    return _monitoring_service

