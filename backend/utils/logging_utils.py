
"""
Enhanced Logging and Performance Monitoring System
=================================================

This module provides comprehensive logging, performance monitoring, and agent communication
tracking for the Automotas AI system. It includes:

- Structured workflow logging with step-by-step visibility
- Agent communication and handoff tracking
- Performance metrics and timing data
- Token usage and cost monitoring
- Real-time progress updates for UI display
- Error handling and recovery logging
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict, deque
import psutil
import os

# Configure structured logging
class WorkflowLogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AGENT_COMM = "AGENT_COMM"
    PERFORMANCE = "PERFORMANCE"
    PROGRESS = "PROGRESS"

@dataclass
class AgentStatus:
    agent_id: str
    agent_type: str
    current_task: str
    status: str  # idle, working, waiting, completed, error
    progress_percentage: float
    start_time: datetime
    last_update: datetime
    metadata: Dict[str, Any] = None

@dataclass
class WorkflowStep:
    step_id: str
    step_name: str
    agent_id: str
    status: str  # pending, in_progress, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    progress_percentage: float = 0.0
    sub_steps: List['WorkflowStep'] = None
    metadata: Dict[str, Any] = None
    error_details: Optional[str] = None

@dataclass
class PerformanceMetrics:
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    success: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class AgentCommunication:
    from_agent: str
    to_agent: str
    message_type: str  # handoff, request, response, status_update
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

class EnhancedWorkflowLogger:
    """Enhanced logging system with workflow visibility and agent communication tracking"""
    
    def __init__(self, workflow_id: str, log_file: str = None):
        self.workflow_id = workflow_id
        self.log_file = log_file or f"workflow_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Initialize logging
        self.logger = logging.getLogger(f"workflow_{workflow_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(console_handler)
        
        # Workflow tracking
        self.workflow_steps: Dict[str, WorkflowStep] = {}
        self.active_agents: Dict[str, AgentStatus] = {}
        self.agent_communications: List[AgentCommunication] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        
        # Progress tracking
        self.overall_progress = 0.0
        self.current_phase = "Initializing"
        self.total_steps = 0
        self.completed_steps = 0
        
        # Performance monitoring
        self.start_time = datetime.now()
        self.token_usage = defaultdict(int)
        self.cost_tracking = defaultdict(float)
        
        # Thread-safe logging
        self._lock = threading.Lock()
        
        self.log_workflow_start()
    
    def log_workflow_start(self):
        """Log the start of a workflow"""
        self.logger.info(f"🚀 WORKFLOW STARTED: {self.workflow_id}")
        self.logger.info(f"📁 Log file: {self.log_file}")
        self.logger.info(f"⏰ Start time: {self.start_time}")
        
    def log_workflow_step(self, step_id: str, step_name: str, agent_id: str, status: str = "pending", metadata: Dict[str, Any] = None):
        """Log a workflow step with detailed information"""
        with self._lock:
            if step_id not in self.workflow_steps:
                self.workflow_steps[step_id] = WorkflowStep(
                    step_id=step_id,
                    step_name=step_name,
                    agent_id=agent_id,
                    status=status,
                    metadata=metadata or {}
                )
                self.total_steps += 1
            else:
                self.workflow_steps[step_id].status = status
                if metadata:
                    self.workflow_steps[step_id].metadata.update(metadata)
            
            step = self.workflow_steps[step_id]
            
            if status == "in_progress":
                step.start_time = datetime.now()
                self.logger.info(f"▶️  STEP STARTED: [{step_id}] {step_name} (Agent: {agent_id})")
                
            elif status == "completed":
                step.end_time = datetime.now()
                if step.start_time:
                    step.duration_seconds = (step.end_time - step.start_time).total_seconds()
                else:
                    step.duration_seconds = 0.0
                step.progress_percentage = 100.0
                self.completed_steps += 1
                self.logger.info(f"✅ STEP COMPLETED: [{step_id}] {step_name} (Duration: {step.duration_seconds:.2f}s)")
                
            elif status == "failed":
                step.end_time = datetime.now()
                if step.start_time:
                    step.duration_seconds = (step.end_time - step.start_time).total_seconds()
                self.logger.error(f"❌ STEP FAILED: [{step_id}] {step_name}")
                
            # Update overall progress
            if self.total_steps > 0:
                self.overall_progress = (self.completed_steps / self.total_steps) * 100
    
    def log_agent_status(self, agent_id: str, agent_type: str, current_task: str, status: str, progress: float = 0.0, metadata: Dict[str, Any] = None):
        """Log agent status and activity"""
        with self._lock:
            now = datetime.now()
            
            if agent_id not in self.active_agents:
                self.active_agents[agent_id] = AgentStatus(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    current_task=current_task,
                    status=status,
                    progress_percentage=progress,
                    start_time=now,
                    last_update=now,
                    metadata=metadata or {}
                )
                self.logger.info(f"🤖 AGENT INITIALIZED: {agent_id} ({agent_type})")
            else:
                agent = self.active_agents[agent_id]
                agent.current_task = current_task
                agent.status = status
                agent.progress_percentage = progress
                agent.last_update = now
                if metadata:
                    agent.metadata.update(metadata)
            
            # Log status change
            status_emoji = {
                "idle": "⏸️",
                "working": "⚡",
                "waiting": "⏳",
                "completed": "✅",
                "error": "❌"
            }.get(status, "🔄")
            
            self.logger.info(f"{status_emoji} AGENT STATUS: {agent_id} | {status.upper()} | {current_task} ({progress:.1f}%)")
    
    def log_agent_communication(self, from_agent: str, to_agent: str, message_type: str, content: str, metadata: Dict[str, Any] = None):
        """Log communication between agents"""
        with self._lock:
            comm = AgentCommunication(
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=message_type,
                content=content,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            self.agent_communications.append(comm)
            
            # Log the communication
            comm_emoji = {
                "handoff": "🔄",
                "request": "📤",
                "response": "📥",
                "status_update": "📊"
            }.get(message_type, "💬")
            
            self.logger.info(f"{comm_emoji} AGENT COMM: {from_agent} → {to_agent} | {message_type.upper()}: {content[:100]}...")
    
    def log_performance_metric(self, operation_name: str, duration_seconds: float, tokens_used: int = 0, estimated_cost: float = 0.0, success: bool = True, metadata: Dict[str, Any] = None):
        """Log performance metrics for operations"""
        with self._lock:
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / (1024 * 1024)
            
            metric = PerformanceMetrics(
                operation_name=operation_name,
                start_time=datetime.now() - timedelta(seconds=duration_seconds),
                end_time=datetime.now(),
                duration_seconds=duration_seconds,
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory_mb,
                tokens_used=tokens_used,
                estimated_cost_usd=estimated_cost,
                success=success,
                metadata=metadata or {}
            )
            
            self.performance_metrics.append(metric)
            
            # Update tracking
            self.token_usage[operation_name] += tokens_used
            self.cost_tracking[operation_name] += estimated_cost
            
            # Log performance
            status_emoji = "✅" if success else "❌"
            self.logger.info(f"📊 PERFORMANCE: {status_emoji} {operation_name} | {duration_seconds:.2f}s | {tokens_used} tokens | ${estimated_cost:.4f}")
    
    def log_code_generation_progress(self, file_path: str, status: str, lines_generated: int = 0, metadata: Dict[str, Any] = None):
        """Log code generation progress with file-by-file updates"""
        status_emoji = {
            "started": "📝",
            "generating": "⚡",
            "completed": "✅",
            "failed": "❌"
        }.get(status, "🔄")
        
        self.logger.info(f"{status_emoji} CODE GEN: {file_path} | {status.upper()} | {lines_generated} lines")
        
        if metadata:
            for key, value in metadata.items():
                self.logger.debug(f"    {key}: {value}")
    
    def log_git_operation(self, operation: str, files: List[str], commit_message: str = None, success: bool = True, details: str = None):
        """Log Git operations with commit details"""
        status_emoji = "✅" if success else "❌"
        self.logger.info(f"🔧 GIT: {status_emoji} {operation.upper()}")
        
        if files:
            self.logger.info(f"    Files: {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")
        
        if commit_message:
            self.logger.info(f"    Commit: {commit_message}")
        
        if details:
            self.logger.debug(f"    Details: {details}")
    
    def log_error_with_recovery(self, error: Exception, recovery_action: str = None, context: Dict[str, Any] = None):
        """Log errors with recovery information"""
        self.logger.error(f"❌ ERROR: {type(error).__name__}: {str(error)}")
        
        if context:
            self.logger.error(f"    Context: {json.dumps(context, indent=2)}")
        
        if recovery_action:
            self.logger.info(f"🔄 RECOVERY: {recovery_action}")
    
    def update_phase(self, phase_name: str, progress: float = None):
        """Update the current workflow phase"""
        self.current_phase = phase_name
        if progress is not None:
            self.overall_progress = progress
        
        self.logger.info(f"🔄 PHASE: {phase_name} ({self.overall_progress:.1f}%)")
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a comprehensive workflow summary"""
        with self._lock:
            now = datetime.now()
            total_duration = (now - self.start_time).total_seconds()
            
            return {
                "workflow_id": self.workflow_id,
                "start_time": self.start_time.isoformat(),
                "current_time": now.isoformat(),
                "total_duration_seconds": total_duration,
                "current_phase": self.current_phase,
                "overall_progress": self.overall_progress,
                "total_steps": self.total_steps,
                "completed_steps": self.completed_steps,
                "active_agents": len([a for a in self.active_agents.values() if a.status == "working"]),
                "total_agents": len(self.active_agents),
                "total_tokens_used": sum(self.token_usage.values()),
                "total_estimated_cost": sum(self.cost_tracking.values()),
                "agent_statuses": {aid: asdict(agent) for aid, agent in self.active_agents.items()},
                "recent_communications": [asdict(comm) for comm in self.agent_communications[-10:]],
                "performance_summary": {
                    "total_operations": len(self.performance_metrics),
                    "successful_operations": len([m for m in self.performance_metrics if m.success]),
                    "average_duration": sum(m.duration_seconds for m in self.performance_metrics if m.duration_seconds) / max(len(self.performance_metrics), 1),
                    "total_cost": sum(self.cost_tracking.values())
                }
            }
    
    def export_detailed_log(self, export_path: str = None) -> str:
        """Export detailed workflow log as JSON"""
        if not export_path:
            export_path = f"workflow_export_{self.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with self._lock:
            export_data = {
                "workflow_summary": self.get_workflow_summary(),
                "workflow_steps": {sid: asdict(step) for sid, step in self.workflow_steps.items()},
                "agent_communications": [asdict(comm) for comm in self.agent_communications],
                "performance_metrics": [asdict(metric) for metric in self.performance_metrics],
                "token_usage_by_operation": dict(self.token_usage),
                "cost_tracking_by_operation": dict(self.cost_tracking)
            }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"📄 EXPORT: Detailed log exported to {export_path}")
        return export_path
    
    def log_workflow_completion(self, success: bool = True, final_message: str = None):
        """Log workflow completion with summary"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        status_emoji = "🎉" if success else "💥"
        status_text = "COMPLETED" if success else "FAILED"
        
        self.logger.info(f"{status_emoji} WORKFLOW {status_text}: {self.workflow_id}")
        self.logger.info(f"⏱️  Total Duration: {total_duration:.2f} seconds")
        self.logger.info(f"📊 Steps Completed: {self.completed_steps}/{self.total_steps}")
        self.logger.info(f"🤖 Agents Used: {len(self.active_agents)}")
        self.logger.info(f"💰 Total Cost: ${sum(self.cost_tracking.values()):.4f}")
        self.logger.info(f"🔤 Total Tokens: {sum(self.token_usage.values())}")
        
        if final_message:
            self.logger.info(f"📝 Final Message: {final_message}")
        
        # Export detailed log
        export_path = self.export_detailed_log()
        self.logger.info(f"📄 Detailed log available at: {export_path}")

class PerformanceTimer:
    """Context manager for timing operations with automatic logging"""
    
    def __init__(self, logger: EnhancedWorkflowLogger, operation_name: str, metadata: Dict[str, Any] = None):
        self.logger = logger
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.start_time = None
        self.tokens_used = 0
        self.estimated_cost = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        
        self.logger.log_performance_metric(
            operation_name=self.operation_name,
            duration_seconds=duration,
            tokens_used=self.tokens_used,
            estimated_cost=self.estimated_cost,
            success=success,
            metadata=self.metadata
        )
    
    def add_tokens(self, tokens: int, cost: float = 0.0):
        """Add token usage and cost information"""
        self.tokens_used += tokens
        self.estimated_cost += cost

# Decorator for automatic performance timing
def timed_operation(operation_name: str, logger_attr: str = "workflow_logger"):
    """Decorator to automatically time and log operations"""
    def decorator(func):
        async def async_wrapper(self, *args, **kwargs):
            logger = getattr(self, logger_attr)
            with PerformanceTimer(logger, operation_name) as timer:
                result = await func(self, *args, **kwargs)
                # Extract token usage from result if available
                if isinstance(result, dict) and "usage_stats" in result:
                    usage = result["usage_stats"]
                    if isinstance(usage, dict):
                        tokens = usage.get("total_tokens", 0)
                        cost = usage.get("estimated_cost", 0.0)
                        timer.add_tokens(tokens, cost)
                return result
        
        def sync_wrapper(self, *args, **kwargs):
            logger = getattr(self, logger_attr)
            with PerformanceTimer(logger, operation_name) as timer:
                result = func(self, *args, **kwargs)
                # Extract token usage from result if available
                if isinstance(result, dict) and "usage_stats" in result:
                    usage = result["usage_stats"]
                    if isinstance(usage, dict):
                        tokens = usage.get("total_tokens", 0)
                        cost = usage.get("estimated_cost", 0.0)
                        timer.add_tokens(tokens, cost)
                return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Token usage tracking utilities
class TokenTracker:
    """Utility class for tracking token usage and costs"""
    
    # Pricing per 1K tokens (approximate, as of 2024)
    TOKEN_COSTS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
    }
    
    @classmethod
    def calculate_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for token usage"""
        if model not in cls.TOKEN_COSTS:
            # Default to GPT-4 pricing for unknown models
            model = "gpt-4"
        
        costs = cls.TOKEN_COSTS[model]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    @classmethod
    def format_usage_summary(cls, usage_data: Dict[str, Any]) -> str:
        """Format token usage data for logging"""
        if not usage_data:
            return "No usage data available"
        
        total_tokens = usage_data.get("total_tokens", 0)
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        
        return f"Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})"

# Example usage and testing
if __name__ == "__main__":
    # Test the logging system
    logger = EnhancedWorkflowLogger("test_workflow_001")
    
    # Simulate workflow steps
    logger.log_workflow_step("step_1", "Initialize System", "orchestrator", "in_progress")
    logger.log_agent_status("orchestrator", "main", "System initialization", "working", 25.0)
    
    time.sleep(1)
    
    logger.log_workflow_step("step_1", "Initialize System", "orchestrator", "completed")
    logger.log_agent_communication("orchestrator", "code_generator", "handoff", "Please generate the main application file")
    
    logger.log_workflow_step("step_2", "Generate Code", "code_generator", "in_progress")
    logger.log_agent_status("code_generator", "specialist", "Generating main.py", "working", 50.0)
    
    logger.log_code_generation_progress("main.py", "generating", 45)
    
    time.sleep(1)
    
    logger.log_code_generation_progress("main.py", "completed", 120)
    logger.log_workflow_step("step_2", "Generate Code", "code_generator", "completed")
    
    logger.log_performance_metric("code_generation", 2.5, tokens_used=1500, estimated_cost=0.045)
    
    # Complete workflow
    logger.log_workflow_completion(success=True, final_message="Test workflow completed successfully")
    
    print(f"Test completed. Log file: {logger.log_file}")
