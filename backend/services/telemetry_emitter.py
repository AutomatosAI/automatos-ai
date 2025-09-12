
"""
Telemetry Emission Service - Core Intelligence Foundation
Automatically tracks agent executions and emits telemetry data for pattern mining.
"""
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from sqlalchemy.orm import Session
from database import get_db_session
from models.run_telemetry import RunTrace, RunEvent


class TelemetryEmitter:
    """Service for emitting and managing agent execution telemetry."""
    
    def __init__(self):
        self.current_run_id: Optional[str] = None
        self.run_stack: List[str] = []
    
    def start_run(
        self,
        agent_id: str,
        task_type: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start tracking a new agent run."""
        run_id = str(uuid.uuid4())
        
        with get_db_session() as db:
            run_trace = RunTrace(
                run_id=run_id,
                agent_id=agent_id,
                task_type=task_type,
                input_data=input_data,
                context_data=context or {},
                start_time=datetime.utcnow(),
                status='running'
            )
            db.add(run_trace)
            db.commit()
        
        # Emit run start event
        self.emit_event(
            run_id=run_id,
            event_type='run_start',
            event_data={
                'agent_id': agent_id,
                'task_type': task_type,
                'input_size': len(str(input_data))
            }
        )
        
        self.current_run_id = run_id
        self.run_stack.append(run_id)
        
        return run_id
    
    def end_run(
        self,
        run_id: str,
        success: bool,
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """End tracking of an agent run."""
        end_time = datetime.utcnow()
        
        with get_db_session() as db:
            run_trace = db.query(RunTrace).filter(RunTrace.run_id == run_id).first()
            if run_trace:
                run_trace.end_time = end_time
                run_trace.duration_ms = int((end_time - run_trace.start_time).total_seconds() * 1000)
                run_trace.success = success
                run_trace.output_data = output_data or {}
                run_trace.error_message = error
                run_trace.metadata = metadata or {}
                run_trace.status = 'completed' if success else 'failed'
                db.commit()
        
        # Emit run end event
        self.emit_event(
            run_id=run_id,
            event_type='run_end',
            event_data={
                'success': success,
                'duration_ms': run_trace.duration_ms if run_trace else 0,
                'output_size': len(str(output_data)) if output_data else 0,
                'error': error
            }
        )
        
        # Update run stack
        if run_id in self.run_stack:
            self.run_stack.remove(run_id)
        
        if self.current_run_id == run_id:
            self.current_run_id = self.run_stack[-1] if self.run_stack else None
    
    def emit_event(
        self,
        run_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        step_name: Optional[str] = None
    ):
        """Emit a telemetry event for the current run."""
        with get_db_session() as db:
            event = RunEvent(
                run_id=run_id,
                event_type=event_type,
                step_name=step_name,
                event_data=event_data,
                timestamp=datetime.utcnow()
            )
            db.add(event)
            db.commit()
    
    def emit_step_start(self, step_name: str, step_data: Optional[Dict[str, Any]] = None):
        """Emit step start event for current run."""
        if not self.current_run_id:
            return
            
        self.emit_event(
            run_id=self.current_run_id,
            event_type='step_start',
            event_data=step_data or {},
            step_name=step_name
        )
    
    def emit_step_end(self, step_name: str, success: bool, step_data: Optional[Dict[str, Any]] = None):
        """Emit step end event for current run."""
        if not self.current_run_id:
            return
            
        self.emit_event(
            run_id=self.current_run_id,
            event_type='step_end',
            event_data={
                'success': success,
                **(step_data or {})
            },
            step_name=step_name
        )
    
    def emit_tool_usage(self, tool_name: str, input_data: Dict[str, Any], output_data: Dict[str, Any], cost: float = 0.0):
        """Emit tool usage event."""
        if not self.current_run_id:
            return
            
        self.emit_event(
            run_id=self.current_run_id,
            event_type='tool_usage',
            event_data={
                'tool_name': tool_name,
                'input_size': len(str(input_data)),
                'output_size': len(str(output_data)),
                'cost': cost
            }
        )
    
    def emit_context_usage(self, context_type: str, context_size: int, relevance_score: float = 0.0):
        """Emit context usage event."""
        if not self.current_run_id:
            return
            
        self.emit_event(
            run_id=self.current_run_id,
            event_type='context_usage',
            event_data={
                'context_type': context_type,
                'context_size': context_size,
                'relevance_score': relevance_score
            }
        )
    
    @contextmanager
    def track_run(self, agent_id: str, task_type: str, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        """Context manager for tracking a complete agent run."""
        run_id = self.start_run(agent_id, task_type, input_data, context)
        
        try:
            yield run_id
            self.end_run(run_id, success=True)
        except Exception as e:
            self.end_run(run_id, success=False, error=str(e))
            raise
    
    @contextmanager
    def track_step(self, step_name: str, step_data: Optional[Dict[str, Any]] = None):
        """Context manager for tracking a step within a run."""
        self.emit_step_start(step_name, step_data)
        
        try:
            yield
            self.emit_step_end(step_name, success=True, step_data=step_data)
        except Exception as e:
            self.emit_step_end(step_name, success=False, step_data={
                'error': str(e),
                **(step_data or {})
            })
            raise


# Global telemetry emitter instance
telemetry = TelemetryEmitter()


# Convenience functions for easy integration
def emit_run_start(agent_id: str, task_type: str, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    """Start tracking a new agent run."""
    return telemetry.start_run(agent_id, task_type, input_data, context)


def emit_run_end(run_id: str, success: bool, output_data: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    """End tracking of an agent run."""
    telemetry.end_run(run_id, success, output_data, error)


def emit_step_start(step_name: str, step_data: Optional[Dict[str, Any]] = None):
    """Emit step start event."""
    telemetry.emit_step_start(step_name, step_data)


def emit_step_end(step_name: str, success: bool, step_data: Optional[Dict[str, Any]] = None):
    """Emit step end event."""
    telemetry.emit_step_end(step_name, success, step_data)


def emit_tool_usage(tool_name: str, input_data: Dict[str, Any], output_data: Dict[str, Any], cost: float = 0.0):
    """Emit tool usage event."""
    telemetry.emit_tool_usage(tool_name, input_data, output_data, cost)


def emit_context_usage(context_type: str, context_size: int, relevance_score: float = 0.0):
    """Emit context usage event."""
    telemetry.emit_context_usage(context_type, context_size, relevance_score)


# Integration example:
"""
# In your agent execution code, add:
from services.telemetry_emitter import telemetry

# Method 1: Context manager (recommended)
with telemetry.track_run(agent_id="agent-007", task_type="document_analysis", input_data={"doc_id": "123"}):
    with telemetry.track_step("load_document"):
        # Your document loading code here
        pass
    
    with telemetry.track_step("analyze_content"):
        # Your analysis code here
        telemetry.emit_tool_usage("gpt-4", input_data, output_data, cost=0.05)

# Method 2: Manual tracking
run_id = emit_run_start("agent-007", "document_analysis", {"doc_id": "123"})
try:
    # Your agent code here
    emit_run_end(run_id, success=True, output_data={"analysis": "complete"})
except Exception as e:
    emit_run_end(run_id, success=False, error=str(e))
"""
