
"""
Run Telemetry Models - Database Schema for Agent Execution Tracking
Stores detailed telemetry data for pattern mining and performance analysis.
"""
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from database import Base


class RunTrace(Base):
    """
    Main table for tracking agent execution runs.
    Each row represents a complete agent execution from start to finish.
    """
    __tablename__ = "run_traces"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # Agent and task information
    agent_id = Column(String(255), index=True, nullable=False)
    task_type = Column(String(255), index=True, nullable=False)
    
    # Execution timing
    start_time = Column(DateTime, default=datetime.utcnow, index=True)
    end_time = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)  # Calculated duration in milliseconds
    
    # Execution status
    status = Column(String(50), default='running', index=True)  # running, completed, failed
    success = Column(Boolean, nullable=True, index=True)
    
    # Data payloads
    input_data = Column(JSON)  # Input data provided to agent
    output_data = Column(JSON)  # Output data produced by agent
    context_data = Column(JSON)  # Context/environment data
    metadata = Column(JSON)  # Additional metadata (costs, resource usage, etc.)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_agent_task_time', 'agent_id', 'task_type', 'start_time'),
        Index('idx_success_time', 'success', 'start_time'),
        Index('idx_task_success', 'task_type', 'success'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'run_id': self.run_id,
            'agent_id': self.agent_id,
            'task_type': self.task_type,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'status': self.status,
            'success': self.success,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'context_data': self.context_data,
            'metadata': self.metadata,
            'error_message': self.error_message
        }


class RunEvent(Base):
    """
    Detailed events within each agent run.
    Tracks step-by-step execution for pattern analysis.
    """
    __tablename__ = "run_events"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(255), index=True, nullable=False)
    
    # Event information
    event_type = Column(String(100), index=True, nullable=False)  # step_start, step_end, tool_usage, context_usage, etc.
    step_name = Column(String(255), index=True, nullable=True)  # Name of the step/operation
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Event data
    event_data = Column(JSON)  # Flexible data for each event type
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_run_event_time', 'run_id', 'event_type', 'timestamp'),
        Index('idx_run_step_time', 'run_id', 'step_name', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'run_id': self.run_id,
            'event_type': self.event_type,
            'step_name': self.step_name,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'event_data': self.event_data
        }


class PlaybookExecution(Base):
    """
    Tracks execution of discovered playbooks.
    Links playbooks to actual runs for performance measurement.
    """
    __tablename__ = "playbook_executions"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    playbook_id = Column(Integer, index=True, nullable=False)  # Links to playbooks table
    run_id = Column(String(255), index=True, nullable=False)  # Links to run_traces
    
    # Execution information
    executed_at = Column(DateTime, default=datetime.utcnow, index=True)
    success = Column(Boolean, nullable=True, index=True)
    duration_ms = Column(Integer, nullable=True)
    cost = Column(Float, default=0.0)
    
    # Performance metrics
    steps_completed = Column(Integer, default=0)
    steps_total = Column(Integer, default=0)
    efficiency_score = Column(Float, nullable=True)  # Compared to pattern average
    
    # Deviation analysis
    deviations = Column(JSON)  # Steps that deviated from playbook
    improvements = Column(JSON)  # Potential improvements identified
    
    # Indexes
    __table_args__ = (
        Index('idx_playbook_success_time', 'playbook_id', 'success', 'executed_at'),
        Index('idx_playbook_performance', 'playbook_id', 'efficiency_score'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'playbook_id': self.playbook_id,
            'run_id': self.run_id,
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'success': self.success,
            'duration_ms': self.duration_ms,
            'cost': self.cost,
            'steps_completed': self.steps_completed,
            'steps_total': self.steps_total,
            'efficiency_score': self.efficiency_score,
            'deviations': self.deviations,
            'improvements': self.improvements
        }


# Example event types for reference:
EVENT_TYPES = {
    'run_start': 'Agent run started',
    'run_end': 'Agent run completed',
    'step_start': 'Individual step started',
    'step_end': 'Individual step completed',
    'tool_usage': 'Tool/API called',
    'context_usage': 'Context/knowledge retrieved',
    'error': 'Error occurred',
    'decision': 'Decision point reached',
    'branch': 'Execution path chosen',
    'loop_start': 'Loop iteration started',
    'loop_end': 'Loop iteration completed',
    'checkpoint': 'Progress checkpoint'
}

# Example step names for reference:
COMMON_STEP_NAMES = {
    'load_document': 'Load and parse document',
    'analyze_content': 'Analyze document content',
    'extract_entities': 'Extract named entities',
    'generate_summary': 'Generate content summary',
    'classify_document': 'Classify document type',
    'search_knowledge': 'Search knowledge base',
    'retrieve_context': 'Retrieve relevant context',
    'format_response': 'Format final response',
    'validate_output': 'Validate output quality',
    'save_results': 'Save results to database'
}

# Example telemetry integration:
"""
# In your agent execution code:
from services.telemetry_emitter import telemetry

# Track complete agent run
with telemetry.track_run(
    agent_id="agent-007", 
    task_type="document_analysis", 
    input_data={"document_id": "doc123"},
    context={"user_id": "user456", "project": "legal_review"}
):
    # Track individual steps
    with telemetry.track_step("load_document"):
        document = load_document("doc123")
    
    with telemetry.track_step("analyze_content"):
        analysis = analyze_document(document)
        
        # Track tool usage within step
        telemetry.emit_tool_usage(
            tool_name="gpt-4",
            input_data={"text": document.content},
            output_data={"analysis": analysis},
            cost=0.05
        )
    
    with telemetry.track_step("save_results"):
        save_analysis(analysis)

# This automatically creates:
# 1. RunTrace record with timing, success, input/output data
# 2. Multiple RunEvent records for each step and tool usage
# 3. Pattern mining can then discover successful workflows
"""
