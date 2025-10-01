"""
Agent Execution Manager
=======================

Manages execution of workflow subtasks using selected agents.
Provides real-time WebSocket monitoring and result tracking.

PHASE 2 ENHANCED: Now includes inter-agent communication
- Message passing between agents
- Shared context for team coordination
- Real-time knowledge sharing
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from sqlalchemy.orm import Session

from services.agent_factory import AgentFactory, AgentRuntime, AgentMetadata
from database.models import Agent

# PHASE 2: Import communication components
try:
    from services.inter_agent_communication import (
        AgentCommunicationProtocol,
        SharedContextManager,
        MessageType
    )
    COMMUNICATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Inter-agent communication not available: {e}")
    COMMUNICATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class SubtaskStatus(Enum):
    """Subtask execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class SubtaskExecution:
    """Execution result for a single subtask"""
    subtask_id: str
    subtask_description: str
    agent_id: int
    agent_name: str
    status: SubtaskStatus
    llm_response: Optional[str] = None
    tokens_used: int = 0
    execution_time_ms: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    context_quality: float = 0.0
    prompt_used: str = ""


@dataclass
class ExecutionPlan:
    """Plan for executing workflow subtasks"""
    total_subtasks: int
    parallel_groups: List[List[str]]  # Groups of subtasks that can run in parallel
    dependencies: Dict[str, List[str]]  # subtask_id -> list of dependency subtask_ids
    estimated_duration_seconds: float


class AgentExecutionManager:
    """
    Manages execution of workflow subtasks with selected agents.
    
    Features:
    - Parallel execution where possible
    - Real-time WebSocket updates
    - Error handling and retries
    - Result aggregation
    - Performance tracking
    """
    
    def __init__(
        self,
        db_session: Session,
        agent_factory: Optional[AgentFactory] = None,
        max_parallel_executions: int = 3,
        max_retries: int = 2,
        enable_communication: bool = True  # PHASE 2
    ):
        self.db = db_session
        self.agent_factory = agent_factory or AgentFactory(db_session)
        self.max_parallel_executions = max_parallel_executions
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        
        # Execution tracking
        self.active_executions: Dict[str, SubtaskExecution] = {}
        self.completed_executions: Dict[str, SubtaskExecution] = {}
        self.websocket_manager = None  # Set externally
        
        # PHASE 2: Inter-agent communication
        self.enable_communication = enable_communication and COMMUNICATION_AVAILABLE
        if self.enable_communication:
            try:
                self.communication = AgentCommunicationProtocol()
                self.context_manager = SharedContextManager(db_session)
                self.logger.info("✅ Inter-agent communication ENABLED")
            except Exception as e:
                self.logger.warning(f"Failed to initialize communication: {e}")
                self.enable_communication = False
        else:
            self.communication = None
            self.context_manager = None
    
    async def execute_workflow_subtasks(
        self,
        subtasks: List[Dict[str, Any]],
        agent_assignments: Dict[str, Any],
        context_enhancements: Dict[str, Any],
        execution_id: int,
        workflow_id: int
    ) -> Dict[str, SubtaskExecution]:
        """
        Execute all workflow subtasks with assigned agents.
        
        Args:
            subtasks: List of subtasks from decomposer
            agent_assignments: Dict of agent selections per subtask
            context_enhancements: Dict of context enhancements per subtask
            execution_id: Workflow execution ID for tracking
            workflow_id: Workflow ID
            
        Returns:
            Dict mapping subtask_id to execution results
        """
        self.logger.info(f"🚀 Starting execution of {len(subtasks)} subtasks for workflow {workflow_id}")
        
        # PHASE 2: Create shared context for agent team
        shared_context = None
        if self.enable_communication and self.context_manager:
            try:
                # Get all agent IDs from assignments
                agent_ids = []
                for subtask_id, agent_match in agent_assignments.items():
                    if isinstance(agent_match, list) and agent_match:
                        agent_ids.append(agent_match[0].get("agent_id"))
                    elif isinstance(agent_match, dict):
                        agent_ids.append(agent_match.get("agent_id"))
                
                # Remove duplicates and None values
                agent_ids = list(set(filter(None, agent_ids)))
                
                if agent_ids:
                    shared_context = await self.context_manager.create_shared_context(
                        team=agent_ids,
                        initial_context={
                            "workflow_id": workflow_id,
                            "execution_id": execution_id,
                            "total_subtasks": len(subtasks),
                            "subtask_descriptions": [st.get("description", "") for st in subtasks]
                        }
                    )
                    self.logger.info(f"✅ Shared context created for {len(agent_ids)} agents")
            except Exception as e:
                self.logger.warning(f"Failed to create shared context: {e}")
        
        # 1. Create execution plan
        execution_plan = self._create_execution_plan(subtasks)
        
        # 2. Execute subtasks in parallel groups
        all_results = {}
        self.shared_context = shared_context  # Store for use in subtask execution
        
        for group_idx, parallel_group in enumerate(execution_plan.parallel_groups):
            self.logger.info(f"📦 Executing group {group_idx + 1}/{len(execution_plan.parallel_groups)} ({len(parallel_group)} subtasks)")
            
            # Execute this group in parallel
            group_tasks = []
            for subtask_id in parallel_group:
                subtask_idx = int(subtask_id.split("_")[1])
                subtask = subtasks[subtask_idx]
                
                # Get agent assignment
                agent_match = agent_assignments.get(subtask_id, {})
                if isinstance(agent_match, list) and agent_match:
                    agent_match = agent_match[0]  # First match
                
                # Get context enhancement
                context_enh = context_enhancements.get(subtask_id, {})
                
                # Create execution task
                task = self._execute_single_subtask(
                    subtask_id=subtask_id,
                    subtask=subtask,
                    agent_match=agent_match,
                    context_enh=context_enh,
                    execution_id=execution_id,
                    workflow_id=workflow_id
                )
                group_tasks.append(task)
            
            # Wait for all subtasks in this group to complete
            group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
            
            # Process results
            for subtask_id, result in zip(parallel_group, group_results):
                if isinstance(result, Exception):
                    self.logger.error(f"❌ Subtask {subtask_id} failed with exception: {result}")
                    all_results[subtask_id] = self._create_failed_execution(subtask_id, str(result))
                else:
                    all_results[subtask_id] = result
        
        self.logger.info(f"✅ Completed execution of {len(all_results)} subtasks")
        
        return all_results
    
    def _create_execution_plan(self, subtasks: List[Dict[str, Any]]) -> ExecutionPlan:
        """
        Create execution plan with parallelization strategy.
        
        For now: simple sequential plan (can be enhanced with dependency analysis)
        """
        
        # Simple strategy: execute in order, max N in parallel
        parallel_groups = []
        current_group = []
        
        for idx in range(len(subtasks)):
            subtask_id = f"subtask_{idx}"
            current_group.append(subtask_id)
            
            if len(current_group) >= self.max_parallel_executions:
                parallel_groups.append(current_group)
                current_group = []
        
        if current_group:
            parallel_groups.append(current_group)
        
        # Estimate duration
        total_duration = sum(
            self._parse_duration(st.get("estimated_duration", 30))
            for st in subtasks
        )
        
        return ExecutionPlan(
            total_subtasks=len(subtasks),
            parallel_groups=parallel_groups,
            dependencies={},  # No dependencies for now
            estimated_duration_seconds=total_duration
        )
    
    async def _execute_single_subtask(
        self,
        subtask_id: str,
        subtask: Dict[str, Any],
        agent_match: Dict[str, Any],
        context_enh: Dict[str, Any],
        execution_id: int,
        workflow_id: int
    ) -> SubtaskExecution:
        """Execute a single subtask with assigned agent"""
        
        description = subtask.get("description", subtask.get("name", "Unknown"))
        agent_id = agent_match.get("agent_id") if agent_match else None
        agent_name = agent_match.get("agent_name", "Unknown") if agent_match else "No Agent"
        
        self.logger.info(f"🔧 Executing subtask: {description[:50]}... with agent {agent_name}")
        
        # Create execution tracking
        execution = SubtaskExecution(
            subtask_id=subtask_id,
            subtask_description=description,
            agent_id=agent_id or 0,
            agent_name=agent_name,
            status=SubtaskStatus.RUNNING,
            start_time=datetime.now(),
            context_quality=context_enh.get("context_quality", 0.0) if context_enh else 0.0,
            prompt_used=context_enh.get("enhanced_prompt", description) if context_enh else description
        )
        
        self.active_executions[subtask_id] = execution
        
        # Send WebSocket update
        await self._broadcast_execution_update(execution, execution_id, workflow_id)
        
        # PHASE 2: Notify team that task is starting
        await self._notify_task_start(execution, subtask_id)
        
        # Execute with agent
        try:
            if not agent_id:
                raise Exception("No agent assigned to subtask")
            
            # Get enhanced prompt
            enhanced_prompt = execution.prompt_used
            
            # Execute with agent factory (with retries)
            result = await self._execute_with_retries(
                agent_id,
                enhanced_prompt,
                execution
            )
            
            # Update execution with result
            execution.status = SubtaskStatus.COMPLETED
            execution.llm_response = result.get("response", "")
            execution.tokens_used = result.get("tokens_used", 0)
            execution.end_time = datetime.now()
            execution.execution_time_ms = int(
                (execution.end_time - execution.start_time).total_seconds() * 1000
            )
            
            self.logger.info(
                f"✅ Subtask completed: {description[:50]}... "
                f"({execution.tokens_used} tokens, {execution.execution_time_ms}ms)"
            )
            
            # PHASE 2: Share result with team
            await self._share_result_with_team(execution, subtask_id)
            
        except Exception as e:
            self.logger.error(f"❌ Subtask failed: {description[:50]}... - {e}")
            
            execution.status = SubtaskStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.now()
            execution.execution_time_ms = int(
                (execution.end_time - execution.start_time).total_seconds() * 1000
            )
            
            # PHASE 2: Request help from team when task fails
            await self._request_help_from_team(execution, subtask_id, str(e))
        
        # Move to completed
        self.completed_executions[subtask_id] = execution
        del self.active_executions[subtask_id]
        
        # Send final WebSocket update
        await self._broadcast_execution_update(execution, execution_id, workflow_id)
        
        return execution
    
    async def _execute_with_retries(
        self,
        agent_id: int,
        prompt: str,
        execution: SubtaskExecution
    ) -> Dict[str, Any]:
        """Execute task with agent, with retry logic"""
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    execution.status = SubtaskStatus.RETRYING
                    execution.retry_count = attempt
                    self.logger.info(f"🔄 Retry {attempt}/{self.max_retries} for {execution.subtask_description[:30]}...")
                
                # Execute with agent factory
                result = await self.agent_factory.execute_with_prompt(
                    agent=agent_id,
                    prompt=prompt,
                    system_prompt="You are a helpful AI assistant. Complete the task accurately and concisely.",
                    use_memory=True,
                    max_retries=0  # We handle retries here
                )
                
                if result.get("status") == "error":
                    raise Exception(result.get("error", "Unknown error"))
                
                return result
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        # All retries failed
        raise Exception(f"Failed after {self.max_retries + 1} attempts: {last_error}")
    
    async def _broadcast_execution_update(
        self,
        execution: SubtaskExecution,
        execution_id: int,
        workflow_id: int
    ):
        """Send WebSocket update for subtask execution"""
        
        if not self.websocket_manager:
            return
        
        try:
            await self.websocket_manager.broadcast({
                "type": "subtask_execution_update",
                "data": {
                    "execution_id": execution_id,
                    "workflow_id": workflow_id,
                    "subtask_id": execution.subtask_id,
                    "subtask_description": execution.subtask_description[:100],
                    "agent_name": execution.agent_name,
                    "status": execution.status.value,
                    "tokens_used": execution.tokens_used,
                    "execution_time_ms": execution.execution_time_ms,
                    "error_message": execution.error_message,
                    "retry_count": execution.retry_count,
                    "timestamp": datetime.now().isoformat()
                }
            })
        except Exception as e:
            self.logger.error(f"Failed to broadcast execution update: {e}")
    
    def _parse_duration(self, duration_str: Any) -> float:
        """Parse duration string to seconds"""
        try:
            if isinstance(duration_str, (int, float)):
                return float(duration_str)
            if 'second' in str(duration_str).lower():
                parts = str(duration_str).split('-')
                if len(parts) == 2:
                    low = int(''.join(filter(str.isdigit, parts[0])))
                    high = int(''.join(filter(str.isdigit, parts[1])))
                    return (low + high) / 2
                return int(''.join(filter(str.isdigit, str(duration_str))))
            return 30.0
        except:
            return 30.0
    
    def _create_failed_execution(self, subtask_id: str, error: str) -> SubtaskExecution:
        """Create failed execution record"""
        return SubtaskExecution(
            subtask_id=subtask_id,
            subtask_description="Unknown",
            agent_id=0,
            agent_name="Unknown",
            status=SubtaskStatus.FAILED,
            error_message=error,
            execution_time_ms=0
        )
    
    def get_execution_summary(
        self,
        executions: Dict[str, SubtaskExecution]
    ) -> Dict[str, Any]:
        """Generate summary of all executions"""
        
        total = len(executions)
        completed = sum(1 for e in executions.values() if e.status == SubtaskStatus.COMPLETED)
        failed = sum(1 for e in executions.values() if e.status == SubtaskStatus.FAILED)
        
        total_tokens = sum(e.tokens_used for e in executions.values())
        total_time_ms = sum(e.execution_time_ms for e in executions.values())
        total_retries = sum(e.retry_count for e in executions.values())
        
        avg_context_quality = sum(
            e.context_quality for e in executions.values()
        ) / max(total, 1)
        
        return {
            "total_subtasks": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0,
            "total_tokens_used": total_tokens,
            "total_execution_time_ms": total_time_ms,
            "avg_execution_time_ms": total_time_ms / total if total > 0 else 0,
            "total_retries": total_retries,
            "avg_context_quality": avg_context_quality,
            "timestamp": datetime.now().isoformat()
        }
    
    # ======================================================================
    # PHASE 2: INTER-AGENT COMMUNICATION METHODS
    # ======================================================================
    
    async def _notify_task_start(
        self,
        execution: SubtaskExecution,
        subtask_id: str
    ):
        """Notify team that an agent is starting a task"""
        if not self.enable_communication or not self.communication:
            return
        
        try:
            # Broadcast task start to all team members
            await self.communication.broadcast(
                from_agent=execution.agent_id,
                message_type=MessageType.COORDINATION,
                content={
                    "event": "task_started",
                    "subtask_id": subtask_id,
                    "description": execution.subtask_description,
                    "agent_name": execution.agent_name,
                    "estimated_duration": "30-60 seconds"
                },
                priority=5
            )
            
            self.logger.debug(f"📢 Notified team: {execution.agent_name} starting {subtask_id}")
        except Exception as e:
            self.logger.warning(f"Failed to notify task start: {e}")
    
    async def _share_result_with_team(
        self,
        execution: SubtaskExecution,
        subtask_id: str
    ):
        """
        Share completed task result with team via inter-agent communication.
        This allows other agents to learn from and build upon this work.
        """
        if not self.enable_communication or not self.communication:
            return
        
        try:
            # 1. Broadcast result to team
            await self.communication.broadcast(
                from_agent=execution.agent_id,
                message_type=MessageType.RESULT_SHARE,
                content={
                    "subtask_id": subtask_id,
                    "description": execution.subtask_description,
                    "result": execution.llm_response[:500],  # First 500 chars
                    "status": execution.status.value,
                    "tokens_used": execution.tokens_used,
                    "execution_time_ms": execution.execution_time_ms,
                    "context_quality": execution.context_quality
                },
                priority=7
            )
            
            # 2. Update shared context with result
            if self.context_manager and hasattr(self, 'shared_context') and self.shared_context:
                await self.context_manager.update_context(
                    context_id=self.shared_context.context_id,
                    updates={
                        f"subtask_{subtask_id}_result": {
                            "response": execution.llm_response[:1000],
                            "status": execution.status.value,
                            "completed_at": datetime.now().isoformat()
                        }
                    }
                )
            
            self.logger.info(f"✅ Shared result with team: {subtask_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to share result with team: {e}")
    
    async def _request_help_from_team(
        self,
        execution: SubtaskExecution,
        subtask_id: str,
        error_message: str
    ):
        """
        Request help from team when a task fails.
        Other agents with relevant expertise might provide guidance.
        """
        if not self.enable_communication or not self.communication:
            return
        
        try:
            await self.communication.broadcast(
                from_agent=execution.agent_id,
                message_type=MessageType.TASK_REQUEST,
                content={
                    "event": "help_requested",
                    "subtask_id": subtask_id,
                    "description": execution.subtask_description,
                    "error": error_message,
                    "agent_name": execution.agent_name,
                    "retry_count": execution.retry_count
                },
                priority=8  # High priority for failures
            )
            
            self.logger.info(f"🆘 Requested help from team for {subtask_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to request help from team: {e}")
    
    async def _share_knowledge(
        self,
        agent_id: int,
        knowledge_type: str,
        content: Any
    ):
        """
        Share general knowledge with team (patterns, learnings, insights).
        """
        if not self.enable_communication or not self.communication:
            return
        
        try:
            await self.communication.broadcast(
                from_agent=agent_id,
                message_type=MessageType.KNOWLEDGE_SHARE,
                content={
                    "knowledge_type": knowledge_type,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                },
                priority=3  # Lower priority for general knowledge
            )
            
            self.logger.debug(f"💡 Shared knowledge: {knowledge_type}")
            
        except Exception as e:
            self.logger.warning(f"Failed to share knowledge: {e}")

