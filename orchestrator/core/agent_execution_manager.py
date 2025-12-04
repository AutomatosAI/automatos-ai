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
import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text

import numpy as np

from modules.agents import AgentFactory, AgentRuntime, AgentMetadata
from modules.agents.factory import get_action_executor
from models import Agent
from core.memory_prompt_injector import MemoryPromptInjector

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
    
    @staticmethod
    def _create_secure_temp_dir() -> str:
        """
        Create a secure temporary directory with proper permissions.
        
        Returns:
            Path to the created temporary directory
            
        Security features:
        - Unique per execution (tempfile.mkdtemp)
        - Restricted permissions (0o700)
        - Symlink validation
        - Owned by running user
        """
        # Create unique temporary directory
        temp_dir = tempfile.mkdtemp(prefix="automatos_workspace_")
        
        # Set restrictive permissions (0o700 = rwx------)
        os.chmod(temp_dir, 0o700)
        
        # Validate path is not a symlink (security check)
        if os.path.islink(temp_dir):
            # This shouldn't happen with mkdtemp, but check anyway
            raise ValueError(f"Temporary directory path is a symlink: {temp_dir}")
        
        # Resolve any symlinks in parent path
        real_path = os.path.realpath(temp_dir)
        if real_path != temp_dir:
            # Parent directory had symlinks, use resolved path
            temp_dir = real_path
        
        return temp_dir
    
    def __init__(
        self,
        db_session: Session,
        agent_factory: Optional[AgentFactory] = None,
        max_parallel_executions: int = 3,
        max_retries: int = 2,
        enable_communication: bool = True,  # PHASE 2
        workspace_dir: Optional[str] = None  # Unique workspace per execution
    ):
        self.db = db_session
        self.db_session = db_session  # FIX: Also store as db_session for compatibility
        self.agent_factory = agent_factory or AgentFactory(db_session)
        self.max_parallel_executions = max_parallel_executions
        self.max_retries = max_retries
        
        # Create secure temporary directory if not provided
        if workspace_dir is None:
            self.workspace_dir = self._create_secure_temp_dir()
            self._workspace_created_by_manager = True  # Track for cleanup
        else:
            # Validate provided workspace directory
            if not os.path.isabs(workspace_dir):
                raise ValueError(f"Workspace directory must be absolute path: {workspace_dir}")
            if os.path.islink(workspace_dir):
                raise ValueError(f"Workspace directory cannot be a symlink: {workspace_dir}")
            # Ensure directory exists with proper permissions
            os.makedirs(workspace_dir, mode=0o700, exist_ok=True)
            self.workspace_dir = workspace_dir
            self._workspace_created_by_manager = False
        
        self.logger = logging.getLogger(__name__)
        self.memory_injector = MemoryPromptInjector()  # Memory injection support
        
        # Execution tracking
        self.active_executions: Dict[str, SubtaskExecution] = {}
        self.completed_executions: Dict[str, SubtaskExecution] = {}
        self.websocket_manager = None  # Set externally
        
        # Log workspace info
        self.logger.info(f"📁 Agent execution workspace: {self.workspace_dir}")
        
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
                self.communication = None
                self.context_manager = None  # FIX: Always set context_manager
        else:
            self.communication = None
            self.context_manager = None
    
    def _cleanup_workspace(self):
        """
        Clean up the temporary workspace directory if it was created by this manager.
        
        This method is called automatically after execution completes or fails.
        Only cleans up directories created by _create_secure_temp_dir().
        """
        if hasattr(self, '_workspace_created_by_manager') and self._workspace_created_by_manager:
            try:
                if os.path.exists(self.workspace_dir):
                    # Validate it's still our temp directory (security check)
                    if self.workspace_dir.startswith(tempfile.gettempdir()):
                        shutil.rmtree(self.workspace_dir)
                        self.logger.info(f"🧹 Cleaned up temporary workspace: {self.workspace_dir}")
                    else:
                        self.logger.warning(f"⚠️  Skipping cleanup of non-temp directory: {self.workspace_dir}")
            except Exception as e:
                self.logger.error(f"❌ Failed to cleanup workspace {self.workspace_dir}: {e}", exc_info=True)
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion"""
        try:
            self._cleanup_workspace()
        except Exception:
            pass  # Ignore errors in destructor
    
    async def execute_workflow_subtasks(
        self,
        subtasks: List[Dict[str, Any]],
        agent_assignments: Dict[str, Any],
        context_enhancements: Dict[str, Any],
        execution_id: int,
        workflow_id: int,
        memory_retrieval_results: Optional[Dict[str, Any]] = None,
        execution_strategy: str = "parallel"  # Add strategy parameter
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
        
        try:
            # PHASE 2: Create shared context for agent team
            shared_context = None
            if self.enable_communication and self.context_manager:
                try:
                    # Get all agent IDs from assignments
                    agent_ids = []
                    for subtask_id, agent_match in agent_assignments.items():
                        if isinstance(agent_match, list) and agent_match:
                            agent_ids.append(agent_match[0].agent_id if hasattr(agent_match[0], 'agent_id') else agent_match[0].get("agent_id"))
                        elif isinstance(agent_match, dict):
                            agent_ids.append(agent_match.get("agent_id"))
                        elif hasattr(agent_match, 'agent_id'):
                            agent_ids.append(agent_match.agent_id)
                    
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
            
            # 1. Create execution plan based on strategy
            execution_plan = self._create_execution_plan(subtasks, execution_strategy)
            
            # 2. Execute subtasks based on strategy
            all_results = {}
            self.shared_context = shared_context  # Store for use in subtask execution
            
            # Store results for memory sharing between sequential tasks
            if execution_strategy == "sequential":
                self.execution_memory = {}  # Store outputs for next task
            
            for group_idx, parallel_group in enumerate(execution_plan.parallel_groups):
                if execution_strategy == "sequential":
                    self.logger.info(f"📦 Executing task {group_idx + 1}/{len(execution_plan.parallel_groups)} sequentially")
                else:
                    self.logger.info(f"📦 Executing group {group_idx + 1}/{len(execution_plan.parallel_groups)} ({len(parallel_group)} subtasks)")
                
                # Execute this group (1 task if sequential, multiple if parallel)
                group_tasks = []
                for subtask_id in parallel_group:
                    subtask_idx = int(subtask_id.split("_")[1])
                    subtask = subtasks[subtask_idx]
                    
                    # Get agent assignment
                    agent_match = agent_assignments.get(subtask_id, {})
                    self.logger.info(f"🔍 DEBUG: Raw agent_match for {subtask_id}: type={type(agent_match)}, value={agent_match}")
                    if isinstance(agent_match, list) and agent_match:
                        agent_match = agent_match[0]  # First match
                        self.logger.info(f"🔍 DEBUG: After list extraction: type={type(agent_match)}, value={agent_match}")
                    
                    # Get context enhancement
                    context_enh = context_enhancements.get(subtask_id, {})
                    
                    # Create execution task
                    task = self._execute_single_subtask(
                        subtask_id=subtask_id,
                        subtask=subtask,
                        agent_match=agent_match,
                        context_enh=context_enh,
                        execution_id=execution_id,
                        workflow_id=workflow_id,
                        memory_retrieval_results=memory_retrieval_results
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
                    
                    # ✨ REAL-TIME UPDATE: Save to database immediately after each subtask
                    await self._update_execution_output_data(execution_id, subtasks, all_results)
            
            self.logger.info(f"✅ Completed execution of {len(all_results)} subtasks")
            
            return all_results
        finally:
            # Cleanup temporary workspace if we created it (on completion or failure)
            self._cleanup_workspace()
    
    def _create_execution_plan(self, subtasks: List[Dict[str, Any]], execution_strategy: str = "parallel") -> ExecutionPlan:
        """
        Create execution plan based on execution strategy.
        PHASE 4A: Enhanced with graph-based scheduling using dependency analysis.
        
        Args:
            subtasks: List of subtasks to execute
            execution_strategy: "sequential" or "parallel"
        """
        
        parallel_groups = []
        dependencies = {}
        
        # PHASE 4A: Check if decomposition included graph analysis
        graph_analysis = None
        if subtasks and isinstance(subtasks, list) and len(subtasks) > 0:
            # Look for graph_analysis in the parent decomposition metadata
            # (This would be passed from orchestrator_service)
            for subtask in subtasks:
                if 'graph_metadata' in subtask:
                    graph_analysis = subtask['graph_metadata']
                    break
        
        if execution_strategy == "sequential":
            # Sequential: each task in its own group
            for idx in range(len(subtasks)):
                subtask_id = f"subtask_{idx}"
                parallel_groups.append([subtask_id])
                if idx < len(subtasks):
                    task_desc = subtasks[idx].get('description', '')[:50] if idx < len(subtasks) else ''
                    self.logger.info(f"📝 Task {idx + 1} will execute sequentially: {task_desc}")
        elif graph_analysis and 'parallel_groups' in graph_analysis:
            # PHASE 4A: Use graph-optimized parallel groups from decomposition
            self.logger.info("📊 Using graph-based execution scheduling")
            parallel_groups = graph_analysis['parallel_groups']
            
            # Build dependency map
            for i, subtask in enumerate(subtasks):
                subtask_id = f"subtask_{i}"
                deps = subtask.get('dependencies', [])
                if deps:
                    # Map dependency IDs to indices
                    dep_indices = []
                    for dep in deps:
                        for j, st in enumerate(subtasks):
                            if st.get('subtask_id') == dep or st.get('id') == dep:
                                dep_indices.append(f"subtask_{j}")
                    dependencies[subtask_id] = dep_indices
            
            self.logger.info(f"  📈 Parallel Levels: {len(parallel_groups)}")
            self.logger.info(f"  ⚡ Max Parallelism: {graph_analysis.get('max_parallelism', 1)}")
            
        else:
            # Fallback: Simple parallel grouping
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
            dependencies=dependencies,
            estimated_duration_seconds=total_duration
        )
    
    async def _execute_single_subtask(
        self,
        subtask_id: str,
        subtask: Dict[str, Any],
        agent_match: Dict[str, Any],
        context_enh: Dict[str, Any],
        execution_id: int,
        workflow_id: int,
        memory_retrieval_results: Optional[Dict[str, Any]] = None
    ) -> SubtaskExecution:
        """Execute a single subtask with assigned agent"""
        
        description = subtask.get("description", subtask.get("name", "Unknown"))
        
        # Enhanced Debug Logging for Agent Extraction
        self.logger.info(f"🔍 DEBUG: Extracting agent from agent_match (type: {type(agent_match)})")
        
        # Handle different agent_match types
        if hasattr(agent_match, 'agent_id'):  # AgentMatch object
            agent_id = agent_match.agent_id
            agent_name = agent_match.agent_name
            self.logger.info(f"🔍 DEBUG: Extracted from AgentMatch object: agent_id={agent_id}, agent_name='{agent_name}'")
        elif isinstance(agent_match, dict):
            agent_id = agent_match.get("agent_id")
            agent_name = agent_match.get("agent_name", "Unknown")
            self.logger.info(f"🔍 DEBUG: Extracted from dict: agent_id={agent_id}, agent_name='{agent_name}'")
            self.logger.info(f"🔍 DEBUG: agent_match keys: {list(agent_match.keys())}")
        else:
            agent_id = None
            agent_name = "No Agent"
            self.logger.error(f"❌ ERROR: Unrecognized agent_match type: {type(agent_match)}")
        
        if not agent_id:
            self.logger.error(
                f"❌ CRITICAL: No agent_id extracted for subtask! "
                f"agent_match type: {type(agent_match)}, "
                f"agent_name: '{agent_name}'"
            )
        
        self.logger.info(f"🔧 Executing subtask: {description}")
        self.logger.info(f"   Agent: {agent_name} (ID: {agent_id})")
        
        # Create execution tracking
        # Handle context_enh as dataclass or dict
        self.logger.info(f"🔍 DEBUG: Processing context enhancement (provided: {context_enh is not None})")
        
        if context_enh:
            if hasattr(context_enh, 'context_quality_score'):  # ContextEnhancement object
                context_quality = context_enh.context_quality_score
                prompt_used = context_enh.enhanced_prompt
                self.logger.info(
                    f"🔍 DEBUG: Context from ContextEnhancement object: "
                    f"quality={context_quality:.2%}, "
                    f"prompt_length={len(prompt_used)}"
                )
            else:  # dict
                context_quality = context_enh.get("context_quality", 0.0)
                prompt_used = context_enh.get("enhanced_prompt", description)
                self.logger.info(
                    f"🔍 DEBUG: Context from dict: "
                    f"quality={context_quality:.2%}, "
                    f"prompt_length={len(prompt_used)}, "
                    f"keys={list(context_enh.keys())}"
                )
        else:
            context_quality = 0.0
            prompt_used = description
            self.logger.warning(
                f"⚠️  WARNING: No context enhancement provided! "
                f"context_quality will be 0%, using raw description as prompt"
            )
        
        # Inject memory into the prompt if available
        self.logger.info(
            f"🔍 DEBUG: Memory injection check: "
            f"memory_retrieval_results={'provided' if memory_retrieval_results else 'none'}, "
            f"agent_id={agent_id}"
        )
        
        if memory_retrieval_results and agent_id:
            # Try both int and str keys (memory system uses int keys)
            agent_memories_dict = memory_retrieval_results.get('results', {}).get('agent_memories', {})
            memories = agent_memories_dict.get(agent_id) or agent_memories_dict.get(str(agent_id))
            
            if memories:
                self.logger.info(
                    f"✅ Injecting {memories.get('count', 0)} memories for agent {agent_id}"
                )
            else:
                self.logger.warning(
                    f"⚠️  Memory retrieval results exist but no memories found for agent_id={agent_id}"
                )
                self.logger.debug(f"   Available agent IDs in memories: {list(agent_memories_dict.keys())}")
            
            prompt_used = self.memory_injector.inject_memory_into_prompt(
                original_prompt=prompt_used,
                agent_id=agent_id,
                memory_retrieval_results=memory_retrieval_results,
                subtask_id=subtask_id
            )
        elif not memory_retrieval_results:
            self.logger.warning("⚠️  No memory retrieval results available")
        elif not agent_id:
            self.logger.warning("⚠️  Cannot inject memories: agent_id is missing")
        
        # Inject previous subtask results from shared context (for sequential dependencies)
        if self.context_manager and hasattr(self, 'shared_context') and self.shared_context:
            try:
                # CRITICAL FIX: Refresh context from database to get latest results from previous groups
                fresh_context = await self.context_manager.get_shared_context(self.shared_context.id)
                if fresh_context:
                    context_data = fresh_context.context_data or {}
                    self.logger.info(f"🔄 Refreshed shared context from database: {len(context_data)} keys")
                else:
                    # Fallback to cached context if refresh fails
                    context_data = self.shared_context.context_data or {}
                    self.logger.warning("⚠️  Failed to refresh context, using cached version")
                
                previous_results = []
                
                # DEBUG: Log all keys to see what's actually stored
                self.logger.info(f"🔍 DEBUG: Shared context keys: {list(context_data.keys())}")
                
                # Look for results from previous subtasks
                for key, value in context_data.items():
                    self.logger.info(f"🔍 DEBUG: Checking key '{key}' - type: {type(value)}")
                    if key.startswith("subtask_") and key.endswith("_result"):
                        # Handle both dict and string responses
                        if isinstance(value, dict):
                            # Don't truncate - provide full context for synthesis
                            result_content = value.get('response', 'No response')
                            result_length = len(result_content)
                        else:
                            result_content = str(value)
                            result_length = len(result_content)
                        
                        previous_results.append({
                            'key': key,
                            'content': result_content,
                            'length': result_length
                        })
                        self.logger.info(f"  📦 Found result: {key} ({result_length} chars)")
                
                if previous_results:
                    self.logger.info(f"📨 Injecting {len(previous_results)} previous task results into prompt")
                    context_section = "\n\n" + "="*80 + "\n"
                    context_section += "🔥 RESEARCH FINDINGS FROM YOUR TEAM - BASE YOUR WORK ON THIS!\n"
                    context_section += "="*80 + "\n\n"
                    context_section += "Your teammates have already completed research tasks. Their findings are below.\n"
                    context_section += "⚠️  CRITICAL INSTRUCTIONS:\n"
                    context_section += "- Use the ACTUAL findings below (not generic knowledge)\n"
                    context_section += "- Quote specific facts, code examples, and sources from the research\n"
                    context_section += "- Cite which subtask each piece of information came from\n"
                    context_section += "- Do NOT make up information or use general knowledge when specific research exists\n"
                    context_section += "- If research is incomplete, use tools to fill gaps (search_knowledge, etc.)\n\n"
                    context_section += "="*80 + "\n\n"
                    
                    # Format each previous result clearly
                    for idx, result in enumerate(previous_results, 1):
                        subtask_num = result['key'].replace('subtask_', '').replace('_result', '')
                        context_section += f"📋 FINDING #{idx} (from subtask_{subtask_num}):\n"
                        context_section += "─" * 80 + "\n"
                        context_section += result['content'] + "\n"
                        context_section += "─" * 80 + "\n\n"
                    
                    context_section += "="*80 + "\n"
                    context_section += "END OF TEAM RESEARCH - Now complete your task using the findings above.\n"
                    context_section += "="*80 + "\n\n"
                    
                    prompt_used = prompt_used + context_section
                else:
                    self.logger.info(f"ℹ️  No previous results found in shared context (checked {len(context_data)} keys)")
            except Exception as e:
                self.logger.warning(f"Failed to inject previous results: {e}")
            
        # Enhanced Debug Logging for Agent ID
        self.logger.info(
            f"🔍 DEBUG: Creating SubtaskExecution for {subtask_id}: "
            f"agent_id={agent_id} (type: {type(agent_id)}), "
            f"agent_name='{agent_name}'"
        )
        if not agent_id or agent_id == 0:
            self.logger.warning(
                f"⚠️  WARNING: agent_id is {agent_id} for {subtask_id}! "
                f"This will prevent memory consolidation."
            )
        
        execution = SubtaskExecution(
            subtask_id=subtask_id,
            subtask_description=description,
            agent_id=agent_id or 0,
            agent_name=agent_name,
            status=SubtaskStatus.RUNNING,
            start_time=datetime.now(),
            context_quality=context_quality,
            prompt_used=prompt_used
        )
        
        self.logger.info(
            f"✅ SubtaskExecution created: agent_id={execution.agent_id}, "
            f"agent_name='{execution.agent_name}'"
        )
        
        self.active_executions[subtask_id] = execution
        
        # Send WebSocket update
        await self._broadcast_execution_update(execution, execution_id, workflow_id)
        # Force event loop to yield and flush WebSocket immediately
        await asyncio.sleep(0)
        
        # PHASE 2: Notify team that task is starting
        await self._notify_task_start(execution, subtask_id)
        
        # Execute with agent
        try:
            if not agent_id:
                raise Exception("No agent assigned to subtask")
            
            # 🔧 FIX: Load agent with skills and tool assignments from database
            agent = self.db.query(Agent).options(
                joinedload(Agent.skills),
                joinedload(Agent.tool_assignments)
            ).filter(Agent.id == agent_id).first()
            
            if not agent:
                raise Exception(f"Agent {agent_id} not found in database")
            
            # Get enhanced prompt
            enhanced_prompt = execution.prompt_used
            
            # PRD-17: Extract required_tools from subtask
            # Map skills_required to tool categories (skills come from LLM, required_tools is what we need)
            skills = subtask.get("skills_required", subtask.get("skills", []))
            if isinstance(skills, str):
                skills = [skills]
            
            # If subtask has explicit required_tools, use it; otherwise map from skills
            if "required_tools" in subtask:
                required_tools = subtask.get("required_tools", [])
                if not isinstance(required_tools, list):
                    required_tools = [required_tools] if required_tools else []
            else:
                # Map common skills to tool categories
                required_tools = []
                skill_to_tool_map = {
                    "research": "research",
                    "file_ops": "file_ops",
                    "file_operations": "file_ops",
                    "shell": "shell",
                    "shell_commands": "shell",
                    "mcp": "mcp",
                    "database": "database"
                }
                for skill in skills:
                    skill_lower = skill.lower().replace(" ", "_").replace("-", "_")
                    if skill_lower in skill_to_tool_map:
                        required_tools.append(skill_to_tool_map[skill_lower])
            
            # 🦸 PRD-22 FIX: Extract tools from agent's ACTUAL skills (tools_schema)
            agent_skill_tool_names = []
            if agent.skills:
                for skill in agent.skills:
                    if skill.tools_schema and isinstance(skill.tools_schema, dict):
                        skill_tools = skill.tools_schema.get('tools', [])
                        for tool_def in skill_tools:
                            tool_name = tool_def.get('name')
                            if tool_name:
                                agent_skill_tool_names.append(tool_name)
                
                if agent_skill_tool_names:
                    self.logger.info(f"🦸 Agent {agent_id} has {len(agent_skill_tool_names)} skill tools: {agent_skill_tool_names}")
            
            # 🔧 PRD-20 FIX: Extract MCP tools from agent's tool assignments
            agent_mcp_tool_names = []
            if agent.tool_assignments:
                for assignment in agent.tool_assignments:
                    if assignment.tool and assignment.tool.name:
                        agent_mcp_tool_names.append(assignment.tool.name)
                
                if agent_mcp_tool_names:
                    self.logger.info(f"🔧 Agent {agent_id} has {len(agent_mcp_tool_names)} MCP tools: {agent_mcp_tool_names}")
            
            # Merge all tools together: subtask tools + agent skill tools + agent MCP tools
            all_tools = set(required_tools)
            all_tools.update(agent_skill_tool_names)
            all_tools.update(agent_mcp_tool_names)
            required_tools = list(all_tools)
            
            # Default to research only if agent has absolutely NO tools
            if not required_tools:
                required_tools = ["research"]
                self.logger.warning(f"⚠️  Agent {agent_id} has no tools, defaulting to research")
            else:
                self.logger.info(f"🛠️  Agent {agent_id} executing with {len(required_tools)} tools: {required_tools}")
            
            # Execute with agent factory (with retries)
            result = await self._execute_with_retries(
                agent_id,
                enhanced_prompt,
                execution,
                required_tools
            )
            
            # Update execution with result
            execution.status = SubtaskStatus.COMPLETED
            execution.llm_response = result.get("result", "")  # agent_factory returns "result" not "response"
            execution.tokens_used = result.get("execution", {}).get("tokens_used", 0)
            execution.end_time = datetime.now()
            execution.execution_time_ms = int(
                (execution.end_time - execution.start_time).total_seconds() * 1000
            )
            
            self.logger.info(f"✅ Subtask completed: {description}")
            self.logger.info(f"   Tokens: {execution.tokens_used}, Duration: {execution.execution_time_ms}ms")
            if execution.llm_response:
                result_preview = execution.llm_response[:200] + "..." if len(execution.llm_response) > 200 else execution.llm_response
                self.logger.debug(f"   Result: {result_preview}")
            
            # PHASE 2: Share result with team
            await self._share_result_with_team(execution, subtask_id)
            
        except Exception as e:
            self.logger.error(f"❌ Subtask failed: {description}")
            self.logger.error(f"   Error: {e}")
            
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
        # Force event loop to yield and flush WebSocket immediately
        await asyncio.sleep(0)
        
        # ✨ CRITICAL: Update database immediately with THIS subtask's result
        await self._update_single_subtask_in_db(
            execution_id=execution_id,
            subtask_id=subtask_id,
            subtask=subtask,
            result=execution
        )
        
        return execution
    
    async def _execute_with_retries(
        self,
        agent_id: int,
        prompt: str,
        execution: SubtaskExecution,
        required_tools: List[str] = None
    ) -> Dict[str, Any]:
        """Execute task with agent, with retry logic"""
        
        if required_tools is None:
            required_tools = ["research"]  # Default
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    execution.status = SubtaskStatus.RETRYING
                    execution.retry_count = attempt
                    self.logger.info(f"🔄 Retry {attempt}/{self.max_retries} for: {execution.subtask_description}")
                
                # Execute with agent factory - include action_executor for tool calls
                action_executor = get_action_executor()
                
                # PRD-21: ENHANCED SYSTEM PROMPT - Confident, Professional, Tool-Using Agent
                professional_system_prompt = """You are a professional AI agent with full access to the codebase, documentation, and tools.

🎯 CRITICAL INSTRUCTIONS:
- YOU HAVE TOOLS: Use search_knowledge, search_codebase, read_file, list_directory to find information
- BE CONFIDENT: State findings authoritatively based on evidence from tools
- BE COMPLETE: Finish the ENTIRE task, don't leave placeholders or "to be continued"
- USE CODE EVIDENCE: Include specific file paths, line numbers, and code snippets
- NO APOLOGIES: Never say "I'm sorry", "unfortunately", or "I don't have access to"
- NO META-COMMENTARY: Don't talk about your process, just deliver the result

❌ ABSOLUTELY FORBIDDEN:
- "I'm sorry, but as an AI..."
- "Unfortunately, without access to..."
- "I'm unable to..."
- "To be continued..."
- "Please let me know if..."
- "You could use..."
- Any incomplete or draft outputs

✅ REQUIRED APPROACH:
1. If you need information: USE YOUR TOOLS (search_codebase, read_file, etc.)
2. Find concrete evidence from the codebase
3. Write confidently: "The platform implements..." not "It appears that..."
4. Include code snippets with file paths
5. Complete the FULL deliverable - no drafts

REMEMBER: You are a PROFESSIONAL delivering a FINAL PRODUCT, not a draft."""
                
                # PRD-22: Build agent prompt with skills BEFORE calling execute_with_prompt
                # Get agent record with skills
                agent_record = (
                    self.db_session.query(Agent)
                    .options(joinedload(Agent.skills))
                    .filter(Agent.id == agent_id)
                    .first()
                )
                
                # Build system prompt with skills and extract tool schemas
                full_system_prompt = professional_system_prompt
                all_required_tools = list(required_tools or [])
                skill_tool_schemas = []
                if agent_record and agent_record.skills:
                    skill_enhanced_prompt, skill_tool_schemas = self.agent_factory._build_agent_system_prompt(
                        agent=agent_record,
                        task_context=prompt[:500],
                        db=self.db_session,
                        required_tools=required_tools,
                        enable_smart_skill_selection=True
                    )
                    full_system_prompt = skill_enhanced_prompt + "\n\n" + professional_system_prompt

                if skill_tool_schemas:
                    skill_tool_names = [t['function']['name'] for t in skill_tool_schemas]
                    all_required_tools.extend(skill_tool_names)
                    self.logger.info(f"✅ Added {len(skill_tool_names)} skill tools: {skill_tool_names}")

                # Append workspace and deliverable guidance for every agent
                workspace_guidance_lines = [
                    "## Workspace & Deliverable Rules",
                    f"- Workspace root: {self.workspace_dir}",
                    "- Keep every file operation inside this workspace (use relative paths).",
                    "- Persist all artifacts with write_file (or the appropriate skill tool) so the orchestrator can save them.",
                    "- In your final answer, list the relative paths of every file you created or updated."
                ]

                conversion_tools = {
                    "create_pdf": "pdf",
                    "create_docx": "docx",
                    "create_pptx": "pptx",
                    "create_xlsx": "xlsx"
                }
                base_stub = execution.subtask_id.replace("subtask_", "task")

                for tool_name, extension in conversion_tools.items():
                    if tool_name in all_required_tools:
                        source_name = f"{base_stub}.md"
                        output_name = f"{base_stub}.{extension}"
                        workspace_guidance_lines.extend([
                            "",
                            f"- This task includes the `{tool_name}` capability. Follow this workflow:",
                            f"  1. Draft the complete deliverable into `{source_name}` using `write_file`.",
                            f"  2. Call `{tool_name}` with {{\"source_file\": \"{source_name}\", \"output_file\": \"{output_name}\", \"title\": \"{execution.subtask_description}\"}}.",
                            f"  3. Verify the tool succeeds and report the saved file `{output_name}`."
                        ])

                full_system_prompt = full_system_prompt + "\n\n" + "\n".join(workspace_guidance_lines)

                result = await self.agent_factory.execute_with_prompt(
                    agent=agent_id,
                    prompt=prompt,
                    system_prompt=full_system_prompt,
                    use_memory=True,
                    max_retries=0,  # We handle retries here
                    action_executor=action_executor,  # Enable platform tools
                    required_tools=all_required_tools,  # Platform + Skill tools
                    workspace_dir=self.workspace_dir  # Unique workspace per execution
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
        """
        Broadcast subtask update via SSE (PRD-28 - replaced Redis/WebSocket).
        Direct streaming to frontend with < 500ms latency.
        """
        
        self.logger.info(f"🔔 Broadcasting SSE update for subtask {execution.subtask_id} - status: {execution.status.value}")
        
        try:
            from services.workflow_streaming_service import get_stream_manager
            
            manager = get_stream_manager()
            
            # Only broadcast if there are active SSE streams
            if not manager.has_active_streams(execution_id):
                self.logger.debug(f"⏭️ No active SSE streams for execution {execution_id}, skipping broadcast")
                return
            
            # Prepare message payload (FULL data, no truncation - PRD-28)
            message_data = {
                "subtask_id": execution.subtask_id,
                "subtask_description": execution.subtask_description,  # FULL description
                "agent_name": execution.agent_name,
                "status": execution.status.value,
                "tokens_used": execution.tokens_used,
                "execution_time_ms": execution.execution_time_ms,
                "error_message": execution.error_message,
                "retry_count": execution.retry_count,
                "result_preview": execution.llm_response[:500] if execution.llm_response and len(execution.llm_response) > 500 else execution.llm_response,
                "result_truncated": len(execution.llm_response) > 500 if execution.llm_response else False,
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast via SSE (non-blocking, immediate delivery)
            await manager.broadcast_event(
                execution_id=execution_id,
                event_type="subtask_update",
                data=message_data
            )
            
            stream_count = manager.get_active_stream_count(execution_id)
            self.logger.debug(f"✅ SSE broadcast sent to {stream_count} client(s) for subtask {execution.subtask_id}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to broadcast SSE update: {e}", exc_info=True)
    
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
    
    async def _update_single_subtask_in_db(
        self,
        execution_id: int,
        subtask_id: str,
        subtask: Dict[str, Any],
        result: SubtaskExecution
    ):
        """Update a SINGLE subtask in database immediately after completion"""
        try:
            from database.database import get_db_session
            from models import WorkflowExecution
            
            with get_db_session() as db:
                execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.id == execution_id
                ).first()
                
                if not execution:
                    self.logger.warning(f"Execution {execution_id} not found for update")
                    return
                
                # Initialize output_data if needed
                if execution.output_data is None:
                    execution.output_data = {}
                if "subtasks" not in execution.output_data:
                    execution.output_data["subtasks"] = []
                
                # Find or create this subtask in the list
                subtask_index = int(subtask_id.split("_")[1])
                subtasks_list = execution.output_data["subtasks"]
                
                # Extend list if needed
                while len(subtasks_list) <= subtask_index:
                    subtasks_list.append({})
                
                # Update this specific subtask (only serialize JSON-safe fields)
                subtasks_list[subtask_index] = {
                    "subtask_id": subtask.get("subtask_id"),
                    "description": subtask.get("description"),
                    "agent_type": subtask.get("agent_type"),
                    "priority": subtask.get("priority"),
                    "required_tools": subtask.get("required_tools"),
                    "dependencies": subtask.get("dependencies"),
                    "context_quality": subtask.get("context_quality", 0),
                    "context_type": subtask.get("context_type"),
                    "selected_agent": subtask.get("selected_agent"),
                    "execution_result": {
                        "status": result.status.value,
                        "llm_response": result.llm_response,
                        "tokens_used": result.tokens_used,
                        "execution_time_ms": result.execution_time_ms
                    },
                    "completed": (result.status == SubtaskStatus.COMPLETED)
                }
                
                execution.output_data["subtasks"] = subtasks_list
                
                # Mark as modified for JSONB update
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(execution, "output_data")
                
                # Commit to database
                db.commit()
                self.logger.info(f"🔥 LIVE UPDATE: Execution {execution_id}, subtask {subtask_id} - {result.tokens_used} tokens")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update single subtask: {e}", exc_info=True)
    
    async def _update_execution_output_data(
        self,
        execution_id: int,
        subtasks: List[Dict[str, Any]],
        results: Dict[str, SubtaskExecution]
    ):
        """Update execution output_data in database with latest subtask results"""
        try:
            from database.database import get_db_session
            from models import WorkflowExecution
            
            with get_db_session() as db:
                execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.id == execution_id
                ).first()
                
                if not execution:
                    self.logger.warning(f"Execution {execution_id} not found for update")
                    return
                
                # Update subtasks with execution results (only JSON-safe fields)
                updated_subtasks = []
                for idx, subtask in enumerate(subtasks):
                    subtask_id = f"subtask_{idx}"
                    result = results.get(subtask_id)
                    
                    # Copy only JSON-serializable subtask data
                    updated_subtask = {
                        "subtask_id": subtask.get("subtask_id"),
                        "description": subtask.get("description"),
                        "agent_type": subtask.get("agent_type"),
                        "priority": subtask.get("priority"),
                        "required_tools": subtask.get("required_tools"),
                        "dependencies": subtask.get("dependencies"),
                        "context_quality": subtask.get("context_quality", 0),
                        "context_type": subtask.get("context_type"),
                        "selected_agent": subtask.get("selected_agent")
                    }
                    
                    # Add execution result if available
                    if result:
                        updated_subtask["execution_result"] = {
                            "status": result.status.value,
                            "llm_response": result.llm_response,
                            "tokens_used": result.tokens_used,
                            "execution_time_ms": result.execution_time_ms
                        }
                        updated_subtask["completed"] = (result.status == SubtaskStatus.COMPLETED)
                    
                    updated_subtasks.append(updated_subtask)
                
                # Update output_data with latest subtasks
                if execution.output_data is None:
                    execution.output_data = {}
                execution.output_data["subtasks"] = updated_subtasks
                
                # Commit to database
                db.commit()
                self.logger.info(f"✅ Updated execution {execution_id} with {len(results)} completed subtasks")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update execution output_data: {e}")
    
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
            from services.inter_agent_communication import Message
            message = Message(
                from_agent_id=execution.agent_id,
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
            await self.communication.broadcast(
                from_agent=execution.agent_id,
                team=self.shared_context.team_agents if self.shared_context else [],
                message=message
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
        
        CRITICAL: Always updates shared_context even if communication fails!
        """
        try:
            # 1. CRITICAL: Always store result in shared context (even if communication fails)
            if self.context_manager and hasattr(self, 'shared_context') and self.shared_context:
                result_key = f"{subtask_id}_result"
                await self.context_manager.update_shared_context(
                    context_id=self.shared_context.id,
                    agent=execution.agent_id,
                    updates={
                        result_key: {
                            "response": execution.llm_response[:5000],  # Store up to 5000 chars
                            "status": execution.status.value,
                            "completed_at": datetime.now().isoformat()
                        }
                    },
                    merge_strategy="override"
                )
                self.logger.info(f"📝 Updated shared context with key: {result_key} ({len(execution.llm_response or '')} chars)")
            
            # 2. Optional: Broadcast result to team (only if communication enabled)
            if self.enable_communication and self.communication:
                from services.inter_agent_communication import Message
                message = Message(
                    from_agent_id=execution.agent_id,
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
                await self.communication.broadcast(
                    from_agent=execution.agent_id,
                    team=self.shared_context.team_agents if self.shared_context else [],
                    message=message
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
            from services.inter_agent_communication import Message
            message = Message(
                from_agent_id=execution.agent_id,
                message_type=MessageType.TASK_REQUEST,
                content={
                    "event": "help_requested",
                    "subtask_id": subtask_id,
                    "description": execution.subtask_description,
                    "error": error_message,
                    "agent_name": execution.agent_name,
                    "retry_count": execution.retry_count
                },
                priority=8,  # High priority for failures
                requires_ack=True
            )
            await self.communication.broadcast(
                from_agent=execution.agent_id,
                team=self.shared_context.team_agents if self.shared_context else [],
                message=message
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
            from services.inter_agent_communication import Message
            message = Message(
                from_agent_id=agent_id,
                message_type=MessageType.KNOWLEDGE_SHARE,
                content={
                    "knowledge_type": knowledge_type,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                },
                priority=3  # Lower priority for general knowledge
            )
            await self.communication.broadcast(
                from_agent=agent_id,
                team=self.shared_context.team_agents if self.shared_context else [],
                message=message
            )
            
            self.logger.debug(f"💡 Shared knowledge: {knowledge_type}")
            
        except Exception as e:
            self.logger.warning(f"Failed to share knowledge: {e}")

