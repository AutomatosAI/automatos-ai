"""
Enhanced Orchestrator Service
============================

Implements the fully functional 9-stage workflow pipeline for Automatos AI.
Replaces previous placeholder implementation with real, mathematically grounded logic.

9-Stage Pipeline:
1. Task Decomposition (RealTaskDecomposer)
2. Agent Selection (IntelligentAgentSelector)
3. Context Engineering (ContextEngineeringIntegrator)
4. Agent Execution (AgentExecutionManager)
5. Result Aggregation (ResultAggregator)
6. Learning Update (WorkflowMemoryIntegrator)
7. Quality Assessment (OutputQualityAssessor)
8. Memory Storage (WorkflowMemoryIntegrator)
9. Response Generation (Synthesis)
"""

import logging
import time
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

# Core Workflow Components
from core.real_task_decomposer import RealTaskDecomposer
from core.intelligent_agent_selector import IntelligentAgentSelector
from core.context_engineering_integrator import ContextEngineeringIntegrator
from core.agent_execution_manager import AgentExecutionManager
from core.result_aggregator import ResultAggregator
from core.output_quality_assessor import OutputQualityAssessor
from core.workflow_memory_integrator import WorkflowMemoryIntegrator
from core.llm.semantic_skill_matcher import SemanticSkillMatcher

# Services (Assumed available based on imports)
from services.llm_provider import create_llm_manager
from services.rag_service import RAGService
from services.agent_factory import AgentFactory
from services.agent_performance_service import get_performance_service

logger = logging.getLogger(__name__)

class EnhancedOrchestratorService:
    """
    Orchestrates the 9-stage intelligent agent workflow.
    """
    
    def __init__(self, db_session: Session = None):
        self.logger = logging.getLogger(__name__)
        self.db = db_session
        
        # Initialize services (lazy-load LLM provider to avoid startup crashes)
        try:
            self.llm_provider = create_llm_manager(service_name="orchestrator")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM provider at startup: {e}. Will retry on first use.")
            self.llm_provider = None
        self.rag_service = RAGService()
        self.agent_factory = AgentFactory(db_session=self.db)
        self.performance_service = get_performance_service(self.db)
        
        # Initialize Stage Components (handle None LLM provider)
        if self.llm_provider:
            self.task_decomposer = RealTaskDecomposer(self.llm_provider)
            self.quality_assessor = OutputQualityAssessor(self.llm_provider)
        else:
            self.task_decomposer = None
            self.quality_assessor = None
            self.logger.warning("Task decomposer and quality assessor not initialized - LLM provider unavailable")
        
        # SemanticSkillMatcher uses global embedding provider from General Settings
        self.skill_matcher = SemanticSkillMatcher()
        self.agent_selector = IntelligentAgentSelector(self.skill_matcher)
        self.context_integrator = ContextEngineeringIntegrator(db_session=self.db)
        self.execution_manager = AgentExecutionManager(db_session=self.db)
        self.result_aggregator = ResultAggregator()
        
        # Memory Integrator needs memory system, assuming it's available or passed
        # For now, we'll instantiate with None and let it handle missing deps gracefully or mock
        # In a real scenario, we'd pass the actual memory system
        from services.memory_knowledge_system import HierarchicalMemorySystem
        self.memory_system = HierarchicalMemorySystem() # Placeholder instantiation
        self.memory_integrator = WorkflowMemoryIntegrator(self.memory_system)

        self.logger.info("✅ Enhanced Orchestrator Service initialized with 9-stage pipeline")
    
    def _ensure_llm_provider(self):
        """Lazy initialization of LLM provider if it wasn't available at startup"""
        if self.llm_provider is None:
            try:
                self.llm_provider = create_llm_manager(service_name="orchestrator")
                if self.task_decomposer is None:
                    self.task_decomposer = RealTaskDecomposer(self.llm_provider)
                if self.quality_assessor is None:
                    self.quality_assessor = OutputQualityAssessor(self.llm_provider)
                self.logger.info("✅ LLM provider initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM provider: {e}")
                raise

    async def execute_workflow(self, user_prompt: str, user_id: int, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the full 9-stage workflow for a user request.
        
        Args:
            user_prompt: The user's natural language request
            user_id: ID of the requesting user
            context: Optional additional context
            
        Returns:
            Dict containing the final response and execution metadata
        """
        workflow_id = int(time.time()) # Simple ID generation
        start_time = time.time()
        self.logger.info(f"🚀 STARTING WORKFLOW {workflow_id}: {user_prompt[:50]}...")
        
        try:
            # Ensure LLM provider is initialized
            self._ensure_llm_provider()
            
            # --- STAGE 1: Task Decomposition ---
            self.logger.info("--- STAGE 1: Task Decomposition ---")
            decomposition = await self.task_decomposer.decompose_task(user_prompt)
            subtasks = decomposition.get("subtasks", [])
            self.logger.info(f"✅ Decomposed into {len(subtasks)} subtasks")

            # --- STAGE 2: Agent Selection ---
            self.logger.info("--- STAGE 2: Agent Selection ---")
            # Get available agents from database
            available_agents = await self.agent_factory.get_available_agents() 
            
            # Get real performance data and workload from database
            agent_ids = [a["id"] for a in available_agents]
            agent_performance_data = await self.performance_service.get_agent_performance_data(agent_ids)
            current_workloads = await self.performance_service.get_agent_workload(agent_ids)
            
            agent_assignments = {}
            for subtask in subtasks:
                selected_agent = await self.agent_selector.select_best_agent(
                    subtask_description=subtask.get("description", ""),
                    required_skills=subtask.get("required_skills", []),
                    available_agents=available_agents,
                    agent_performance_data=agent_performance_data,
                    current_workloads=current_workloads
                )
                if selected_agent:
                    agent_assignments[subtask["id"]] = selected_agent
                else:
                    self.logger.warning(f"⚠️ No agent found for subtask {subtask['id']}")
            
            # --- STAGE 3: Context Engineering ---
            self.logger.info("--- STAGE 3: Context Engineering ---")
            context_enhancements = await self.context_integrator.enhance_context(
                subtasks=subtasks,
                workflow_description=user_prompt,
                agent_assignments=agent_assignments
            )

            # --- STAGE 4: Agent Execution ---
            self.logger.info("--- STAGE 4: Agent Execution ---")
            execution_results = await self.execution_manager.execute_workflow_subtasks(
                workflow_id=str(workflow_id),
                subtasks=subtasks,
                agent_assignments=agent_assignments,
                context_enhancements=context_enhancements,
                execution_strategy=decomposition.get("execution_strategy", "parallel")
            )

            # --- STAGE 5: Result Aggregation ---
            self.logger.info("--- STAGE 5: Result Aggregation ---")
            aggregated_results = await self.result_aggregator.aggregate_workflow_results(
                workflow_id=str(workflow_id),
                subtask_executions=execution_results,
                original_request=user_prompt
            )

            # --- STAGE 6: Learning Update (Consolidation) ---
            self.logger.info("--- STAGE 6: Learning Update ---")
            learning_summary = await self.memory_integrator.consolidate_workflow_learnings(
                workflow_id=workflow_id,
                execution_id=workflow_id,
                aggregated_results=aggregated_results,
                decomposition_metadata=decomposition,
                subtask_executions=execution_results
            )

            # --- STAGE 7: Quality Assessment ---
            self.logger.info("--- STAGE 7: Quality Assessment ---")
            quality_assessment = await self.quality_assessor.assess_workflow_quality(
                request=user_prompt,
                aggregated_results=aggregated_results,
                execution_metadata={
                    "duration": time.time() - start_time,
                    "steps": len(subtasks)
                }
            )

            # --- STAGE 8: Memory Storage (Experiences) ---
            self.logger.info("--- STAGE 8: Memory Storage ---")
            memory_storage = await self.memory_integrator.store_execution_experiences(
                workflow_id=workflow_id,
                execution_id=workflow_id,
                subtask_executions=execution_results,
                aggregated_results=aggregated_results
            )

            # --- STAGE 9: Response Generation ---
            self.logger.info("--- STAGE 9: Response Generation ---")
            final_response = {
                "workflow_id": workflow_id,
                "status": "completed",
                "result": aggregated_results.final_summary,
                "quality_score": quality_assessment.get("overall_score", 0.0),
                "execution_time": time.time() - start_time,
                "stages_completed": 9,
                "details": {
                    "subtasks": len(subtasks),
                    "agents_used": list(set(a["name"] for a in agent_assignments.values())),
                    "memories_stored": memory_storage.get("total_experiences", 0)
                }
            }
            
            self.logger.info(f"🏁 WORKFLOW COMPLETED in {final_response['execution_time']:.2f}s")
            return final_response

        except Exception as e:
            self.logger.error(f"❌ WORKFLOW FAILED: {str(e)}", exc_info=True)
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    # --- Legacy / Placeholder Methods (Kept for compatibility) ---
    
    async def create_task(self, task_data: Any, user_id: int, db: Session) -> Dict[str, Any]:
        logger.warning("Legacy create_task called")
        return {"error": "Use execute_workflow instead"}
    
    async def execute_task(self, task_id: int, user_id: int, db: Session) -> Dict[str, Any]:
        logger.warning("Legacy execute_task called")
        return {"error": "Use execute_workflow instead"}
    
    async def get_tasks(self, *args, **kwargs) -> List[Any]:
        return []

    def get_task(self, *args, **kwargs) -> Optional[Any]:
        return None
        
    def update_task(self, *args, **kwargs) -> Optional[Any]:
        return None

    async def update_field_context(self, session_id: str, context_data: dict = None, field_type: str = None, context: dict = None):
        """Legacy field context update"""
        return {"status": "legacy_mock"}

    async def model_field_interactions(self, fields: list):
        """Legacy interaction model"""
        return {"status": "legacy_mock"}

    async def propagate_field_influence(self, *args, **kwargs):
        """Legacy propagation"""
        return {"status": "legacy_mock"}

    async def perform_collaborative_reasoning(self, *args, **kwargs):
        """Legacy reasoning"""
        return {"status": "legacy_mock"}

    async def monitor_emergent_behavior(self, *args, **kwargs):
        """Legacy monitoring"""
        return {"status": "legacy_mock"}

    async def coordinate_multi_agent(self, *args, **kwargs):
        """Legacy coordination"""
        return {"status": "legacy_mock"}

    async def coordinate_multi_agents(self, *args, **kwargs):
        """Legacy coordination plural"""
        return {"status": "legacy_mock"}

    async def manage_dynamic_fields(self, *args, **kwargs):
        """Legacy field management"""
        return {"status": "legacy_mock"}

    async def optimize_multi_agent_system(self, *args, **kwargs):
        """Legacy optimization"""
        return {"status": "legacy_mock"}


# --- Legacy Classes (Kept for import compatibility) ---

class FieldManager:
    def __init__(self): pass
    async def get_context(self, session_id): return {}
    def get_field_statistics(self): return {}

class CoordinationManager:
    def __init__(self): pass
    def get_statistics(self): return {}
    async def rebalance_agents(self, data): return {}
    def get_coordination_statistics(self): return {}

class BehaviorManager:
    def __init__(self): pass
    def get_statistics(self): return {}

class OptimizationManager:
    def __init__(self): pass
    def get_statistics(self): return {}
    async def rebalance_agents(self, data): return {}
    def get_coordination_statistics(self): return {}

class BehaviorMonitor:
    def __init__(self): pass
    def get_monitoring_statistics(self): return {}

class MultiAgentOptimizer:
    def __init__(self): pass
    def get_optimization_statistics(self): return {}

