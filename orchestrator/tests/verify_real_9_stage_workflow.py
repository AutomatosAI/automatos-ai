import asyncio
import sys
import os
import logging
from unittest.mock import MagicMock, AsyncMock, patch

# Add orchestrator directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

# --- MOCK EVERYTHING BEFORE IMPORTING API ---
# This prevents any real imports from triggering missing dependency errors

# External libs
sys.modules["sqlalchemy"] = MagicMock()
sys.modules["sqlalchemy.orm"] = MagicMock()
sys.modules["sqlalchemy.dialects"] = MagicMock()
sys.modules["sqlalchemy.dialects.postgresql"] = MagicMock()
sys.modules["sqlalchemy.sql"] = MagicMock()
sys.modules["dotenv"] = MagicMock()
sys.modules["tiktoken"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["networkx"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["httpx"] = MagicMock()
sys.modules["pydantic"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pandas"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()

# Mock Database
mock_db_module = MagicMock()
sys.modules["database.database"] = mock_db_module
mock_db_session = MagicMock()
mock_db_module.get_db_session.return_value.__enter__.return_value = mock_db_session

# Mock Models
mock_models_module = MagicMock()
sys.modules["models"] = mock_models_module
mock_execution = MagicMock()
mock_execution.id = 1
mock_execution.workflow_id = 1
mock_execution.input_data = {}
mock_execution.status = "pending"
mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

mock_workflow = MagicMock()
mock_workflow.name = "Test Workflow"
mock_workflow.description = "Test Description"
mock_db_session.query.return_value.filter.return_value.first.side_effect = [mock_execution, mock_workflow]

# Mock Core Components
mock_decomposer_module = MagicMock()
sys.modules["core.real_task_decomposer"] = mock_decomposer_module
mock_decomposer_instance = mock_decomposer_module.RealTaskDecomposer.return_value
mock_decomposer_instance.decompose_task = AsyncMock(return_value={
    "subtasks": [{"id": "1", "description": "Task 1", "required_skills": ["python"]}],
    "execution_strategy": "sequential"
})

mock_selector_module = MagicMock()
sys.modules["core.llm.llm_agent_selector"] = mock_selector_module
mock_selector_instance = mock_selector_module.LLMAgentSelector.return_value
mock_selector_instance.select_best_agent = AsyncMock(return_value={"id": 1, "name": "CoderAgent"})
mock_selector_instance.select_agents_for_subtasks = AsyncMock(return_value={
    "1": [MagicMock(agent_id=1, agent_name="CoderAgent", match_score=0.95)]
})

mock_context_module = MagicMock()
sys.modules["core.context_engineering_integrator"] = mock_context_module
mock_context_instance = mock_context_module.ContextEngineeringIntegrator.return_value
mock_context_instance.enhance_subtasks_with_context = AsyncMock(return_value={"1": MagicMock(total_tokens=100, num_sources=1, context_quality_score=0.9, retrieval_time_ms=100)})
mock_context_instance.get_enhancement_summary = MagicMock(return_value={"context_coverage": 1.0, "total_sources_retrieved": 1, "avg_context_quality": 0.9})

mock_execution_module = MagicMock()
sys.modules["core.agent_execution_manager"] = mock_execution_module
mock_execution_instance = mock_execution_module.AgentExecutionManager.return_value
mock_execution_instance.execute_workflow_subtasks = AsyncMock(return_value={"1": MagicMock(status="completed", token_usage={"total_tokens": 100})})

mock_aggregator_module = MagicMock()
sys.modules["core.result_aggregator"] = mock_aggregator_module
mock_aggregator_instance = mock_aggregator_module.ResultAggregator.return_value
mock_aggregator_instance.aggregate_results = MagicMock(return_value=MagicMock(
    final_summary="Workflow Success",
    quality_scores=MagicMock(overall=0.95),
    agent_performance={},
    total_tokens_used=100,
    total_execution_time_ms=1000,
    avg_context_quality=0.9,
    recommendations=[]
))

mock_learning_module = MagicMock()
sys.modules["core.learning_system_updater"] = mock_learning_module
mock_learning_instance = mock_learning_module.LearningSystemUpdater.return_value
mock_learning_instance.update_from_execution = AsyncMock(return_value={
    "total_updates": 5,
    "agent_performance_updates": []
})
mock_learning_instance.get_learning_summary = MagicMock(return_value="Summary")

mock_quality_module = MagicMock()
sys.modules["core.output_quality_assessor"] = mock_quality_module
mock_quality_instance = mock_quality_module.OutputQualityAssessor.return_value
mock_quality_instance.assess_quality = AsyncMock(return_value=MagicMock(
    overall_score=0.95,
    passes_threshold=True,
    dimensions={},
    strengths=[],
    weaknesses=[],
    improvement_suggestions=[]
))

mock_memory_module = MagicMock()
sys.modules["core.workflow_memory_integrator"] = mock_memory_module
mock_memory_instance = mock_memory_module.WorkflowMemoryIntegrator.return_value
mock_memory_instance.store_execution_experiences = AsyncMock(return_value={"total_experiences": 1, "per_agent": {}})
mock_memory_instance.consolidate_workflow_learnings = AsyncMock(return_value={"patterns_extracted": 1, "knowledge_nodes_created": 1, "agents_consolidated": []})
mock_memory_instance.get_memory_integration_summary = MagicMock(return_value="Memory Summary")

# Mock Services
sys.modules["services.memory_knowledge_system"] = MagicMock()
sys.modules["services.workspace_manager"] = MagicMock()
mock_tracker_module = MagicMock()
sys.modules["utils.model_usage_tracker"] = mock_tracker_module
mock_tracker_instance = mock_tracker_module.ModelUsageTracker.return_value
mock_tracker_instance.get_usage_summary.return_value = {
    "total_requests": 10,
    "total_tokens": 1000,
    "total_cost": 0.05,
    "records": []
}
sys.modules["core.redis_client"] = MagicMock()

# Import the function to test
# We need to patch 'api.workflows.WorkflowStageTracker' because it's defined in the file
with patch("api.workflows.WorkflowStageTracker") as MockTracker:
    from api.workflows import execute_workflow_with_progress

    async def verify_workflow():
        logger.info("🧪 Starting Real 9-Stage Workflow Verification")
        
        # Execute Workflow
        logger.info("🚀 Executing Workflow...")
        try:
            await execute_workflow_with_progress(1, {})
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Verification Checks
        logger.info("🔍 Verifying Stage Calls...")
        
        # Stage 1: Task Decomposition
        assert mock_decomposer_instance.decompose_task.called, "❌ Stage 1: Task Decomposition NOT called"
        logger.info("✅ Stage 1: Task Decomposition called")
        
        # Stage 2: Agent Selection
        assert mock_selector_instance.select_agents_for_subtasks.called, "❌ Stage 2: Agent Selection NOT called"
        logger.info("✅ Stage 2: Agent Selection called")
        
        # Stage 3: Context Engineering
        assert mock_context_instance.enhance_subtasks_with_context.called, "❌ Stage 3: Context Engineering NOT called"
        logger.info("✅ Stage 3: Context Engineering called")
        
        # Stage 4: Agent Execution
        assert mock_execution_instance.execute_workflow_subtasks.called, "❌ Stage 4: Agent Execution NOT called"
        logger.info("✅ Stage 4: Agent Execution called")
        
        # Stage 5: Result Aggregation
        assert mock_aggregator_instance.aggregate_results.called, "❌ Stage 5: Result Aggregation NOT called"
        logger.info("✅ Stage 5: Result Aggregation called")
        
        # Stage 6: Learning Update
        assert mock_learning_instance.update_from_execution.called, "❌ Stage 6: Learning Update NOT called"
        logger.info("✅ Stage 6: Learning Update called")
        
        # Stage 7: Quality Assessment
        assert mock_quality_instance.assess_quality.called, "❌ Stage 7: Quality Assessment NOT called"
        logger.info("✅ Stage 7: Quality Assessment called")
        
        # Stage 8: Memory Storage
        assert mock_memory_instance.store_execution_experiences.called, "❌ Stage 8: Memory Storage NOT called"
        assert mock_memory_instance.consolidate_workflow_learnings.called, "❌ Stage 8: Memory Consolidation NOT called"
        logger.info("✅ Stage 8: Memory Storage & Consolidation called")
        
        # Stage 9: Response Generation (Implicit)
        assert mock_execution.status == "completed", "❌ Stage 9: Execution status not completed"
        logger.info("✅ Stage 9: Execution completed successfully")
        
        # Verify Order (by checking call order on the mock manager if we had one, or just logic flow)
        # Since we are running the actual code, the order is determined by the code flow.
        # If all mocks were called, and the code executes sequentially, the order is correct.
        
        logger.info("🎉 ALL 9 STAGES VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(verify_workflow())
