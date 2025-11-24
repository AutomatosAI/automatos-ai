import asyncio
import sys
import os
import logging
from unittest.mock import MagicMock, AsyncMock

# Add orchestrator directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

# --- MOCK EVERYTHING BEFORE IMPORTING SERVICE ---
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

# Internal Core Components
mock_decomposer_module = MagicMock()
sys.modules["core.real_task_decomposer"] = mock_decomposer_module

mock_selector_module = MagicMock()
sys.modules["core.intelligent_agent_selector"] = mock_selector_module

# Mock ContextOptimizer
sys.modules["context_engineering.context_optimizer"] = MagicMock()
sys.modules["context_engineering.context_optimizer"].ContextOptimizer = MagicMock
sys.modules["context_engineering.context_optimizer"].ContextItem = MagicMock

mock_context_module = MagicMock()
sys.modules["core.context_engineering_integrator"] = mock_context_module

mock_execution_module = MagicMock()
sys.modules["core.agent_execution_manager"] = mock_execution_module

mock_aggregator_module = MagicMock()
sys.modules["core.result_aggregator"] = mock_aggregator_module

mock_quality_module = MagicMock()
sys.modules["core.output_quality_assessor"] = mock_quality_module

mock_memory_module = MagicMock()
sys.modules["core.workflow_memory_integrator"] = mock_memory_module

mock_matcher_module = MagicMock()
sys.modules["core.llm.semantic_skill_matcher"] = mock_matcher_module

# Internal Services
mock_llm_module = MagicMock()
sys.modules["services.llm_provider"] = mock_llm_module

mock_rag_module = MagicMock()
sys.modules["services.rag_service"] = mock_rag_module

mock_factory_module = MagicMock()
sys.modules["services.agent_factory"] = mock_factory_module

mock_mem_sys_module = MagicMock()
sys.modules["services.memory_knowledge_system"] = mock_mem_sys_module

mock_performance_service_module = MagicMock()
sys.modules["services.agent_performance_service"] = mock_performance_service_module

# NOW Import the service under test
from services.orchestrator_service import EnhancedOrchestratorService

async def verify_workflow():
    logger.info("🧪 Starting 9-Stage Workflow Verification (Isolated)")
    
    # Setup Mock Behaviors
    
    # 1. Decomposer
    mock_decomposer_instance = mock_decomposer_module.RealTaskDecomposer.return_value
    mock_decomposer_instance.decompose_task = AsyncMock(return_value={
        "subtasks": [{"id": "1", "description": "Task 1", "required_skills": ["python"]}],
        "execution_strategy": "sequential"
    })
    
    # 2. Selector
    mock_selector_instance = mock_selector_module.IntelligentAgentSelector.return_value
    mock_selector_instance.select_best_agent = AsyncMock(return_value={"id": 1, "name": "CoderAgent"})
    
    # 3. Context
    mock_context_instance = mock_context_module.ContextEngineeringIntegrator.return_value
    mock_context_instance.enhance_context = AsyncMock(return_value={"1": "Context"})
    
    # 4. Execution
    mock_execution_instance = mock_execution_module.AgentExecutionManager.return_value
    mock_execution_instance.execute_workflow_subtasks = AsyncMock(return_value={"1": "Result"})
    
    # 5. Aggregator
    mock_aggregator_instance = mock_aggregator_module.ResultAggregator.return_value
    mock_aggregator_instance.aggregate_workflow_results = AsyncMock(return_value=MagicMock(
        final_summary="Workflow Success",
        quality_scores=MagicMock(overall=0.95),
        agent_performance={}
    ))
    
    # Mock AgentPerformanceService
    mock_performance_service_instance = mock_performance_service_module.AgentPerformanceService.return_value
    mock_performance_service_instance.get_agent_performance_data = AsyncMock(return_value={
        "Test Agent": {
            "success_rate": 0.95,
            "avg_quality": 0.87,
            "total_executions": 42,
            "avg_tokens": 1500,
            "avg_execution_time": 3.2
        }
    })
    mock_performance_service_instance.get_agent_workload = AsyncMock(return_value={1: 2})
    
    # 6. Memory (Learning & Storage)
    mock_memory_instance = mock_memory_module.WorkflowMemoryIntegrator.return_value
    mock_memory_instance.consolidate_workflow_learnings = AsyncMock(return_value={"patterns": 1})
    mock_memory_instance.store_execution_experiences = AsyncMock(return_value={"total_experiences": 1})
    
    # 7. Quality
    mock_quality_instance = mock_quality_module.OutputQualityAssessor.return_value
    mock_quality_instance.assess_workflow_quality = AsyncMock(return_value={"overall_score": 0.95})
    
    # Factory
    mock_factory_instance = mock_factory_module.AgentFactory.return_value
    mock_factory_instance.get_available_agents = AsyncMock(return_value=[{"id": 1, "name": "TestAgent"}])
    
    # Performance Service
    mock_perf_service_instance = MagicMock()
    mock_perf_service_instance.get_agent_performance_data = AsyncMock(return_value={
        "TestAgent": {
            "success_rate": 0.95,
            "avg_quality": 0.87,
            "total_executions": 42,
            "avg_tokens": 1500,
            "avg_execution_time": 3.2
        }
    })
    mock_perf_service_instance.get_agent_workload = AsyncMock(return_value={1: 2})
    mock_performance_service_module.get_performance_service.return_value = mock_perf_service_instance

    # Instantiate Service
    service = EnhancedOrchestratorService()
    
    # Execute Workflow
    logger.info("🚀 Executing Workflow...")
    result = await service.execute_workflow("Test Prompt", 1)
    
    # Verification Checks
    logger.info("🔍 Verifying Stage Calls...")
    
    # Stage 1
    assert mock_decomposer_instance.decompose_task.called, "❌ Stage 1: Task Decomposition NOT called"
    logger.info("✅ Stage 1: Task Decomposition called")
    
    # Stage 2
    assert mock_selector_instance.select_best_agent.called, "❌ Stage 2: Agent Selection NOT called"
    logger.info("✅ Stage 2: Agent Selection called")
    
    # Stage 3
    assert mock_context_instance.enhance_context.called, "❌ Stage 3: Context Engineering NOT called"
    logger.info("✅ Stage 3: Context Engineering called")
    
    # Stage 4
    assert mock_execution_instance.execute_workflow_subtasks.called, "❌ Stage 4: Agent Execution NOT called"
    logger.info("✅ Stage 4: Agent Execution called")
    
    # Stage 5
    assert mock_aggregator_instance.aggregate_workflow_results.called, "❌ Stage 5: Result Aggregation NOT called"
    logger.info("✅ Stage 5: Result Aggregation called")
    
    # Stage 6
    assert mock_memory_instance.consolidate_workflow_learnings.called, "❌ Stage 6: Learning Update NOT called"
    logger.info("✅ Stage 6: Learning Update called")
    
    # Stage 7
    assert mock_quality_instance.assess_workflow_quality.called, "❌ Stage 7: Quality Assessment NOT called"
    logger.info("✅ Stage 7: Quality Assessment called")
    
    # Stage 8
    assert mock_memory_instance.store_execution_experiences.called, "❌ Stage 8: Memory Storage NOT called"
    logger.info("✅ Stage 8: Memory Storage called")
    
    # Stage 9 (Implicit in return)
    assert result["status"] == "completed", "❌ Stage 9: Response Generation failed (status not completed)"
    assert result["stages_completed"] == 9, "❌ Stage 9: Incorrect stage count"
    logger.info("✅ Stage 9: Response Generation successful")
    
    logger.info("🎉 ALL 9 STAGES VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(verify_workflow())
