"""
Orchestrator Pipeline Stages
============================

The workflow pipeline stages, now with dynamic phase support (PRD-59).

Original 9 stages:
  1. Task Decomposition   2. Agent Selection     3. Context Engineering
  4. Agent Execution      5. Result Aggregation  6. Learning Update
  7. Quality Assessment   8. Memory Storage      9. Response Generation

PRD-59 additions:
  2b. Inter-Agent Negotiation (3+ agents)
  3b. Prompt Optimization (PRD-58 integration)
"""

from modules.orchestrator.stages.task_decomposer import RealTaskDecomposer, get_decomposer
from modules.orchestrator.stages.agent_selector import IntelligentAgentSelector
from modules.orchestrator.stages.context_engineering import ContextEngineeringIntegrator
from modules.orchestrator.stages.result_aggregator import ResultAggregator, AggregatedResults
from modules.orchestrator.stages.quality_assessor import OutputQualityAssessor, OutputType
from modules.orchestrator.stages.memory_integrator import WorkflowMemoryIntegrator
from modules.orchestrator.stages.complexity_analyzer import ComplexityAnalyzer
from modules.orchestrator.stages.token_budget_manager import TokenBudgetManager

# PRD-59: New stages
from modules.orchestrator.stages.agent_negotiation import InterAgentNegotiation, NegotiationResult
from modules.orchestrator.stages.prompt_optimization import PromptOptimizationStage, PromptOptimizationResult

__all__ = [
    # Stage 1: Task Decomposition
    'RealTaskDecomposer',
    'get_decomposer',

    # Stage 2: Agent Selection
    'IntelligentAgentSelector',

    # Stage 2b: Inter-Agent Negotiation (PRD-59)
    'InterAgentNegotiation',
    'NegotiationResult',

    # Stage 3: Context Engineering
    'ContextEngineeringIntegrator',

    # Stage 3b: Prompt Optimization (PRD-59/PRD-58)
    'PromptOptimizationStage',
    'PromptOptimizationResult',

    # Stage 5: Result Aggregation
    'ResultAggregator',
    'AggregatedResults',

    # Stage 7: Quality Assessment
    'OutputQualityAssessor',
    'OutputType',

    # Stages 6 & 8: Learning & Memory
    'WorkflowMemoryIntegrator',

    # Supporting
    'ComplexityAnalyzer',
    'TokenBudgetManager',
]

