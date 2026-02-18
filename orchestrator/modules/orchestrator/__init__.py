"""
Orchestrator Module — Dynamic Phase Workflow Engine
=====================================================

PRD-59: Evolved from fixed 9-stage pipeline to dynamic phase architecture.

Phases (expand into stages as needed):
  PLAN     → 1. Task Decomposition, 2. Agent Selection, 2b. Negotiation
  PREPARE  → 3. Context Engineering, 3b. Prompt Optimization
  EXECUTE  → 4. Agent Execution, 4b. Inter-Agent Coordination
  EVALUATE → 5. Result Aggregation, 6. Quality Assessment
  LEARN    → 7. Learning Update, 8. Memory Storage, 9. Response Generation

Components:
  stages/         — Individual pipeline stages (original + PRD-59 additions)
  llm/            — LLM-based orchestration (PRD-16)
  phase_selector  — Dynamic phase selection (PRD-59)
  pipeline        — Composable stage executor (PRD-59)
  service.py      — Main orchestrator service
  tracker.py      — Execution tracking

Sellable as: automatos-orchestrator
"""

from modules.orchestrator.service import EnhancedOrchestratorService
from modules.orchestrator.tracker import orchestration_tracker

# PRD-59: Phase architecture
from modules.orchestrator.phase_selector import (
    PhaseSelector,
    ExecutionMode,
    ComplexityLevel,
    Phase,
    PhaseSpec,
)
from modules.orchestrator.pipeline import (
    WorkflowPipeline,
    WorkflowContext,
    StageResult,
    StageStatus,
    ErrorStrategy,
)

# Stage components
from modules.orchestrator.stages import (
    RealTaskDecomposer,
    IntelligentAgentSelector,
    ContextEngineeringIntegrator,
    ResultAggregator,
    AggregatedResults,
    OutputQualityAssessor,
    WorkflowMemoryIntegrator,
    ComplexityAnalyzer,
    TokenBudgetManager,
    # PRD-59 new stages
    InterAgentNegotiation,
    NegotiationResult,
    PromptOptimizationStage,
    PromptOptimizationResult,
)

__all__ = [
    # Main service
    'EnhancedOrchestratorService',
    'orchestration_tracker',

    # PRD-59: Phase architecture
    'PhaseSelector',
    'ExecutionMode',
    'ComplexityLevel',
    'Phase',
    'PhaseSpec',
    'WorkflowPipeline',
    'WorkflowContext',
    'StageResult',
    'StageStatus',
    'ErrorStrategy',

    # Stages
    'RealTaskDecomposer',
    'IntelligentAgentSelector',
    'ContextEngineeringIntegrator',
    'ResultAggregator',
    'AggregatedResults',
    'OutputQualityAssessor',
    'WorkflowMemoryIntegrator',
    'ComplexityAnalyzer',
    'TokenBudgetManager',

    # PRD-59 new stages
    'InterAgentNegotiation',
    'NegotiationResult',
    'PromptOptimizationStage',
    'PromptOptimizationResult',
]

