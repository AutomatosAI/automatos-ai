"""
Orchestrator Module - 9-Stage Workflow Pipeline
================================================

The brain of Automatos AI - coordinates all workflow execution.

Stages:
1. Task Decomposition - Break complex tasks into subtasks
2. Agent Selection - Choose best agents for each subtask
3. Context Engineering - Build optimal context for each agent
4. Agent Execution - Execute agents (uses modules.agents)
5. Result Aggregation - Combine agent outputs
6. Learning Update - Update learning systems
7. Quality Assessment - Assess output quality
8. Memory Storage - Store results in memory
9. Response Generation - Synthesize final response

Usage:
    from modules.orchestrator import EnhancedOrchestratorService
    
    orchestrator = EnhancedOrchestratorService(db_session)
    result = await orchestrator.execute_workflow(workflow_id, prompt)

Components:
- stages/      - Individual pipeline stages
- llm/         - LLM-based orchestration (PRD-16)
- service.py   - Main orchestrator service
- tracker.py   - Execution tracking

Sellable as: automatos-orchestrator
"""

from modules.orchestrator.service import EnhancedOrchestratorService
from modules.orchestrator.tracker import orchestration_tracker

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
)

__all__ = [
    # Main service
    'EnhancedOrchestratorService',
    'orchestration_tracker',
    
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
]

