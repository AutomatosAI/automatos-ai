"""
Master Orchestrator - Meta-coordination of all LLM stages
==========================================================

PRD-16: Master Orchestrator that coordinates all stage LLMs.
Implements meta-reasoning about orchestration quality and adaptive strategies.

This is the highest level of the Software 3.0 orchestration system:
- Coordinates all 9 workflow stages
- Makes strategic decisions about orchestration
- Adapts strategies based on results
- Learns from execution patterns
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import all stage components
# Orchestration LLM components (local imports)
from modules.orchestrator.llm.orchestrator_llm import OrchestratorLLM, ReasoningMode
from modules.orchestrator.llm.llm_agent_selector import LLMAgentSelector
from modules.orchestrator.llm.llm_context_strategy import LLMContextStrategySelector
from modules.orchestrator.llm.adaptive_execution_monitor import AdaptiveExecutionMonitor
from modules.orchestrator.llm.llm_result_aggregator import LLMResultAggregator

# Stage components
from modules.orchestrator.stages import RealTaskDecomposer, WorkflowMemoryIntegrator
from modules.learning import LearningSystemUpdater
from modules.memory.storage import HierarchicalMemorySystem

logger = logging.getLogger(__name__)


class OrchestrationStrategy(Enum):
    """Overall orchestration strategies"""
    SPEED_OPTIMIZED = "speed"          # Minimize execution time
    QUALITY_OPTIMIZED = "quality"      # Maximize output quality
    COST_OPTIMIZED = "cost"           # Minimize token usage
    BALANCED = "balanced"              # Balance all factors
    ADAPTIVE = "adaptive"              # Adapt based on context


class ExecutionMode(Enum):
    """
    Workflow execution modes based on user pre-configuration
    """
    AUTONOMOUS = "autonomous"  # User provides goal, system figures everything out (full 9-stage)
    RECIPE = "recipe"          # User pre-defines all steps and agents (smart stage skipping)
    HYBRID = "hybrid"          # User pre-defines some steps, system fills in the rest


@dataclass
class StageResult:
    """Result from a workflow stage"""
    stage_number: int
    stage_name: str
    status: str  # "success", "partial", "failed"
    result: Any
    execution_time: float
    tokens_used: int = 0
    cost: float = 0.0
    confidence: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class OrchestrationPlan:
    """Plan for orchestrating a workflow"""
    strategy: OrchestrationStrategy
    stage_configurations: Dict[int, Dict[str, Any]]
    parallel_stages: List[List[int]]  # Groups of stages to run in parallel
    quality_thresholds: Dict[int, float]
    time_budget: Optional[int]
    token_budget: Optional[int]
    reasoning: str


@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    workflow_id: int
    status: str
    execution_mode: ExecutionMode  # NEW: Track which mode was used
    stages_completed: List[StageResult]
    stages_skipped: List[Tuple[int, str]]  # NEW: Track skipped stages with reasons
    final_output: Any
    total_execution_time: float
    total_tokens_used: int
    total_cost: float
    orchestration_quality: float
    meta_insights: List[str]
    improvements_identified: List[str]


def determine_execution_mode(workflow_definition: Dict[str, Any]) -> ExecutionMode:
    """
    Detect execution mode based on workflow/recipe configuration.

    Logic:
    - AUTONOMOUS: No steps defined OR no agents assigned to any step
    - RECIPE: All steps have agents assigned
    - HYBRID: Some steps have agents, some don't

    Args:
        workflow_definition: Workflow configuration with optional steps

    Returns:
        ExecutionMode enum value
    """
    steps = workflow_definition.get('steps', [])

    # No steps defined = AUTONOMOUS (user just provided a goal)
    if not steps or len(steps) == 0:
        logger.info("ExecutionMode: AUTONOMOUS (no steps defined)")
        return ExecutionMode.AUTONOMOUS

    # Count how many steps have agents assigned
    steps_with_agents = sum(1 for step in steps if step.get('agent_id'))
    total_steps = len(steps)

    # All steps have agents = RECIPE (fully pre-configured)
    if steps_with_agents == total_steps:
        logger.info(f"ExecutionMode: RECIPE ({total_steps}/{total_steps} steps have agents)")
        return ExecutionMode.RECIPE

    # Some steps have agents = HYBRID (mixed mode)
    elif steps_with_agents > 0:
        logger.info(f"ExecutionMode: HYBRID ({steps_with_agents}/{total_steps} steps have agents)")
        return ExecutionMode.HYBRID

    # No steps have agents = AUTONOMOUS (user defined steps but let system assign agents)
    else:
        logger.info(f"ExecutionMode: AUTONOMOUS ({total_steps} steps but no agents assigned)")
        return ExecutionMode.AUTONOMOUS


def should_run_stage(
    stage_num: int,
    stage_name: str,
    mode: ExecutionMode,
    context: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Decide if a stage should run based on execution mode.

    Args:
        stage_num: Stage number (1-9)
        stage_name: Stage name
        mode: Execution mode
        context: Additional context (steps, agents, etc.)

    Returns:
        Tuple of (should_run: bool, reason: str)
    """

    # AUTONOMOUS mode: Run all stages
    if mode == ExecutionMode.AUTONOMOUS:
        return (True, "")

    # RECIPE mode: Smart stage skipping
    elif mode == ExecutionMode.RECIPE:
        if stage_num == 1:  # Task Decomposition
            return (False, "Recipe already has pre-defined steps")

        elif stage_num == 2:  # Agent Selection
            return (False, "Recipe already has pre-assigned agents")

        elif stage_num == 3:  # Context Engineering
            # Conditional: Check if steps need internal context
            needs_context = requires_internal_context(context.get('steps', []))
            if needs_context:
                return (True, "")
            else:
                return (False, "Steps only use external tools (no RAG/NL2SQL needed)")

        else:  # Stages 4-9: Always run
            return (True, "")

    # HYBRID mode: Skip stage 1, but run agent selection for unassigned steps
    elif mode == ExecutionMode.HYBRID:
        if stage_num == 1:  # Task Decomposition
            return (False, "Recipe already has pre-defined steps structure")

        elif stage_num == 2:  # Agent Selection
            # Only select agents for steps that don't have them
            return (True, "Partial agent selection needed for unassigned steps")

        elif stage_num == 3:  # Context Engineering
            needs_context = requires_internal_context(context.get('steps', []))
            return (needs_context, "" if needs_context else "Steps only use external tools")

        else:  # Stages 4-9: Always run
            return (True, "")

    return (True, "")


def requires_internal_context(steps: List[Dict[str, Any]]) -> bool:
    """
    Check if any step requires internal context tools (RAG, NL2SQL, CodeGraph).

    Keywords that indicate need for internal context:
    - Document/knowledge search: "find document", "search", "knowledge base"
    - Database queries: "how many", "count", "analyze", "query", "sql"
    - Code search: "code", "implementation", "codebase", "function"

    Args:
        steps: List of workflow steps

    Returns:
        True if any step needs internal context, False otherwise
    """
    context_keywords = [
        # RAG indicators
        'find document', 'search document', 'knowledge base', 'search knowledge',
        'find information', 'lookup', 'retrieve document',

        # NL2SQL indicators
        'how many', 'count', 'total', 'sum', 'average', 'query database',
        'analyze data', 'sql', 'database', 'table', 'records',

        # CodeGraph indicators
        'code', 'codebase', 'implementation', 'function', 'class',
        'search code', 'find code', 'code search'
    ]

    for step in steps:
        prompt = step.get('prompt_template', '').lower()
        description = step.get('description', '').lower()
        combined_text = f"{prompt} {description}"

        # Check if any context keyword appears in step text
        if any(keyword in combined_text for keyword in context_keywords):
            logger.info(f"Step requires internal context: '{step.get('description', 'N/A')[:50]}...'")
            return True

    logger.info("No internal context required - steps use external tools only")
    return False


class MasterOrchestrator:
    """
    Meta-orchestrator that coordinates all stage LLMs.
    
    This implements the highest level of the Software 3.0 paradigm,
    where the orchestrator itself reasons about orchestration strategies.
    """
    
    def __init__(
        self,
        db_session,
        default_strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE
    ):
        """
        Initialize Master Orchestrator.
        
        Args:
            db_session: Database session
            default_strategy: Default orchestration strategy
        """
        self.db = db_session
        self.default_strategy = default_strategy
        
        # Initialize master LLM with meta-cognitive mode
        self.master_llm = OrchestratorLLM(
            temperature=0.7,
            reasoning_mode=ReasoningMode.META_COGNITIVE
        )
        
        # Initialize stage components
        self.stage_llms = self._initialize_stage_llms()
        
        # Orchestration state
        self.active_workflows: Dict[int, OrchestrationPlan] = {}
        self.execution_history: List[WorkflowResult] = []
        self.learning_patterns: Dict[str, Any] = {}
        
        logger.info(f"MasterOrchestrator initialized with strategy: {default_strategy.value}")
    
    def _initialize_stage_llms(self) -> Dict[str, Any]:
        """Initialize all stage LLM components"""
        return {
            'decomposition': RealTaskDecomposer(llm_manager=None),  # Uses own LLM
            'context_strategy': LLMContextStrategySelector(llm=None),
            'agent_selection': LLMAgentSelector(self.db, llm=None),
            'execution_monitor': AdaptiveExecutionMonitor(llm=None),
            'result_aggregation': LLMResultAggregator(llm=None),
            'learning_updater': LearningSystemUpdater(self.db),
            'memory_integrator': WorkflowMemoryIntegrator(self.db)
        }
    
    async def orchestrate_workflow(
        self,
        workflow_id: int,
        task_description: str,
        workflow_context: Optional[Dict[str, Any]] = None,
        strategy_override: Optional[OrchestrationStrategy] = None
    ) -> WorkflowResult:
        """
        Orchestrate complete workflow execution with meta-reasoning.
        
        Args:
            workflow_id: Workflow identifier
            task_description: Main task description
            workflow_context: Additional context
            strategy_override: Override default strategy
            
        Returns:
            Complete workflow result
        """
        start_time = asyncio.get_event_loop().time()
        logger.info(f"🎯 Master Orchestrator starting workflow {workflow_id}")

        # NEW: Detect execution mode based on workflow configuration
        execution_mode = determine_execution_mode(workflow_context or {})
        logger.info(f"🔍 Execution Mode: {execution_mode.value.upper()}")

        # Step 1: Plan orchestration strategy
        plan = await self.plan_orchestration_strategy(
            task_description,
            workflow_context,
            strategy_override or self.default_strategy
        )

        self.active_workflows[workflow_id] = plan
        logger.info(f"📋 Orchestration Plan: {plan.strategy.value}")
        logger.info(f"   Reasoning: {plan.reasoning[:200]}...")
        
        # Step 2: Execute stages according to plan (with conditional execution)
        stage_results = []
        stages_skipped = []  # NEW: Track skipped stages
        total_tokens = 0
        total_cost = 0.0

        try:
            # Stage 1: Task Decomposition
            should_run, skip_reason = should_run_stage(
                1, "Task Decomposition", execution_mode, workflow_context or {}
            )

            if should_run:
                stage1_result = await self._execute_stage_1(
                    task_description,
                    plan.stage_configurations.get(1, {})
                )
                stage_results.append(stage1_result)
                total_tokens += stage1_result.tokens_used
                total_cost += stage1_result.cost

                if stage1_result.status == "failed":
                    raise Exception(f"Stage 1 failed: {stage1_result.issues}")

                subtasks = stage1_result.result.get('subtasks', [])
            else:
                logger.info(f"⏭️  SKIPPED Stage 1 (Task Decomposition): {skip_reason}")
                stages_skipped.append((1, skip_reason))
                # Use pre-defined steps from workflow_context
                subtasks = workflow_context.get('steps', [])
            
            # Stage 2: Agent Selection
            should_run, skip_reason = should_run_stage(
                2, "Agent Selection", execution_mode, {'steps': subtasks}
            )

            agent_assignments = {}
            if should_run:
                stage2_result = await self._execute_stage_2(
                    subtasks,
                    plan.stage_configurations.get(2, {})
                )
                stage_results.append(stage2_result)
                total_tokens += stage2_result.tokens_used
                total_cost += stage2_result.cost
                agent_assignments = stage2_result.result
            else:
                logger.info(f"⏭️  SKIPPED Stage 2 (Agent Selection): {skip_reason}")
                stages_skipped.append((2, skip_reason))
                # Use pre-assigned agents from recipe steps
                agent_assignments = {
                    f"subtask_{i}": [{"agent_id": step.get('agent_id')}]
                    for i, step in enumerate(subtasks)
                    if step.get('agent_id')
                }

            # Stage 3: Context Engineering (LLM-driven)
            should_run, skip_reason = should_run_stage(
                3, "Context Engineering", execution_mode, {'steps': subtasks}
            )

            if should_run:
                stage3_result = await self._execute_stage_3(
                    subtasks,
                    workflow_context,
                    plan.stage_configurations.get(3, {})
                )
                stage_results.append(stage3_result)
                total_tokens += stage3_result.tokens_used
                total_cost += stage3_result.cost
            else:
                logger.info(f"⏭️  SKIPPED Stage 3 (Context Engineering): {skip_reason}")
                stages_skipped.append((3, skip_reason))
            
            # Stage 4: Execution with Monitoring (always runs)
            stage4_result = await self._execute_stage_4(
                subtasks,
                agent_assignments,  # Use agent_assignments from Stage 2 or pre-defined
                plan.stage_configurations.get(4, {})
            )
            stage_results.append(stage4_result)
            total_tokens += stage4_result.tokens_used
            total_cost += stage4_result.cost
            
            # Stage 5: Result Aggregation (LLM-driven)
            stage5_result = await self._execute_stage_5(
                stage4_result.result,  # Execution results
                task_description,
                plan.stage_configurations.get(5, {})
            )
            stage_results.append(stage5_result)
            total_tokens += stage5_result.tokens_used
            total_cost += stage5_result.cost
            
            # Meta-evaluation of orchestration quality
            quality_assessment = await self.evaluate_orchestration_quality(
                stage_results,
                plan
            )
            
            # Learn from this execution
            await self.learn_from_execution(
                workflow_id,
                stage_results,
                quality_assessment
            )
            
            # Build final result
            execution_time = asyncio.get_event_loop().time() - start_time
            
            result = WorkflowResult(
                workflow_id=workflow_id,
                status="completed",
                execution_mode=execution_mode,  # NEW
                stages_completed=stage_results,
                stages_skipped=stages_skipped,  # NEW
                final_output=stage5_result.result.get('synthesized_result'),
                total_execution_time=execution_time,
                total_tokens_used=total_tokens,
                total_cost=total_cost,
                orchestration_quality=quality_assessment['overall_quality'],
                meta_insights=quality_assessment['insights'],
                improvements_identified=quality_assessment['improvements']
            )
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            execution_time = asyncio.get_event_loop().time() - start_time
            
            result = WorkflowResult(
                workflow_id=workflow_id,
                status="failed",
                execution_mode=execution_mode,  # NEW
                stages_completed=stage_results,
                stages_skipped=stages_skipped,  # NEW
                final_output=None,
                total_execution_time=execution_time,
                total_tokens_used=total_tokens,
                total_cost=total_cost,
                orchestration_quality=0.0,
                meta_insights=[f"Workflow failed: {str(e)}"],
                improvements_identified=["Improve error handling"]
            )
        
        # Store result
        self.execution_history.append(result)
        
        # Log summary
        self._log_workflow_summary(result)
        
        return result
    
    async def plan_orchestration_strategy(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]],
        base_strategy: OrchestrationStrategy
    ) -> OrchestrationPlan:
        """
        Plan orchestration strategy using meta-reasoning.
        
        The master LLM reasons about:
        - Task complexity and requirements
        - Available resources and constraints
        - Historical performance patterns
        - Optimal stage configurations
        """
        prompt = f"""
You are planning the orchestration strategy for a workflow.

TASK: {task_description}

CONTEXT:
- Base Strategy: {base_strategy.value}
- Time Constraints: {context.get('time_limit', 'None')}
- Quality Requirements: {context.get('quality_requirement', 'Standard')}
- Budget: {context.get('budget', 'Standard')}

AVAILABLE STRATEGIES:
1. SPEED_OPTIMIZED: Minimize time, may sacrifice some quality
2. QUALITY_OPTIMIZED: Maximum quality, may take longer
3. COST_OPTIMIZED: Minimize token usage, simpler approaches
4. BALANCED: Balance all factors equally
5. ADAPTIVE: Adjust strategy based on intermediate results

STAGE CONFIGURATION OPTIONS:
- Parallel vs Sequential execution
- Aggressive vs Conservative context optimization
- Single vs Multiple agent assignments
- Continuous vs Periodic monitoring
- Simple vs Deep result synthesis

Consider:
1. What strategy best fits this task?
2. Which stages can run in parallel?
3. What quality thresholds for each stage?
4. What time/token budgets to allocate?

Provide your orchestration plan with reasoning.
"""
        
        response = await self.master_llm.generate_with_reasoning(
            prompt=prompt,
            context=context,
            reasoning_mode=ReasoningMode.META_COGNITIVE
        )
        
        # Parse response into plan
        # For now, create a default plan
        plan = OrchestrationPlan(
            strategy=base_strategy,
            stage_configurations={
                1: {"max_subtasks": 10, "complexity_threshold": 0.7},
                2: {"optimization_level": "balanced"},
                3: {"use_llm": True, "confidence_threshold": 0.7},
                4: {"monitoring_interval": 5, "intervention_threshold": 0.6},
                5: {"synthesis_depth": "comprehensive"}
            },
            parallel_stages=[[2, 3]],  # Context and agent selection can be parallel
            quality_thresholds={i: 0.7 for i in range(1, 6)},
            time_budget=300,  # 5 minutes
            token_budget=50000,
            reasoning=response.reasoning or "Default orchestration plan"
        )
        
        return plan
    
    async def _execute_stage_1(
        self,
        task_description: str,
        config: Dict[str, Any]
    ) -> StageResult:
        """Execute Stage 1: Task Decomposition"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            decomposer = self.stage_llms['decomposition']
            result = await decomposer.decompose_task(
                task_description=task_description,
                max_subtasks=config.get('max_subtasks', 10)
            )
            
            return StageResult(
                stage_number=1,
                stage_name="Task Decomposition",
                status="success",
                result=result,
                execution_time=asyncio.get_event_loop().time() - start_time,
                tokens_used=result.get('tokens_used', 1000),
                cost=result.get('cost', 0.01),
                confidence=0.85
            )
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            return StageResult(
                stage_number=1,
                stage_name="Task Decomposition",
                status="failed",
                result=None,
                execution_time=asyncio.get_event_loop().time() - start_time,
                issues=[str(e)]
            )
    
    async def _execute_stage_2(
        self,
        subtasks: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> StageResult:
        """Execute Stage 2: Context Strategy Selection"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            selector = self.stage_llms['context_strategy']
            strategies = {}
            
            for subtask in subtasks:
                strategy = await selector.select_optimization_strategy(
                    subtask=subtask,
                    available_context={"source_count": 5, "total_tokens": 2000},
                    token_limit=config.get('token_limit', 4000)
                )
                strategies[subtask.get('subtask_id')] = strategy
            
            return StageResult(
                stage_number=2,
                stage_name="Context Strategy Selection",
                status="success",
                result={"strategies": strategies},
                execution_time=asyncio.get_event_loop().time() - start_time,
                tokens_used=len(subtasks) * 500,
                cost=len(subtasks) * 0.005,
                confidence=0.8
            )
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            return StageResult(
                stage_number=2,
                stage_name="Context Strategy Selection",
                status="failed",
                result=None,
                execution_time=asyncio.get_event_loop().time() - start_time,
                issues=[str(e)]
            )
    
    async def _execute_stage_3(
        self,
        subtasks: List[Dict[str, Any]],
        workflow_context: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> StageResult:
        """Execute Stage 3: LLM Agent Selection"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            selector = self.stage_llms['agent_selection']
            
            # Use LLM-driven selection
            agent_assignments = await selector.select_agents_for_subtasks(
                subtasks=subtasks,
                workflow_context=workflow_context
            )
            
            total_tokens = sum(
                500 for _ in subtasks  # Estimate
            )
            
            return StageResult(
                stage_number=3,
                stage_name="Agent Selection (LLM)",
                status="success",
                result=agent_assignments,
                execution_time=asyncio.get_event_loop().time() - start_time,
                tokens_used=total_tokens,
                cost=total_tokens * 0.00001,
                confidence=0.85
            )
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            return StageResult(
                stage_number=3,
                stage_name="Agent Selection (LLM)",
                status="failed",
                result=None,
                execution_time=asyncio.get_event_loop().time() - start_time,
                issues=[str(e)]
            )
    
    async def _execute_stage_4(
        self,
        subtasks: List[Dict[str, Any]],
        agent_assignments: Dict[str, Any],
        config: Dict[str, Any]
    ) -> StageResult:
        """Execute Stage 4: Execution with Monitoring"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            monitor = self.stage_llms['execution_monitor']
            
            # Simulate execution results
            execution_results = {}
            for subtask in subtasks:
                subtask_id = subtask.get('subtask_id')
                # Simulate execution
                execution_results[subtask_id] = {
                    'status': 'completed',
                    'result': f"Result for {subtask_id}",
                    'quality_score': 0.8
                }
            
            return StageResult(
                stage_number=4,
                stage_name="Execution & Monitoring",
                status="success",
                result=execution_results,
                execution_time=asyncio.get_event_loop().time() - start_time,
                tokens_used=len(subtasks) * 200,
                cost=len(subtasks) * 0.002,
                confidence=0.75
            )
        except Exception as e:
            logger.error(f"Stage 4 failed: {e}")
            return StageResult(
                stage_number=4,
                stage_name="Execution & Monitoring",
                status="failed",
                result=None,
                execution_time=asyncio.get_event_loop().time() - start_time,
                issues=[str(e)]
            )
    
    async def _execute_stage_5(
        self,
        execution_results: Dict[str, Any],
        workflow_goal: str,
        config: Dict[str, Any]
    ) -> StageResult:
        """Execute Stage 5: Result Aggregation"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            aggregator = self.stage_llms['result_aggregation']
            
            # Convert to expected format
            from modules.agents import SubtaskExecution, SubtaskStatus
            
            subtask_executions = {}
            for subtask_id, result in execution_results.items():
                subtask_executions[subtask_id] = SubtaskExecution(
                    subtask_id=subtask_id,
                    subtask_description=f"Subtask {subtask_id}",
                    agent_id=1,
                    agent_name="Agent",
                    status=SubtaskStatus.COMPLETED,
                    result=result.get('result'),
                    quality_score=result.get('quality_score', 0.7)
                )
            
            aggregated = await aggregator.aggregate_with_reasoning(
                subtask_results=subtask_executions,
                workflow_goal=workflow_goal,
                workflow_context={}
            )
            
            return StageResult(
                stage_number=5,
                stage_name="Result Aggregation (LLM)",
                status="success",
                result=aggregated.to_dict(),
                execution_time=asyncio.get_event_loop().time() - start_time,
                tokens_used=2000,
                cost=0.02,
                confidence=aggregated.confidence_score
            )
        except Exception as e:
            logger.error(f"Stage 5 failed: {e}")
            return StageResult(
                stage_number=5,
                stage_name="Result Aggregation (LLM)",
                status="failed",
                result=None,
                execution_time=asyncio.get_event_loop().time() - start_time,
                issues=[str(e)]
            )
    
    async def evaluate_orchestration_quality(
        self,
        stage_results: List[StageResult],
        plan: OrchestrationPlan
    ) -> Dict[str, Any]:
        """
        Meta-evaluation of orchestration quality.
        The master LLM reflects on the orchestration performance.
        """
        # Analyze stage performance
        successful_stages = sum(1 for s in stage_results if s.status == "success")
        total_stages = len(stage_results)
        avg_confidence = sum(s.confidence for s in stage_results) / total_stages if total_stages > 0 else 0
        
        # Meta-reasoning prompt
        stage_summary = "\n".join([
            f"Stage {s.stage_number} ({s.stage_name}): {s.status} - {s.execution_time:.2f}s"
            for s in stage_results
        ])
        
        prompt = f"""
Evaluate the orchestration quality of this workflow execution.

ORCHESTRATION PLAN:
- Strategy: {plan.strategy.value}
- Reasoning: {plan.reasoning}

STAGE RESULTS:
{stage_summary}

METRICS:
- Success Rate: {successful_stages}/{total_stages}
- Average Confidence: {avg_confidence:.2%}
- Total Time: {sum(s.execution_time for s in stage_results):.2f}s
- Total Tokens: {sum(s.tokens_used for s in stage_results)}

Analyze:
1. Did the orchestration strategy work well?
2. Which stages performed best/worst?
3. What improvements could be made?
4. Should we adjust the strategy for similar tasks?

Provide insights and recommendations.
"""
        
        response = await self.master_llm.generate_with_reasoning(
            prompt=prompt,
            reasoning_mode=ReasoningMode.META_COGNITIVE
        )
        
        # Calculate overall quality
        overall_quality = (
            (successful_stages / total_stages) * 0.4 +
            avg_confidence * 0.3 +
            (0.8 if sum(s.execution_time for s in stage_results) < plan.time_budget else 0.5) * 0.3
        )
        
        return {
            'overall_quality': overall_quality,
            'success_rate': successful_stages / total_stages,
            'avg_confidence': avg_confidence,
            'insights': [
                f"Strategy {plan.strategy.value} achieved {overall_quality:.2%} quality",
                response.reasoning or "Meta-evaluation complete"
            ],
            'improvements': [
                "Consider parallel execution for independent stages",
                "Optimize token usage in high-frequency stages"
            ]
        }
    
    async def learn_from_execution(
        self,
        workflow_id: int,
        stage_results: List[StageResult],
        quality_assessment: Dict[str, Any]
    ):
        """
        Learn from this execution to improve future orchestrations.
        Updates learning patterns for strategy selection.
        """
        # Extract patterns
        pattern = {
            'workflow_id': workflow_id,
            'timestamp': datetime.now().isoformat(),
            'strategy': self.active_workflows[workflow_id].strategy.value,
            'quality': quality_assessment['overall_quality'],
            'stage_performance': {
                s.stage_number: {
                    'status': s.status,
                    'confidence': s.confidence,
                    'time': s.execution_time
                }
                for s in stage_results
            }
        }
        
        # Update learning patterns
        strategy = self.active_workflows[workflow_id].strategy.value
        if strategy not in self.learning_patterns:
            self.learning_patterns[strategy] = []
        
        self.learning_patterns[strategy].append(pattern)
        
        # Trim history
        if len(self.learning_patterns[strategy]) > 100:
            self.learning_patterns[strategy] = self.learning_patterns[strategy][-100:]
        
        logger.info(f"📚 Learned from workflow {workflow_id}: Quality {quality_assessment['overall_quality']:.2%}")
    
    def _log_workflow_summary(self, result: WorkflowResult):
        """Log workflow execution summary"""
        logger.info("\n" + "="*60)
        logger.info("🎯 WORKFLOW EXECUTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Workflow ID: {result.workflow_id}")
        logger.info(f"Status: {result.status}")
        logger.info(f"Stages Completed: {len(result.stages_completed)}")
        logger.info(f"Total Time: {result.total_execution_time:.2f}s")
        logger.info(f"Total Tokens: {result.total_tokens_used:,}")
        logger.info(f"Total Cost: ${result.total_cost:.4f}")
        logger.info(f"Orchestration Quality: {result.orchestration_quality:.2%}")
        
        if result.meta_insights:
            logger.info("\n📊 Meta Insights:")
            for insight in result.meta_insights:
                logger.info(f"  - {insight}")
        
        if result.improvements_identified:
            logger.info("\n💡 Improvements Identified:")
            for improvement in result.improvements_identified:
                logger.info(f"  - {improvement}")
        
        logger.info("="*60 + "\n")
    
    async def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics"""
        if not self.execution_history:
            return {
                "total_workflows": 0,
                "message": "No workflows executed yet"
            }
        
        total = len(self.execution_history)
        successful = sum(1 for w in self.execution_history if w.status == "completed")
        
        avg_quality = sum(w.orchestration_quality for w in self.execution_history) / total
        avg_time = sum(w.total_execution_time for w in self.execution_history) / total
        avg_cost = sum(w.total_cost for w in self.execution_history) / total
        
        # Strategy performance
        strategy_stats = {}
        for pattern_strategy, patterns in self.learning_patterns.items():
            if patterns:
                strategy_stats[pattern_strategy] = {
                    'count': len(patterns),
                    'avg_quality': sum(p['quality'] for p in patterns) / len(patterns)
                }
        
        return {
            'total_workflows': total,
            'success_rate': successful / total,
            'avg_orchestration_quality': avg_quality,
            'avg_execution_time': avg_time,
            'avg_cost': avg_cost,
            'strategy_performance': strategy_stats,
            'best_strategy': max(strategy_stats, key=lambda s: strategy_stats[s]['avg_quality']) if strategy_stats else None
        }

