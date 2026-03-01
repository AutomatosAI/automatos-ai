"""
Adaptive Execution Monitor - Stage 4 Enhancement
=================================================

PRD-16: Stage 4 - LLM monitors execution and can intervene when needed.
Provides real-time monitoring, failure detection, and adaptive recovery.

The LLM watches execution and can:
- Detect failures and performance issues
- Decide on intervention strategies
- Retry with different agents
- Modify task parameters
- Abort or skip non-critical tasks
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from modules.agents import SubtaskExecution, SubtaskStatus
from core.llm import (
    OrchestratorLLM,
    FunctionRegistry,
    FunctionSpec,
    FunctionParameter,
    FunctionCategory
)

logger = logging.getLogger(__name__)


class InterventionAction(Enum):
    """Possible intervention actions"""
    CONTINUE = "continue"          # No intervention needed
    RETRY_SAME = "retry_same"      # Retry with same agent
    RETRY_DIFFERENT = "retry_different"  # Try different agent
    MODIFY_TASK = "modify_task"    # Modify task parameters
    SKIP = "skip"                  # Skip this subtask
    ABORT = "abort"                # Abort entire workflow
    ESCALATE = "escalate"          # Escalate to human


@dataclass
class ExecutionIssue:
    """Detected execution issue"""
    issue_type: str                # "failure", "timeout", "quality", "performance"
    severity: str                  # "low", "medium", "high", "critical"
    description: str
    detected_at: datetime
    subtask_id: str
    agent_id: int
    error_details: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationDecision:
    """Decision on how to adapt"""
    action: InterventionAction
    reasoning: str
    confidence: float
    next_steps: List[str]
    alternative_agent_id: Optional[int] = None
    modified_parameters: Optional[Dict[str, Any]] = None
    estimated_recovery_time: Optional[int] = None
    risk_assessment: Optional[str] = None


class AdaptiveExecutionMonitor:
    """
    LLM monitors execution and adapts when needed.
    Reference: Tool-Augmented Reasoning - Adaptive intervention
    """
    
    def __init__(
        self,
        llm: Optional[OrchestratorLLM] = None,
        intervention_threshold: float = 0.7,
        max_retries: int = 2
    ):
        """
        Initialize adaptive execution monitor.
        
        Args:
            llm: Orchestrator LLM instance
            intervention_threshold: Confidence threshold for intervention
            max_retries: Maximum retry attempts per subtask
        """
        self.llm = llm or OrchestratorLLM(temperature=0.5)  # Lower temp for critical decisions
        self.intervention_threshold = intervention_threshold
        self.max_retries = max_retries
        
        # Initialize function registry
        self.registry = FunctionRegistry()
        self._register_functions()
        
        # Monitoring state
        self.active_executions: Dict[str, SubtaskExecution] = {}
        self.intervention_history: List[Dict[str, Any]] = []
        self.retry_counts: Dict[str, int] = {}
        
        logger.info("AdaptiveExecutionMonitor initialized")
    
    def _register_functions(self):
        """Register monitoring functions"""
        
        # 1. Analyze failure cause
        self.registry.register(FunctionSpec(
            name="analyze_failure_cause",
            description="Diagnose why execution failed",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="execution",
                    type="object",
                    description="Execution details",
                    required=True
                ),
                FunctionParameter(
                    name="error_message",
                    type="string",
                    description="Error message if available",
                    required=False
                )
            ],
            implementation=self._analyze_failure
        ))
        
        # 2. Get alternative agents
        self.registry.register(FunctionSpec(
            name="get_alternative_agents",
            description="Find other agents that could handle this task",
            category=FunctionCategory.QUERY,
            parameters=[
                FunctionParameter(
                    name="subtask",
                    type="object",
                    description="Subtask details",
                    required=True
                ),
                FunctionParameter(
                    name="exclude_agent_id",
                    type="number",
                    description="Agent to exclude (failed agent)",
                    required=False
                )
            ],
            implementation=self._get_alternatives
        ))
        
        # 3. Estimate retry success
        self.registry.register(FunctionSpec(
            name="estimate_retry_success",
            description="Predict probability of retry success",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="agent_id",
                    type="number",
                    description="Agent ID",
                    required=True
                ),
                FunctionParameter(
                    name="subtask",
                    type="object",
                    description="Subtask details",
                    required=True
                ),
                FunctionParameter(
                    name="previous_failures",
                    type="array",
                    description="Previous failure details",
                    required=False,
                    items={"type": "object"}
                )
            ],
            implementation=self._estimate_success
        ))
        
        # 4. Check dependency impact
        self.registry.register(FunctionSpec(
            name="check_dependency_impact",
            description="Analyze impact of skipping this subtask",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="subtask_id",
                    type="string",
                    description="Subtask identifier",
                    required=True
                ),
                FunctionParameter(
                    name="workflow_context",
                    type="object",
                    description="Workflow context",
                    required=True
                )
            ],
            implementation=self._check_dependencies
        ))
        
        # 5. Get execution metrics
        self.registry.register(FunctionSpec(
            name="get_execution_metrics",
            description="Get current execution performance metrics",
            category=FunctionCategory.MONITORING,
            parameters=[
                FunctionParameter(
                    name="execution_id",
                    type="string",
                    description="Execution identifier",
                    required=True
                )
            ],
            implementation=self._get_metrics
        ))
    
    async def monitor_and_adapt(
        self,
        subtask_execution: SubtaskExecution,
        workflow_context: Dict[str, Any]
    ) -> AdaptationDecision:
        """
        Monitor execution and decide on adaptation.
        
        Args:
            subtask_execution: Current execution state
            workflow_context: Workflow context
            
        Returns:
            Adaptation decision
        """
        # Check if intervention is needed
        issue = self._detect_issue(subtask_execution)
        
        if not issue:
            return AdaptationDecision(
                action=InterventionAction.CONTINUE,
                reasoning="Execution proceeding normally",
                confidence=1.0,
                next_steps=["Continue monitoring"]
            )
        
        # Log issue
        logger.warning(f"⚠️ Issue detected: {issue.issue_type} - {issue.description}")
        
        # Build intervention prompt
        prompt = self._build_intervention_prompt(
            issue,
            subtask_execution,
            workflow_context
        )
        
        # Get functions for monitoring
        functions = self.registry.get_for_prompt()
        
        # LLM reasoning about intervention
        response = await self.llm.generate_with_functions(
            prompt=prompt,
            functions=functions,
            function_executor=self._execute_function,
            max_function_calls=6,
            context={
                "issue": issue,
                "execution": subtask_execution,
                "workflow": workflow_context
            }
        )
        
        # Parse response
        try:
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {
                    "decision": "continue",
                    "reasoning": response.content,
                    "confidence": response.confidence,
                    "next_steps": []
                }
            
            # Map decision to action
            action_map = {
                "continue": InterventionAction.CONTINUE,
                "retry_same": InterventionAction.RETRY_SAME,
                "retry_different": InterventionAction.RETRY_DIFFERENT,
                "modify": InterventionAction.MODIFY_TASK,
                "skip": InterventionAction.SKIP,
                "abort": InterventionAction.ABORT,
                "escalate": InterventionAction.ESCALATE
            }
            
            action = action_map.get(
                result.get("decision", "continue").lower(),
                InterventionAction.CONTINUE
            )
            
            decision = AdaptationDecision(
                action=action,
                reasoning=result.get("reasoning", ""),
                confidence=float(result.get("confidence", 0.5)),
                next_steps=result.get("next_steps", []),
                alternative_agent_id=result.get("alternative_agent_id"),
                modified_parameters=result.get("modified_parameters"),
                estimated_recovery_time=result.get("estimated_recovery_time"),
                risk_assessment=result.get("risk_assessment")
            )
            
        except Exception as e:
            logger.error(f"Failed to parse intervention response: {e}")
            decision = AdaptationDecision(
                action=InterventionAction.CONTINUE,
                reasoning="Parse error, defaulting to continue",
                confidence=0.3,
                next_steps=["Manual review recommended"]
            )
        
        # Log decision
        logger.info(f"🎯 Intervention Decision: {decision.action.value}")
        logger.info(f"   Confidence: {decision.confidence:.2%}")
        logger.info(f"   Reasoning: {decision.reasoning[:200]}...")
        
        # Update history
        self._update_intervention_history(
            subtask_execution.subtask_id,
            issue,
            decision
        )
        
        return decision
    
    async def evaluate_stage_outcome(
        self,
        stage_name: str,
        stage_results: Dict[str, Any],
        workflow_context: Dict[str, Any],
        attempt_number: int = 1,
    ) -> AdaptationDecision:
        """
        Stage-level evaluation: assess whether a pipeline stage produced
        acceptable results, and decide whether to continue, retry, or abort.

        Two-tier approach: fast heuristic first, LLM only when needed.

        Args:
            stage_name: "execution" or "quality"
            stage_results: execution_summary or quality_assessment dict
            workflow_context: task_description, steps, agent_assignments
            attempt_number: current attempt (1 or 2)

        Returns:
            AdaptationDecision with action and reasoning
        """
        # ── Tier 1: Heuristic fast-path ──
        if stage_name == "execution":
            success_rate = stage_results.get("success_rate", 0)
            # High success on first attempt → continue without LLM
            if success_rate >= 0.8:
                return AdaptationDecision(
                    action=InterventionAction.CONTINUE,
                    reasoning=f"Execution success rate {success_rate:.0%} is acceptable",
                    confidence=0.95,
                    next_steps=["Proceed to result aggregation"],
                )
            # Second attempt with moderate success → accept and move on
            if attempt_number >= 2 and success_rate >= 0.4:
                return AdaptationDecision(
                    action=InterventionAction.CONTINUE,
                    reasoning=f"Retry attempt {attempt_number}: success rate {success_rate:.0%} accepted (max retries reached)",
                    confidence=0.7,
                    next_steps=["Accept partial results, proceed"],
                )
        elif stage_name == "quality":
            passes = stage_results.get("passes_threshold", False)
            overall_score = stage_results.get("overall_score", 0)
            # Already passing → continue
            if passes:
                return AdaptationDecision(
                    action=InterventionAction.CONTINUE,
                    reasoning=f"Quality score {overall_score:.0%} passes threshold",
                    confidence=0.95,
                    next_steps=["Proceed to memory storage"],
                )
            # Second attempt → accept whatever we have
            if attempt_number >= 2:
                return AdaptationDecision(
                    action=InterventionAction.CONTINUE,
                    reasoning=f"Retry attempt {attempt_number}: quality {overall_score:.0%} accepted (max retries reached)",
                    confidence=0.6,
                    next_steps=["Accept current quality, proceed"],
                )

        # ── Tier 2: LLM evaluation for ambiguous cases ──
        try:
            from core.llm import create_llm_manager

            llm = create_llm_manager(service_name="complexity_assessor")

            prompt = self._build_stage_evaluation_prompt(
                stage_name, stage_results, workflow_context, attempt_number
            )

            response = await llm.generate(prompt)
            content = response if isinstance(response, str) else getattr(response, "content", str(response))

            return self._parse_stage_evaluation(content, stage_name, attempt_number)

        except Exception as e:
            logger.error(f"LLM stage evaluation failed: {e}", exc_info=True)
            # Fallback: conservative decision based on stage
            if stage_name == "execution":
                success_rate = stage_results.get("success_rate", 0)
                if success_rate < 0.3 and attempt_number < 2:
                    return AdaptationDecision(
                        action=InterventionAction.RETRY_SAME,
                        reasoning=f"LLM unavailable; heuristic retry (success_rate={success_rate:.0%})",
                        confidence=0.5,
                        next_steps=["Retry execution with same agents"],
                    )
            return AdaptationDecision(
                action=InterventionAction.CONTINUE,
                reasoning=f"LLM evaluation failed, defaulting to continue: {e}",
                confidence=0.4,
                next_steps=["Proceed with current results"],
            )

    def _build_stage_evaluation_prompt(
        self,
        stage_name: str,
        stage_results: Dict[str, Any],
        workflow_context: Dict[str, Any],
        attempt_number: int,
    ) -> str:
        """Build the LLM prompt for stage-level evaluation."""
        task_desc = workflow_context.get("task_description", "Unknown task")[:500]
        steps_summary = ""
        for step in workflow_context.get("steps", [])[:6]:
            name = step.get("name", step.get("description", "?"))[:80]
            status = step.get("execution_result", {}).get("status", "pending")
            steps_summary += f"\n  - {name} → {status}"

        if stage_name == "execution":
            metrics = (
                f"Success Rate: {stage_results.get('success_rate', 0):.0%}\n"
                f"Total Subtasks: {stage_results.get('total_subtasks', '?')}\n"
                f"Completed: {stage_results.get('completed', '?')}\n"
                f"Failed: {stage_results.get('failed', '?')}\n"
                f"Total Tokens: {stage_results.get('total_tokens_used', 0)}\n"
                f"Total Time: {stage_results.get('total_execution_time_ms', 0)}ms"
            )
        else:
            metrics = (
                f"Overall Score: {stage_results.get('overall_score', 0):.0%}\n"
                f"Passes Threshold: {stage_results.get('passes_threshold', False)}\n"
                f"Strengths: {stage_results.get('strengths', [])}\n"
                f"Weaknesses: {stage_results.get('weaknesses', [])}"
            )

        return f"""STAGE EVALUATION — Decide whether to continue, retry, or abort.

STAGE: {stage_name}
ATTEMPT: {attempt_number}/2
TASK: {task_desc}

SUBTASK RESULTS:{steps_summary}

METRICS:
{metrics}

DECISION OPTIONS:
- continue: Results are acceptable, proceed to next stage
- retry_same: Re-run this stage with the same agents (transient failures likely)
- retry_different: Re-select agents and re-run (agent mismatch likely)
- skip: Accept partial results and proceed to response generation
- abort: Mark execution as failed (unrecoverable)

RULES:
- If attempt >= 2, prefer "continue" or "skip" over retry
- If > 50% subtasks succeeded, lean toward "continue"
- Only "abort" if results are completely unusable
- Only "retry_different" if agent mismatch is the likely cause

Respond with ONLY a JSON object:
{{"decision": "continue|retry_same|retry_different|skip|abort", "reasoning": "brief explanation", "confidence": 0.0-1.0}}"""

    def _parse_stage_evaluation(
        self, content: str, stage_name: str, attempt_number: int
    ) -> AdaptationDecision:
        """Parse LLM response into an AdaptationDecision."""
        import json as json_mod
        import re

        action_map = {
            "continue": InterventionAction.CONTINUE,
            "retry_same": InterventionAction.RETRY_SAME,
            "retry_different": InterventionAction.RETRY_DIFFERENT,
            "skip": InterventionAction.SKIP,
            "abort": InterventionAction.ABORT,
        }

        try:
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                result = json_mod.loads(json_match.group())
            else:
                result = {"decision": "continue", "reasoning": content[:200], "confidence": 0.5}

            decision_str = result.get("decision", "continue").lower().strip()
            action = action_map.get(decision_str, InterventionAction.CONTINUE)

            # Safety: don't allow retry on second attempt
            if attempt_number >= 2 and action in (InterventionAction.RETRY_SAME, InterventionAction.RETRY_DIFFERENT):
                action = InterventionAction.CONTINUE
                result["reasoning"] = f"Max retries reached. Original: {result.get('reasoning', '')}"

            return AdaptationDecision(
                action=action,
                reasoning=result.get("reasoning", ""),
                confidence=float(result.get("confidence", 0.5)),
                next_steps=[f"After {stage_name}: {action.value}"],
            )
        except Exception as e:
            logger.error(f"Failed to parse stage evaluation: {e}")
            return AdaptationDecision(
                action=InterventionAction.CONTINUE,
                reasoning=f"Parse error, defaulting to continue: {e}",
                confidence=0.3,
                next_steps=["Continue with current results"],
            )

    async def monitor_batch(
        self,
        executions: List[SubtaskExecution],
        workflow_context: Dict[str, Any],
        parallel: bool = True
    ) -> List[AdaptationDecision]:
        """
        Monitor multiple executions.
        
        Args:
            executions: List of executions to monitor
            workflow_context: Workflow context
            parallel: Monitor in parallel
            
        Returns:
            List of adaptation decisions
        """
        if parallel:
            tasks = [
                self.monitor_and_adapt(exec, workflow_context)
                for exec in executions
            ]
            decisions = await asyncio.gather(*tasks)
        else:
            decisions = []
            for exec in executions:
                decision = await self.monitor_and_adapt(exec, workflow_context)
                decisions.append(decision)
        
        return decisions
    
    def _detect_issue(self, execution: SubtaskExecution) -> Optional[ExecutionIssue]:
        """Detect if there's an issue with execution"""
        
        # Check for failure
        if execution.status == SubtaskStatus.FAILED:
            return ExecutionIssue(
                issue_type="failure",
                severity="high",
                description=f"Task failed: {execution.error or 'Unknown error'}",
                detected_at=datetime.now(),
                subtask_id=execution.subtask_id,
                agent_id=execution.agent_id,
                error_details=execution.error
            )
        
        # Check for timeout
        if execution.execution_time > 300:  # 5 minutes
            return ExecutionIssue(
                issue_type="timeout",
                severity="medium",
                description=f"Execution taking too long: {execution.execution_time}s",
                detected_at=datetime.now(),
                subtask_id=execution.subtask_id,
                agent_id=execution.agent_id,
                metrics={"execution_time": execution.execution_time}
            )
        
        # Check for quality issues
        if execution.quality_score and execution.quality_score < 0.5:
            return ExecutionIssue(
                issue_type="quality",
                severity="medium",
                description=f"Low quality score: {execution.quality_score}",
                detected_at=datetime.now(),
                subtask_id=execution.subtask_id,
                agent_id=execution.agent_id,
                metrics={"quality_score": execution.quality_score}
            )
        
        # Check for performance issues
        if execution.status == SubtaskStatus.RUNNING and execution.execution_time > 180:
            return ExecutionIssue(
                issue_type="performance",
                severity="low",
                description="Slow execution detected",
                detected_at=datetime.now(),
                subtask_id=execution.subtask_id,
                agent_id=execution.agent_id,
                metrics={"execution_time": execution.execution_time}
            )
        
        return None
    
    def _build_intervention_prompt(
        self,
        issue: ExecutionIssue,
        execution: SubtaskExecution,
        workflow_context: Dict[str, Any]
    ) -> str:
        """Build prompt for intervention decision"""
        
        # Get retry count
        retry_count = self.retry_counts.get(execution.subtask_id, 0)
        
        prompt = f"""
EXECUTION ISSUE DETECTED:

ISSUE DETAILS:
- Type: {issue.issue_type}
- Severity: {issue.severity}
- Description: {issue.description}
- Error: {issue.error_details or 'None'}

SUBTASK: {execution.subtask_description}
AGENT: {execution.agent_name} (ID: {execution.agent_id})
STATUS: {execution.status.value}
QUALITY SCORE: {execution.quality_score or 'N/A'}
EXECUTION TIME: {execution.execution_time}s
RETRY COUNT: {retry_count}/{self.max_retries}

WORKFLOW CONTEXT:
- Workflow ID: {workflow_context.get('workflow_id')}
- Subtasks Completed: {len(workflow_context.get('completed_subtasks', []))}
- Subtasks Remaining: {workflow_context.get('remaining_subtasks', 'Unknown')}
- Time Remaining: {workflow_context.get('time_remaining', 'Unlimited')}
- Budget Remaining: {workflow_context.get('budget_remaining', 'Unlimited')}

FUNCTIONS AVAILABLE:
- analyze_failure_cause(execution) → Diagnose why it failed
- get_alternative_agents(subtask) → Find other agents
- estimate_retry_success(agent, subtask) → Predict retry outcome
- check_dependency_impact(subtask_id) → See what depends on this
- get_execution_metrics(execution_id) → Get performance data

YOUR TASK:
Decide what to do about this failed/poor execution.

OPTIONS:
A) continue: No intervention needed (minor issue)
B) retry_same: Retry with same agent (transient error)
C) retry_different: Retry with different agent (agent not suitable)
D) modify: Modify task parameters and retry
E) skip: Skip this subtask (not critical)
F) abort: Abort entire workflow (critical failure)
G) escalate: Escalate to human operator

DECISION PROCESS:
1. Call analyze_failure_cause() to understand what went wrong
2. Consider: Is this a transient error or fundamental mismatch?
3. If retrying, call get_alternative_agents() to find better options
4. Call estimate_retry_success() to predict outcomes
5. Call check_dependency_impact() to assess criticality
6. Make decision based on analysis

FACTORS TO CONSIDER:
- Is error transient (timeout, rate limit) or permanent (wrong agent)?
- Have we already retried? (Current: {retry_count}/{self.max_retries})
- How critical is this subtask to overall workflow?
- Do we have time/budget for retry?
- Are there better agents available?
- Will failure cascade to dependent subtasks?

Provide your response as:
{{
    "decision": "continue|retry_same|retry_different|modify|skip|abort|escalate",
    "reasoning": "Detailed explanation",
    "confidence": 0.0-1.0,
    "next_steps": ["Step 1", "Step 2"],
    "alternative_agent_id": <if retry_different>,
    "modified_parameters": {{...}} <if modify>,
    "estimated_recovery_time": <seconds>,
    "risk_assessment": "Assessment of risks"
}}
"""
        return prompt
    
    async def _execute_function(
        self,
        function_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute function calls"""
        logger.info(f"  🔧 Executing: {function_name}({parameters})")
        
        spec = self.registry.get(function_name)
        if not spec or not spec.implementation:
            return {"error": f"Function {function_name} not found"}
        
        try:
            result = await spec.implementation(parameters)
            return result
        except Exception as e:
            logger.error(f"Function execution failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_failure(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze failure cause"""
        execution = parameters['execution']
        error_message = parameters.get('error_message', '')
        
        # Pattern matching for common errors
        transient_errors = [
            "timeout", "rate limit", "connection", "temporary",
            "unavailable", "retry", "busy"
        ]
        
        permanent_errors = [
            "not found", "invalid", "unauthorized", "forbidden",
            "not supported", "incompatible"
        ]
        
        error_lower = error_message.lower()
        
        is_transient = any(pattern in error_lower for pattern in transient_errors)
        is_permanent = any(pattern in error_lower for pattern in permanent_errors)
        
        if is_transient:
            failure_type = "transient"
            recommendation = "retry with same agent"
        elif is_permanent:
            failure_type = "permanent"
            recommendation = "try different agent"
        else:
            failure_type = "unknown"
            recommendation = "investigate further"
        
        return {
            "failure_type": failure_type,
            "is_transient": is_transient,
            "is_permanent": is_permanent,
            "error_category": "timeout" if "timeout" in error_lower else "general",
            "recommendation": recommendation,
            "confidence": 0.8 if is_transient or is_permanent else 0.5
        }
    
    async def _get_alternatives(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get alternative agents"""
        subtask = parameters['subtask']
        exclude_id = parameters.get('exclude_agent_id')
        
        # Simulate alternative agents
        alternatives = [
            {
                "agent_id": 101,
                "agent_name": "Backup Agent A",
                "match_score": 0.75,
                "availability": "available"
            },
            {
                "agent_id": 102,
                "agent_name": "Backup Agent B",
                "match_score": 0.68,
                "availability": "busy"
            }
        ]
        
        # Filter out excluded agent
        if exclude_id:
            alternatives = [a for a in alternatives if a['agent_id'] != exclude_id]
        
        return {
            "alternatives": alternatives,
            "best_alternative": alternatives[0] if alternatives else None,
            "total_found": len(alternatives)
        }
    
    async def _estimate_success(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate retry success probability"""
        agent_id = parameters['agent_id']
        subtask = parameters['subtask']
        previous_failures = parameters.get('previous_failures', [])
        
        # Simple heuristic
        base_probability = 0.7
        
        # Reduce for each previous failure
        failure_penalty = len(previous_failures) * 0.2
        
        # Adjust based on agent (simulation)
        agent_factor = 0.1 if agent_id > 100 else 0.0  # New agents slightly better
        
        success_probability = max(0.1, min(0.95, base_probability - failure_penalty + agent_factor))
        
        return {
            "agent_id": agent_id,
            "success_probability": success_probability,
            "previous_failures": len(previous_failures),
            "recommendation": "retry" if success_probability > 0.5 else "skip or escalate"
        }
    
    async def _check_dependencies(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check dependency impact"""
        subtask_id = parameters['subtask_id']
        workflow_context = parameters['workflow_context']
        
        # Simulate dependency analysis
        dependent_tasks = []
        for task in workflow_context.get('remaining_tasks', []):
            if subtask_id in task.get('dependencies', []):
                dependent_tasks.append(task['id'])
        
        criticality = "high" if len(dependent_tasks) > 2 else "medium" if dependent_tasks else "low"
        
        return {
            "subtask_id": subtask_id,
            "dependent_tasks": dependent_tasks,
            "dependency_count": len(dependent_tasks),
            "criticality": criticality,
            "can_skip": criticality == "low",
            "impact_assessment": f"{len(dependent_tasks)} tasks depend on this"
        }
    
    async def _get_metrics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get execution metrics"""
        execution_id = parameters['execution_id']
        
        # Simulate metrics
        return {
            "execution_id": execution_id,
            "cpu_usage": 45.2,
            "memory_usage": 62.8,
            "tokens_used": 1250,
            "api_calls": 5,
            "error_rate": 0.02,
            "avg_response_time": 1.2
        }
    
    def _update_intervention_history(
        self,
        subtask_id: str,
        issue: ExecutionIssue,
        decision: AdaptationDecision
    ):
        """Update intervention history"""
        self.intervention_history.append({
            "timestamp": datetime.now().isoformat(),
            "subtask_id": subtask_id,
            "issue_type": issue.issue_type,
            "severity": issue.severity,
            "action": decision.action.value,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning[:200]
        })
        
        # Update retry count if retrying
        if decision.action in [InterventionAction.RETRY_SAME, InterventionAction.RETRY_DIFFERENT]:
            self.retry_counts[subtask_id] = self.retry_counts.get(subtask_id, 0) + 1
    
    async def get_intervention_summary(self) -> Dict[str, Any]:
        """Get summary of interventions"""
        if not self.intervention_history:
            return {
                "total_interventions": 0,
                "message": "No interventions yet"
            }
        
        # Count by action type
        action_counts = {}
        for entry in self.intervention_history:
            action = entry['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Success rate (simplified)
        total = len(self.intervention_history)
        successful = sum(
            1 for e in self.intervention_history
            if e['action'] in ['continue', 'retry_same', 'retry_different']
            and e['confidence'] > 0.7
        )
        
        return {
            "total_interventions": total,
            "action_counts": action_counts,
            "success_rate": successful / total if total > 0 else 0,
            "most_common_issue": max(
                set(e['issue_type'] for e in self.intervention_history),
                key=lambda x: sum(1 for e in self.intervention_history if e['issue_type'] == x)
            ),
            "avg_confidence": sum(e['confidence'] for e in self.intervention_history) / total
        }




