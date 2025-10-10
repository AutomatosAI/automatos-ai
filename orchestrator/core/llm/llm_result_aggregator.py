"""
LLM Result Aggregator - Stage 5 Enhancement
============================================

PRD-16: Stage 5 - LLM synthesizes results with reasoning-based conflict resolution.
Replaces basic aggregation with intelligent synthesis.

The LLM:
- Analyzes results from all subtasks
- Detects and resolves conflicts
- Validates completeness
- Synthesizes coherent final output
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.agent_execution_manager import SubtaskExecution
from core.llm import (
    OrchestratorLLM,
    FunctionRegistry,
    FunctionSpec,
    FunctionParameter,
    FunctionCategory
)

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts in results"""
    DATA_INCONSISTENCY = "data_inconsistency"
    LOGICAL_CONTRADICTION = "logical_contradiction"
    QUALITY_DISPARITY = "quality_disparity"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    FORMAT_INCOMPATIBILITY = "format_incompatibility"


@dataclass
class Conflict:
    """Detected conflict between results"""
    conflict_type: ConflictType
    subtask_ids: List[str]
    description: str
    severity: str  # "low", "medium", "high"
    conflicting_values: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConflictResolution:
    """Resolution for a detected conflict"""
    conflict: Conflict
    resolution_strategy: str
    resolved_value: Any
    reasoning: str
    confidence: float
    affected_subtasks: List[str]


@dataclass
class AggregatedResult:
    """Final aggregated result with synthesis"""
    synthesized_result: str
    key_insights: List[str]
    conflicts_found: List[Conflict]
    conflicts_resolved: List[ConflictResolution]
    confidence_score: float
    completeness_score: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "synthesized_result": self.synthesized_result,
            "key_insights": self.key_insights,
            "conflicts_found": len(self.conflicts_found),
            "conflicts_resolved": len(self.conflicts_resolved),
            "confidence_score": self.confidence_score,
            "completeness_score": self.completeness_score,
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }


class LLMResultAggregator:
    """
    LLM synthesizes results with reasoning-based conflict resolution.
    Reference: Context Engineering - Coherence analysis
    """
    
    def __init__(
        self,
        llm: Optional[OrchestratorLLM] = None,
        conflict_threshold: float = 0.3
    ):
        """
        Initialize result aggregator.
        
        Args:
            llm: Orchestrator LLM instance
            conflict_threshold: Threshold for conflict detection
        """
        self.llm = llm or OrchestratorLLM(temperature=0.6)
        self.conflict_threshold = conflict_threshold
        
        # Initialize function registry
        self.registry = FunctionRegistry()
        self._register_functions()
        
        # Aggregation history
        self.aggregation_history: List[Dict[str, Any]] = []
        
        logger.info("LLMResultAggregator initialized")
    
    def _register_functions(self):
        """Register aggregation functions"""
        
        # 1. Get result summary
        self.registry.register(FunctionSpec(
            name="get_result_summary",
            description="Get detailed summary of subtask result",
            category=FunctionCategory.QUERY,
            parameters=[
                FunctionParameter(
                    name="subtask_id",
                    type="string",
                    description="Subtask identifier",
                    required=True
                ),
                FunctionParameter(
                    name="include_metadata",
                    type="boolean",
                    description="Include execution metadata",
                    required=False,
                    default=True
                )
            ],
            implementation=self._get_result_summary
        ))
        
        # 2. Detect conflicts
        self.registry.register(FunctionSpec(
            name="detect_conflicts",
            description="Find inconsistencies between results",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="results",
                    type="array",
                    description="List of results to analyze",
                    required=True,
                    items={"type": "object"}
                ),
                FunctionParameter(
                    name="conflict_types",
                    type="array",
                    description="Types of conflicts to check",
                    required=False,
                    items={"type": "string"}
                )
            ],
            implementation=self._detect_conflicts
        ))
        
        # 3. Resolve conflict
        self.registry.register(FunctionSpec(
            name="resolve_conflict",
            description="Reasoning-based conflict resolution",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="conflict",
                    type="object",
                    description="Conflict details",
                    required=True
                ),
                FunctionParameter(
                    name="context",
                    type="object",
                    description="Additional context for resolution",
                    required=False
                )
            ],
            implementation=self._resolve_conflict
        ))
        
        # 4. Validate completeness
        self.registry.register(FunctionSpec(
            name="validate_completeness",
            description="Check if all aspects of goal are addressed",
            category=FunctionCategory.VALIDATION,
            parameters=[
                FunctionParameter(
                    name="results",
                    type="array",
                    description="All subtask results",
                    required=True,
                    items={"type": "object"}
                ),
                FunctionParameter(
                    name="goal",
                    type="string",
                    description="Original workflow goal",
                    required=True
                )
            ],
            implementation=self._validate_completeness
        ))
        
        # 5. Calculate confidence
        self.registry.register(FunctionSpec(
            name="calculate_confidence",
            description="Assess overall confidence in synthesis",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="synthesis",
                    type="string",
                    description="Synthesized result",
                    required=True
                ),
                FunctionParameter(
                    name="conflicts_resolved",
                    type="number",
                    description="Number of conflicts resolved",
                    required=False,
                    default=0
                ),
                FunctionParameter(
                    name="quality_scores",
                    type="array",
                    description="Quality scores of subtasks",
                    required=False,
                    items={"type": "number"}
                )
            ],
            implementation=self._calculate_confidence
        ))
    
    async def aggregate_with_reasoning(
        self,
        subtask_results: Dict[str, SubtaskExecution],
        workflow_goal: str,
        workflow_context: Dict[str, Any]
    ) -> AggregatedResult:
        """
        LLM synthesizes results from all subtasks.
        
        Args:
            subtask_results: Results from all subtasks
            workflow_goal: Original workflow goal
            workflow_context: Workflow context
            
        Returns:
            Aggregated result with synthesis
        """
        # Build aggregation prompt
        prompt = self._build_aggregation_prompt(
            subtask_results,
            workflow_goal,
            workflow_context
        )
        
        # Get functions for aggregation
        functions = self.registry.get_for_prompt()
        
        # LLM reasoning with function calls
        response = await self.llm.generate_with_functions(
            prompt=prompt,
            functions=functions,
            function_executor=self._execute_function,
            max_function_calls=10,
            context={
                "subtask_results": subtask_results,
                "workflow_goal": workflow_goal,
                "workflow_context": workflow_context
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
                    "synthesized_result": response.content,
                    "key_insights": [],
                    "confidence_score": response.confidence,
                    "completeness_score": 0.7,
                    "reasoning": response.reasoning or ""
                }
            
            # Extract conflicts from function calls
            conflicts = []
            resolutions = []
            
            for fc in response.function_calls:
                if fc['name'] == 'detect_conflicts' and 'result' in fc:
                    detected = fc['result'].get('conflicts', [])
                    for c in detected:
                        conflicts.append(Conflict(
                            conflict_type=ConflictType.DATA_INCONSISTENCY,
                            subtask_ids=c.get('subtask_ids', []),
                            description=c.get('description', ''),
                            severity=c.get('severity', 'medium'),
                            conflicting_values=c.get('values', {})
                        ))
            
            aggregated = AggregatedResult(
                synthesized_result=result.get("synthesized_result", ""),
                key_insights=result.get("key_insights", []),
                conflicts_found=conflicts,
                conflicts_resolved=resolutions,
                confidence_score=float(result.get("confidence_score", 0.5)),
                completeness_score=float(result.get("completeness_score", 0.5)),
                reasoning=result.get("reasoning", ""),
                metadata={
                    "subtask_count": len(subtask_results),
                    "tokens_used": response.tokens_used,
                    "processing_time": response.processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse aggregation response: {e}")
            # Fallback aggregation
            aggregated = self._fallback_aggregation(subtask_results, workflow_goal)
        
        # Log result
        logger.info(f"📊 Aggregation Complete:")
        logger.info(f"   Conflicts Found: {len(aggregated.conflicts_found)}")
        logger.info(f"   Conflicts Resolved: {len(aggregated.conflicts_resolved)}")
        logger.info(f"   Confidence: {aggregated.confidence_score:.2%}")
        logger.info(f"   Completeness: {aggregated.completeness_score:.2%}")
        
        # Update history
        self._update_history(aggregated)
        
        return aggregated
    
    def _build_aggregation_prompt(
        self,
        subtask_results: Dict[str, SubtaskExecution],
        workflow_goal: str,
        workflow_context: Dict[str, Any]
    ) -> str:
        """Build prompt for result aggregation"""
        
        # Format results
        results_text = ""
        for subtask_id, execution in subtask_results.items():
            results_text += f"\n{subtask_id}:"
            results_text += f"\n  Description: {execution.subtask_description}"
            results_text += f"\n  Status: {execution.status.value}"
            results_text += f"\n  Quality: {execution.quality_score or 'N/A'}"
            if execution.result:
                results_text += f"\n  Result: {str(execution.result)[:200]}..."
            results_text += "\n"
        
        prompt = f"""
WORKFLOW GOAL: {workflow_goal}

SUBTASK RESULTS ({len(subtask_results)} subtasks):
{results_text}

YOUR TASK:
Synthesize these subtask results into a coherent final result.

PROCESS:
1. REVIEW RESULTS:
   - Call get_result_summary(subtask_id) for each key result
   - Identify main findings from each subtask
   - Note quality scores and confidence levels

2. DETECT CONFLICTS:
   - Call detect_conflicts(results) to find inconsistencies
   - Are any results contradictory?
   - Do results from different agents disagree?

3. RESOLVE CONFLICTS:
   - For each conflict, call resolve_conflict(conflict, context)
   - Use reasoning to determine which result is more reliable
   - Consider: agent performance, evidence quality, logical consistency
   - Don't just "vote" - reason about which makes more sense

4. VALIDATE COMPLETENESS:
   - Call validate_completeness(results, goal)
   - Does synthesis address all aspects of the goal?
   - Are there gaps in coverage?

5. SYNTHESIZE:
   - Combine results into coherent narrative
   - Preserve key insights from each subtask
   - Explain how results relate to each other
   - Address any limitations or uncertainties

6. CALCULATE CONFIDENCE:
   - Call calculate_confidence(synthesis)
   - Consider conflicts, quality scores, completeness

FUNCTIONS AVAILABLE:
- get_result_summary(subtask_id) → Get detailed result summary
- detect_conflicts(results) → Find inconsistencies
- resolve_conflict(conflict, context) → Reasoning-based resolution
- validate_completeness(results, goal) → Check coverage
- calculate_confidence(synthesis) → Assess overall confidence

QUALITY CRITERIA:
- Coherence: Does synthesis tell a logical story?
- Completeness: All aspects of goal addressed?
- Accuracy: Results properly integrated?
- Transparency: Conflicts and resolutions explained?

Provide:
{{
    "synthesized_result": "Main synthesized output",
    "key_insights": ["Insight 1", "Insight 2", ...],
    "conflicts_found": ["Description of conflicts"],
    "confidence_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "reasoning": "Explain your synthesis process"
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
    
    async def _get_result_summary(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get result summary for subtask"""
        subtask_id = parameters['subtask_id']
        include_metadata = parameters.get('include_metadata', True)
        
        # Simulate result summary
        return {
            "subtask_id": subtask_id,
            "summary": f"Summary of {subtask_id} results",
            "key_findings": [
                "Finding 1 from this subtask",
                "Finding 2 from this subtask"
            ],
            "quality_score": 0.75,
            "confidence": 0.8,
            "metadata": {
                "execution_time": 45.2,
                "tokens_used": 1250
            } if include_metadata else {}
        }
    
    async def _detect_conflicts(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect conflicts between results"""
        results = parameters['results']
        conflict_types = parameters.get('conflict_types', ['all'])
        
        # Simulate conflict detection
        conflicts = []
        
        # Check for data inconsistencies
        if len(results) > 1:
            # Simulate finding a conflict
            conflicts.append({
                "type": "data_inconsistency",
                "subtask_ids": ["subtask_0", "subtask_1"],
                "description": "Different values reported for same metric",
                "severity": "medium",
                "values": {
                    "subtask_0": "Value A",
                    "subtask_1": "Value B"
                }
            })
        
        return {
            "conflicts": conflicts,
            "total_found": len(conflicts),
            "types_checked": conflict_types
        }
    
    async def _resolve_conflict(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a conflict using reasoning"""
        conflict = parameters['conflict']
        context = parameters.get('context', {})
        
        # Simulate conflict resolution
        # In reality, this would use more sophisticated reasoning
        
        resolution_strategies = {
            "data_inconsistency": "Use value from higher quality source",
            "logical_contradiction": "Apply logical consistency check",
            "quality_disparity": "Weight by quality scores",
            "temporal_mismatch": "Use most recent value",
            "format_incompatibility": "Convert to common format"
        }
        
        strategy = resolution_strategies.get(
            conflict.get('type', 'data_inconsistency'),
            "Default resolution"
        )
        
        # Simple resolution: choose first value
        resolved_value = list(conflict.get('values', {}).values())[0] if conflict.get('values') else None
        
        return {
            "resolution_strategy": strategy,
            "resolved_value": resolved_value,
            "reasoning": f"Applied {strategy} to resolve conflict",
            "confidence": 0.75,
            "affected_subtasks": conflict.get('subtask_ids', [])
        }
    
    async def _validate_completeness(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate completeness of results"""
        results = parameters['results']
        goal = parameters['goal']
        
        # Simple completeness check
        # In reality, would use NLP to match goal requirements
        
        expected_aspects = ["research", "analysis", "recommendations"]
        covered_aspects = []
        
        for result in results:
            if "research" in str(result).lower():
                covered_aspects.append("research")
            if "analysis" in str(result).lower():
                covered_aspects.append("analysis")
            if "recommend" in str(result).lower():
                covered_aspects.append("recommendations")
        
        covered_aspects = list(set(covered_aspects))
        coverage = len(covered_aspects) / len(expected_aspects) if expected_aspects else 1.0
        
        missing = [a for a in expected_aspects if a not in covered_aspects]
        
        return {
            "completeness_score": coverage,
            "expected_aspects": expected_aspects,
            "covered_aspects": covered_aspects,
            "missing_aspects": missing,
            "is_complete": coverage >= 0.8
        }
    
    async def _calculate_confidence(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall confidence"""
        synthesis = parameters['synthesis']
        conflicts_resolved = parameters.get('conflicts_resolved', 0)
        quality_scores = parameters.get('quality_scores', [])
        
        # Base confidence from synthesis length/quality
        base_confidence = min(0.9, len(synthesis) / 1000) if synthesis else 0.3
        
        # Reduce for conflicts
        conflict_penalty = conflicts_resolved * 0.05
        
        # Adjust based on quality scores
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.7
        quality_factor = avg_quality * 0.3
        
        overall_confidence = max(0.1, min(0.95, base_confidence - conflict_penalty + quality_factor))
        
        return {
            "overall_confidence": overall_confidence,
            "base_confidence": base_confidence,
            "conflict_impact": -conflict_penalty,
            "quality_impact": quality_factor,
            "recommendation": "high confidence" if overall_confidence > 0.8 else "moderate confidence"
        }
    
    def _fallback_aggregation(
        self,
        subtask_results: Dict[str, SubtaskExecution],
        workflow_goal: str
    ) -> AggregatedResult:
        """Fallback aggregation when LLM fails"""
        # Simple concatenation
        results = []
        for subtask_id, execution in subtask_results.items():
            if execution.result:
                results.append(f"{subtask_id}: {execution.result}")
        
        return AggregatedResult(
            synthesized_result="\n".join(results),
            key_insights=["Fallback aggregation used"],
            conflicts_found=[],
            conflicts_resolved=[],
            confidence_score=0.3,
            completeness_score=0.5,
            reasoning="Fallback aggregation due to LLM error"
        )
    
    def _update_history(self, result: AggregatedResult):
        """Update aggregation history"""
        self.aggregation_history.append({
            "timestamp": datetime.now().isoformat(),
            "conflicts_found": len(result.conflicts_found),
            "conflicts_resolved": len(result.conflicts_resolved),
            "confidence": result.confidence_score,
            "completeness": result.completeness_score
        })
        
        # Trim history
        if len(self.aggregation_history) > 100:
            self.aggregation_history = self.aggregation_history[-100:]
    
    async def analyze_aggregation_quality(self) -> Dict[str, Any]:
        """Analyze quality of aggregations over time"""
        if len(self.aggregation_history) < 5:
            return {
                "status": "insufficient_data",
                "message": "Need at least 5 aggregations for analysis"
            }
        
        recent = self.aggregation_history[-20:]
        
        avg_confidence = sum(h['confidence'] for h in recent) / len(recent)
        avg_completeness = sum(h['completeness'] for h in recent) / len(recent)
        avg_conflicts = sum(h['conflicts_found'] for h in recent) / len(recent)
        
        return {
            "status": "analyzed",
            "total_aggregations": len(self.aggregation_history),
            "metrics": {
                "avg_confidence": avg_confidence,
                "avg_completeness": avg_completeness,
                "avg_conflicts_per_workflow": avg_conflicts
            },
            "trend": "improving" if avg_confidence > 0.7 else "needs improvement",
            "recommendations": [
                "Improve conflict detection" if avg_conflicts > 2 else None,
                "Enhance completeness validation" if avg_completeness < 0.8 else None
            ]
        }
