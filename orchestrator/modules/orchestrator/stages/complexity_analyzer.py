"""
Intelligent Task Complexity Analyzer
=====================================

Phase 1: Complexity Analysis Component

Analyzes task complexity using:
1. LLM-based reasoning (context-aware analysis)
2. Heuristic analysis (length, keywords, structure)
3. Historical patterns (optional, for future enhancement)

Returns:
- Complexity score (0.0-1.0)
- Suggested subtask count
- Suggested agent count
- Complexity metadata (technical, coordination, resource requirements)

Author: Automatos AI - Phase 1 Implementation
"""

import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from config import config

logger = logging.getLogger(__name__)


@dataclass
class ComplexityAnalysis:
    """Result of complexity analysis"""
    
    # Core metrics
    complexity_score: float  # 0.0 (simple) to 1.0 (very complex)
    complexity_level: str  # "low", "medium", "high", "very_high"
    
    # Recommendations
    recommended_subtasks: int
    recommended_agents: int
    
    # Detailed breakdown
    technical_complexity: float  # 0.0-1.0
    coordination_complexity: float  # 0.0-1.0
    resource_requirements: float  # 0.0-1.0
    
    # Metadata
    analysis_method: str  # "llm", "heuristic", "hybrid"
    confidence: float  # 0.0-1.0
    reasoning: str  # Explanation of the analysis
    
    # Constraints consideration
    respects_user_constraints: bool
    constraint_adjustments: Dict[str, Any]
    
    # Timing
    analysis_duration_ms: float
    analyzed_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "complexity_score": self.complexity_score,
            "complexity_level": self.complexity_level,
            "recommended_subtasks": self.recommended_subtasks,
            "recommended_agents": self.recommended_agents,
            "technical_complexity": self.technical_complexity,
            "coordination_complexity": self.coordination_complexity,
            "resource_requirements": self.resource_requirements,
            "analysis_method": self.analysis_method,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "respects_user_constraints": self.respects_user_constraints,
            "constraint_adjustments": self.constraint_adjustments,
            "analysis_duration_ms": self.analysis_duration_ms,
            "analyzed_at": self.analyzed_at
        }


class ComplexityAnalyzer:
    """
    Intelligent task complexity analyzer
    
    Combines LLM reasoning with heuristics to determine:
    - How complex is this task?
    - How many subtasks should we create?
    - How many agents do we need?
    
    Uses context engineering principles to make informed decisions.
    """
    
    # Complexity keywords for heuristic analysis
    HIGH_COMPLEXITY_KEYWORDS = [
        "architect", "design", "comprehensive", "full", "complete", "entire",
        "system", "integrate", "multiple", "complex", "advanced", "scalable",
        "distributed", "microservices", "production", "enterprise",
        # Report-related keywords (critical for proper complexity detection)
        "report", "analysis", "research", "documentation", "document", 
        "technical", "detailed", "analyze", "investigate", "study", 
        "review", "assessment", "thorough", "in-depth", "extensive"
    ]
    
    MEDIUM_COMPLEXITY_KEYWORDS = [
        "implement", "create", "build", "develop", "analyze", "refactor",
        "optimize", "enhance", "extend", "modify", "update", "fix"
    ]
    
    LOW_COMPLEXITY_KEYWORDS = [
        "simple", "basic", "quick", "small", "single", "one", "just",
        "only", "read", "list", "show", "display", "check"
    ]
    
    COORDINATION_KEYWORDS = [
        "coordinate", "integrate", "synchronize", "collaborate", "multiple",
        "across", "between", "various", "different", "several"
    ]
    
    def __init__(self, llm_manager=None, use_llm: bool = True):
        """
        Initialize complexity analyzer
        
        Args:
            llm_manager: LLM manager instance (optional)
            use_llm: Whether to use LLM analysis (can be disabled for testing)
        """
        self.use_llm = use_llm
        self.llm = llm_manager
        
        # Initialize LLM if not provided and use_llm is True
        if self.use_llm and not self.llm:
            try:
                from config import orchestrator_config
                if orchestrator_config.ENABLE_COMPLEXITY_ANALYSIS:
                    from core.llm import create_llm_manager
                    # Use service_name to get settings from database (NO hardcoded defaults)
                    self.llm = create_llm_manager(service_name="orchestrator")
                    logger.info("ComplexityAnalyzer initialized with LLM")
                else:
                    self.use_llm = False
                    logger.info("ComplexityAnalyzer: LLM analysis disabled by config")
            except Exception as e:
                logger.warning(f"Could not initialize LLM for complexity analysis: {e}")
                self.use_llm = False
        
        logger.info(f"ComplexityAnalyzer initialized (LLM: {self.use_llm})")
    
    async def analyze(
        self,
        task_description: str,
        user_constraints: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ComplexityAnalysis:
        """
        Analyze task complexity
        
        Args:
            task_description: The task to analyze
            user_constraints: Optional user-defined constraints
            context: Optional additional context (task type, requirements, etc.)
        
        Returns:
            ComplexityAnalysis with recommendations
        """
        start_time = datetime.now()
        
        # Default constraints
        if user_constraints is None:
            from config import orchestrator_config
            user_constraints = orchestrator_config.get_user_constraints()
        
        # Default context
        if context is None:
            context = {}
        
        logger.info(f"🔍 Analyzing task complexity: {task_description[:100]}...")
        
        # Run both analyses
        heuristic_result = self._heuristic_analysis(task_description, context)
        
        if self.use_llm and self.llm:
            try:
                llm_result = await self._llm_analysis(task_description, context, user_constraints)
                # Combine results (weighted average: 70% LLM, 30% heuristic)
                final_result = self._combine_analyses(llm_result, heuristic_result, weights=(0.7, 0.3))
                final_result.analysis_method = "hybrid"
            except Exception as e:
                logger.warning(f"LLM analysis failed, using heuristic only: {e}")
                final_result = heuristic_result
                final_result.analysis_method = "heuristic"
        else:
            final_result = heuristic_result
            final_result.analysis_method = "heuristic"
        
        # Apply user constraints
        final_result = self._apply_constraints(final_result, user_constraints)
        
        # Calculate duration
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        final_result.analysis_duration_ms = duration_ms
        final_result.analyzed_at = datetime.now().isoformat()
        
        logger.info(f"✅ Complexity analysis complete: {final_result.complexity_level} "
                   f"({final_result.complexity_score:.2f}) - "
                   f"{final_result.recommended_subtasks} subtasks, "
                   f"{final_result.recommended_agents} agents")
        
        return final_result
    
    def _heuristic_analysis(
        self,
        task_description: str,
        context: Dict[str, Any]
    ) -> ComplexityAnalysis:
        """
        Heuristic-based complexity analysis
        
        Uses text analysis, keywords, and patterns to estimate complexity
        """
        description_lower = task_description.lower()
        
        # Length analysis (longer descriptions often indicate more complex tasks)
        length_score = min(len(task_description) / 1000, 1.0)  # Normalize to 0-1
        
        # Keyword analysis
        high_count = sum(1 for kw in self.HIGH_COMPLEXITY_KEYWORDS if kw in description_lower)
        medium_count = sum(1 for kw in self.MEDIUM_COMPLEXITY_KEYWORDS if kw in description_lower)
        low_count = sum(1 for kw in self.LOW_COMPLEXITY_KEYWORDS if kw in description_lower)
        coord_count = sum(1 for kw in self.COORDINATION_KEYWORDS if kw in description_lower)
        
        # Calculate keyword score
        keyword_score = (high_count * 1.0 + medium_count * 0.5 + low_count * 0.0) / max(high_count + medium_count + low_count, 1)
        
        # Structural analysis
        has_multiple_steps = bool(re.search(r'\d+\.|step \d+|first.*then.*finally', description_lower))
        has_requirements = "requirement" in description_lower or "must" in description_lower
        has_integration = "integrate" in description_lower or "connect" in description_lower
        
        structure_score = (
            (0.3 if has_multiple_steps else 0.0) +
            (0.2 if has_requirements else 0.0) +
            (0.3 if has_integration else 0.0)
        )
        
        # Technical complexity (from context or keywords)
        tech_keywords = ["algorithm", "optimization", "performance", "security", "scalable", "distributed"]
        technical_complexity = min(
            sum(0.2 for kw in tech_keywords if kw in description_lower),
            1.0
        )
        
        # Coordination complexity
        coordination_complexity = min(coord_count * 0.25, 1.0)
        
        # Resource requirements (estimate from task size)
        resource_requirements = (length_score + keyword_score) / 2
        
        # Overall complexity score (weighted average)
        complexity_score = (
            length_score * 0.2 +
            keyword_score * 0.4 +
            structure_score * 0.2 +
            technical_complexity * 0.1 +
            coordination_complexity * 0.1
        )
        
        # Determine complexity level
        if complexity_score < 0.3:
            complexity_level = "low"
        elif complexity_score < 0.6:
            complexity_level = "medium"
        elif complexity_score < 0.85:
            complexity_level = "high"
        else:
            complexity_level = "very_high"
        
        # Recommend subtasks based on complexity
        from config import orchestrator_config
        
        # CRITICAL FIX: Ensure minimum 5 subtasks for quality output
        # Previous logic gave only 2 subtasks for "low complexity", causing generic output
        # Phase 1 success required 5+ subtasks even for report tasks
        if complexity_score < 0.3:
            # Raised minimum from 2 to 5 to ensure adequate subtask decomposition
            recommended_subtasks = max(orchestrator_config.MIN_SUBTASKS, 5)
        elif complexity_score < 0.6:
            recommended_subtasks = max(orchestrator_config.DEFAULT_SUBTASKS, 5)
        elif complexity_score < 0.85:
            recommended_subtasks = min(orchestrator_config.DEFAULT_SUBTASKS + 5, orchestrator_config.MAX_SUBTASKS)
        else:
            recommended_subtasks = min(orchestrator_config.DEFAULT_SUBTASKS + 10, orchestrator_config.MAX_SUBTASKS)
        
        # Recommend agents based on coordination complexity
        if coordination_complexity < 0.3:
            recommended_agents = orchestrator_config.MIN_AGENTS_PER_SUBTASK
        elif coordination_complexity < 0.6:
            recommended_agents = orchestrator_config.DEFAULT_AGENTS_PER_SUBTASK
        else:
            recommended_agents = min(
                orchestrator_config.DEFAULT_AGENTS_PER_SUBTASK + 2,
                orchestrator_config.MAX_AGENTS_PER_SUBTASK
            )
        
        return ComplexityAnalysis(
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            recommended_subtasks=recommended_subtasks,
            recommended_agents=recommended_agents,
            technical_complexity=technical_complexity,
            coordination_complexity=coordination_complexity,
            resource_requirements=resource_requirements,
            analysis_method="heuristic",
            confidence=0.65,  # Heuristic analysis has moderate confidence
            reasoning=f"Heuristic analysis: length={length_score:.2f}, keywords={keyword_score:.2f}, "
                     f"structure={structure_score:.2f}, technical={technical_complexity:.2f}, "
                     f"coordination={coordination_complexity:.2f}",
            respects_user_constraints=True,
            constraint_adjustments={},
            analysis_duration_ms=0,  # Will be set by caller
            analyzed_at=""  # Will be set by caller
        )
    
    async def _llm_analysis(
        self,
        task_description: str,
        context: Dict[str, Any],
        user_constraints: Dict[str, Any]
    ) -> ComplexityAnalysis:
        """
        LLM-based complexity analysis
        
        Uses LLM reasoning to provide more nuanced complexity assessment
        """
        from config import orchestrator_config
        
        prompt = f"""You are an expert task complexity analyzer for a multi-agent orchestration system.

Analyze this task and provide a detailed complexity assessment:

TASK: {task_description}

CONTEXT:
- Task Type: {context.get('task_type', 'general')}
- Requirements: {context.get('requirements', [])}

CONSTRAINTS:
- Max Tokens: {user_constraints.get('max_tokens', 'N/A')}
- Max Cost: ${user_constraints.get('max_cost', 'N/A')}
- Max Duration: {user_constraints.get('max_duration', 'N/A')}s

ANALYSIS DIMENSIONS:
1. **Technical Complexity**: How technically challenging is this task?
   - Low: Simple operations, well-defined APIs
   - Medium: Some complexity, multiple components
   - High: Advanced algorithms, optimization, architecture
   - Very High: Novel solutions, research-level complexity

2. **Coordination Complexity**: How much inter-agent coordination is needed?
   - Low: Independent subtasks
   - Medium: Some dependencies, sequential execution
   - High: Parallel execution with coordination
   - Very High: Complex dependencies, tight coordination

3. **Resource Requirements**: What resources are needed?
   - Low: Minimal tokens, fast execution
   - Medium: Moderate tokens, standard execution
   - High: Significant tokens, longer execution
   - Very High: Maximum tokens, extended execution

CONFIGURATION LIMITS:
- Subtasks: {orchestrator_config.MIN_SUBTASKS}-{orchestrator_config.MAX_SUBTASKS} (default: {orchestrator_config.DEFAULT_SUBTASKS})
- Agents per subtask: {orchestrator_config.MIN_AGENTS_PER_SUBTASK}-{orchestrator_config.MAX_AGENTS_PER_SUBTASK} (default: {orchestrator_config.DEFAULT_AGENTS_PER_SUBTASK})

Provide your analysis in JSON format:
{{
  "complexity_score": 0.0-1.0,
  "complexity_level": "low|medium|high|very_high",
  "recommended_subtasks": <number within limits>,
  "recommended_agents": <number within limits>,
  "technical_complexity": 0.0-1.0,
  "coordination_complexity": 0.0-1.0,
  "resource_requirements": 0.0-1.0,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your analysis and recommendations"
}}

Be practical and consider the user's constraints. Don't over-engineer simple tasks."""

        messages = [
            {"role": "system", "content": "You are an expert task complexity analyzer. Always return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.llm.generate_response(messages)
        
        try:
            # Parse response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content)
            
            return ComplexityAnalysis(
                complexity_score=float(result.get("complexity_score", 0.5)),
                complexity_level=result.get("complexity_level", "medium"),
                recommended_subtasks=int(result.get("recommended_subtasks", orchestrator_config.DEFAULT_SUBTASKS)),
                recommended_agents=int(result.get("recommended_agents", orchestrator_config.DEFAULT_AGENTS_PER_SUBTASK)),
                technical_complexity=float(result.get("technical_complexity", 0.5)),
                coordination_complexity=float(result.get("coordination_complexity", 0.5)),
                resource_requirements=float(result.get("resource_requirements", 0.5)),
                analysis_method="llm",
                confidence=float(result.get("confidence", 0.85)),
                reasoning=result.get("reasoning", "LLM-based analysis"),
                respects_user_constraints=True,
                constraint_adjustments={},
                analysis_duration_ms=0,
                analyzed_at=""
            )
        
        except Exception as e:
            logger.error(f"Failed to parse LLM complexity analysis: {e}")
            raise
    
    def _combine_analyses(
        self,
        llm_result: ComplexityAnalysis,
        heuristic_result: ComplexityAnalysis,
        weights: Tuple[float, float] = (0.7, 0.3)
    ) -> ComplexityAnalysis:
        """
        Combine LLM and heuristic analyses with weighted average
        
        Args:
            llm_result: Result from LLM analysis
            heuristic_result: Result from heuristic analysis
            weights: (llm_weight, heuristic_weight)
        
        Returns:
            Combined analysis
        """
        llm_weight, heuristic_weight = weights
        
        # Weighted average of scores
        combined_score = (
            llm_result.complexity_score * llm_weight +
            heuristic_result.complexity_score * heuristic_weight
        )
        
        # Determine final complexity level
        if combined_score < 0.3:
            complexity_level = "low"
        elif combined_score < 0.6:
            complexity_level = "medium"
        elif combined_score < 0.85:
            complexity_level = "high"
        else:
            complexity_level = "very_high"
        
        # Use LLM recommendations (more reliable)
        recommended_subtasks = llm_result.recommended_subtasks
        recommended_agents = llm_result.recommended_agents
        
        # Average other dimensions
        technical_complexity = (
            llm_result.technical_complexity * llm_weight +
            heuristic_result.technical_complexity * heuristic_weight
        )
        coordination_complexity = (
            llm_result.coordination_complexity * llm_weight +
            heuristic_result.coordination_complexity * heuristic_weight
        )
        resource_requirements = (
            llm_result.resource_requirements * llm_weight +
            heuristic_result.resource_requirements * heuristic_weight
        )
        
        # Combined confidence
        confidence = (llm_result.confidence * llm_weight + heuristic_result.confidence * heuristic_weight)
        
        return ComplexityAnalysis(
            complexity_score=combined_score,
            complexity_level=complexity_level,
            recommended_subtasks=recommended_subtasks,
            recommended_agents=recommended_agents,
            technical_complexity=technical_complexity,
            coordination_complexity=coordination_complexity,
            resource_requirements=resource_requirements,
            analysis_method="hybrid",
            confidence=confidence,
            reasoning=f"Combined analysis (LLM: {llm_result.confidence:.2f}, Heuristic: {heuristic_result.confidence:.2f}). "
                     f"LLM reasoning: {llm_result.reasoning[:200]}",
            respects_user_constraints=True,
            constraint_adjustments={},
            analysis_duration_ms=0,
            analyzed_at=""
        )
    
    def _apply_constraints(
        self,
        analysis: ComplexityAnalysis,
        user_constraints: Dict[str, Any]
    ) -> ComplexityAnalysis:
        """
        Apply user constraints to the analysis recommendations
        
        Adjusts recommendations to respect token/cost/time limits
        """
        from config import orchestrator_config
        
        adjustments = {}
        
        # Estimate token usage for recommendations
        estimated_tokens_per_subtask = orchestrator_config.TOKEN_BUDGET_MAX_SUBTASK
        total_estimated_tokens = analysis.recommended_subtasks * estimated_tokens_per_subtask
        
        max_tokens = user_constraints.get('max_tokens', orchestrator_config.TOKEN_BUDGET_DEFAULT)
        
        # If we exceed token budget, reduce subtasks
        if total_estimated_tokens > max_tokens:
            original_subtasks = analysis.recommended_subtasks
            adjusted_subtasks = max(
                orchestrator_config.MIN_SUBTASKS,
                int(max_tokens / estimated_tokens_per_subtask)
            )
            if adjusted_subtasks != original_subtasks:
                analysis.recommended_subtasks = adjusted_subtasks
                adjustments['subtasks'] = {
                    'original': original_subtasks,
                    'adjusted': adjusted_subtasks,
                    'reason': f'Reduced to fit token budget ({max_tokens:,} tokens)'
                }
        
        # Estimate cost
        model = config.LLM_MODEL  # Default from system config
        estimated_cost = orchestrator_config.estimate_cost(total_estimated_tokens, model)
        max_cost = user_constraints.get('max_cost', orchestrator_config.MAX_COST_USD)
        
        if estimated_cost > max_cost:
            # Further reduce subtasks to fit cost
            cost_reduction_factor = max_cost / estimated_cost
            original_subtasks = analysis.recommended_subtasks
            adjusted_subtasks = max(
                orchestrator_config.MIN_SUBTASKS,
                int(analysis.recommended_subtasks * cost_reduction_factor)
            )
            if adjusted_subtasks != original_subtasks:
                analysis.recommended_subtasks = adjusted_subtasks
                adjustments['subtasks_cost'] = {
                    'original': original_subtasks,
                    'adjusted': adjusted_subtasks,
                    'reason': f'Reduced to fit cost budget (${max_cost:.2f})'
                }
        
        # Update analysis with adjustments
        analysis.constraint_adjustments = adjustments
        analysis.respects_user_constraints = True
        
        return analysis


# Singleton instance
_analyzer_instance = None


def get_complexity_analyzer(llm_manager=None, use_llm: bool = True) -> ComplexityAnalyzer:
    """Get or create singleton complexity analyzer"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ComplexityAnalyzer(llm_manager, use_llm)
    return _analyzer_instance


# Export
__all__ = ['ComplexityAnalyzer', 'ComplexityAnalysis', 'get_complexity_analyzer']
