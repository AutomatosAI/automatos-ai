"""
LLM Context Strategy Selection - Stage 2 Enhancement
=====================================================

PRD-16: Stage 2 - LLM decides WHICH context optimizations to use and HOW.
Keeps existing mathematical optimizations but adds reasoning layer on top.

The LLM analyzes the task and context to select optimal strategies:
- MMR (Maximal Marginal Relevance) for diversity
- Shannon Entropy for information filtering
- Knapsack Optimization for token budget
- Field Propagation for context spreading
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Import existing context engineering
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import config
from modules.orchestrator.stages.context_engineering import (
    ContextEngineeringIntegrator,
    ContextEnhancement
)
from core.llm import (
    OrchestratorLLM,
    FunctionRegistry,
    FunctionSpec,
    FunctionParameter,
    FunctionCategory
)

logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """Available context optimization strategies"""
    MMR = "mmr"                        # Maximal Marginal Relevance
    SHANNON_ENTROPY = "shannon_entropy" # Information filtering
    KNAPSACK = "knapsack"              # Token budget optimization
    FIELD_PROPAGATION = "field"        # Context spreading
    HYBRID = "hybrid"                  # Combination of strategies


@dataclass
class OptimizationStrategy:
    """Selected optimization strategy with parameters"""
    strategy: ContextStrategy
    parameters: Dict[str, Any]
    reasoning: str
    expected_benefit: str
    confidence: float
    estimated_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
            "expected_benefit": self.expected_benefit,
            "confidence": self.confidence,
            "estimated_tokens": self.estimated_tokens
        }


@dataclass
class ContextAnalysis:
    """Analysis of available context"""
    total_sources: int
    total_tokens: int
    information_density: float
    redundancy_level: float
    relevance_scores: Dict[str, float]
    quality_assessment: str
    recommendations: List[str]


class LLMContextStrategySelector:
    """
    LLM selects optimal context engineering strategy.
    Reference: Context Engineering - Adaptive optimization
    """
    
    def __init__(
        self,
        llm: Optional[OrchestratorLLM] = None,
        context_integrator: Optional[ContextEngineeringIntegrator] = None
    ):
        """
        Initialize context strategy selector.
        
        Args:
            llm: Orchestrator LLM instance
            context_integrator: Existing context engineering integrator
        """
        self.llm = llm or OrchestratorLLM(temperature=0.6)  # Lower temp for strategic decisions
        self.context_integrator = context_integrator
        
        # Initialize function registry
        self.registry = FunctionRegistry()
        self._register_functions()
        
        # Strategy usage history
        self.strategy_history: List[Dict[str, Any]] = []
        self.max_history = 100
        
        logger.info("LLMContextStrategySelector initialized")
    
    def _register_functions(self):
        """Register context analysis functions"""
        
        # 1. Get available context sources
        self.registry.register(FunctionSpec(
            name="get_available_context_sources",
            description="Get all available context sources for subtask",
            category=FunctionCategory.QUERY,
            parameters=[
                FunctionParameter(
                    name="subtask_id",
                    type="string",
                    description="Subtask identifier",
                    required=True
                ),
                FunctionParameter(
                    name="include_metrics",
                    type="boolean",
                    description="Include quality metrics",
                    required=False,
                    default=True
                )
            ],
            implementation=self._get_context_sources
        ))
        
        # 2. Estimate information density
        self.registry.register(FunctionSpec(
            name="estimate_information_density",
            description="Calculate Shannon entropy for context sources",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="sources",
                    type="array",
                    description="List of context sources",
                    required=True,
                    items={"type": "object"}
                )
            ],
            implementation=self._estimate_information_density
        ))
        
        # 3. Get token budget
        self.registry.register(FunctionSpec(
            name="get_token_budget",
            description="Get remaining token budget for this subtask",
            category=FunctionCategory.MONITORING,
            parameters=[
                FunctionParameter(
                    name="subtask_id",
                    type="string",
                    description="Subtask identifier",
                    required=True
                ),
                FunctionParameter(
                    name="model",
                    type="string",
                    description="Target model",
                    required=False,
                    default="gpt-4o"
                )
            ],
            implementation=self._get_token_budget
        ))
        
        # 4. Analyze task complexity
        self.registry.register(FunctionSpec(
            name="analyze_task_complexity",
            description="Analyze how complex the subtask is",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="description",
                    type="string",
                    description="Task description",
                    required=True
                ),
                FunctionParameter(
                    name="dependencies",
                    type="array",
                    description="Task dependencies",
                    required=False,
                    items={"type": "string"}
                )
            ],
            implementation=self._analyze_complexity
        ))
        
        # 5. Simulate strategy outcome
        self.registry.register(FunctionSpec(
            name="simulate_strategy_outcome",
            description="Predict outcome of applying a strategy",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="strategy",
                    type="string",
                    description="Strategy name",
                    required=True,
                    enum=["mmr", "shannon_entropy", "knapsack", "field"]
                ),
                FunctionParameter(
                    name="parameters",
                    type="object",
                    description="Strategy parameters",
                    required=True
                ),
                FunctionParameter(
                    name="context_size",
                    type="number",
                    description="Current context size in tokens",
                    required=True
                )
            ],
            implementation=self._simulate_outcome
        ))
    
    async def select_optimization_strategy(
        self,
        subtask: Dict[str, Any],
        available_context: Dict[str, Any],
        token_limit: Optional[int] = None
    ) -> OptimizationStrategy:
        """
        LLM reasons about which optimization to use.
        
        Args:
            subtask: Subtask details
            available_context: Available context sources
            token_limit: Maximum token budget
            
        Returns:
            Selected optimization strategy with parameters
        """
        # Build prompt for strategy selection
        prompt = self._build_strategy_prompt(subtask, available_context, token_limit)
        
        # Get functions for strategy selection
        functions = self.registry.get_for_prompt(
            categories=[FunctionCategory.QUERY, FunctionCategory.ANALYSIS, FunctionCategory.MONITORING]
        )
        
        # LLM reasoning with function calls
        response = await self.llm.generate_with_functions(
            prompt=prompt,
            functions=functions,
            function_executor=self._execute_function,
            max_function_calls=8,
            context={
                "subtask": subtask,
                "available_context": available_context
            }
        )
        
        # Parse response
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback to default strategy
                result = {
                    "strategy_name": "mmr",
                    "parameters": {"lambda": 0.7},
                    "reasoning": "Default MMR strategy",
                    "expected_benefit": "Balanced relevance and diversity"
                }
            
            # Map strategy name to enum
            strategy_map = {
                "mmr": ContextStrategy.MMR,
                "shannon_entropy": ContextStrategy.SHANNON_ENTROPY,
                "shannon": ContextStrategy.SHANNON_ENTROPY,
                "entropy": ContextStrategy.SHANNON_ENTROPY,
                "knapsack": ContextStrategy.KNAPSACK,
                "field": ContextStrategy.FIELD_PROPAGATION,
                "field_propagation": ContextStrategy.FIELD_PROPAGATION,
                "hybrid": ContextStrategy.HYBRID
            }
            
            strategy_enum = strategy_map.get(
                result.get("strategy_name", "mmr").lower(),
                ContextStrategy.MMR
            )
            
            optimization = OptimizationStrategy(
                strategy=strategy_enum,
                parameters=result.get("parameters", {}),
                reasoning=result.get("reasoning", response.reasoning or ""),
                expected_benefit=result.get("expected_benefit", ""),
                confidence=float(result.get("confidence", response.confidence)),
                estimated_tokens=result.get("estimated_tokens", 0)
            )
            
        except Exception as e:
            logger.error(f"Failed to parse strategy response: {e}")
            # Fallback to MMR
            optimization = OptimizationStrategy(
                strategy=ContextStrategy.MMR,
                parameters={"lambda": 0.7},
                reasoning="Fallback to MMR due to parsing error",
                expected_benefit="Balanced approach",
                confidence=0.5
            )
        
        # Log decision
        logger.info(f"📊 Context Strategy Selected: {optimization.strategy.value}")
        logger.info(f"   Parameters: {optimization.parameters}")
        logger.info(f"   Confidence: {optimization.confidence:.2%}")
        logger.info(f"   Reasoning: {optimization.reasoning[:200]}...")
        
        # Update history
        self._update_history(subtask.get("subtask_id"), optimization)
        
        return optimization
    
    async def apply_selected_strategy(
        self,
        strategy: OptimizationStrategy,
        subtask: Dict[str, Any],
        context_sources: List[Dict[str, Any]]
    ) -> ContextEnhancement:
        """
        Apply the selected optimization strategy using existing integrator.
        
        Args:
            strategy: Selected optimization strategy
            subtask: Subtask to enhance
            context_sources: Available context sources
            
        Returns:
            Enhanced context
        """
        if not self.context_integrator:
            raise ValueError("Context integrator not initialized")
        
        # Map strategy to existing integrator methods
        if strategy.strategy == ContextStrategy.MMR:
            # Use MMR optimization
            lambda_param = strategy.parameters.get("lambda", 0.7)
            enhanced = await self.context_integrator.enhance_with_mmr(
                subtask,
                context_sources,
                lambda_param=lambda_param
            )
            
        elif strategy.strategy == ContextStrategy.SHANNON_ENTROPY:
            # Use entropy filtering
            threshold = strategy.parameters.get("threshold", 0.5)
            enhanced = await self.context_integrator.enhance_with_entropy(
                subtask,
                context_sources,
                entropy_threshold=threshold
            )
            
        elif strategy.strategy == ContextStrategy.KNAPSACK:
            # Use knapsack optimization
            budget = strategy.parameters.get("budget", 4000)
            enhanced = await self.context_integrator.enhance_with_knapsack(
                subtask,
                context_sources,
                token_budget=budget
            )
            
        elif strategy.strategy == ContextStrategy.FIELD_PROPAGATION:
            # Use field propagation
            radius = strategy.parameters.get("radius", 2)
            enhanced = await self.context_integrator.enhance_with_field(
                subtask,
                context_sources,
                propagation_radius=radius
            )
            
        else:  # HYBRID
            # Combine multiple strategies
            enhanced = await self.context_integrator.enhance_context(
                subtask,
                optimization_level="aggressive"
            )
        
        # Add strategy metadata
        enhanced.metadata = enhanced.metadata or {}
        enhanced.metadata["strategy_used"] = strategy.strategy.value
        enhanced.metadata["strategy_parameters"] = strategy.parameters
        enhanced.metadata["strategy_reasoning"] = strategy.reasoning
        
        return enhanced
    
    def _build_strategy_prompt(
        self,
        subtask: Dict[str, Any],
        available_context: Dict[str, Any],
        token_limit: Optional[int]
    ) -> str:
        """Build prompt for strategy selection"""
        
        prompt = f"""
SUBTASK: {subtask.get('description', '')}
REQUIRED SKILLS: {subtask.get('skills_required', [])}
PRIORITY: {subtask.get('priority', 'medium')}

CONTEXT ENGINEERING TASK:
You need to optimize the prompt for this subtask using mathematical optimizations.

AVAILABLE OPTIMIZATIONS:
1. MMR (Maximal Marginal Relevance):
   - Purpose: Balance relevance vs diversity
   - Parameter λ (0-1): 0=diverse, 1=relevant
   - Best for: Tasks needing varied perspectives
   - Cost: O(n²) in context size

2. Shannon Entropy Filtering:
   - Purpose: Remove low-information content
   - Parameter threshold (0-1): higher=more aggressive
   - Best for: Noisy context sources
   - Cost: O(n) in context size

3. Knapsack Optimization:
   - Purpose: Maximize value within token budget
   - Parameter budget (tokens): strict limit
   - Best for: Token-constrained scenarios
   - Cost: O(n×budget) optimization time

4. Field Propagation:
   - Purpose: Spread context to related subtasks
   - Parameter radius (1-5): propagation distance
   - Best for: Interdependent subtasks
   - Cost: O(n×radius) computation

5. Hybrid:
   - Purpose: Combine multiple strategies
   - Best for: Complex tasks requiring multiple optimizations
   - Cost: Higher but more comprehensive

AVAILABLE CONTEXT:
- Sources: {available_context.get('source_count', 0)}
- Total tokens: {available_context.get('total_tokens', 0)}
- Quality: {available_context.get('quality', 'unknown')}

TOKEN LIMIT: {token_limit or 'No strict limit'}

FUNCTIONS YOU CAN CALL:
- get_available_context_sources() → See what context is available
- estimate_information_density() → Calculate entropy
- get_token_budget() → Check budget constraints
- analyze_task_complexity() → Understand task needs
- simulate_strategy_outcome() → Predict strategy effectiveness

YOUR DECISION PROCESS:
1. Call functions to understand the situation
2. Analyze subtask requirements
3. Consider available context quality
4. Evaluate token budget constraints
5. Select optimal optimization(s)
6. Set appropriate parameters
7. Explain your reasoning

Provide: strategy_name, parameters, reasoning, expected_benefit, confidence, estimated_tokens

RESPONSE FORMAT:
{{
    "strategy_name": "mmr|shannon_entropy|knapsack|field|hybrid",
    "parameters": {{...}},
    "reasoning": "Why this strategy is optimal",
    "expected_benefit": "What improvement to expect",
    "confidence": 0.0-1.0,
    "estimated_tokens": <number>
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
    
    async def _get_context_sources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get available context sources"""
        subtask_id = parameters['subtask_id']
        include_metrics = parameters.get('include_metrics', True)
        
        # Simulate context sources (would query real system)
        sources = [
            {
                "source_id": "memory_1",
                "type": "episodic_memory",
                "relevance": 0.85,
                "tokens": 250,
                "quality": "high"
            },
            {
                "source_id": "knowledge_1",
                "type": "knowledge_graph",
                "relevance": 0.72,
                "tokens": 180,
                "quality": "medium"
            },
            {
                "source_id": "doc_1",
                "type": "documentation",
                "relevance": 0.65,
                "tokens": 420,
                "quality": "high"
            }
        ]
        
        total_tokens = sum(s['tokens'] for s in sources)
        avg_relevance = sum(s['relevance'] for s in sources) / len(sources) if sources else 0
        
        return {
            "subtask_id": subtask_id,
            "sources": sources,
            "total_sources": len(sources),
            "total_tokens": total_tokens,
            "avg_relevance": avg_relevance,
            "metrics_included": include_metrics
        }
    
    async def _estimate_information_density(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate information density using Shannon entropy"""
        sources = parameters['sources']
        
        # Simplified entropy calculation
        import math
        
        entropies = []
        for source in sources:
            # Simulate entropy based on type and quality
            base_entropy = {
                "episodic_memory": 0.8,
                "knowledge_graph": 0.7,
                "documentation": 0.6,
                "code": 0.9
            }.get(source.get('type', 'unknown'), 0.5)
            
            quality_factor = {
                "high": 1.2,
                "medium": 1.0,
                "low": 0.8
            }.get(source.get('quality', 'medium'), 1.0)
            
            entropy = base_entropy * quality_factor
            entropies.append(entropy)
        
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0
        
        return {
            "sources_analyzed": len(sources),
            "avg_entropy": avg_entropy,
            "max_entropy": max(entropies) if entropies else 0,
            "min_entropy": min(entropies) if entropies else 0,
            "recommendation": "high diversity" if avg_entropy > 0.7 else "needs filtering"
        }
    
    async def _get_token_budget(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get token budget for subtask"""
        subtask_id = parameters['subtask_id']
        model = parameters.get('model', config.LLM_MODEL)
        
        # Try to get context window from database model registry
        max_context = None
        try:
            # Lazy import to avoid circular dependencies
            import sys
            if 'database.database' in sys.modules and hasattr(sys.modules.get('database.database'), 'SessionLocal'):
                from core.database.database import SessionLocal
                from core.llm.model_registry import ModelRegistry
                
                db = SessionLocal()
                try:
                    registry = ModelRegistry(db)
                    model_info = registry.get_model(model)
                    if model_info:
                        # Validate context_window attribute exists and has valid value
                        if (hasattr(model_info, 'context_window') and 
                            model_info.context_window is not None and
                            isinstance(model_info.context_window, int) and
                            model_info.context_window > 0):
                            max_context = model_info.context_window
                            logger.debug(f"Found context window for model '{model}': {max_context} tokens")
                        else:
                            # context_window is missing or invalid
                            context_window_value = getattr(model_info, 'context_window', 'missing')
                            logger.warning(
                                f"⚠️  Invalid or missing context_window for model '{model}' "
                                f"(value: {context_window_value}). "
                                f"Falling back to default context window. "
                                f"Please verify the model registry entry for '{model}'."
                            )
                finally:
                    db.close()
        except Exception as e:
            logger.debug(f"Could not query model registry for '{model}': {e}")
        
        # Fallback to conservative default if model not found in registry
        if max_context is None:
            # Conservative default: 8192 tokens (safe for most legacy/smaller models)
            max_context = 8192
            logger.warning(
                f"⚠️  Model '{model}' not found in model registry. "
                f"Using conservative default context window of {max_context} tokens. "
                f"This may be too small for larger models. Please verify the model name is correct "
                f"or add it to the model registry with the correct context_window value."
            )
        
        # Reserve tokens for response
        response_reserve = 2000
        
        # Calculate available budget
        available_budget = max_context - response_reserve
        
        return {
            "subtask_id": subtask_id,
            "model": model,
            "max_context": max_context,
            "response_reserve": response_reserve,
            "available_budget": available_budget,
            "recommended_budget": int(available_budget * 0.8)  # Use 80% for safety
        }
    
    async def _analyze_complexity(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task complexity"""
        description = parameters['description']
        dependencies = parameters.get('dependencies', [])
        
        # Simple heuristic analysis
        complexity_indicators = {
            "multiple_steps": len(description.split('.')) > 2,
            "technical_terms": any(term in description.lower() for term in ['implement', 'optimize', 'analyze', 'integrate']),
            "dependencies": len(dependencies) > 0,
            "coordination": 'coordinate' in description.lower() or 'collaborate' in description.lower(),
            "research": 'research' in description.lower() or 'investigate' in description.lower()
        }
        
        complexity_score = sum(complexity_indicators.values()) / len(complexity_indicators)
        
        if complexity_score > 0.6:
            complexity_level = "high"
            recommended_strategy = "hybrid"
        elif complexity_score > 0.3:
            complexity_level = "medium"
            recommended_strategy = "mmr"
        else:
            complexity_level = "low"
            recommended_strategy = "shannon_entropy"
        
        return {
            "description_length": len(description),
            "dependency_count": len(dependencies),
            "complexity_indicators": complexity_indicators,
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "recommended_strategy": recommended_strategy
        }
    
    async def _simulate_outcome(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate strategy outcome"""
        strategy = parameters['strategy']
        strategy_params = parameters['parameters']
        context_size = parameters['context_size']
        
        # Simulate different strategies
        if strategy == "mmr":
            lambda_val = strategy_params.get('lambda', 0.7)
            reduction_factor = 0.7 + (0.2 * lambda_val)  # Higher lambda = less reduction
            final_size = int(context_size * reduction_factor)
            quality_score = 0.8 + (0.1 * lambda_val)
            
        elif strategy == "shannon_entropy":
            threshold = strategy_params.get('threshold', 0.5)
            reduction_factor = 1.0 - (0.5 * threshold)  # Higher threshold = more reduction
            final_size = int(context_size * reduction_factor)
            quality_score = 0.7 + (0.2 * threshold)
            
        elif strategy == "knapsack":
            budget = strategy_params.get('budget', 4000)
            final_size = min(context_size, budget)
            quality_score = 0.85 if final_size >= context_size * 0.8 else 0.7
            
        elif strategy == "field":
            radius = strategy_params.get('radius', 2)
            expansion_factor = 1.0 + (0.2 * radius)
            final_size = int(context_size * expansion_factor)
            quality_score = 0.75
            
        else:
            final_size = context_size
            quality_score = 0.7
        
        return {
            "strategy": strategy,
            "initial_size": context_size,
            "final_size": final_size,
            "size_change": final_size - context_size,
            "quality_score": quality_score,
            "efficiency": quality_score / (final_size / 1000),  # Quality per 1K tokens
            "recommendation": "good fit" if quality_score > 0.75 else "consider alternatives"
        }
    
    def _update_history(self, subtask_id: str, strategy: OptimizationStrategy):
        """Update strategy usage history"""
        self.strategy_history.append({
            "timestamp": datetime.now().isoformat(),
            "subtask_id": subtask_id,
            "strategy": strategy.strategy.value,
            "parameters": strategy.parameters,
            "confidence": strategy.confidence
        })
        
        # Trim history
        if len(self.strategy_history) > self.max_history:
            self.strategy_history = self.strategy_history[-self.max_history:]
    
    async def analyze_strategy_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of different strategies"""
        if len(self.strategy_history) < 5:
            return {
                "status": "insufficient_data",
                "message": "Need at least 5 strategy selections for analysis"
            }
        
        # Count strategy usage
        strategy_counts = {}
        confidence_by_strategy = {}
        
        for entry in self.strategy_history:
            strategy = entry['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            if strategy not in confidence_by_strategy:
                confidence_by_strategy[strategy] = []
            confidence_by_strategy[strategy].append(entry['confidence'])
        
        # Calculate averages
        strategy_stats = {}
        for strategy, confidences in confidence_by_strategy.items():
            strategy_stats[strategy] = {
                "usage_count": strategy_counts[strategy],
                "avg_confidence": sum(confidences) / len(confidences),
                "usage_percentage": strategy_counts[strategy] / len(self.strategy_history)
            }
        
        # Find most effective
        most_used = max(strategy_counts, key=strategy_counts.get)
        highest_confidence = max(
            confidence_by_strategy,
            key=lambda s: sum(confidence_by_strategy[s]) / len(confidence_by_strategy[s])
        )
        
        return {
            "status": "analyzed",
            "total_selections": len(self.strategy_history),
            "strategy_stats": strategy_stats,
            "most_used": most_used,
            "highest_confidence": highest_confidence,
            "recommendation": f"Consider using {highest_confidence} more often"
        }




