
"""
Tool Selection Optimization Module
Advanced tool ranking and selection algorithms for intelligent orchestration
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import numpy as np
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ToolCapabilityType(Enum):
    """Types of tool capabilities"""
    COMPUTATIONAL = "computational"
    DATA_PROCESSING = "data_processing"
    API_INTEGRATION = "api_integration"
    FILE_OPERATIONS = "file_operations"
    NETWORK_OPERATIONS = "network_operations"
    DATABASE_OPERATIONS = "database_operations"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    TRANSFORMATION = "transformation"

@dataclass
class ToolMetrics:
    """Performance metrics for a tool"""
    success_rate: float = 0.0
    average_latency: float = 0.0
    error_count: int = 0
    total_executions: int = 0
    last_used: Optional[datetime] = None
    reliability_score: float = 0.0
    cost_score: float = 1.0  # Lower is better
    complexity_score: float = 0.5  # 0 = simple, 1 = complex

@dataclass
class ToolDefinition:
    """Comprehensive tool definition"""
    name: str
    description: str
    capabilities: List[ToolCapabilityType]
    parameters: Dict[str, Any]
    dependencies: List[str]
    version: str
    metrics: ToolMetrics = field(default_factory=ToolMetrics)
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool definition to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": [cap.value for cap in self.capabilities],
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "version": self.version,
            "category": self.category,
            "tags": self.tags,
            "documentation_url": self.documentation_url,
            "metrics": {
                "success_rate": self.metrics.success_rate,
                "average_latency": self.metrics.average_latency,
                "error_count": self.metrics.error_count,
                "total_executions": self.metrics.total_executions,
                "reliability_score": self.metrics.reliability_score,
                "cost_score": self.metrics.cost_score,
                "complexity_score": self.metrics.complexity_score,
                "last_used": self.metrics.last_used.isoformat() if self.metrics.last_used else None
            }
        }

@dataclass
class TaskRequirements:
    """Requirements for a specific task"""
    task_id: str
    description: str
    required_capabilities: List[ToolCapabilityType]
    priority: float = 0.5  # 0 = low, 1 = high
    deadline: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    performance_requirements: Dict[str, float] = field(default_factory=dict)

@dataclass
class ToolSelectionResult:
    """Result of tool selection process"""
    selected_tools: List[Tuple[str, float]]  # (tool_name, score)
    ranking_explanation: Dict[str, Any]
    alternative_tools: List[Tuple[str, float]]
    selection_confidence: float
    estimated_performance: Dict[str, float]
    warnings: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

class ToolSelectionOptimizer:
    """Advanced tool selection with multi-criteria optimization"""
    
    def __init__(self):
        # Tool registry
        self.tools: Dict[str, ToolDefinition] = {}
        
        # Selection history and learning
        self.selection_history: deque = deque(maxlen=10000)
        self.performance_feedback: Dict[str, List[float]] = defaultdict(list)
        
        # Optimization weights (can be dynamically adjusted)
        self.default_weights = {
            "capability_match": 0.30,
            "performance": 0.25,
            "reliability": 0.20,
            "cost": 0.15,
            "availability": 0.10
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.05
        
        logger.info("Initialized ToolSelectionOptimizer")
    
    async def register_tool(self, tool: ToolDefinition):
        """Register a new tool in the system"""
        try:
            self.tools[tool.name] = tool
            logger.info(f"Registered tool: {tool.name} (v{tool.version})")
            
            # Initialize performance tracking
            if tool.name not in self.performance_feedback:
                self.performance_feedback[tool.name] = []
            
        except Exception as e:
            logger.error(f"Error registering tool {tool.name}: {e}")
    
    async def update_tool_metrics(self, tool_name: str, execution_result: Dict[str, Any]):
        """Update tool performance metrics based on execution results"""
        try:
            if tool_name not in self.tools:
                logger.warning(f"Tool {tool_name} not found for metrics update")
                return
            
            tool = self.tools[tool_name]
            metrics = tool.metrics
            
            # Update execution count
            metrics.total_executions += 1
            metrics.last_used = datetime.utcnow()
            
            # Update success rate
            success = execution_result.get("success", False)
            if success:
                metrics.success_rate = (
                    metrics.success_rate * (metrics.total_executions - 1) + 1.0
                ) / metrics.total_executions
            else:
                metrics.error_count += 1
                metrics.success_rate = (
                    metrics.success_rate * (metrics.total_executions - 1)
                ) / metrics.total_executions
            
            # Update latency
            latency = execution_result.get("duration", 0.0)
            if latency > 0:
                metrics.average_latency = (
                    metrics.average_latency * (metrics.total_executions - 1) + latency
                ) / metrics.total_executions
            
            # Update reliability score (weighted combination of success rate and consistency)
            metrics.reliability_score = (
                metrics.success_rate * 0.7 + 
                (1.0 - min(metrics.average_latency / 10.0, 1.0)) * 0.3
            )
            
            # Store performance feedback for learning
            performance_score = 1.0 if success else 0.0
            if latency > 0:
                performance_score *= max(0.0, 1.0 - latency / 30.0)  # Penalize high latency
            
            self.performance_feedback[tool_name].append(performance_score)
            
            # Keep only recent feedback
            if len(self.performance_feedback[tool_name]) > 100:
                self.performance_feedback[tool_name] = self.performance_feedback[tool_name][-100:]
            
            logger.debug(f"Updated metrics for {tool_name}: success_rate={metrics.success_rate:.3f}, latency={metrics.average_latency:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating tool metrics for {tool_name}: {e}")
    
    async def select_tools(self, requirements: TaskRequirements, 
                          max_tools: int = 5, 
                          custom_weights: Optional[Dict[str, float]] = None) -> ToolSelectionResult:
        """Select optimal tools for given task requirements"""
        try:
            logger.info(f"Selecting tools for task: {requirements.task_id}")
            
            # Use custom weights or defaults
            weights = custom_weights or self.default_weights
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # Calculate scores for all tools
            tool_scores = []
            warnings = []
            
            for tool_name, tool in self.tools.items():
                try:
                    score = await self._calculate_tool_score(tool, requirements, weights)
                    tool_scores.append((tool_name, score))
                except Exception as e:
                    warnings.append(f"Error scoring tool {tool_name}: {e}")
            
            # Sort by score (descending)
            tool_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top tools
            selected_tools = tool_scores[:max_tools]
            alternative_tools = tool_scores[max_tools:max_tools*2]
            
            # Calculate selection confidence
            confidence = await self._calculate_selection_confidence(selected_tools, requirements)
            
            # Estimate performance
            estimated_performance = await self._estimate_performance(selected_tools, requirements)
            
            # Create ranking explanation
            ranking_explanation = await self._create_ranking_explanation(
                selected_tools, requirements, weights
            )
            
            # Record selection for learning
            selection_record = {
                "task_id": requirements.task_id,
                "selected_tools": [name for name, _ in selected_tools],
                "scores": dict(selected_tools),
                "requirements": requirements,
                "weights": weights,
                "timestamp": datetime.utcnow()
            }
            self.selection_history.append(selection_record)
            
            result = ToolSelectionResult(
                selected_tools=selected_tools,
                alternative_tools=alternative_tools,
                ranking_explanation=ranking_explanation,
                selection_confidence=confidence,
                estimated_performance=estimated_performance,
                warnings=warnings
            )
            
            logger.info(f"Selected {len(selected_tools)} tools for task {requirements.task_id} with confidence {confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in tool selection: {e}")
            return ToolSelectionResult(
                selected_tools=[],
                alternative_tools=[],
                ranking_explanation={"error": str(e)},
                selection_confidence=0.0,
                estimated_performance={},
                warnings=[f"Tool selection failed: {e}"]
            )
    
    async def _calculate_tool_score(self, tool: ToolDefinition, 
                                  requirements: TaskRequirements, 
                                  weights: Dict[str, float]) -> float:
        """Calculate comprehensive score for a tool given requirements"""
        try:
            # 1. Capability Match Score
            capability_score = await self._calculate_capability_match(tool, requirements)
            
            # 2. Performance Score
            performance_score = await self._calculate_performance_score(tool, requirements)
            
            # 3. Reliability Score
            reliability_score = tool.metrics.reliability_score
            
            # 4. Cost Score (inverted - lower cost is better)
            cost_score = 1.0 - tool.metrics.cost_score
            
            # 5. Availability Score
            availability_score = await self._calculate_availability_score(tool, requirements)
            
            # Weighted combination
            total_score = (
                capability_score * weights.get("capability_match", 0.3) +
                performance_score * weights.get("performance", 0.25) +
                reliability_score * weights.get("reliability", 0.2) +
                cost_score * weights.get("cost", 0.15) +
                availability_score * weights.get("availability", 0.1)
            )
            
            return min(1.0, max(0.0, total_score))
            
        except Exception as e:
            logger.error(f"Error calculating score for tool {tool.name}: {e}")
            return 0.0
    
    async def _calculate_capability_match(self, tool: ToolDefinition, 
                                        requirements: TaskRequirements) -> float:
        """Calculate how well tool capabilities match requirements"""
        try:
            if not requirements.required_capabilities:
                return 1.0  # No specific requirements
            
            required_caps = set(requirements.required_capabilities)
            tool_caps = set(tool.capabilities)
            
            # Calculate overlap
            matching_caps = required_caps.intersection(tool_caps)
            
            if not required_caps:
                return 1.0
            
            # Base match ratio
            match_ratio = len(matching_caps) / len(required_caps)
            
            # Bonus for having additional relevant capabilities
            bonus_caps = tool_caps - required_caps
            bonus_score = min(0.2, len(bonus_caps) * 0.05)
            
            # Penalty for missing critical capabilities
            missing_caps = required_caps - tool_caps
            penalty = len(missing_caps) * 0.3
            
            final_score = match_ratio + bonus_score - penalty
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating capability match: {e}")
            return 0.0
    
    async def _calculate_performance_score(self, tool: ToolDefinition, 
                                         requirements: TaskRequirements) -> float:
        """Calculate performance score based on requirements and tool metrics"""
        try:
            metrics = tool.metrics
            
            # Base performance from success rate and latency
            base_performance = metrics.success_rate * 0.7
            
            # Latency factor
            required_latency = requirements.performance_requirements.get("max_latency", 10.0)
            if metrics.average_latency > 0:
                latency_factor = min(1.0, required_latency / metrics.average_latency)
            else:
                latency_factor = 1.0
            
            base_performance += latency_factor * 0.3
            
            # Recent performance trend
            if tool.name in self.performance_feedback and self.performance_feedback[tool.name]:
                recent_feedback = self.performance_feedback[tool.name][-20:]  # Last 20 executions
                recent_avg = sum(recent_feedback) / len(recent_feedback)
                
                # Weight recent performance
                base_performance = base_performance * 0.7 + recent_avg * 0.3
            
            return min(1.0, max(0.0, base_performance))
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.5
    
    async def _calculate_availability_score(self, tool: ToolDefinition, 
                                          requirements: TaskRequirements) -> float:
        """Calculate tool availability score"""
        try:
            # Check if tool has been used recently (indicates availability)
            if tool.metrics.last_used:
                time_since_last_use = datetime.utcnow() - tool.metrics.last_used
                hours_since_use = time_since_last_use.total_seconds() / 3600
                
                # Tools used recently are more likely to be available
                if hours_since_use < 1:
                    availability = 1.0
                elif hours_since_use < 24:
                    availability = 0.9
                elif hours_since_use < 168:  # 1 week
                    availability = 0.7
                else:
                    availability = 0.5
            else:
                availability = 0.6  # Unknown availability
            
            # Check dependency availability (simplified)
            if tool.dependencies:
                # Assume 90% availability if dependencies exist
                dependency_factor = 0.9
            else:
                dependency_factor = 1.0
            
            return availability * dependency_factor
            
        except Exception as e:
            logger.error(f"Error calculating availability score: {e}")
            return 0.5
    
    async def _calculate_selection_confidence(self, selected_tools: List[Tuple[str, float]], 
                                            requirements: TaskRequirements) -> float:
        """Calculate confidence in tool selection"""
        try:
            if not selected_tools:
                return 0.0
            
            # Base confidence from top tool score
            top_score = selected_tools[0][1]
            base_confidence = top_score
            
            # Confidence boost from score distribution
            if len(selected_tools) > 1:
                scores = [score for _, score in selected_tools]
                score_range = max(scores) - min(scores)
                
                # If scores are well-separated, confidence is higher
                if score_range > 0.2:
                    base_confidence += 0.1
            
            # Confidence adjustment based on requirements specificity
            if requirements.required_capabilities:
                specificity_bonus = min(0.1, len(requirements.required_capabilities) * 0.02)
                base_confidence += specificity_bonus
            
            return min(1.0, base_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating selection confidence: {e}")
            return 0.5
    
    async def _estimate_performance(self, selected_tools: List[Tuple[str, float]], 
                                  requirements: TaskRequirements) -> Dict[str, float]:
        """Estimate performance metrics for selected tools"""
        try:
            if not selected_tools:
                return {}
            
            # Calculate aggregate metrics
            total_success_rate = 0.0
            total_latency = 0.0
            total_reliability = 0.0
            
            for tool_name, score in selected_tools:
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    total_success_rate += tool.metrics.success_rate
                    total_latency += tool.metrics.average_latency
                    total_reliability += tool.metrics.reliability_score
            
            num_tools = len(selected_tools)
            
            return {
                "estimated_success_rate": total_success_rate / num_tools,
                "estimated_average_latency": total_latency / num_tools,
                "estimated_reliability": total_reliability / num_tools,
                "estimated_total_latency": total_latency  # If tools run sequentially
            }
            
        except Exception as e:
            logger.error(f"Error estimating performance: {e}")
            return {}
    
    async def _create_ranking_explanation(self, selected_tools: List[Tuple[str, float]], 
                                        requirements: TaskRequirements, 
                                        weights: Dict[str, float]) -> Dict[str, Any]:
        """Create detailed explanation of tool ranking"""
        try:
            explanations = {}
            
            for tool_name, score in selected_tools:
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    
                    # Calculate individual score components
                    capability_score = await self._calculate_capability_match(tool, requirements)
                    performance_score = await self._calculate_performance_score(tool, requirements)
                    reliability_score = tool.metrics.reliability_score
                    cost_score = 1.0 - tool.metrics.cost_score
                    availability_score = await self._calculate_availability_score(tool, requirements)
                    
                    explanations[tool_name] = {
                        "total_score": score,
                        "components": {
                            "capability_match": {
                                "score": capability_score,
                                "weight": weights.get("capability_match", 0.3),
                                "contribution": capability_score * weights.get("capability_match", 0.3)
                            },
                            "performance": {
                                "score": performance_score,
                                "weight": weights.get("performance", 0.25),
                                "contribution": performance_score * weights.get("performance", 0.25)
                            },
                            "reliability": {
                                "score": reliability_score,
                                "weight": weights.get("reliability", 0.2),
                                "contribution": reliability_score * weights.get("reliability", 0.2)
                            },
                            "cost": {
                                "score": cost_score,
                                "weight": weights.get("cost", 0.15),
                                "contribution": cost_score * weights.get("cost", 0.15)
                            },
                            "availability": {
                                "score": availability_score,
                                "weight": weights.get("availability", 0.1),
                                "contribution": availability_score * weights.get("availability", 0.1)
                            }
                        },
                        "tool_info": {
                            "description": tool.description,
                            "capabilities": [cap.value for cap in tool.capabilities],
                            "success_rate": tool.metrics.success_rate,
                            "average_latency": tool.metrics.average_latency,
                            "total_executions": tool.metrics.total_executions
                        }
                    }
            
            return {
                "tool_rankings": explanations,
                "selection_criteria": {
                    "weights": weights,
                    "required_capabilities": [cap.value for cap in requirements.required_capabilities],
                    "performance_requirements": requirements.performance_requirements
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating ranking explanation: {e}")
            return {"error": str(e)}
    
    async def learn_from_feedback(self, task_id: str, tool_name: str, 
                                feedback_score: float, context: Optional[Dict[str, Any]] = None):
        """Learn from execution feedback to improve future selections"""
        try:
            # Update tool metrics based on feedback
            if tool_name in self.tools:
                # Store feedback
                self.performance_feedback[tool_name].append(feedback_score)
                
                # Adjust tool cost/reliability based on feedback
                tool = self.tools[tool_name]
                
                # Exponential moving average for reliability adjustment
                current_reliability = tool.metrics.reliability_score
                new_reliability = (
                    current_reliability * (1 - self.learning_rate) + 
                    feedback_score * self.learning_rate
                )
                tool.metrics.reliability_score = new_reliability
                
                logger.debug(f"Learning feedback for {tool_name}: {feedback_score:.3f}, new reliability: {new_reliability:.3f}")
            
            # Adapt selection weights based on feedback patterns
            await self._adapt_selection_weights()
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
    
    async def _adapt_selection_weights(self):
        """Adapt selection weights based on historical performance"""
        try:
            # Only adapt if we have sufficient history
            if len(self.selection_history) < 50:
                return
            
            # Analyze recent selections and their outcomes
            recent_selections = list(self.selection_history)[-50:]
            
            # Calculate correlation between weight components and success
            # This is a simplified adaptation mechanism
            # In production, you'd use more sophisticated ML techniques
            
            capability_successes = []
            performance_successes = []
            reliability_successes = []
            
            for selection in recent_selections:
                # Get feedback for selected tools
                for tool_name in selection["selected_tools"]:
                    if tool_name in self.performance_feedback:
                        recent_feedback = self.performance_feedback[tool_name][-5:]
                        avg_feedback = sum(recent_feedback) / len(recent_feedback) if recent_feedback else 0.5
                        
                        # Correlate with weight components (simplified)
                        capability_successes.append(avg_feedback)
                        performance_successes.append(avg_feedback)
                        reliability_successes.append(avg_feedback)
            
            # Adjust weights slightly based on success patterns
            if capability_successes and len(capability_successes) > 10:
                avg_capability_success = sum(capability_successes) / len(capability_successes)
                if avg_capability_success > 0.8:
                    self.default_weights["capability_match"] = min(0.4, self.default_weights["capability_match"] + 0.01)
                elif avg_capability_success < 0.6:
                    self.default_weights["capability_match"] = max(0.2, self.default_weights["capability_match"] - 0.01)
            
            # Renormalize weights
            total_weight = sum(self.default_weights.values())
            if total_weight > 0:
                self.default_weights = {k: v / total_weight for k, v in self.default_weights.items()}
            
        except Exception as e:
            logger.error(f"Error adapting selection weights: {e}")
    
    async def get_tool_registry_status(self) -> Dict[str, Any]:
        """Get status of tool registry"""
        try:
            total_tools = len(self.tools)
            tools_by_category = defaultdict(int)
            tools_by_capability = defaultdict(int)
            
            total_executions = 0
            avg_success_rate = 0.0
            
            for tool in self.tools.values():
                tools_by_category[tool.category] += 1
                for capability in tool.capabilities:
                    tools_by_capability[capability.value] += 1
                
                total_executions += tool.metrics.total_executions
                avg_success_rate += tool.metrics.success_rate
            
            if total_tools > 0:
                avg_success_rate /= total_tools
            
            return {
                "total_tools": total_tools,
                "tools_by_category": dict(tools_by_category),
                "tools_by_capability": dict(tools_by_capability),
                "total_executions": total_executions,
                "average_success_rate": avg_success_rate,
                "selection_history_size": len(self.selection_history),
                "current_weights": dict(self.default_weights)
            }
            
        except Exception as e:
            logger.error(f"Error getting tool registry status: {e}")
            return {"error": str(e)}
