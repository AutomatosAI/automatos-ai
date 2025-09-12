
"""
Reasoning System Manager
Comprehensive manager for all reasoning components with unified API
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import logging
import json

from .tool_selection import ToolSelectionOptimizer, ToolDefinition, TaskRequirements, ToolCapabilityType
from .execution_orchestrator import ToolExecutionOrchestrator, WorkflowDefinition, ExecutionNode
from .output_processing import OutputProcessor, OutputSchema, OutputType
from .reasoning_engine import (
    IntegratedReasoningEngine, 
    ReasoningContext, 
    ReasoningStrategy
)

logger = logging.getLogger(__name__)

class ReasoningSystemManager:
    """Unified manager for all reasoning system components"""
    
    def __init__(self):
        # Initialize all components
        self.tool_selector = ToolSelectionOptimizer()
        self.execution_orchestrator = ToolExecutionOrchestrator()
        self.output_processor = OutputProcessor()
        self.reasoning_engine = IntegratedReasoningEngine()
        
        # System statistics
        self.total_operations = 0
        self.success_count = 0
        self.last_health_check = None
        
        logger.info("Initialized ReasoningSystemManager")
    
    async def initialize_system(self) -> Dict[str, Any]:
        """Initialize the complete reasoning system"""
        try:
            initialization_results = {
                "tool_selector": "initialized",
                "execution_orchestrator": "initialized", 
                "output_processor": "initialized",
                "reasoning_engine": "initialized",
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Register some default tools for demonstration
            await self._register_default_tools()
            
            logger.info("Reasoning system initialization completed successfully")
            return initialization_results
            
        except Exception as e:
            logger.error(f"Error initializing reasoning system: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _register_default_tools(self):
        """Register some default tools for the system"""
        try:
            default_tools = [
                ToolDefinition(
                    name="text_analyzer",
                    description="Analyze text content for insights",
                    capabilities=[ToolCapabilityType.ANALYSIS, ToolCapabilityType.DATA_PROCESSING],
                    parameters={"input_text": str, "analysis_type": str},
                    dependencies=[],
                    version="1.0.0",
                    category="analysis"
                ),
                ToolDefinition(
                    name="data_processor",
                    description="Process and transform data",
                    capabilities=[ToolCapabilityType.DATA_PROCESSING, ToolCapabilityType.TRANSFORMATION],
                    parameters={"data": dict, "operations": list},
                    dependencies=[],
                    version="1.0.0",
                    category="processing"
                ),
                ToolDefinition(
                    name="api_caller",
                    description="Make API calls and handle responses",
                    capabilities=[ToolCapabilityType.API_INTEGRATION, ToolCapabilityType.NETWORK_OPERATIONS],
                    parameters={"url": str, "method": str, "headers": dict},
                    dependencies=[],
                    version="1.0.0",
                    category="integration"
                )
            ]
            
            for tool in default_tools:
                await self.tool_selector.register_tool(tool)
                
                # Register mock executor
                await self.execution_orchestrator.register_tool_executor(
                    tool.name,
                    self._create_mock_executor(tool.name)
                )
            
            logger.info(f"Registered {len(default_tools)} default tools")
            
        except Exception as e:
            logger.error(f"Error registering default tools: {e}")
    
    def _create_mock_executor(self, tool_name: str):
        """Create a mock executor for demonstration purposes"""
        async def mock_executor(parameters: Dict[str, Any]) -> Dict[str, Any]:
            # Simulate tool execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            return {
                "tool": tool_name,
                "status": "success",
                "output": f"Mock output from {tool_name}",
                "parameters_received": parameters,
                "execution_time": 0.1
            }
        
        return mock_executor
    
    async def execute_reasoning_task(self, 
                                   task_description: str,
                                   goals: Optional[List[str]] = None,
                                   context_data: Optional[Dict[str, Any]] = None,
                                   strategy: Optional[ReasoningStrategy] = None,
                                   quality_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Execute a complete reasoning task using all components"""
        try:
            self.total_operations += 1
            start_time = datetime.utcnow()
            
            logger.info(f"Executing reasoning task: {task_description[:100]}")
            
            # Prepare reasoning context
            available_tools = list(self.tool_selector.tools.keys())
            
            reasoning_context = ReasoningContext(
                task_description=task_description,
                goals=goals or [task_description],
                constraints={},
                available_tools=available_tools,
                context_data=context_data or {},
                quality_requirements=quality_thresholds or {}
            )
            
            # Execute reasoning
            reasoning_result = await self.reasoning_engine.reason(
                reasoning_context, 
                preferred_strategy=strategy,
                max_iterations=3
            )
            
            # Process outputs if available
            processed_outputs = []
            if reasoning_result.final_output:
                processing_result = await self.output_processor.process_output(
                    reasoning_result.final_output,
                    {"task_description": task_description},
                    quality_thresholds=quality_thresholds
                )
                processed_outputs.append(processing_result)
            
            # Compile final result
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            success = (len(reasoning_result.errors) == 0 and 
                      reasoning_result.confidence_score > 0.5)
            
            if success:
                self.success_count += 1
            
            final_result = {
                "task_id": reasoning_result.reasoning_id,
                "task_description": task_description,
                "success": success,
                "confidence_score": reasoning_result.confidence_score,
                "strategy_used": reasoning_result.strategy_used.value,
                "execution_time": execution_time,
                "reasoning_steps": reasoning_result.steps_taken,
                "tools_used": reasoning_result.tools_used,
                "iterations_performed": reasoning_result.iterations_performed,
                "final_output": reasoning_result.final_output,
                "processed_outputs": [
                    {
                        "validation_status": po.validation_status.value,
                        "quality_scores": po.quality_scores,
                        "transformations_applied": po.transformations_applied,
                        "warnings": po.warnings,
                        "errors": po.errors
                    } for po in processed_outputs
                ],
                "quality_metrics": reasoning_result.quality_metrics,
                "warnings": reasoning_result.warnings,
                "errors": reasoning_result.errors,
                "metadata": reasoning_result.metadata,
                "timestamp": reasoning_result.timestamp.isoformat()
            }
            
            logger.info(f"Reasoning task completed: success={success}, "
                       f"confidence={reasoning_result.confidence_score:.3f}, "
                       f"time={execution_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error executing reasoning task: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def register_custom_tool(self, tool_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Register a custom tool in the system"""
        try:
            # Convert dict to ToolDefinition
            tool = ToolDefinition(
                name=tool_definition["name"],
                description=tool_definition["description"],
                capabilities=[ToolCapabilityType(cap) for cap in tool_definition.get("capabilities", [])],
                parameters=tool_definition.get("parameters", {}),
                dependencies=tool_definition.get("dependencies", []),
                version=tool_definition.get("version", "1.0.0"),
                category=tool_definition.get("category", "custom")
            )
            
            await self.tool_selector.register_tool(tool)
            
            # Register executor if provided
            if "executor" in tool_definition:
                await self.execution_orchestrator.register_tool_executor(
                    tool.name, tool_definition["executor"]
                )
            
            logger.info(f"Registered custom tool: {tool.name}")
            
            return {
                "success": True,
                "tool_name": tool.name,
                "message": f"Tool {tool.name} registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Error registering custom tool: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_tool_performance(self, tool_name: str) -> Dict[str, Any]:
        """Analyze performance of a specific tool"""
        try:
            if tool_name not in self.tool_selector.tools:
                return {"error": f"Tool {tool_name} not found"}
            
            tool = self.tool_selector.tools[tool_name]
            metrics = tool.metrics
            
            # Get performance data
            feedback_history = self.tool_selector.performance_feedback.get(tool_name, [])
            
            analysis = {
                "tool_name": tool_name,
                "basic_metrics": {
                    "success_rate": metrics.success_rate,
                    "average_latency": metrics.average_latency,
                    "total_executions": metrics.total_executions,
                    "error_count": metrics.error_count,
                    "reliability_score": metrics.reliability_score,
                    "last_used": metrics.last_used.isoformat() if metrics.last_used else None
                },
                "performance_trend": {
                    "recent_feedback_count": len(feedback_history),
                    "average_recent_score": sum(feedback_history) / len(feedback_history) if feedback_history else 0.0,
                    "trend": self._calculate_performance_trend(feedback_history)
                },
                "recommendations": self._generate_tool_recommendations(tool, feedback_history)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing tool performance: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_trend(self, feedback_history: List[float]) -> str:
        """Calculate performance trend from feedback history"""
        if len(feedback_history) < 10:
            return "insufficient_data"
        
        # Compare recent vs older performance
        recent = feedback_history[-5:]
        older = feedback_history[-10:-5]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        diff = recent_avg - older_avg
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _generate_tool_recommendations(self, tool: ToolDefinition, 
                                     feedback_history: List[float]) -> List[str]:
        """Generate recommendations for tool optimization"""
        recommendations = []
        
        metrics = tool.metrics
        
        if metrics.success_rate < 0.8:
            recommendations.append("Consider improving tool reliability - success rate below 80%")
        
        if metrics.average_latency > 5.0:
            recommendations.append("High average latency detected - optimize for speed")
        
        if feedback_history and len(feedback_history) > 5:
            recent_avg = sum(feedback_history[-5:]) / 5
            if recent_avg < 0.6:
                recommendations.append("Recent performance declining - review tool implementation")
        
        if metrics.total_executions < 10:
            recommendations.append("Limited usage data - increase tool utilization for better metrics")
        
        if not recommendations:
            recommendations.append("Tool performance is satisfactory")
        
        return recommendations
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            self.last_health_check = datetime.utcnow()
            
            # Get component statistics
            tool_registry_status = await self.tool_selector.get_tool_registry_status()
            orchestrator_stats = await self.execution_orchestrator.get_orchestrator_statistics()
            processor_stats = await self.output_processor.get_processing_statistics()
            reasoning_stats = await self.reasoning_engine.get_reasoning_statistics()
            
            # Calculate overall health metrics
            overall_success_rate = self.success_count / self.total_operations if self.total_operations > 0 else 0.0
            
            system_health = "healthy"
            if overall_success_rate < 0.5:
                system_health = "critical"
            elif overall_success_rate < 0.7:
                system_health = "warning"
            
            return {
                "system_status": system_health,
                "overall_success_rate": overall_success_rate,
                "total_operations": self.total_operations,
                "successful_operations": self.success_count,
                "last_health_check": self.last_health_check.isoformat(),
                "components": {
                    "tool_selector": {
                        "status": "operational",
                        "statistics": tool_registry_status
                    },
                    "execution_orchestrator": {
                        "status": "operational", 
                        "statistics": orchestrator_stats
                    },
                    "output_processor": {
                        "status": "operational",
                        "statistics": processor_stats
                    },
                    "reasoning_engine": {
                        "status": "operational",
                        "statistics": reasoning_stats
                    }
                },
                "recommendations": self._generate_system_recommendations(
                    overall_success_rate, tool_registry_status, orchestrator_stats
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "system_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _generate_system_recommendations(self, success_rate: float, 
                                       tool_stats: Dict[str, Any],
                                       orchestrator_stats: Dict[str, Any]) -> List[str]:
        """Generate system-wide optimization recommendations"""
        recommendations = []
        
        if success_rate < 0.7:
            recommendations.append("Low success rate - review tool selection and execution strategies")
        
        if tool_stats.get("total_tools", 0) < 5:
            recommendations.append("Limited tool registry - consider adding more specialized tools")
        
        if orchestrator_stats.get("success_rate", 0) < 0.8:
            recommendations.append("Orchestrator success rate below optimal - review execution workflows")
        
        if not recommendations:
            recommendations.append("System is operating optimally")
        
        return recommendations
    
    async def optimize_system_performance(self) -> Dict[str, Any]:
        """Perform system-wide performance optimization"""
        try:
            optimization_results = {
                "optimizations_applied": [],
                "performance_improvements": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Optimize tool selection weights
            await self.tool_selector._adapt_selection_weights()
            optimization_results["optimizations_applied"].append("tool_selection_weights_adapted")
            
            # Optimize reasoning strategy adaptation
            await self.reasoning_engine._update_strategy_performance(
                # Use dummy result for adaptation
                type('obj', (object,), {
                    'strategy_used': ReasoningStrategy.ADAPTIVE,
                    'confidence_score': 0.8,
                    'errors': []
                })()
            )
            optimization_results["optimizations_applied"].append("reasoning_strategies_optimized")
            
            # Clear old cache entries in output processor
            if len(self.output_processor.validation_cache) > 1000:
                # Keep only recent entries
                recent_keys = list(self.output_processor.validation_cache.keys())[-500:]
                new_cache = {k: self.output_processor.validation_cache[k] for k in recent_keys}
                self.output_processor.validation_cache = new_cache
                optimization_results["optimizations_applied"].append("output_processor_cache_optimized")
            
            logger.info("System performance optimization completed")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing system performance: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def export_system_configuration(self) -> Dict[str, Any]:
        """Export current system configuration"""
        try:
            config = {
                "version": "1.0.0",
                "export_timestamp": datetime.utcnow().isoformat(),
                "tools": {
                    name: tool.to_dict() for name, tool in self.tool_selector.tools.items()
                },
                "reasoning_strategies": {
                    strategy.value: {
                        "success_rate": perf["success_rate"],
                        "avg_confidence": perf["avg_confidence"],
                        "usage_count": perf["usage_count"]
                    } for strategy, perf in self.reasoning_engine.strategy_performance.items()
                },
                "system_statistics": {
                    "total_operations": self.total_operations,
                    "success_count": self.success_count,
                    "overall_success_rate": self.success_count / max(1, self.total_operations)
                }
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Error exporting system configuration: {e}")
            return {"error": str(e)}
    
    async def shutdown_system(self) -> Dict[str, Any]:
        """Gracefully shutdown the reasoning system"""
        try:
            logger.info("Shutting down reasoning system...")
            
            # Cancel any active workflows
            active_workflows = list(self.execution_orchestrator.active_workflows.keys())
            for workflow_id in active_workflows:
                await self.execution_orchestrator.cancel_workflow(workflow_id)
            
            shutdown_summary = {
                "status": "shutdown_complete",
                "cancelled_workflows": len(active_workflows),
                "total_operations_processed": self.total_operations,
                "final_success_rate": self.success_count / max(1, self.total_operations),
                "shutdown_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info("Reasoning system shutdown completed successfully")
            return shutdown_summary
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
            return {
                "status": "shutdown_error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
