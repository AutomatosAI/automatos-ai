
"""
Reasoning System Service
=======================

Service layer for tool-integrated reasoning with enhanced capabilities.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from reasoning.manager import ReasoningSystemManager
from models import Task

logger = logging.getLogger(__name__)

class ReasoningSystemService:
    """
    Service for managing reasoning system operations
    """
    
    def __init__(self):
        self.reasoning_manager = ReasoningSystemManager()
        logger.info("Reasoning System Service initialized")
    
    async def perform_comprehensive_reasoning(
        self,
        db: Session,
        task_id: int,
        context: Dict[str, Any],
        reasoning_type: str = "integrated"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive reasoning for a task
        """
        try:
            # Get task from database
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                return {"error": "Task not found"}
            
            # Prepare reasoning context for the existing method
            reasoning_parameters = {
                "task_context": {
                    "task_id": task_id,
                    "title": task.title,
                    "description": task.description,
                    "importance": task.importance
                },
                "available_tools": task.tools or [],
                "execution_results": task.execution_status or {},
                "user_context": context
            }
            
            # Use the existing execute_reasoning_task method
            reasoning_result = await self.reasoning_manager.execute_reasoning_task(
                task_description=f"Reasoning for task: {task.title}",
                parameters=reasoning_parameters,
                max_iterations=5
            )
            
            # Store reasoning results in task
            task.reasoning = {
                **reasoning_result,
                "reasoning_type": reasoning_type,
                "performed_at": datetime.utcnow().isoformat()
            }
            
            db.commit()
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Failed to perform comprehensive reasoning: {e}")
            return {"error": str(e)}
    
    async def optimize_tool_selection(
        self,
        available_tools: List[Dict[str, Any]],
        task_requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Optimize tool selection using reasoning system
        """
        try:
            # Use simple optimization based on task requirements
            optimized_tools = []
            
            for tool in available_tools:
                tool_score = 0.5  # Base score
                
                # Boost score based on task requirements
                if "priority" in task_requirements:
                    if task_requirements["priority"] == "high" and tool.get("capability", 0) > 0.8:
                        tool_score += 0.3
                
                if "complexity" in task_requirements:
                    if task_requirements["complexity"] == "high" and tool.get("complexity_handling", False):
                        tool_score += 0.2
                
                optimized_tool = {**tool, "optimization_score": tool_score}
                optimized_tools.append(optimized_tool)
            
            # Sort by optimization score
            optimized_tools.sort(key=lambda x: x.get("optimization_score", 0), reverse=True)
            
            return optimized_tools
            
        except Exception as e:
            logger.error(f"Failed to optimize tool selection: {e}")
            return available_tools
    
    async def analyze_execution_results(
        self,
        execution_results: List[Dict[str, Any]],
        expected_outcomes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze execution results using reasoning system
        """
        try:
            analysis = {
                "total_executions": len(execution_results),
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0,
                "outcome_match": False,
                "recommendations": []
            }
            
            execution_times = []
            
            for result in execution_results:
                if result.get("status") == "completed":
                    analysis["successful_executions"] += 1
                else:
                    analysis["failed_executions"] += 1
                
                if "execution_time" in result:
                    execution_times.append(result["execution_time"])
            
            if execution_times:
                analysis["average_execution_time"] = sum(execution_times) / len(execution_times)
            
            # Check outcome match
            success_rate = analysis["successful_executions"] / len(execution_results) if execution_results else 0
            expected_success_rate = expected_outcomes.get("expected_success_rate", 0.8)
            
            analysis["outcome_match"] = success_rate >= expected_success_rate
            
            # Generate recommendations
            if success_rate < 0.7:
                analysis["recommendations"].append("Consider reviewing tool selection and parameters")
            
            if analysis["average_execution_time"] > 30:
                analysis["recommendations"].append("Execution time is high, consider optimization")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze execution results: {e}")
            return {"error": str(e)}
    
    async def get_reasoning_insights(
        self,
        db: Session,
        task_id: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get reasoning insights and statistics
        """
        try:
            # Use available manager methods
            insights = {
                "system_status": "operational",
                "reasoning_operations": self.reasoning_manager.total_operations,
                "success_rate": (self.reasoning_manager.success_count / max(self.reasoning_manager.total_operations, 1)) * 100,
                "last_health_check": self.reasoning_manager.last_health_check.isoformat() if self.reasoning_manager.last_health_check else None
            }
            
            if task_id:
                task = db.query(Task).filter(Task.id == task_id).first()
                if task and task.reasoning:
                    insights["task_reasoning"] = task.reasoning
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get reasoning insights: {e}")
            return {"error": str(e)}
