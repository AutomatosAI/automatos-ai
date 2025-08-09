
"""
Enhanced Orchestrator Service
============================

Advanced orchestrator service implementing features from code reviews:
- Tool selection optimization
- Task execution with memory integration
- Performance monitoring and optimization
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import numpy as np
import networkx as nx

# Import models and database
from models import Task, User, TaskCreate, TaskUpdate, TaskResponse
from database import get_db

# Import memory and reasoning systems
from memory.manager import AdvancedMemoryManager
from memory.memory_types import MemoryType
from reasoning.manager import ReasoningSystemManager

# Import multi-agent systems
from multi_agent.collaborative_reasoning import CollaborativeReasoningEngine
from multi_agent.coordination_manager import CoordinationManager
from multi_agent.behavior_monitor import EmergentBehaviorMonitor
from multi_agent.optimization_engine import MultiAgentOptimizer

# Import field theory integration
from field_theory.field_manager import FieldContextManager, FieldType

logger = logging.getLogger(__name__)

class EnhancedOrchestratorService:
    """
    Enhanced orchestrator service with advanced memory and reasoning integration
    """
    
    def __init__(self):
        # Initialize advanced components
        self.memory_manager = AdvancedMemoryManager()
        self.reasoning_manager = ReasoningSystemManager()
        
        # Initialize multi-agent systems
        self.collaborative_reasoning = CollaborativeReasoningEngine()
        self.coordination_manager = CoordinationManager()
        self.behavior_monitor = EmergentBehaviorMonitor()
        self.multi_agent_optimizer = MultiAgentOptimizer()
        
        # Initialize field theory integration
        self.field_manager = FieldContextManager()
        
        # Performance tracking
        self.operation_metrics = {
            "task_creation": [],
            "task_execution": [],
            "tool_selection": [],
            "memory_operations": [],
            "multi_agent_operations": [],
            "field_operations": []
        }
        
        logger.info("Enhanced Orchestrator Service with Multi-Agent & Field Theory initialized")
    
    # Enhanced Task Management (based on code review 06_tool_integrated_reasoning)
    
    async def create_task_with_intelligence(
        self,
        db: Session,
        task_data: TaskCreate,
        user_id: int,
        auto_tool_selection: bool = True,
        memory_integration: bool = True
    ) -> TaskResponse:
        """
        Create task with intelligent tool selection and memory integration
        """
        start_time = time.time()
        
        try:
            # Create base task
            db_task = Task(
                title=task_data.title,
                description=task_data.description,
                owner_id=user_id,
                importance=task_data.importance,
                status="created"
            )
            
            db.add(db_task)
            db.commit()
            db.refresh(db_task)
            
            # Store task in memory system if enabled
            if memory_integration:
                memory_content = {
                    "task_id": db_task.id,
                    "title": db_task.title,
                    "description": db_task.description,
                    "status": db_task.status
                }
                
                memory_id = await self.memory_manager.store_memory(
                    session_id=f"task_{user_id}",
                    content=memory_content,
                    memory_type=MemoryType.PROCEDURAL,
                    importance=task_data.importance,
                    tags=["task", "creation", str(db_task.id)]
                )
                
                # Store memory reference in task
                db_task.working_memory = {"memory_id": memory_id}
                db.commit()
            
            # Auto-select tools if enabled
            if auto_tool_selection:
                selected_tools = await self.select_tools_for_task(db, db_task.id, user_id)
                db_task.tools = [tool["name"] for tool in selected_tools]
                db_task.tool_scores = selected_tools
                db.commit()
            
            # Record performance
            execution_time = time.time() - start_time
            self.operation_metrics["task_creation"].append(execution_time)
            
            # Convert to response
            return TaskResponse(
                id=db_task.id,
                title=db_task.title,
                description=db_task.description,
                status=db_task.status,
                owner_id=db_task.owner_id,
                importance=db_task.importance,
                tools=db_task.tools,
                reasoning=db_task.reasoning,
                created_at=db_task.created_at,
                updated_at=db_task.updated_at
            )
            
        except Exception as e:
            logger.error(f"Failed to create intelligent task: {e}")
            db.rollback()
            raise
    
    async def select_tools_for_task(
        self,
        db: Session,
        task_id: int,
        user_id: int
    ) -> List[Dict]:
        """
        Implement tool selection optimization from code review
        """
        try:
            db_task = db.query(Task).filter(
                and_(Task.id == task_id, Task.owner_id == user_id)
            ).first()
            
            if not db_task:
                return []
            
            # Define available tools with capabilities, latency, and cost
            available_tools = [
                {"name": "web_search", "capability": 0.9, "latency": 0.3, "cost": 0.1},
                {"name": "code_analysis", "capability": 0.8, "latency": 0.5, "cost": 0.2},
                {"name": "document_processing", "capability": 0.7, "latency": 0.2, "cost": 0.05},
                {"name": "data_visualization", "capability": 0.6, "latency": 0.4, "cost": 0.15},
                {"name": "ai_reasoning", "capability": 0.95, "latency": 0.6, "cost": 0.3}
            ]
            
            # Calculate weighted scores: Score(T) = w1 * Capability + w2 * Performance + w3 * Cost
            weights = {"capability": 0.5, "latency": 0.3, "cost": 0.2}
            scored_tools = []
            
            for tool in available_tools:
                # Higher capability is better, lower latency is better, lower cost is better
                score = (
                    weights["capability"] * tool["capability"] - 
                    weights["latency"] * tool["latency"] - 
                    weights["cost"] * tool["cost"]
                )
                
                # Adjust score based on task content
                task_content = f"{db_task.title} {db_task.description or ''}".lower()
                
                # Boost specific tools based on task content
                if "web" in task_content and tool["name"] == "web_search":
                    score += 0.2
                elif "code" in task_content and tool["name"] == "code_analysis":
                    score += 0.2
                elif "document" in task_content and tool["name"] == "document_processing":
                    score += 0.2
                elif "visualiz" in task_content and tool["name"] == "data_visualization":
                    score += 0.2
                elif "analy" in task_content and tool["name"] == "ai_reasoning":
                    score += 0.2
                
                scored_tools.append({
                    "name": tool["name"],
                    "score": score,
                    "capability": tool["capability"],
                    "latency": tool["latency"],
                    "cost": tool["cost"]
                })
            
            # Sort by score and return top tools
            scored_tools.sort(key=lambda x: x["score"], reverse=True)
            selected_tools = scored_tools[:3]  # Select top 3 tools
            
            # Record performance
            self.operation_metrics["tool_selection"].append(time.time())
            
            return selected_tools
            
        except Exception as e:
            logger.error(f"Failed to select tools for task {task_id}: {e}")
            return []
    
    async def execute_task_with_orchestration(
        self,
        db: Session,
        task_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Execute task with tool orchestration and dependency management
        """
        start_time = time.time()
        
        try:
            db_task = db.query(Task).filter(
                and_(Task.id == task_id, Task.owner_id == user_id)
            ).first()
            
            if not db_task:
                return {"error": "Task not found"}
            
            # Create execution dependency graph
            tools = db_task.tools or []
            if not tools:
                return {"error": "No tools selected for task"}
            
            # Build dependency graph G = (T, D)
            g = nx.DiGraph()
            for i, tool in enumerate(tools):
                g.add_node(tool, status="pending", index=i)
                # Create sequential dependencies for now (can be enhanced)
                if i > 0:
                    g.add_edge(tools[i-1], tool)
            
            # Get execution plan using topological sort
            try:
                execution_plan = list(nx.topological_sort(g))
            except nx.NetworkXError:
                # Handle cycles by using original order
                execution_plan = tools
            
            # Execute tools in planned order
            execution_results = []
            for tool_name in execution_plan:
                # Simulate tool execution (integrate with actual tools in production)
                result = await self._execute_tool(tool_name, db_task)
                execution_results.append({
                    "tool": tool_name,
                    "status": "completed",
                    "output": result,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Update graph status
                g.nodes[tool_name]["status"] = "completed"
            
            # Store execution results
            db_task.execution_status = {
                "plan": execution_plan,
                "results": execution_results,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            # Update dependencies
            db_task.dependencies = {
                "graph": {
                    "nodes": [{"name": n, "status": g.nodes[n]["status"]} for n in g.nodes()],
                    "edges": [{"from": u, "to": v} for u, v in g.edges()]
                }
            }
            
            # Calculate efficiency: Efficiency = (Completed_Tasks / Total_Resources)
            completed_tasks = len([r for r in execution_results if r["status"] == "completed"])
            efficiency = completed_tasks / len(execution_plan) if execution_plan else 0
            
            db_task.status = "completed"
            db.commit()
            
            # Record performance
            execution_time = time.time() - start_time
            self.operation_metrics["task_execution"].append(execution_time)
            
            return {
                "task_id": task_id,
                "execution_plan": execution_plan,
                "results": execution_results,
                "efficiency": efficiency,
                "execution_time": execution_time,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Failed to execute task {task_id}: {e}")
            return {"error": str(e)}
    
    async def _execute_tool(self, tool_name: str, task: Task) -> Dict[str, Any]:
        """
        Execute individual tool (placeholder for actual tool integration)
        """
        # Simulate tool execution based on tool type
        simulation_results = {
            "web_search": {
                "type": "search_results",
                "results": ["Result 1", "Result 2", "Result 3"],
                "count": 3
            },
            "code_analysis": {
                "type": "code_metrics",
                "complexity": 0.7,
                "quality_score": 0.85,
                "issues": []
            },
            "document_processing": {
                "type": "document_summary",
                "summary": f"Processed document for task: {task.title}",
                "word_count": 1250
            },
            "data_visualization": {
                "type": "chart_data",
                "chart_type": "bar",
                "data_points": 10
            },
            "ai_reasoning": {
                "type": "reasoning_result",
                "conclusion": f"Analysis of task '{task.title}' suggests proceeding with planned approach",
                "confidence": 0.92
            }
        }
        
        return simulation_results.get(tool_name, {"type": "generic", "output": f"Executed {tool_name}"})
    
    async def reason_with_tools_for_task(
        self,
        db: Session,
        task_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Perform integrated reasoning based on tool outputs
        """
        try:
            db_task = db.query(Task).filter(
                and_(Task.id == task_id, Task.owner_id == user_id)
            ).first()
            
            if not db_task:
                return {"error": "Task not found"}
            
            tools = db_task.tools or []
            execution_status = db_task.execution_status or {}
            results = execution_status.get("results", [])
            
            if not results:
                return {"error": "No tool execution results to reason about"}
            
            # Perform reasoning with iterative refinement
            reasoning_steps = []
            
            for result in results:
                tool_name = result["tool"]
                output = result["output"]
                
                # Calculate reasoning score based on output quality
                reasoning_score = self._calculate_reasoning_score(output)
                
                step = {
                    "tool": tool_name,
                    "output_summary": str(output)[:200],  # Truncate for storage
                    "reasoning_score": reasoning_score,
                    "analysis": self._analyze_tool_output(tool_name, output)
                }
                reasoning_steps.append(step)
            
            # Apply meta-reasoning: R_{t+1} = R_t + α * ∇_R Q(R_t)
            overall_score = np.mean([step["reasoning_score"] for step in reasoning_steps])
            
            # Strategy selection: P(Strategy | Context, Tools)
            context_tokens = f"{db_task.title} {db_task.description or ''}".split()
            strategy_probability = min(len(context_tokens) / 100.0, 1.0)  # Simple heuristic
            
            final_reasoning = {
                "steps": reasoning_steps,
                "overall_score": overall_score,
                "strategy_probability": strategy_probability,
                "recommendation": self._generate_recommendation(reasoning_steps, overall_score),
                "confidence": overall_score * strategy_probability,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store reasoning results
            db_task.reasoning = final_reasoning
            db.commit()
            
            return final_reasoning
            
        except Exception as e:
            logger.error(f"Failed to perform reasoning for task {task_id}: {e}")
            return {"error": str(e)}
    
    def _calculate_reasoning_score(self, output: Dict[str, Any]) -> float:
        """Calculate reasoning score for tool output"""
        base_score = 0.5
        
        # Adjust score based on output characteristics
        if output.get("type") == "search_results":
            result_count = output.get("count", 0)
            base_score += min(result_count * 0.1, 0.4)
        
        elif output.get("type") == "code_metrics":
            quality = output.get("quality_score", 0.5)
            base_score = quality
        
        elif output.get("type") == "reasoning_result":
            confidence = output.get("confidence", 0.5)
            base_score = confidence
        
        return min(base_score, 1.0)
    
    def _analyze_tool_output(self, tool_name: str, output: Dict[str, Any]) -> str:
        """Generate analysis of tool output"""
        analyses = {
            "web_search": f"Found {output.get('count', 0)} relevant search results",
            "code_analysis": f"Code quality score: {output.get('quality_score', 0):.2f}",
            "document_processing": f"Processed {output.get('word_count', 0)} words",
            "data_visualization": f"Generated {output.get('chart_type', 'unknown')} chart",
            "ai_reasoning": f"Reasoning confidence: {output.get('confidence', 0):.2f}"
        }
        
        return analyses.get(tool_name, f"Executed {tool_name} successfully")
    
    def _generate_recommendation(self, reasoning_steps: List[Dict], overall_score: float) -> str:
        """Generate task recommendation based on reasoning"""
        if overall_score > 0.8:
            return "High confidence in task completion. Proceed with implementation."
        elif overall_score > 0.6:
            return "Moderate confidence. Consider additional validation steps."
        else:
            return "Low confidence. Review task requirements and tool selection."
    
    # Performance optimization methods (from code review 03_context_management)
    
    async def optimize_task_performance(
        self,
        db: Session,
        task_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Optimize task performance using adaptive protocols
        """
        try:
            db_task = db.query(Task).filter(
                and_(Task.id == task_id, Task.owner_id == user_id)
            ).first()
            
            if not db_task:
                return {"error": "Task not found"}
            
            # Adaptive optimization based on task content and history
            content_length = len(f"{db_task.description or ''}")
            
            # Adjust priority based on content and importance
            if content_length < 500 and db_task.importance:
                optimized_priority = min(db_task.importance * 1.2, 1.0)
            else:
                optimized_priority = db_task.importance * 0.9 if db_task.importance else 0.5
            
            # Update task with optimized settings
            db_task.importance = optimized_priority
            
            # Store optimization metadata
            optimization_data = {
                "original_importance": db_task.importance,
                "optimized_importance": optimized_priority,
                "content_length": content_length,
                "optimization_factor": optimized_priority / db_task.importance if db_task.importance else 1.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            db_task.working_memory = {
                **(db_task.working_memory or {}),
                "optimization": optimization_data
            }
            
            db.commit()
            
            return {
                "task_id": task_id,
                "optimization_applied": True,
                "optimization_data": optimization_data
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize task {task_id}: {e}")
            return {"error": str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics
        """
        try:
            metrics = {}
            
            for operation, times in self.operation_metrics.items():
                if times:
                    metrics[operation] = {
                        "count": len(times),
                        "avg_time": np.mean(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "std_dev": np.std(times)
                    }
                else:
                    metrics[operation] = {
                        "count": 0,
                        "avg_time": 0,
                        "min_time": 0,
                        "max_time": 0,
                        "std_dev": 0
                    }
            
            # Get memory system stats
            memory_stats = await self.memory_manager.get_comprehensive_stats()
            
            return {
                "operation_metrics": metrics,
                "memory_system_stats": {
                    "total_memories": memory_stats.total_memories,
                    "memory_levels": memory_stats.memory_levels,
                    "system_performance": memory_stats.system_performance
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    # Standard CRUD operations with enhancements
    
    def get_tasks(
        self,
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        status_filter: Optional[str] = None,
        importance_threshold: Optional[float] = None
    ) -> List[Task]:
        """
        Get tasks with enhanced filtering and caching
        """
        try:
            query = db.query(Task).filter(Task.owner_id == user_id)
            
            if status_filter:
                query = query.filter(Task.status == status_filter)
            
            if importance_threshold:
                query = query.filter(Task.importance >= importance_threshold)
            
            tasks = query.offset(skip).limit(limit).all()
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to get tasks: {e}")
            return []
    
    def get_task(self, db: Session, task_id: int, user_id: int) -> Optional[Task]:
        """Get single task with user verification"""
        return db.query(Task).filter(
            and_(Task.id == task_id, Task.owner_id == user_id)
        ).first()
    
    def update_task(
        self,
        db: Session,
        task_id: int,
        user_id: int,
        task_update: TaskUpdate
    ) -> Optional[Task]:
        """Update task with change tracking"""
        try:
            db_task = self.get_task(db, task_id, user_id)
            if not db_task:
                return None
            
            update_data = task_update.dict(exclude_unset=True)
            
            for field, value in update_data.items():
                setattr(db_task, field, value)
            
            db_task.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_task)
            
            return db_task
            
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            db.rollback()
            return None
    
    def delete_task(self, db: Session, task_id: int, user_id: int) -> bool:
        """Delete task with cleanup"""
        try:
            db_task = self.get_task(db, task_id, user_id)
            if not db_task:
                return False
            
            db.delete(db_task)
            db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            db.rollback()
            return False
    
    # Multi-Agent System Methods
    
    async def perform_collaborative_reasoning(
        self,
        db: Session,
        task_id: int,
        user_id: int,
        agents: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform collaborative reasoning across multiple agents
        """
        start_time = time.time()
        
        try:
            result = await self.collaborative_reasoning.collaborative_reasoning(
                db, task_id, user_id, agents, context
            )
            
            # Record performance
            execution_time = time.time() - start_time
            self.operation_metrics["multi_agent_operations"].append(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Collaborative reasoning failed for task {task_id}: {e}")
            raise
    
    async def coordinate_multi_agents(
        self,
        db: Session,
        task_id: int,
        user_id: int,
        agents: List[str],
        strategy: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents for task execution
        """
        start_time = time.time()
        
        try:
            from multi_agent.coordination_manager import CoordinationStrategy
            
            # Convert strategy string to enum
            coord_strategy = None
            if strategy:
                try:
                    coord_strategy = CoordinationStrategy(strategy.lower())
                except ValueError:
                    logger.warning(f"Invalid coordination strategy: {strategy}")
            
            result = await self.coordination_manager.coordinate_agents(
                db, task_id, user_id, agents, coord_strategy, context
            )
            
            # Record performance
            execution_time = time.time() - start_time
            self.operation_metrics["multi_agent_operations"].append(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Agent coordination failed for task {task_id}: {e}")
            raise
    
    async def monitor_emergent_behavior(
        self,
        session_id: str,
        agents: List[str],
        interactions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Monitor and analyze emergent behaviors in agent interactions
        """
        start_time = time.time()
        
        try:
            result = await self.behavior_monitor.monitor_emergent_behavior(
                session_id, agents, interactions, context
            )
            
            # Record performance
            execution_time = time.time() - start_time
            self.operation_metrics["multi_agent_operations"].append(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Behavior monitoring failed for session {session_id}: {e}")
            raise
    
    async def optimize_multi_agent_system(
        self,
        db: Session,
        task_id: int,
        user_id: int,
        agents: List[str],
        config: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize multi-agent system configuration and performance
        """
        start_time = time.time()
        
        try:
            from multi_agent.optimization_engine import OptimizationConfig, OptimizationStrategy, OptimizationObjective
            
            # Convert config dict to OptimizationConfig if provided
            optimization_config = None
            if config:
                try:
                    strategy = OptimizationStrategy(config.get("strategy", "adaptive"))
                    objectives = [OptimizationObjective(obj) for obj in config.get("objectives", ["performance", "scalability", "robustness"])]
                    weights = config.get("weights", {})
                    
                    optimization_config = OptimizationConfig(
                        strategy=strategy,
                        objectives=objectives,
                        weights=weights,
                        constraints=config.get("constraints", {}),
                        max_iterations=config.get("max_iterations", 100),
                        tolerance=config.get("tolerance", 1e-6),
                        population_size=config.get("population_size", 20)
                    )
                except Exception as e:
                    logger.warning(f"Invalid optimization config: {e}")
            
            result = await self.multi_agent_optimizer.optimize_agents(
                db, task_id, user_id, agents, optimization_config, context
            )
            
            # Record performance
            execution_time = time.time() - start_time
            self.operation_metrics["multi_agent_operations"].append(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-agent optimization failed for task {task_id}: {e}")
            raise
    
    # Field Theory Integration Methods
    
    async def update_field_context(
        self,
        session_id: str,
        context_data: Dict[str, Any],
        field_type: str = "scalar"
    ) -> Dict[str, Any]:
        """
        Update field representation for context data
        """
        start_time = time.time()
        
        try:
            # Convert field type string to enum
            field_type_enum = FieldType.SCALAR
            if field_type.lower() == "vector":
                field_type_enum = FieldType.VECTOR
            elif field_type.lower() == "tensor":
                field_type_enum = FieldType.TENSOR
            
            result = await self.field_manager.update_field(
                session_id, context_data, field_type_enum
            )
            
            # Record performance
            execution_time = time.time() - start_time
            self.operation_metrics["field_operations"].append(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Field context update failed for session {session_id}: {e}")
            raise
    
    async def propagate_field_influence(
        self,
        session_id: str,
        propagation_steps: int = 3
    ) -> Dict[str, Any]:
        """
        Propagate field influence using gradient calculations
        """
        start_time = time.time()
        
        try:
            result = await self.field_manager.propagate_influence(
                session_id, propagation_steps
            )
            
            # Record performance
            execution_time = time.time() - start_time
            self.operation_metrics["field_operations"].append(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Field influence propagation failed for session {session_id}: {e}")
            raise
    
    async def model_field_interactions(
        self,
        db: Session,
        task_id: int,
        user_id: int,
        similarity_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Model interactions between task contexts using field theory
        """
        start_time = time.time()
        
        try:
            result = await self.field_manager.model_context_interactions(
                db, task_id, user_id, similarity_threshold
            )
            
            # Record performance
            execution_time = time.time() - start_time
            self.operation_metrics["field_operations"].append(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Field interaction modeling failed for task {task_id}: {e}")
            raise
    
    async def manage_dynamic_fields(
        self,
        session_id: str,
        alpha: float = 0.1,
        beta: float = 0.2
    ) -> Dict[str, Any]:
        """
        Dynamic field management with real-time updates
        """
        start_time = time.time()
        
        try:
            result = await self.field_manager.manage_dynamic_field(
                session_id, alpha, beta
            )
            
            # Record performance
            execution_time = time.time() - start_time
            self.operation_metrics["field_operations"].append(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Dynamic field management failed for session {session_id}: {e}")
            raise
    
    async def optimize_field_configuration(
        self,
        db: Session,
        task_id: int,
        user_id: int,
        objectives: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Multi-objective field optimization
        """
        start_time = time.time()
        
        try:
            result = await self.field_manager.optimize_field(
                db, task_id, user_id, objectives
            )
            
            # Record performance
            execution_time = time.time() - start_time
            self.operation_metrics["field_operations"].append(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Field optimization failed for task {task_id}: {e}")
            raise
    
    # Comprehensive Analytics
    
    async def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive system analytics including multi-agent and field theory metrics
        """
        try:
            # Base performance metrics
            base_metrics = await self.get_performance_metrics()
            
            # Multi-agent system statistics
            collaborative_stats = self.collaborative_reasoning.get_reasoning_statistics()
            coordination_stats = self.coordination_manager.get_coordination_statistics()
            behavior_stats = self.behavior_monitor.get_monitoring_statistics()
            optimization_stats = self.multi_agent_optimizer.get_optimization_statistics()
            
            # Field theory statistics
            field_stats = self.field_manager.get_field_statistics()
            
            return {
                "base_metrics": base_metrics,
                "multi_agent_systems": {
                    "collaborative_reasoning": collaborative_stats,
                    "coordination_management": coordination_stats,
                    "behavior_monitoring": behavior_stats,
                    "optimization": optimization_stats
                },
                "field_theory": field_stats,
                "system_integration": {
                    "total_operations": sum(len(ops) for ops in self.operation_metrics.values()),
                    "operation_breakdown": {
                        op_type: len(ops) for op_type, ops in self.operation_metrics.items()
                    },
                    "average_operation_times": {
                        op_type: np.mean(ops) if ops else 0.0 
                        for op_type, ops in self.operation_metrics.items()
                    }
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive analytics: {e}")
            return {"error": str(e)}
