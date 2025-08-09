
"""
Tool Execution Orchestration Module
Advanced workflow orchestration with dependency management and parallel execution
"""
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
from collections import defaultdict, deque
import networkx as nx
import uuid

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Status of tool execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"

class DependencyType(Enum):
    """Types of dependencies between tools"""
    SEQUENTIAL = "sequential"  # Tool B runs after Tool A completes
    DATA_DEPENDENCY = "data_dependency"  # Tool B needs output from Tool A
    CONDITIONAL = "conditional"  # Tool B runs only if Tool A meets condition
    PARALLEL_SAFE = "parallel_safe"  # Tools can run in parallel
    EXCLUSIVE = "exclusive"  # Tools cannot run simultaneously

@dataclass
class ExecutionNode:
    """Represents a tool execution in the workflow graph"""
    node_id: str
    tool_name: str
    parameters: Dict[str, Any]
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    priority: float = 0.5
    
    def duration(self) -> float:
        """Calculate execution duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def is_terminal(self) -> bool:
        """Check if node is in terminal state"""
        return self.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, 
                              ExecutionStatus.CANCELLED, ExecutionStatus.TIMEOUT]

@dataclass
class ExecutionDependency:
    """Represents dependency between execution nodes"""
    from_node: str
    to_node: str
    dependency_type: DependencyType
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    data_mapping: Optional[Dict[str, str]] = None  # Maps output keys to input keys

@dataclass
class WorkflowDefinition:
    """Defines a complete workflow for execution"""
    workflow_id: str
    description: str
    nodes: List[ExecutionNode]
    dependencies: List[ExecutionDependency]
    global_timeout: int = 1800  # 30 minutes
    failure_strategy: str = "fail_fast"  # "fail_fast", "continue", "retry"
    max_parallel: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Result of workflow execution"""
    workflow_id: str
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime]
    total_duration: float
    node_results: Dict[str, Dict[str, Any]]
    execution_graph: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]

class ToolExecutionOrchestrator:
    """Advanced tool execution orchestrator with dependency management"""
    
    def __init__(self, max_concurrent_workflows: int = 10):
        # Workflow management
        self.active_workflows: Dict[str, WorkflowDefinition] = {}
        self.workflow_results: Dict[str, ExecutionResult] = {}
        self.max_concurrent_workflows = max_concurrent_workflows
        
        # Execution tracking
        self.execution_history: deque = deque(maxlen=10000)
        self.performance_stats: Dict[str, List[float]] = defaultdict(list)
        
        # Registered tool executors
        self.tool_executors: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}
        
        # Resource management
        self.resource_usage: Dict[str, float] = defaultdict(float)
        self.resource_limits: Dict[str, float] = {
            "cpu": 80.0,  # Max CPU usage percentage
            "memory": 90.0,  # Max memory usage percentage
            "network": 100.0,  # Max network bandwidth
            "concurrent_executions": 50  # Max concurrent tool executions
        }
        
        logger.info("Initialized ToolExecutionOrchestrator")
    
    async def register_tool_executor(self, tool_name: str, 
                                   executor: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        """Register an executor function for a specific tool"""
        try:
            self.tool_executors[tool_name] = executor
            logger.info(f"Registered executor for tool: {tool_name}")
        except Exception as e:
            logger.error(f"Error registering tool executor for {tool_name}: {e}")
    
    async def execute_workflow(self, workflow: WorkflowDefinition) -> ExecutionResult:
        """Execute a complete workflow with dependency management"""
        try:
            logger.info(f"Starting workflow execution: {workflow.workflow_id}")
            
            # Validate workflow
            validation_result = await self._validate_workflow(workflow)
            if not validation_result["valid"]:
                return ExecutionResult(
                    workflow_id=workflow.workflow_id,
                    status=ExecutionStatus.FAILED,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    total_duration=0.0,
                    node_results={},
                    execution_graph={},
                    performance_metrics={},
                    errors=[f"Workflow validation failed: {validation_result['error']}"],
                    warnings=[]
                )
            
            # Check resource constraints
            if len(self.active_workflows) >= self.max_concurrent_workflows:
                return ExecutionResult(
                    workflow_id=workflow.workflow_id,
                    status=ExecutionStatus.FAILED,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    total_duration=0.0,
                    node_results={},
                    execution_graph={},
                    performance_metrics={},
                    errors=["Maximum concurrent workflows exceeded"],
                    warnings=[]
                )
            
            # Add to active workflows
            self.active_workflows[workflow.workflow_id] = workflow
            
            start_time = datetime.utcnow()
            
            try:
                # Build execution graph
                execution_graph = await self._build_execution_graph(workflow)
                
                # Execute workflow
                node_results, errors, warnings = await self._execute_workflow_graph(
                    workflow, execution_graph
                )
                
                # Determine final status
                final_status = ExecutionStatus.COMPLETED
                if errors:
                    final_status = ExecutionStatus.FAILED
                elif any(node.status == ExecutionStatus.TIMEOUT for node in workflow.nodes):
                    final_status = ExecutionStatus.TIMEOUT
                
                end_time = datetime.utcnow()
                total_duration = (end_time - start_time).total_seconds()
                
                # Calculate performance metrics
                performance_metrics = await self._calculate_workflow_metrics(
                    workflow, node_results, total_duration
                )
                
                result = ExecutionResult(
                    workflow_id=workflow.workflow_id,
                    status=final_status,
                    start_time=start_time,
                    end_time=end_time,
                    total_duration=total_duration,
                    node_results=node_results,
                    execution_graph=execution_graph,
                    performance_metrics=performance_metrics,
                    errors=errors,
                    warnings=warnings
                )
                
                # Store result and update history
                self.workflow_results[workflow.workflow_id] = result
                self.execution_history.append({
                    "workflow_id": workflow.workflow_id,
                    "status": final_status.value,
                    "duration": total_duration,
                    "nodes_executed": len([n for n in workflow.nodes if n.status == ExecutionStatus.COMPLETED]),
                    "timestamp": start_time.isoformat()
                })
                
                logger.info(f"Workflow {workflow.workflow_id} completed with status {final_status.value} in {total_duration:.2f}s")
                
                return result
                
            finally:
                # Clean up active workflow
                if workflow.workflow_id in self.active_workflows:
                    del self.active_workflows[workflow.workflow_id]
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow.workflow_id}: {e}")
            return ExecutionResult(
                workflow_id=workflow.workflow_id,
                status=ExecutionStatus.FAILED,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                total_duration=0.0,
                node_results={},
                execution_graph={},
                performance_metrics={},
                errors=[f"Workflow execution failed: {str(e)}"],
                warnings=[]
            )
    
    async def _validate_workflow(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Validate workflow definition"""
        try:
            # Check for empty workflow
            if not workflow.nodes:
                return {"valid": False, "error": "Workflow has no nodes"}
            
            # Validate node IDs are unique
            node_ids = [node.node_id for node in workflow.nodes]
            if len(node_ids) != len(set(node_ids)):
                return {"valid": False, "error": "Duplicate node IDs found"}
            
            # Validate tool executors exist
            missing_tools = []
            for node in workflow.nodes:
                if node.tool_name not in self.tool_executors:
                    missing_tools.append(node.tool_name)
            
            if missing_tools:
                return {
                    "valid": False, 
                    "error": f"Missing tool executors: {', '.join(missing_tools)}"
                }
            
            # Validate dependencies reference existing nodes
            for dep in workflow.dependencies:
                if dep.from_node not in node_ids:
                    return {"valid": False, "error": f"Dependency references non-existent node: {dep.from_node}"}
                if dep.to_node not in node_ids:
                    return {"valid": False, "error": f"Dependency references non-existent node: {dep.to_node}"}
            
            # Check for circular dependencies
            if await self._has_circular_dependencies(workflow):
                return {"valid": False, "error": "Circular dependencies detected"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    async def _has_circular_dependencies(self, workflow: WorkflowDefinition) -> bool:
        """Check for circular dependencies in workflow"""
        try:
            # Build dependency graph
            G = nx.DiGraph()
            
            # Add nodes
            for node in workflow.nodes:
                G.add_node(node.node_id)
            
            # Add edges
            for dep in workflow.dependencies:
                G.add_edge(dep.from_node, dep.to_node)
            
            # Check for cycles
            return not nx.is_directed_acyclic_graph(G)
            
        except Exception:
            return True  # Assume circular if error
    
    async def _build_execution_graph(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Build execution graph with topological ordering"""
        try:
            # Create networkx graph
            G = nx.DiGraph()
            
            # Add nodes with metadata
            for node in workflow.nodes:
                G.add_node(node.node_id, 
                          tool_name=node.tool_name,
                          parameters=node.parameters,
                          priority=node.priority,
                          max_retries=node.max_retries,
                          timeout=node.timeout_seconds)
            
            # Add dependency edges
            for dep in workflow.dependencies:
                G.add_edge(dep.from_node, dep.to_node,
                          dependency_type=dep.dependency_type.value,
                          condition=dep.condition,
                          data_mapping=dep.data_mapping)
            
            # Calculate execution order
            execution_order = list(nx.topological_sort(G))
            
            # Identify parallel execution groups
            parallel_groups = await self._identify_parallel_groups(G, workflow.dependencies)
            
            return {
                "graph": nx.node_link_data(G),
                "execution_order": execution_order,
                "parallel_groups": parallel_groups,
                "total_nodes": len(workflow.nodes),
                "total_dependencies": len(workflow.dependencies)
            }
            
        except Exception as e:
            logger.error(f"Error building execution graph: {e}")
            raise
    
    async def _identify_parallel_groups(self, graph: nx.DiGraph, 
                                       dependencies: List[ExecutionDependency]) -> List[List[str]]:
        """Identify groups of nodes that can execute in parallel"""
        try:
            parallel_groups = []
            
            # Find nodes with no dependencies (can start immediately)
            roots = [node for node in graph.nodes() if graph.in_degree(node) == 0]
            if roots:
                parallel_groups.append(roots)
            
            # Group nodes by dependency level
            remaining_nodes = set(graph.nodes()) - set(roots)
            
            while remaining_nodes:
                # Find nodes whose dependencies are satisfied
                ready_nodes = []
                
                for node in list(remaining_nodes):
                    # Check if all dependencies are satisfied
                    dependencies_satisfied = True
                    for pred in graph.predecessors(node):
                        # Check if predecessor can run in parallel or is already processed
                        parallel_safe = any(
                            dep.dependency_type == DependencyType.PARALLEL_SAFE 
                            for dep in dependencies 
                            if dep.from_node == pred and dep.to_node == node
                        )
                        
                        if not parallel_safe and pred in remaining_nodes:
                            dependencies_satisfied = False
                            break
                    
                    if dependencies_satisfied:
                        ready_nodes.append(node)
                
                if ready_nodes:
                    parallel_groups.append(ready_nodes)
                    remaining_nodes -= set(ready_nodes)
                else:
                    # Break cycle - add remaining nodes as individual groups
                    for node in remaining_nodes:
                        parallel_groups.append([node])
                    break
            
            return parallel_groups
            
        except Exception as e:
            logger.error(f"Error identifying parallel groups: {e}")
            return [[node] for node in graph.nodes()]  # Fallback to sequential
    
    async def _execute_workflow_graph(self, workflow: WorkflowDefinition, 
                                    execution_graph: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
        """Execute workflow graph with parallel processing"""
        try:
            node_results = {}
            errors = []
            warnings = []
            
            # Create node lookup
            node_lookup = {node.node_id: node for node in workflow.nodes}
            
            # Execute parallel groups
            parallel_groups = execution_graph["parallel_groups"]
            
            for group_index, group in enumerate(parallel_groups):
                logger.debug(f"Executing parallel group {group_index + 1}/{len(parallel_groups)}: {group}")
                
                # Create tasks for parallel execution
                group_tasks = []
                
                for node_id in group:
                    if node_id in node_lookup:
                        node = node_lookup[node_id]
                        
                        # Prepare node parameters with dependency data
                        prepared_params = await self._prepare_node_parameters(
                            node, workflow.dependencies, node_results
                        )
                        
                        # Create execution task
                        task = asyncio.create_task(
                            self._execute_single_node(node, prepared_params),
                            name=f"execute_{node_id}"
                        )
                        group_tasks.append((node_id, node, task))
                
                # Wait for group completion with timeout
                group_timeout = max(node.timeout_seconds for _, node, _ in group_tasks)
                
                try:
                    # Wait for all tasks in group
                    for node_id, node, task in group_tasks:
                        try:
                            result = await asyncio.wait_for(task, timeout=group_timeout)
                            node_results[node_id] = result
                            
                            if result["status"] == "success":
                                node.status = ExecutionStatus.COMPLETED
                                node.result = result
                            else:
                                node.status = ExecutionStatus.FAILED
                                node.error = result.get("error", "Unknown error")
                                errors.append(f"Node {node_id} failed: {node.error}")
                            
                        except asyncio.TimeoutError:
                            node.status = ExecutionStatus.TIMEOUT
                            node.error = f"Execution timeout after {group_timeout}s"
                            errors.append(f"Node {node_id} timeout: {node.error}")
                            
                            # Cancel the task
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                        
                        except Exception as e:
                            node.status = ExecutionStatus.FAILED
                            node.error = str(e)
                            errors.append(f"Node {node_id} error: {node.error}")
                
                except Exception as e:
                    errors.append(f"Group execution error: {str(e)}")
                
                # Check failure strategy
                if errors and workflow.failure_strategy == "fail_fast":
                    break
            
            return node_results, errors, warnings
            
        except Exception as e:
            logger.error(f"Error executing workflow graph: {e}")
            return {}, [f"Workflow graph execution failed: {str(e)}"], []
    
    async def _prepare_node_parameters(self, node: ExecutionNode, 
                                     dependencies: List[ExecutionDependency],
                                     node_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare node parameters with dependency data mapping"""
        try:
            prepared_params = node.parameters.copy()
            
            # Find dependencies that provide data to this node
            for dep in dependencies:
                if dep.to_node == node.node_id and dep.data_mapping:
                    from_result = node_results.get(dep.from_node)
                    
                    if from_result and from_result.get("status") == "success":
                        result_data = from_result.get("data", {})
                        
                        # Map data according to data_mapping
                        for source_key, target_key in dep.data_mapping.items():
                            if source_key in result_data:
                                prepared_params[target_key] = result_data[source_key]
            
            return prepared_params
            
        except Exception as e:
            logger.error(f"Error preparing node parameters: {e}")
            return node.parameters
    
    async def _execute_single_node(self, node: ExecutionNode, 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single node with retry logic"""
        try:
            node.start_time = datetime.utcnow()
            node.status = ExecutionStatus.RUNNING
            
            logger.debug(f"Executing node {node.node_id} with tool {node.tool_name}")
            
            # Get tool executor
            if node.tool_name not in self.tool_executors:
                raise ValueError(f"No executor found for tool: {node.tool_name}")
            
            executor = self.tool_executors[node.tool_name]
            
            # Execute with retries
            last_error = None
            
            for attempt in range(node.max_retries + 1):
                try:
                    # Execute tool
                    result = await asyncio.wait_for(
                        executor(parameters),
                        timeout=node.timeout_seconds
                    )
                    
                    # Success
                    node.end_time = datetime.utcnow()
                    node.status = ExecutionStatus.COMPLETED
                    
                    return {
                        "status": "success",
                        "data": result,
                        "duration": node.duration(),
                        "attempt": attempt + 1,
                        "node_id": node.node_id,
                        "tool_name": node.tool_name
                    }
                
                except asyncio.TimeoutError as e:
                    last_error = f"Timeout after {node.timeout_seconds}s"
                    node.retry_count = attempt + 1
                    
                    if attempt < node.max_retries:
                        await asyncio.sleep(min(2 ** attempt, 10))  # Exponential backoff
                    
                except Exception as e:
                    last_error = str(e)
                    node.retry_count = attempt + 1
                    
                    if attempt < node.max_retries:
                        await asyncio.sleep(min(2 ** attempt, 10))  # Exponential backoff
            
            # All retries exhausted
            node.end_time = datetime.utcnow()
            node.status = ExecutionStatus.FAILED
            node.error = last_error
            
            return {
                "status": "failed",
                "error": last_error,
                "duration": node.duration(),
                "attempts": node.retry_count,
                "node_id": node.node_id,
                "tool_name": node.tool_name
            }
            
        except Exception as e:
            node.end_time = datetime.utcnow()
            node.status = ExecutionStatus.FAILED
            node.error = str(e)
            
            return {
                "status": "failed",
                "error": str(e),
                "duration": node.duration(),
                "attempts": node.retry_count,
                "node_id": node.node_id,
                "tool_name": node.tool_name
            }
    
    async def _calculate_workflow_metrics(self, workflow: WorkflowDefinition,
                                        node_results: Dict[str, Dict[str, Any]],
                                        total_duration: float) -> Dict[str, Any]:
        """Calculate comprehensive workflow performance metrics"""
        try:
            metrics = {
                "total_duration": total_duration,
                "total_nodes": len(workflow.nodes),
                "completed_nodes": 0,
                "failed_nodes": 0,
                "timeout_nodes": 0,
                "average_node_duration": 0.0,
                "parallel_efficiency": 0.0,
                "success_rate": 0.0,
                "retry_rate": 0.0,
                "resource_utilization": {}
            }
            
            node_durations = []
            total_retries = 0
            
            for node in workflow.nodes:
                if node.status == ExecutionStatus.COMPLETED:
                    metrics["completed_nodes"] += 1
                elif node.status == ExecutionStatus.FAILED:
                    metrics["failed_nodes"] += 1
                elif node.status == ExecutionStatus.TIMEOUT:
                    metrics["timeout_nodes"] += 1
                
                if node.duration() > 0:
                    node_durations.append(node.duration())
                
                total_retries += node.retry_count
            
            # Calculate derived metrics
            if node_durations:
                metrics["average_node_duration"] = sum(node_durations) / len(node_durations)
                
                # Parallel efficiency = (sum of individual durations) / (total workflow duration)
                if total_duration > 0:
                    sequential_duration = sum(node_durations)
                    metrics["parallel_efficiency"] = min(1.0, sequential_duration / total_duration)
            
            if metrics["total_nodes"] > 0:
                metrics["success_rate"] = metrics["completed_nodes"] / metrics["total_nodes"]
                metrics["retry_rate"] = total_retries / metrics["total_nodes"]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating workflow metrics: {e}")
            return {"error": str(e)}
    
    async def cancel_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Cancel an active workflow"""
        try:
            if workflow_id not in self.active_workflows:
                return {"success": False, "error": "Workflow not found or not active"}
            
            workflow = self.active_workflows[workflow_id]
            
            # Mark all pending/running nodes as cancelled
            cancelled_count = 0
            for node in workflow.nodes:
                if node.status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
                    node.status = ExecutionStatus.CANCELLED
                    node.end_time = datetime.utcnow()
                    cancelled_count += 1
            
            logger.info(f"Cancelled workflow {workflow_id}: {cancelled_count} nodes cancelled")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "cancelled_nodes": cancelled_count
            }
            
        except Exception as e:
            logger.error(f"Error cancelling workflow {workflow_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""
        try:
            # Check if workflow is active
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                
                node_statuses = {}
                for node in workflow.nodes:
                    node_statuses[node.node_id] = {
                        "status": node.status.value,
                        "tool_name": node.tool_name,
                        "start_time": node.start_time.isoformat() if node.start_time else None,
                        "end_time": node.end_time.isoformat() if node.end_time else None,
                        "duration": node.duration(),
                        "retry_count": node.retry_count,
                        "error": node.error
                    }
                
                return {
                    "workflow_id": workflow_id,
                    "status": "active",
                    "description": workflow.description,
                    "node_statuses": node_statuses,
                    "total_nodes": len(workflow.nodes),
                    "completed_nodes": len([n for n in workflow.nodes if n.status == ExecutionStatus.COMPLETED]),
                    "failed_nodes": len([n for n in workflow.nodes if n.status == ExecutionStatus.FAILED])
                }
            
            # Check if workflow is completed
            elif workflow_id in self.workflow_results:
                result = self.workflow_results[workflow_id]
                return {
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "final_status": result.status.value,
                    "start_time": result.start_time.isoformat(),
                    "end_time": result.end_time.isoformat() if result.end_time else None,
                    "total_duration": result.total_duration,
                    "performance_metrics": result.performance_metrics,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
            
            else:
                return {"error": "Workflow not found"}
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return {"error": str(e)}
    
    async def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator statistics"""
        try:
            # Calculate statistics from execution history
            if self.execution_history:
                total_workflows = len(self.execution_history)
                successful_workflows = len([w for w in self.execution_history if w["status"] == "completed"])
                failed_workflows = len([w for w in self.execution_history if w["status"] == "failed"])
                
                durations = [w["duration"] for w in self.execution_history if w.get("duration")]
                avg_duration = sum(durations) / len(durations) if durations else 0.0
                
                success_rate = successful_workflows / total_workflows if total_workflows > 0 else 0.0
            else:
                total_workflows = successful_workflows = failed_workflows = 0
                avg_duration = success_rate = 0.0
            
            return {
                "active_workflows": len(self.active_workflows),
                "registered_tools": len(self.tool_executors),
                "total_workflows_executed": total_workflows,
                "successful_workflows": successful_workflows,
                "failed_workflows": failed_workflows,
                "success_rate": success_rate,
                "average_workflow_duration": avg_duration,
                "resource_limits": dict(self.resource_limits),
                "current_resource_usage": dict(self.resource_usage)
            }
            
        except Exception as e:
            logger.error(f"Error getting orchestrator statistics: {e}")
            return {"error": str(e)}
