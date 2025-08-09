
"""
Multi-Agent Coordination Manager
================================

Advanced coordination strategies for multi-agent systems with:
- Load balancing and resource allocation
- Dependency-aware task scheduling
- Network topology optimization
- Real-time coordination adjustments

Mathematical foundation:
- Balance = min(Σ |Load_i - Load_avg|)
- Plan* = arg max_P Utility(P, Agents)
- Efficiency = Completed_Tasks / Total_Resources
"""

import logging
import time
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Import models
from models import Task, Agent, Workflow
from database import get_db

logger = logging.getLogger(__name__)

class CoordinationStrategy(Enum):
    """Coordination strategy types"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    ADAPTIVE = "adaptive"

@dataclass
class AgentLoad:
    """Agent load information"""
    agent_id: str
    current_tasks: int
    cpu_usage: float
    memory_usage: float
    response_time: float
    capacity: float
    availability: float

@dataclass
class CoordinationPlan:
    """Coordination execution plan"""
    strategy: CoordinationStrategy
    agents: List[str]
    task_assignments: Dict[str, List[int]]
    dependencies: List[Tuple[str, str]]
    estimated_completion: datetime
    load_balance_score: float
    efficiency_score: float

class CoordinationManager:
    """
    Advanced coordination manager for multi-agent systems
    """
    
    def __init__(self):
        self.agent_loads = {}
        self.coordination_history = []
        self.network_topology = nx.DiGraph()
        self.load_threshold = 0.8
        self.balance_weight = 0.4
        self.efficiency_weight = 0.6
        
        # Performance tracking
        self.coordination_metrics = {
            "plans_created": 0,
            "successful_coordinations": 0,
            "load_balancing_events": 0,
            "topology_optimizations": 0
        }
        
        logger.info("Coordination Manager initialized")
    
    async def coordinate_agents(
        self,
        db: Session,
        task_id: int,
        user_id: int,
        agents: List[str],
        strategy: Optional[CoordinationStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents for task execution
        
        Args:
            db: Database session
            task_id: Task identifier
            user_id: User identifier
            agents: List of agent identifiers
            strategy: Coordination strategy (auto-selected if None)
            context: Optional coordination context
            
        Returns:
            Dict containing coordination plan and execution data
        """
        start_time = time.time()
        
        try:
            # Retrieve task
            db_task = db.query(Task).filter(
                and_(Task.id == task_id, Task.owner_id == user_id)
            ).first()
            
            if not db_task:
                raise ValueError(f"Task {task_id} not found")
            
            # Update agent load information
            await self._update_agent_loads(db, agents)
            
            # Select coordination strategy if not provided
            if strategy is None:
                strategy = self._select_coordination_strategy(agents, db_task, context)
            
            # Create coordination plan
            coordination_plan = await self._create_coordination_plan(
                db, db_task, agents, strategy, context
            )
            
            # Optimize network topology
            topology_updates = await self._optimize_network_topology(
                agents, coordination_plan
            )
            
            # Execute coordination
            execution_result = await self._execute_coordination(
                db, coordination_plan, topology_updates
            )
            
            # Calculate performance metrics
            balance_score = self._calculate_load_balance_score(agents)
            efficiency_score = execution_result.get("efficiency", 0.0)
            
            # Store coordination data
            coordination_data = {
                "strategy": strategy.value,
                "agents": agents,
                "plan": asdict(coordination_plan),
                "topology_updates": topology_updates,
                "execution_result": execution_result,
                "balance_score": balance_score,
                "efficiency_score": efficiency_score,
                "coordination_time": time.time() - start_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update task with coordination data
            db_task.coordination = coordination_data
            db.commit()
            
            # Update coordination history
            self.coordination_history.append({
                "task_id": task_id,
                "strategy": strategy.value,
                "num_agents": len(agents),
                "balance_score": balance_score,
                "efficiency_score": efficiency_score,
                "success": execution_result.get("success", False),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Update metrics
            self.coordination_metrics["plans_created"] += 1
            if execution_result.get("success", False):
                self.coordination_metrics["successful_coordinations"] += 1
            
            logger.info(
                f"Agent coordination completed for task {task_id}: "
                f"strategy={strategy.value}, balance={balance_score:.3f}, "
                f"efficiency={efficiency_score:.3f}"
            )
            
            return coordination_data
            
        except Exception as e:
            logger.error(f"Failed agent coordination for task {task_id}: {e}")
            raise
    
    async def _update_agent_loads(self, db: Session, agents: List[str]):
        """Update current load information for agents"""
        
        for agent in agents:
            try:
                # Query current tasks for agent (simulation)
                current_tasks = len([
                    t for t in self.coordination_history 
                    if agent in t.get("agents", []) and t.get("success", False)
                ][-5:])  # Last 5 successful coordinations
                
                # Simulate system metrics
                cpu_usage = np.random.uniform(0.2, 0.9)
                memory_usage = np.random.uniform(0.3, 0.8)
                response_time = np.random.uniform(0.1, 2.0)
                
                # Calculate capacity and availability
                capacity = max(0.1, 1.0 - (cpu_usage * 0.6 + memory_usage * 0.4))
                availability = max(0.1, 1.0 - (response_time / 2.0))
                
                self.agent_loads[agent] = AgentLoad(
                    agent_id=agent,
                    current_tasks=current_tasks,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    response_time=response_time,
                    capacity=capacity,
                    availability=availability
                )
                
            except Exception as e:
                logger.warning(f"Failed to update load for agent {agent}: {e}")
                # Default load if update fails
                self.agent_loads[agent] = AgentLoad(
                    agent_id=agent,
                    current_tasks=0,
                    cpu_usage=0.5,
                    memory_usage=0.5,
                    response_time=1.0,
                    capacity=0.7,
                    availability=0.8
                )
    
    def _select_coordination_strategy(
        self,
        agents: List[str],
        task: Task,
        context: Optional[Dict[str, Any]] = None
    ) -> CoordinationStrategy:
        """Automatically select optimal coordination strategy"""
        
        num_agents = len(agents)
        task_complexity = self._estimate_task_complexity(task)
        
        # Get average agent load
        if agents and all(agent in self.agent_loads for agent in agents):
            avg_load = np.mean([
                self.agent_loads[agent].cpu_usage + self.agent_loads[agent].memory_usage
                for agent in agents
            ]) / 2.0
        else:
            avg_load = 0.5  # Default
        
        # Strategy selection logic
        if num_agents <= 2:
            return CoordinationStrategy.SEQUENTIAL
        elif num_agents <= 4 and task_complexity < 0.6:
            return CoordinationStrategy.PARALLEL
        elif avg_load > 0.7:
            return CoordinationStrategy.HIERARCHICAL
        elif task_complexity > 0.8:
            return CoordinationStrategy.ADAPTIVE
        else:
            return CoordinationStrategy.MESH
    
    def _estimate_task_complexity(self, task: Task) -> float:
        """Estimate task complexity based on description and requirements"""
        
        # Simple complexity estimation
        content = f"{task.title} {task.description or ''}"
        word_count = len(content.split())
        
        # Complexity factors
        length_factor = min(word_count / 100.0, 1.0)
        
        # Check for complexity indicators
        complexity_keywords = [
            "complex", "advanced", "multiple", "integration", "optimization",
            "analysis", "machine learning", "ai", "algorithm", "distributed"
        ]
        
        keyword_factor = sum(1 for keyword in complexity_keywords if keyword in content.lower())
        keyword_factor = min(keyword_factor / len(complexity_keywords), 1.0)
        
        # Combined complexity score
        complexity = (0.6 * length_factor + 0.4 * keyword_factor)
        
        return min(complexity, 1.0)
    
    async def _create_coordination_plan(
        self,
        db: Session,
        task: Task,
        agents: List[str],
        strategy: CoordinationStrategy,
        context: Optional[Dict[str, Any]] = None
    ) -> CoordinationPlan:
        """Create detailed coordination plan"""
        
        # Create network graph for coordination
        g = nx.DiGraph()
        
        # Add agents as nodes with load information
        for agent in agents:
            load_info = self.agent_loads.get(agent)
            if load_info:
                g.add_node(agent, 
                          load=load_info.cpu_usage + load_info.memory_usage,
                          capacity=load_info.capacity,
                          availability=load_info.availability)
            else:
                g.add_node(agent, load=0.5, capacity=0.7, availability=0.8)
        
        # Create edges based on coordination strategy
        dependencies = []
        
        if strategy == CoordinationStrategy.SEQUENTIAL:
            # Chain agents sequentially
            for i in range(len(agents) - 1):
                g.add_edge(agents[i], agents[i + 1])
                dependencies.append((agents[i], agents[i + 1]))
                
        elif strategy == CoordinationStrategy.PARALLEL:
            # All agents work in parallel (no dependencies)
            # Star topology with optional coordinator
            if len(agents) > 3:
                coordinator = min(agents, 
                    key=lambda a: self.agent_loads.get(a, AgentLoad("", 0, 0.5, 0.5, 1.0, 0.7, 0.8)).cpu_usage
                )
                for agent in agents:
                    if agent != coordinator:
                        g.add_edge(coordinator, agent)
                        dependencies.append((coordinator, agent))
                        
        elif strategy == CoordinationStrategy.HIERARCHICAL:
            # Create hierarchy based on agent capacity
            sorted_agents = sorted(agents, 
                key=lambda a: self.agent_loads.get(a, AgentLoad("", 0, 0.5, 0.5, 1.0, 0.7, 0.8)).capacity,
                reverse=True
            )
            
            # Top agent coordinates others
            coordinator = sorted_agents[0]
            for agent in sorted_agents[1:]:
                g.add_edge(coordinator, agent)
                dependencies.append((coordinator, agent))
                
        elif strategy == CoordinationStrategy.MESH:
            # Full mesh - all agents communicate with all others
            for i, agent_i in enumerate(agents):
                for j, agent_j in enumerate(agents):
                    if i != j:
                        g.add_edge(agent_i, agent_j)
                        dependencies.append((agent_i, agent_j))
                        
        else:  # ADAPTIVE
            # Dynamic strategy based on current conditions
            adaptive_strategy = self._create_adaptive_topology(agents, g)
            dependencies = [(u, v) for u, v in adaptive_strategy.edges()]
            g = adaptive_strategy
        
        # Assign tasks to agents based on load balancing
        task_assignments = await self._assign_tasks_to_agents(agents, task, strategy)
        
        # Calculate load balance score
        loads = [g.nodes[agent]["load"] for agent in agents]
        avg_load = np.mean(loads)
        balance_score = max(0.0, 1.0 - (np.std(loads) / avg_load if avg_load > 0 else 1.0))
        
        # Estimate completion time
        max_load = max(loads) if loads else 0.5
        estimated_duration = timedelta(minutes=max_load * 30)  # Rough estimation
        estimated_completion = datetime.utcnow() + estimated_duration
        
        # Calculate efficiency score
        total_capacity = sum(g.nodes[agent]["capacity"] for agent in agents)
        efficiency_score = min(1.0, len(agents) / total_capacity if total_capacity > 0 else 0.5)
        
        return CoordinationPlan(
            strategy=strategy,
            agents=agents,
            task_assignments=task_assignments,
            dependencies=dependencies,
            estimated_completion=estimated_completion,
            load_balance_score=balance_score,
            efficiency_score=efficiency_score
        )
    
    def _create_adaptive_topology(
        self,
        agents: List[str],
        base_graph: nx.DiGraph
    ) -> nx.DiGraph:
        """Create adaptive network topology based on current conditions"""
        
        g = base_graph.copy()
        
        # Sort agents by availability and capacity
        agent_scores = {}
        for agent in agents:
            load_info = self.agent_loads.get(agent)
            if load_info:
                score = (load_info.availability * 0.6 + load_info.capacity * 0.4)
                agent_scores[agent] = score
            else:
                agent_scores[agent] = 0.7
        
        sorted_agents = sorted(agents, key=lambda a: agent_scores[a], reverse=True)
        
        # Create adaptive connections
        for i, agent in enumerate(sorted_agents):
            # Connect to next 1-2 agents in the sorted list
            connections = min(2, len(sorted_agents) - i - 1)
            for j in range(1, connections + 1):
                if i + j < len(sorted_agents):
                    target_agent = sorted_agents[i + j]
                    g.add_edge(agent, target_agent)
        
        return g
    
    async def _assign_tasks_to_agents(
        self,
        agents: List[str],
        task: Task,
        strategy: CoordinationStrategy
    ) -> Dict[str, List[int]]:
        """Assign task components to agents based on load balancing"""
        
        # For simulation, create subtasks
        task_complexity = self._estimate_task_complexity(task)
        num_subtasks = max(1, int(task_complexity * len(agents) * 2))
        
        subtask_ids = list(range(1, num_subtasks + 1))
        assignments = {agent: [] for agent in agents}
        
        if strategy == CoordinationStrategy.SEQUENTIAL:
            # Distribute subtasks sequentially
            for i, subtask_id in enumerate(subtask_ids):
                agent = agents[i % len(agents)]
                assignments[agent].append(subtask_id)
                
        elif strategy == CoordinationStrategy.PARALLEL:
            # Balance subtasks across agents by load
            agent_loads = [(agent, self.agent_loads.get(agent, AgentLoad("", 0, 0.5, 0.5, 1.0, 0.7, 0.8)).cpu_usage + 
                           self.agent_loads.get(agent, AgentLoad("", 0, 0.5, 0.5, 1.0, 0.7, 0.8)).memory_usage) 
                          for agent in agents]
            agent_loads.sort(key=lambda x: x[1])  # Sort by load (ascending)
            
            for i, subtask_id in enumerate(subtask_ids):
                agent = agent_loads[i % len(agent_loads)][0]
                assignments[agent].append(subtask_id)
                
        else:
            # Default balanced assignment
            for i, subtask_id in enumerate(subtask_ids):
                agent = agents[i % len(agents)]
                assignments[agent].append(subtask_id)
        
        return assignments
    
    async def _optimize_network_topology(
        self,
        agents: List[str],
        plan: CoordinationPlan
    ) -> Dict[str, Any]:
        """Optimize network topology for better performance"""
        
        optimization_start = time.time()
        
        # Create current topology graph
        g = nx.DiGraph()
        for agent in agents:
            load_info = self.agent_loads.get(agent)
            if load_info:
                g.add_node(agent, 
                          load=load_info.cpu_usage + load_info.memory_usage,
                          capacity=load_info.capacity)
        
        for dep in plan.dependencies:
            g.add_edge(dep[0], dep[1])
        
        # Optimization metrics
        original_diameter = nx.diameter(g.to_undirected()) if nx.is_connected(g.to_undirected()) else len(agents)
        original_clustering = nx.average_clustering(g.to_undirected())
        
        # Apply topology optimizations
        optimizations_applied = []
        
        # 1. Remove redundant connections
        redundant_edges = self._find_redundant_edges(g)
        if redundant_edges:
            g.remove_edges_from(redundant_edges[:2])  # Remove up to 2 redundant edges
            optimizations_applied.append(f"Removed {len(redundant_edges[:2])} redundant connections")
        
        # 2. Add strategic connections to improve clustering
        strategic_edges = self._find_strategic_edges(g, agents)
        if strategic_edges:
            g.add_edges_from(strategic_edges[:2])  # Add up to 2 strategic edges
            optimizations_applied.append(f"Added {len(strategic_edges[:2])} strategic connections")
        
        # 3. Load balancing edge weights
        for edge in g.edges():
            source_load = g.nodes[edge[0]]["load"]
            target_load = g.nodes[edge[1]]["load"]
            weight = 1.0 / (1.0 + abs(source_load - target_load))  # Lower weight for unbalanced connections
            g[edge[0]][edge[1]]["weight"] = weight
        
        # Calculate optimization results
        new_diameter = nx.diameter(g.to_undirected()) if nx.is_connected(g.to_undirected()) else len(agents)
        new_clustering = nx.average_clustering(g.to_undirected())
        
        optimization_results = {
            "optimizations_applied": optimizations_applied,
            "topology_improvements": {
                "diameter_reduction": original_diameter - new_diameter,
                "clustering_improvement": new_clustering - original_clustering,
                "optimization_time": time.time() - optimization_start
            },
            "updated_dependencies": [(u, v) for u, v in g.edges()],
            "edge_weights": {f"{u}-{v}": g[u][v].get("weight", 1.0) for u, v in g.edges()}
        }
        
        # Update metrics
        if optimizations_applied:
            self.coordination_metrics["topology_optimizations"] += 1
        
        logger.info(f"Network topology optimization completed: {len(optimizations_applied)} improvements")
        
        return optimization_results
    
    def _find_redundant_edges(self, graph: nx.DiGraph) -> List[Tuple[str, str]]:
        """Find edges that can be removed without affecting connectivity"""
        
        redundant_edges = []
        
        for edge in list(graph.edges()):
            # Temporarily remove edge
            graph.remove_edge(*edge)
            
            # Check if graph is still connected
            if nx.is_weakly_connected(graph):
                redundant_edges.append(edge)
            else:
                # Add edge back if removal breaks connectivity
                graph.add_edge(*edge)
        
        return redundant_edges
    
    def _find_strategic_edges(
        self,
        graph: nx.DiGraph,
        agents: List[str]
    ) -> List[Tuple[str, str]]:
        """Find strategic edges to add for improved topology"""
        
        strategic_edges = []
        
        # Find pairs of nodes with similar load that aren't connected
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i+1:]:
                if not graph.has_edge(agent1, agent2) and not graph.has_edge(agent2, agent1):
                    load1 = graph.nodes[agent1]["load"]
                    load2 = graph.nodes[agent2]["load"]
                    
                    # Add edge if loads are similar (good for load balancing)
                    if abs(load1 - load2) < 0.3:
                        strategic_edges.append((agent1, agent2))
        
        # Sort by load similarity
        strategic_edges.sort(
            key=lambda edge: abs(graph.nodes[edge[0]]["load"] - graph.nodes[edge[1]]["load"])
        )
        
        return strategic_edges
    
    async def _execute_coordination(
        self,
        db: Session,
        plan: CoordinationPlan,
        topology_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the coordination plan"""
        
        execution_start = time.time()
        
        # Simulate coordination execution
        try:
            # Execute task assignments in coordination order
            execution_results = []
            
            if plan.strategy == CoordinationStrategy.SEQUENTIAL:
                # Sequential execution
                for agent in plan.agents:
                    subtasks = plan.task_assignments.get(agent, [])
                    if subtasks:
                        result = await self._execute_agent_subtasks(agent, subtasks)
                        execution_results.append(result)
                        
            else:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=min(len(plan.agents), 4)) as executor:
                    futures = []
                    for agent in plan.agents:
                        subtasks = plan.task_assignments.get(agent, [])
                        if subtasks:
                            future = executor.submit(
                                asyncio.run, 
                                self._execute_agent_subtasks(agent, subtasks)
                            )
                            futures.append((agent, future))
                    
                    # Collect results
                    for agent, future in futures:
                        try:
                            result = future.result(timeout=30)
                            execution_results.append(result)
                        except Exception as e:
                            logger.error(f"Agent {agent} execution failed: {e}")
                            execution_results.append({
                                "agent": agent,
                                "success": False,
                                "error": str(e)
                            })
            
            # Calculate execution metrics
            successful_executions = sum(1 for r in execution_results if r.get("success", False))
            total_executions = len(execution_results)
            success_rate = successful_executions / total_executions if total_executions > 0 else 0.0
            
            # Calculate efficiency
            total_subtasks = sum(len(subtasks) for subtasks in plan.task_assignments.values())
            completed_subtasks = sum(r.get("completed_subtasks", 0) for r in execution_results)
            efficiency = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
            
            return {
                "success": success_rate > 0.5,
                "execution_results": execution_results,
                "success_rate": success_rate,
                "efficiency": efficiency,
                "total_subtasks": total_subtasks,
                "completed_subtasks": completed_subtasks,
                "execution_time": time.time() - execution_start
            }
            
        except Exception as e:
            logger.error(f"Coordination execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - execution_start
            }
    
    async def _execute_agent_subtasks(
        self,
        agent: str,
        subtask_ids: List[int]
    ) -> Dict[str, Any]:
        """Execute subtasks for a specific agent"""
        
        try:
            # Simulate subtask execution
            completed_subtasks = 0
            execution_details = []
            
            for subtask_id in subtask_ids:
                # Simulate execution time based on agent load
                agent_load = self.agent_loads.get(agent)
                if agent_load:
                    execution_time = agent_load.response_time + np.random.uniform(0.1, 0.5)
                else:
                    execution_time = np.random.uniform(0.5, 2.0)
                
                await asyncio.sleep(min(execution_time, 0.1))  # Don't actually sleep too long
                
                # Simulate success based on agent availability
                success_probability = agent_load.availability if agent_load else 0.8
                success = np.random.random() < success_probability
                
                if success:
                    completed_subtasks += 1
                
                execution_details.append({
                    "subtask_id": subtask_id,
                    "success": success,
                    "execution_time": execution_time
                })
            
            return {
                "agent": agent,
                "success": completed_subtasks > 0,
                "completed_subtasks": completed_subtasks,
                "total_subtasks": len(subtask_ids),
                "execution_details": execution_details
            }
            
        except Exception as e:
            logger.error(f"Agent {agent} subtask execution failed: {e}")
            return {
                "agent": agent,
                "success": False,
                "error": str(e),
                "completed_subtasks": 0,
                "total_subtasks": len(subtask_ids)
            }
    
    def _calculate_load_balance_score(self, agents: List[str]) -> float:
        """Calculate load balance score across agents"""
        
        if not agents:
            return 0.0
        
        loads = []
        for agent in agents:
            load_info = self.agent_loads.get(agent)
            if load_info:
                load = load_info.cpu_usage + load_info.memory_usage
                loads.append(load)
            else:
                loads.append(1.0)  # Default high load
        
        if not loads:
            return 0.0
        
        # Calculate balance score: 1 - normalized standard deviation
        avg_load = np.mean(loads)
        if avg_load == 0:
            return 1.0
        
        load_std = np.std(loads)
        balance_score = max(0.0, 1.0 - (load_std / avg_load))
        
        # Update load balancing metrics
        if balance_score < self.load_threshold:
            self.coordination_metrics["load_balancing_events"] += 1
        
        return balance_score
    
    async def rebalance_agents(
        self,
        db: Session,
        agents: List[str],
        target_balance: float = 0.8
    ) -> Dict[str, Any]:
        """Rebalance agent loads to achieve target balance score"""
        
        rebalancing_start = time.time()
        
        try:
            # Update current agent loads
            await self._update_agent_loads(db, agents)
            
            # Calculate current balance
            current_balance = self._calculate_load_balance_score(agents)
            
            if current_balance >= target_balance:
                return {
                    "rebalancing_needed": False,
                    "current_balance": current_balance,
                    "target_balance": target_balance,
                    "message": "System already balanced"
                }
            
            # Identify overloaded and underloaded agents
            loads = []
            for agent in agents:
                load_info = self.agent_loads.get(agent)
                if load_info:
                    load = load_info.cpu_usage + load_info.memory_usage
                    loads.append((agent, load))
                else:
                    loads.append((agent, 1.0))
            
            loads.sort(key=lambda x: x[1], reverse=True)  # Sort by load (descending)
            
            avg_load = np.mean([load for _, load in loads])
            overloaded = [agent for agent, load in loads if load > avg_load * 1.2]
            underloaded = [agent for agent, load in loads if load < avg_load * 0.8]
            
            # Create rebalancing plan
            rebalancing_actions = []
            
            for overloaded_agent in overloaded:
                if underloaded:
                    target_agent = underloaded.pop(0)
                    rebalancing_actions.append({
                        "action": "move_tasks",
                        "from_agent": overloaded_agent,
                        "to_agent": target_agent,
                        "task_count": 1  # Move 1 task for simplicity
                    })
            
            # Execute rebalancing (simulation)
            for action in rebalancing_actions:
                # Update simulated loads
                from_agent = action["from_agent"]
                to_agent = action["to_agent"]
                
                if from_agent in self.agent_loads:
                    self.agent_loads[from_agent].current_tasks -= 1
                    self.agent_loads[from_agent].cpu_usage *= 0.9
                
                if to_agent in self.agent_loads:
                    self.agent_loads[to_agent].current_tasks += 1
                    self.agent_loads[to_agent].cpu_usage *= 1.1
            
            # Recalculate balance score
            new_balance = self._calculate_load_balance_score(agents)
            
            return {
                "rebalancing_needed": True,
                "current_balance": current_balance,
                "new_balance": new_balance,
                "target_balance": target_balance,
                "improvement": new_balance - current_balance,
                "actions_taken": rebalancing_actions,
                "overloaded_agents": overloaded,
                "rebalancing_time": time.time() - rebalancing_start
            }
            
        except Exception as e:
            logger.error(f"Agent rebalancing failed: {e}")
            return {
                "rebalancing_needed": True,
                "error": str(e),
                "rebalancing_time": time.time() - rebalancing_start
            }
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coordination statistics"""
        
        if not self.coordination_history:
            return {
                "message": "No coordination history available",
                "metrics": self.coordination_metrics
            }
        
        # Aggregate statistics
        total_coordinations = len(self.coordination_history)
        successful_coordinations = sum(1 for c in self.coordination_history if c["success"])
        success_rate = successful_coordinations / total_coordinations
        
        avg_balance = np.mean([c["balance_score"] for c in self.coordination_history])
        avg_efficiency = np.mean([c["efficiency_score"] for c in self.coordination_history])
        avg_agents = np.mean([c["num_agents"] for c in self.coordination_history])
        
        # Strategy usage
        strategies = [c["strategy"] for c in self.coordination_history]
        strategy_counts = {s: strategies.count(s) for s in set(strategies)}
        
        # Agent performance
        agent_loads_summary = {
            agent: {
                "current_tasks": load.current_tasks,
                "cpu_usage": load.cpu_usage,
                "memory_usage": load.memory_usage,
                "capacity": load.capacity,
                "availability": load.availability
            }
            for agent, load in self.agent_loads.items()
        }
        
        return {
            "total_coordinations": total_coordinations,
            "success_rate": success_rate,
            "average_balance_score": avg_balance,
            "average_efficiency_score": avg_efficiency,
            "average_agents_per_coordination": avg_agents,
            "strategy_usage": strategy_counts,
            "current_agent_loads": agent_loads_summary,
            "coordination_metrics": self.coordination_metrics,
            "load_threshold": self.load_threshold
        }
