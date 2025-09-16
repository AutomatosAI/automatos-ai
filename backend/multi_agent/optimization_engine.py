
"""
Multi-Agent Optimization Engine
===============================

Advanced optimization engine for multi-agent systems with:
- Multi-objective optimization (performance, scalability, robustness)
- Dynamic resource allocation
- Adaptive algorithm selection
- Performance prediction and optimization

Mathematical foundation:
- O* = arg max_O [Performance(O), Scalability(O), Robustness(O)]
- Resource allocation: R* = arg min_R Cost(R) s.t. Performance(R) ≥ threshold
- Adaptive weights: W_t+1 = W_t + α * ∇_W Objective(W_t)
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from scipy.optimize import minimize, differential_evolution, basinhopping
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Optimization strategy types"""
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    SIMULATED_ANNEALING = "simulated_annealing"
    MULTI_OBJECTIVE = "multi_objective"
    ADAPTIVE = "adaptive"

class OptimizationObjective(Enum):
    """Optimization objectives"""
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"  
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    COST = "cost"
    LATENCY = "latency"

@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    strategy: OptimizationStrategy
    objectives: List[OptimizationObjective]
    weights: Dict[OptimizationObjective, float]
    constraints: Dict[str, Any]
    max_iterations: int
    tolerance: float
    population_size: int  # For genetic algorithms

@dataclass
class OptimizationResult:
    """Optimization result"""
    strategy_used: OptimizationStrategy
    optimal_params: Dict[str, float]
    objective_value: float
    objective_breakdown: Dict[str, float]
    iterations: int
    convergence_time: float
    success: bool
    confidence: float

class MultiAgentOptimizer:
    """
    Advanced multi-agent optimization engine
    """
    
    def __init__(self):
        self.optimization_history = []
        self.performance_models = {}
        self.scaler = StandardScaler()
        
        # Optimization parameters
        self.default_weights = {
            OptimizationObjective.PERFORMANCE: 0.4,
            OptimizationObjective.SCALABILITY: 0.3,
            OptimizationObjective.ROBUSTNESS: 0.3
        }
        
        # Adaptive learning
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_history = []
        
        # Performance tracking
        self.optimization_metrics = {
            "optimizations_performed": 0,
            "successful_optimizations": 0,
            "average_improvement": 0.0,
            "convergence_failures": 0
        }
        
        # Gaussian Process for Bayesian optimization
        self.gp_kernel = ConstantKernel(1.0) * RBF(1.0) + ConstantKernel(0.1)
        self.gp_model = None
        
        logger.info("Multi-Agent Optimizer initialized")
    
    async def optimize_agents(
        self,
        db,
        task_id: int,
        user_id: int,
        agents: List[str],
        config: Optional[OptimizationConfig] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize multi-agent system configuration and performance
        
        Args:
            db: Database session
            task_id: Task identifier
            user_id: User identifier
            agents: List of agent identifiers
            config: Optimization configuration
            context: Optional optimization context
            
        Returns:
            Dict containing optimization results and recommendations
        """
        optimization_start = time.time()
        
        try:
            # Retrieve task
            from models import Task
            from sqlalchemy import and_
            
            db_task = db.query(Task).filter(
                and_(Task.id == task_id, Task.owner_id == user_id)
            ).first()
            
            if not db_task:
                raise ValueError(f"Task {task_id} not found")
            
            # Set default config if not provided
            if config is None:
                config = OptimizationConfig(
                    strategy=OptimizationStrategy.ADAPTIVE,
                    objectives=[
                        OptimizationObjective.PERFORMANCE,
                        OptimizationObjective.SCALABILITY,
                        OptimizationObjective.ROBUSTNESS
                    ],
                    weights=self.default_weights.copy(),
                    constraints={"max_agents": len(agents), "min_performance": 0.6},
                    max_iterations=100,
                    tolerance=1e-6,
                    population_size=20
                )
            
            # Initialize optimization parameters
            initial_params = await self._initialize_optimization_params(
                agents, db_task, context
            )
            
            # Select optimization strategy
            if config.strategy == OptimizationStrategy.ADAPTIVE:
                config.strategy = await self._select_adaptive_strategy(
                    agents, initial_params, config
                )
            
            # Execute optimization
            optimization_result = await self._execute_optimization(
                agents, initial_params, config, db_task, context
            )
            
            # Apply optimization results
            application_result = await self._apply_optimization_results(
                db, db_task, agents, optimization_result
            )
            
            # Update performance models
            await self._update_performance_models(
                agents, optimization_result, application_result
            )
            
            # Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(
                initial_params, optimization_result
            )
            
            # Store optimization data
            optimization_data = {
                "task_id": task_id,
                "agents": agents,
                "strategy_used": optimization_result.strategy_used.value,
                "initial_params": initial_params,
                "optimal_params": optimization_result.optimal_params,
                "objective_value": optimization_result.objective_value,
                "objective_breakdown": optimization_result.objective_breakdown,
                "improvement_metrics": improvement_metrics,
                "application_result": application_result,
                "convergence_time": optimization_result.convergence_time,
                "success": optimization_result.success,
                "confidence": optimization_result.confidence,
                "optimization_time": time.time() - optimization_start,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update task with optimization data
            db_task.optimization = optimization_data
            db.commit()
            
            # Update optimization history
            self.optimization_history.append(optimization_data)
            
            # Update metrics
            self.optimization_metrics["optimizations_performed"] += 1
            if optimization_result.success:
                self.optimization_metrics["successful_optimizations"] += 1
                improvement = improvement_metrics.get("overall_improvement", 0.0)
                self.optimization_metrics["average_improvement"] = (
                    self.optimization_metrics["average_improvement"] * 0.9 + improvement * 0.1
                )
            else:
                self.optimization_metrics["convergence_failures"] += 1
            
            logger.info(
                f"Agent optimization completed for task {task_id}: "
                f"strategy={optimization_result.strategy_used.value}, "
                f"objective={optimization_result.objective_value:.3f}, "
                f"success={optimization_result.success}"
            )
            
            return optimization_data
            
        except Exception as e:
            logger.error(f"Failed agent optimization for task {task_id}: {e}")
            raise
    
    async def _initialize_optimization_params(
        self,
        agents: List[str],
        task,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Initialize optimization parameters"""
        
        # Base parameters for each agent
        params = {}
        
        for i, agent in enumerate(agents):
            # Agent-specific parameters
            params[f"agent_{i}_load_factor"] = 1.0  # Load balancing factor
            params[f"agent_{i}_priority"] = 1.0 / len(agents)  # Priority weight
            params[f"agent_{i}_timeout"] = 30.0  # Timeout in seconds
            params[f"agent_{i}_retry_count"] = 3  # Retry attempts
            
        # Global system parameters
        params["coordination_factor"] = 0.5  # Inter-agent coordination strength
        params["communication_frequency"] = 1.0  # Communication rate
        params["resource_allocation"] = 1.0  # Resource allocation factor
        params["fault_tolerance"] = 0.8  # Fault tolerance level
        
        # Task-specific adjustments
        if hasattr(task, 'importance') and task.importance:
            # Increase resource allocation for important tasks
            params["resource_allocation"] *= (1.0 + task.importance * 0.5)
        
        return params
    
    async def _select_adaptive_strategy(
        self,
        agents: List[str],
        params: Dict[str, float],
        config: OptimizationConfig
    ) -> OptimizationStrategy:
        """Adaptively select optimization strategy based on problem characteristics"""
        
        # Problem characteristics analysis
        param_count = len(params)
        agent_count = len(agents)
        
        # Strategy selection heuristics
        if param_count <= 10 and agent_count <= 3:
            # Small problem - use gradient descent
            return OptimizationStrategy.GRADIENT_DESCENT
        elif param_count <= 20 and agent_count <= 5:
            # Medium problem - use Bayesian optimization
            return OptimizationStrategy.BAYESIAN_OPTIMIZATION
        elif param_count > 20 or agent_count > 5:
            # Large problem - use genetic algorithm
            return OptimizationStrategy.GENETIC_ALGORITHM
        else:
            # Default to simulated annealing for complex landscapes
            return OptimizationStrategy.SIMULATED_ANNEALING
    
    async def _execute_optimization(
        self,
        agents: List[str],
        initial_params: Dict[str, float],
        config: OptimizationConfig,
        task,
        context: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """Execute optimization using selected strategy"""
        
        start_time = time.time()
        
        # Define objective function
        objective_func = self._create_objective_function(
            agents, task, config.objectives, config.weights, context
        )
        
        # Parameter bounds
        param_names = list(initial_params.keys())
        bounds = self._get_parameter_bounds(initial_params)
        
        try:
            if config.strategy == OptimizationStrategy.GRADIENT_DESCENT:
                result = await self._gradient_descent_optimization(
                    objective_func, initial_params, bounds, config
                )
            elif config.strategy == OptimizationStrategy.GENETIC_ALGORITHM:
                result = await self._genetic_algorithm_optimization(
                    objective_func, param_names, bounds, config
                )
            elif config.strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
                result = await self._bayesian_optimization(
                    objective_func, param_names, bounds, config
                )
            elif config.strategy == OptimizationStrategy.SIMULATED_ANNEALING:
                result = await self._simulated_annealing_optimization(
                    objective_func, initial_params, bounds, config
                )
            else:  # MULTI_OBJECTIVE
                result = await self._multi_objective_optimization(
                    objective_func, initial_params, bounds, config
                )
            
            # Calculate objective breakdown
            objective_breakdown = await self._calculate_objective_breakdown(
                result.optimal_params, agents, task, config.objectives, context
            )
            
            # Calculate confidence based on convergence and consistency
            confidence = self._calculate_optimization_confidence(result, config)
            
            return OptimizationResult(
                strategy_used=config.strategy,
                optimal_params=result.optimal_params,
                objective_value=result.objective_value,
                objective_breakdown=objective_breakdown,
                iterations=result.iterations,
                convergence_time=time.time() - start_time,
                success=result.success,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Optimization execution failed: {e}")
            return OptimizationResult(
                strategy_used=config.strategy,
                optimal_params=initial_params,
                objective_value=float('inf'),
                objective_breakdown={},
                iterations=0,
                convergence_time=time.time() - start_time,
                success=False,
                confidence=0.0
            )
    
    def _create_objective_function(
        self,
        agents: List[str],
        task,
        objectives: List[OptimizationObjective],
        weights: Dict[OptimizationObjective, float],
        context: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """Create multi-objective function for optimization"""
        
        def objective_function(params_dict: Dict[str, float]) -> float:
            try:
                total_objective = 0.0
                
                for objective in objectives:
                    weight = weights.get(objective, 0.0)
                    
                    if objective == OptimizationObjective.PERFORMANCE:
                        perf_score = self._calculate_performance_objective(
                            agents, params_dict, task, context
                        )
                        total_objective += weight * perf_score
                        
                    elif objective == OptimizationObjective.SCALABILITY:
                        scale_score = self._calculate_scalability_objective(
                            agents, params_dict, task, context
                        )
                        total_objective += weight * scale_score
                        
                    elif objective == OptimizationObjective.ROBUSTNESS:
                        robust_score = self._calculate_robustness_objective(
                            agents, params_dict, task, context
                        )
                        total_objective += weight * robust_score
                        
                    elif objective == OptimizationObjective.EFFICIENCY:
                        eff_score = self._calculate_efficiency_objective(
                            agents, params_dict, task, context
                        )
                        total_objective += weight * eff_score
                        
                    elif objective == OptimizationObjective.COST:
                        cost_score = self._calculate_cost_objective(
                            agents, params_dict, task, context
                        )
                        total_objective -= weight * cost_score  # Minimize cost
                        
                    elif objective == OptimizationObjective.LATENCY:
                        latency_score = self._calculate_latency_objective(
                            agents, params_dict, task, context
                        )
                        total_objective -= weight * latency_score  # Minimize latency
                
                return -total_objective  # Minimize negative of objectives (maximize objectives)
                
            except Exception as e:
                logger.warning(f"Objective function evaluation failed: {e}")
                return float('inf')  # Return worst possible value on error
        
        return objective_function
    
    def _calculate_performance_objective(
        self,
        agents: List[str],
        params: Dict[str, float],
        task,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate performance objective score"""
        
        # Base performance from agent capabilities
        base_performance = 0.7  # Baseline performance
        
        # Load balancing factor
        load_factors = [params.get(f"agent_{i}_load_factor", 1.0) for i in range(len(agents))]
        load_balance = 1.0 - np.std(load_factors) / (np.mean(load_factors) or 1.0)
        
        # Priority optimization
        priorities = [params.get(f"agent_{i}_priority", 1.0/len(agents)) for i in range(len(agents))]
        priority_sum = sum(priorities)
        priority_penalty = abs(1.0 - priority_sum) * 0.5  # Penalty for priorities not summing to 1
        
        # Coordination benefit
        coordination_factor = params.get("coordination_factor", 0.5)
        coordination_benefit = coordination_factor * len(agents) * 0.1
        
        # Resource allocation efficiency
        resource_allocation = params.get("resource_allocation", 1.0)
        resource_efficiency = min(resource_allocation, 2.0) * 0.3
        
        # Task importance adjustment
        importance_boost = 0.0
        if hasattr(task, 'importance') and task.importance:
            importance_boost = task.importance * 0.2
        
        performance_score = (
            base_performance + 
            load_balance * 0.3 + 
            coordination_benefit + 
            resource_efficiency + 
            importance_boost - 
            priority_penalty
        )
        
        return max(0.0, min(1.0, performance_score))
    
    def _calculate_scalability_objective(
        self,
        agents: List[str],
        params: Dict[str, float],
        task,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate scalability objective score"""
        
        # Base scalability inversely related to agent count
        base_scalability = max(0.2, 1.0 - len(agents) * 0.1)
        
        # Communication efficiency
        comm_frequency = params.get("communication_frequency", 1.0)
        comm_efficiency = max(0.0, 1.0 - comm_frequency * 0.3)  # Lower frequency = better scalability
        
        # Load distribution scalability
        load_factors = [params.get(f"agent_{i}_load_factor", 1.0) for i in range(len(agents))]
        load_variance = np.var(load_factors) if load_factors else 0
        load_scalability = max(0.0, 1.0 - load_variance * 0.5)
        
        # Timeout scalability (shorter timeouts = better scalability)
        timeouts = [params.get(f"agent_{i}_timeout", 30.0) for i in range(len(agents))]
        avg_timeout = np.mean(timeouts)
        timeout_scalability = max(0.0, 1.0 - avg_timeout / 100.0)  # Normalize by max timeout
        
        scalability_score = (
            base_scalability * 0.4 +
            comm_efficiency * 0.3 +
            load_scalability * 0.2 +
            timeout_scalability * 0.1
        )
        
        return max(0.0, min(1.0, scalability_score))
    
    def _calculate_robustness_objective(
        self,
        agents: List[str],
        params: Dict[str, float],
        task,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate robustness objective score"""
        
        # Base robustness from redundancy
        base_robustness = min(1.0, len(agents) * 0.2)  # More agents = more robustness
        
        # Fault tolerance level
        fault_tolerance = params.get("fault_tolerance", 0.8)
        
        # Retry mechanism robustness
        retry_counts = [params.get(f"agent_{i}_retry_count", 3) for i in range(len(agents))]
        avg_retries = np.mean(retry_counts)
        retry_robustness = min(1.0, avg_retries / 5.0)  # Normalize by max reasonable retries
        
        # Timeout robustness (longer timeouts = more robust but less efficient)
        timeouts = [params.get(f"agent_{i}_timeout", 30.0) for i in range(len(agents))]
        timeout_robustness = min(1.0, np.mean(timeouts) / 60.0)  # Normalize by 60 seconds
        
        # Coordination robustness
        coordination_factor = params.get("coordination_factor", 0.5)
        coord_robustness = coordination_factor  # Higher coordination = more robust
        
        robustness_score = (
            base_robustness * 0.3 +
            fault_tolerance * 0.4 +
            retry_robustness * 0.1 +
            timeout_robustness * 0.1 +
            coord_robustness * 0.1
        )
        
        return max(0.0, min(1.0, robustness_score))
    
    def _calculate_efficiency_objective(
        self,
        agents: List[str],
        params: Dict[str, float],
        task,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate efficiency objective score"""
        
        # Resource utilization efficiency
        resource_allocation = params.get("resource_allocation", 1.0)
        resource_efficiency = 1.0 / (1.0 + abs(resource_allocation - 1.0))  # Penalty for over/under allocation
        
        # Communication efficiency
        comm_frequency = params.get("communication_frequency", 1.0)
        comm_efficiency = 1.0 / (1.0 + comm_frequency * 0.5)  # Lower frequency = higher efficiency
        
        # Load balancing efficiency
        load_factors = [params.get(f"agent_{i}_load_factor", 1.0) for i in range(len(agents))]
        load_std = np.std(load_factors) if load_factors else 0
        load_efficiency = 1.0 / (1.0 + load_std)
        
        # Priority allocation efficiency
        priorities = [params.get(f"agent_{i}_priority", 1.0/len(agents)) for i in range(len(agents))]
        priority_efficiency = 1.0 / (1.0 + abs(sum(priorities) - 1.0))
        
        efficiency_score = (
            resource_efficiency * 0.4 +
            comm_efficiency * 0.2 +
            load_efficiency * 0.2 +
            priority_efficiency * 0.2
        )
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _calculate_cost_objective(
        self,
        agents: List[str],
        params: Dict[str, float],
        task,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate cost objective score (to be minimized)"""
        
        # Base cost from number of agents
        base_cost = len(agents) * 0.1
        
        # Resource allocation cost
        resource_allocation = params.get("resource_allocation", 1.0)
        resource_cost = resource_allocation * 0.3
        
        # Communication cost
        comm_frequency = params.get("communication_frequency", 1.0)
        comm_cost = comm_frequency * len(agents) * 0.05
        
        # Timeout cost (longer timeouts consume more resources)
        timeouts = [params.get(f"agent_{i}_timeout", 30.0) for i in range(len(agents))]
        timeout_cost = np.mean(timeouts) * 0.01
        
        # Retry cost
        retry_counts = [params.get(f"agent_{i}_retry_count", 3) for i in range(len(agents))]
        retry_cost = np.mean(retry_counts) * 0.05
        
        total_cost = base_cost + resource_cost + comm_cost + timeout_cost + retry_cost
        
        return max(0.0, total_cost)
    
    def _calculate_latency_objective(
        self,
        agents: List[str],
        params: Dict[str, float],
        task,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate latency objective score (to be minimized)"""
        
        # Base latency from coordination overhead
        coordination_factor = params.get("coordination_factor", 0.5)
        coordination_latency = coordination_factor * len(agents) * 0.5
        
        # Communication latency
        comm_frequency = params.get("communication_frequency", 1.0)
        comm_latency = comm_frequency * 0.3
        
        # Timeout latency (expected waiting time)
        timeouts = [params.get(f"agent_{i}_timeout", 30.0) for i in range(len(agents))]
        avg_timeout = np.mean(timeouts)
        timeout_latency = avg_timeout / 60.0  # Normalize
        
        # Load imbalance latency
        load_factors = [params.get(f"agent_{i}_load_factor", 1.0) for i in range(len(agents))]
        load_imbalance = np.std(load_factors) if load_factors else 0
        imbalance_latency = load_imbalance * 0.5
        
        total_latency = coordination_latency + comm_latency + timeout_latency + imbalance_latency
        
        return max(0.0, total_latency)
    
    def _get_parameter_bounds(
        self,
        initial_params: Dict[str, float]
    ) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization"""
        
        bounds = []
        
        for param_name, initial_value in initial_params.items():
            if "load_factor" in param_name:
                bounds.append((0.1, 3.0))  # Load factor range
            elif "priority" in param_name:
                bounds.append((0.0, 1.0))  # Priority range
            elif "timeout" in param_name:
                bounds.append((5.0, 120.0))  # Timeout range (5s to 2min)
            elif "retry_count" in param_name:
                bounds.append((1, 10))  # Retry count range
            elif "coordination_factor" in param_name:
                bounds.append((0.0, 1.0))  # Coordination factor range
            elif "communication_frequency" in param_name:
                bounds.append((0.1, 5.0))  # Communication frequency range
            elif "resource_allocation" in param_name:
                bounds.append((0.1, 3.0))  # Resource allocation range
            elif "fault_tolerance" in param_name:
                bounds.append((0.1, 1.0))  # Fault tolerance range
            else:
                # Default bounds
                bounds.append((max(0.1, initial_value * 0.1), initial_value * 2.0))
        
        return bounds
    
    async def _gradient_descent_optimization(
        self,
        objective_func: Callable,
        initial_params: Dict[str, float],
        bounds: List[Tuple[float, float]],
        config: OptimizationConfig
    ) -> Any:
        """Gradient descent optimization"""
        
        param_names = list(initial_params.keys())
        x0 = [initial_params[name] for name in param_names]
        
        # Wrapper function for scipy minimize
        def scipy_objective(x):
            param_dict = dict(zip(param_names, x))
            return objective_func(param_dict)
        
        result = minimize(
            scipy_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': config.max_iterations,
                'ftol': config.tolerance
            }
        )
        
        optimal_params = dict(zip(param_names, result.x))
        
        return type('OptResult', (), {
            'optimal_params': optimal_params,
            'objective_value': result.fun,
            'iterations': result.nit,
            'success': result.success
        })()
    
    async def _genetic_algorithm_optimization(
        self,
        objective_func: Callable,
        param_names: List[str],
        bounds: List[Tuple[float, float]],
        config: OptimizationConfig
    ) -> Any:
        """Genetic algorithm optimization"""
        
        def scipy_objective(x):
            param_dict = dict(zip(param_names, x))
            return objective_func(param_dict)
        
        result = differential_evolution(
            scipy_objective,
            bounds,
            maxiter=config.max_iterations,
            popsize=config.population_size,
            tol=config.tolerance,
            seed=42
        )
        
        optimal_params = dict(zip(param_names, result.x))
        
        return type('OptResult', (), {
            'optimal_params': optimal_params,
            'objective_value': result.fun,
            'iterations': result.nit,
            'success': result.success
        })()
    
    async def _bayesian_optimization(
        self,
        objective_func: Callable,
        param_names: List[str],
        bounds: List[Tuple[float, float]],
        config: OptimizationConfig
    ) -> Any:
        """Bayesian optimization using Gaussian Process"""
        
        # Initialize GP model if not exists
        if self.gp_model is None:
            self.gp_model = GaussianProcessRegressor(
                kernel=self.gp_kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=2
            )
        
        # Initial sample points
        n_initial = min(10, len(bounds) * 2)
        X_sample = []
        y_sample = []
        
        for i in range(n_initial):
            x = [np.random.uniform(bound[0], bound[1]) for bound in bounds]
            param_dict = dict(zip(param_names, x))
            y = objective_func(param_dict)
            
            X_sample.append(x)
            y_sample.append(y)
        
        X_sample = np.array(X_sample)
        y_sample = np.array(y_sample)
        
        # Iterative optimization
        best_x = X_sample[np.argmin(y_sample)]
        best_y = np.min(y_sample)
        iterations = n_initial
        
        for iteration in range(config.max_iterations - n_initial):
            # Fit GP model
            self.gp_model.fit(X_sample, y_sample)
            
            # Acquisition function (Expected Improvement)
            def acquisition(x):
                x = np.array(x).reshape(1, -1)
                mu, sigma = self.gp_model.predict(x, return_std=True)
                
                if sigma == 0:
                    return 0
                
                improvement = best_y - mu
                z = improvement / sigma
                
                from scipy import stats
                ei = improvement * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
                return -ei  # Minimize negative EI
            
            # Optimize acquisition function
            acq_result = minimize(
                acquisition,
                best_x,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            # Evaluate new point
            new_x = acq_result.x
            param_dict = dict(zip(param_names, new_x))
            new_y = objective_func(param_dict)
            
            # Update dataset
            X_sample = np.vstack([X_sample, new_x])
            y_sample = np.append(y_sample, new_y)
            
            # Update best
            if new_y < best_y:
                best_x = new_x
                best_y = new_y
            
            iterations += 1
            
            # Convergence check
            if len(y_sample) > 3:
                recent_improvement = np.min(y_sample[-3:]) - np.min(y_sample[-6:-3])
                if abs(recent_improvement) < config.tolerance:
                    break
        
        optimal_params = dict(zip(param_names, best_x))
        
        return type('OptResult', (), {
            'optimal_params': optimal_params,
            'objective_value': best_y,
            'iterations': iterations,
            'success': True
        })()
    
    async def _simulated_annealing_optimization(
        self,
        objective_func: Callable,
        initial_params: Dict[str, float],
        bounds: List[Tuple[float, float]],
        config: OptimizationConfig
    ) -> Any:
        """Simulated annealing optimization"""
        
        param_names = list(initial_params.keys())
        x0 = [initial_params[name] for name in param_names]
        
        def scipy_objective(x):
            param_dict = dict(zip(param_names, x))
            return objective_func(param_dict)
        
        # Custom minimizer for basinhopping
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "options": {"maxiter": 50}
        }
        
        result = basinhopping(
            scipy_objective,
            x0,
            niter=config.max_iterations,
            T=1.0,  # Temperature
            stepsize=0.5,
            minimizer_kwargs=minimizer_kwargs,
            seed=42
        )
        
        optimal_params = dict(zip(param_names, result.x))
        
        return type('OptResult', (), {
            'optimal_params': optimal_params,
            'objective_value': result.fun,
            'iterations': result.nit,
            'success': result.lowest_optimization_result.success if hasattr(result, 'lowest_optimization_result') else True
        })()
    
    async def _multi_objective_optimization(
        self,
        objective_func: Callable,
        initial_params: Dict[str, float],
        bounds: List[Tuple[float, float]],
        config: OptimizationConfig
    ) -> Any:
        """Multi-objective optimization using NSGA-II-like approach"""
        
        # For simplicity, use weighted sum approach
        # In production, consider NSGA-II or MOPSO
        return await self._gradient_descent_optimization(
            objective_func, initial_params, bounds, config
        )
    
    async def _calculate_objective_breakdown(
        self,
        optimal_params: Dict[str, float],
        agents: List[str],
        task,
        objectives: List[OptimizationObjective],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Calculate breakdown of objective scores"""
        
        breakdown = {}
        
        for objective in objectives:
            if objective == OptimizationObjective.PERFORMANCE:
                score = self._calculate_performance_objective(agents, optimal_params, task, context)
            elif objective == OptimizationObjective.SCALABILITY:
                score = self._calculate_scalability_objective(agents, optimal_params, task, context)
            elif objective == OptimizationObjective.ROBUSTNESS:
                score = self._calculate_robustness_objective(agents, optimal_params, task, context)
            elif objective == OptimizationObjective.EFFICIENCY:
                score = self._calculate_efficiency_objective(agents, optimal_params, task, context)
            elif objective == OptimizationObjective.COST:
                score = self._calculate_cost_objective(agents, optimal_params, task, context)
            elif objective == OptimizationObjective.LATENCY:
                score = self._calculate_latency_objective(agents, optimal_params, task, context)
            else:
                score = 0.0
            
            breakdown[objective.value] = score
        
        return breakdown
    
    def _calculate_optimization_confidence(
        self,
        result: Any,
        config: OptimizationConfig
    ) -> float:
        """Calculate confidence in optimization result"""
        
        # Base confidence from convergence
        base_confidence = 0.8 if result.success else 0.3
        
        # Iteration-based confidence
        iteration_ratio = result.iterations / config.max_iterations
        iteration_confidence = max(0.0, 1.0 - iteration_ratio)  # Earlier convergence = higher confidence
        
        # Objective value confidence (assuming we want to minimize)
        objective_confidence = max(0.0, 1.0 / (1.0 + abs(result.objective_value))) if result.objective_value != float('inf') else 0.0
        
        # Combined confidence
        confidence = (
            base_confidence * 0.5 +
            iteration_confidence * 0.3 +
            objective_confidence * 0.2
        )
        
        return max(0.0, min(1.0, confidence))
    
    async def _apply_optimization_results(
        self,
        db,
        task,
        agents: List[str],
        result: OptimizationResult
    ) -> Dict[str, Any]:
        """Apply optimization results to the system"""
        
        try:
            # Extract optimized parameters
            optimal_params = result.optimal_params
            
            # Apply agent-specific optimizations
            agent_updates = {}
            
            for i, agent in enumerate(agents):
                agent_config = {}
                
                # Extract agent-specific parameters
                if f"agent_{i}_load_factor" in optimal_params:
                    agent_config["load_factor"] = optimal_params[f"agent_{i}_load_factor"]
                
                if f"agent_{i}_priority" in optimal_params:
                    agent_config["priority"] = optimal_params[f"agent_{i}_priority"]
                
                if f"agent_{i}_timeout" in optimal_params:
                    agent_config["timeout"] = optimal_params[f"agent_{i}_timeout"]
                
                if f"agent_{i}_retry_count" in optimal_params:
                    agent_config["retry_count"] = int(optimal_params[f"agent_{i}_retry_count"])
                
                agent_updates[agent] = agent_config
            
            # Apply global system optimizations
            system_config = {}
            
            if "coordination_factor" in optimal_params:
                system_config["coordination_factor"] = optimal_params["coordination_factor"]
            
            if "communication_frequency" in optimal_params:
                system_config["communication_frequency"] = optimal_params["communication_frequency"]
            
            if "resource_allocation" in optimal_params:
                system_config["resource_allocation"] = optimal_params["resource_allocation"]
            
            if "fault_tolerance" in optimal_params:
                system_config["fault_tolerance"] = optimal_params["fault_tolerance"]
            
            # Store configuration in task
            task.optimization_config = {
                "agent_configs": agent_updates,
                "system_config": system_config,
                "optimization_timestamp": datetime.utcnow().isoformat()
            }
            
            db.commit()
            
            return {
                "status": "applied",
                "agent_updates": agent_updates,
                "system_config": system_config,
                "agents_updated": len(agent_updates)
            }
            
        except Exception as e:
            logger.error(f"Failed to apply optimization results: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _update_performance_models(
        self,
        agents: List[str],
        optimization_result: OptimizationResult,
        application_result: Dict[str, Any]
    ):
        """Update performance prediction models"""
        
        try:
            # Store performance data for future predictions
            performance_data = {
                "agents": agents,
                "parameters": optimization_result.optimal_params,
                "objective_value": optimization_result.objective_value,
                "objective_breakdown": optimization_result.objective_breakdown,
                "application_success": application_result.get("status") == "applied",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add to performance history
            model_key = f"agents_{len(agents)}"
            if model_key not in self.performance_models:
                self.performance_models[model_key] = []
            
            self.performance_models[model_key].append(performance_data)
            
            # Keep only recent data (last 100 records)
            if len(self.performance_models[model_key]) > 100:
                self.performance_models[model_key] = self.performance_models[model_key][-100:]
            
        except Exception as e:
            logger.warning(f"Failed to update performance models: {e}")
    
    def _calculate_improvement_metrics(
        self,
        initial_params: Dict[str, float],
        result: OptimizationResult
    ) -> Dict[str, float]:
        """Calculate improvement metrics from optimization"""
        
        try:
            # Calculate parameter changes
            param_changes = {}
            total_change = 0.0
            
            for param_name in initial_params:
                initial_val = initial_params[param_name]
                optimal_val = result.optimal_params.get(param_name, initial_val)
                
                if initial_val != 0:
                    change_percent = abs((optimal_val - initial_val) / initial_val) * 100
                else:
                    change_percent = abs(optimal_val) * 100
                
                param_changes[param_name] = change_percent
                total_change += change_percent
            
            avg_param_change = total_change / len(initial_params) if initial_params else 0.0
            
            # Objective improvement (negative objective means better result)
            objective_improvement = 0.0
            if result.objective_value != float('inf'):
                # Assume initial objective would be worse
                initial_objective_estimate = 1.0  # Rough estimate
                if result.objective_value < initial_objective_estimate:
                    objective_improvement = initial_objective_estimate - result.objective_value
            
            return {
                "average_parameter_change": avg_param_change,
                "total_parameter_change": total_change,
                "objective_improvement": objective_improvement,
                "overall_improvement": (objective_improvement + avg_param_change / 100) / 2,
                "convergence_success": result.success,
                "confidence": result.confidence
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate improvement metrics: {e}")
            return {
                "average_parameter_change": 0.0,
                "total_parameter_change": 0.0,
                "objective_improvement": 0.0,
                "overall_improvement": 0.0,
                "convergence_success": False,
                "confidence": 0.0
            }
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        
        if not self.optimization_history:
            return {
                "message": "No optimization history available",
                "metrics": self.optimization_metrics
            }
        
        # Aggregate statistics
        recent_optimizations = self.optimization_history[-20:]  # Last 20 optimizations
        
        avg_objective_value = np.mean([
            o["objective_value"] for o in recent_optimizations 
            if o["objective_value"] != float('inf')
        ])
        
        avg_convergence_time = np.mean([o["convergence_time"] for o in recent_optimizations])
        success_rate = np.mean([o["success"] for o in recent_optimizations])
        avg_confidence = np.mean([o["confidence"] for o in recent_optimizations])
        
        # Strategy usage
        strategies = [o["strategy_used"] for o in recent_optimizations]
        strategy_counts = {s: strategies.count(s) for s in set(strategies)}
        
        # Objective breakdown
        objective_stats = {}
        for opt in recent_optimizations:
            for obj_name, obj_value in opt.get("objective_breakdown", {}).items():
                if obj_name not in objective_stats:
                    objective_stats[obj_name] = []
                objective_stats[obj_name].append(obj_value)
        
        objective_averages = {
            obj: np.mean(values) for obj, values in objective_stats.items()
        }
        
        return {
            "total_optimizations": len(self.optimization_history),
            "recent_optimizations": len(recent_optimizations),
            "average_objective_value": avg_objective_value,
            "average_convergence_time": avg_convergence_time,
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "strategy_usage": strategy_counts,
            "objective_averages": objective_averages,
            "optimization_metrics": self.optimization_metrics,
            "performance_models": {
                model: len(data) for model, data in self.performance_models.items()
            }
        }
