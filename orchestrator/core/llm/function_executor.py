"""
Function Executor - Safe execution of registered functions
===========================================================

PRD-16: Executes functions called by the LLM with proper error handling,
timeouts, retries, and logging.

Safety Features:
- Input validation before execution
- Timeout enforcement
- Automatic retries with exponential backoff
- Error isolation and recovery
- Execution logging for audit
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .function_registry import FunctionRegistry, FunctionSpec

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of function execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"


@dataclass
class FunctionResult:
    """Result of function execution"""
    name: str
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class FunctionExecutor:
    """
    Safely executes registered functions with comprehensive error handling.
    
    Features:
    - Input validation against function spec
    - Timeout enforcement
    - Automatic retries with backoff
    - Detailed execution logging
    - Performance monitoring
    """
    
    def __init__(
        self,
        registry: FunctionRegistry,
        default_timeout: float = 10.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        enable_logging: bool = True
    ):
        """
        Initialize the function executor.
        
        Args:
            registry: Function registry to use
            default_timeout: Default timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay in seconds
            enable_logging: Whether to log executions
        """
        self.registry = registry
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_logging = enable_logging
        
        # Execution history
        self.execution_history: List[FunctionResult] = []
        self.max_history = 1000
        
        # Performance metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_executions": 0,
            "validation_errors": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0
        }
        
        logger.info(
            f"FunctionExecutor initialized: timeout={default_timeout}s, "
            f"max_retries={max_retries}"
        )
    
    async def execute(
        self,
        function_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> FunctionResult:
        """
        Execute a registered function.
        
        Args:
            function_name: Name of the function to execute
            parameters: Parameters to pass to the function
            timeout: Override default timeout
            context: Additional context for logging
            
        Returns:
            FunctionResult with execution details
        """
        start_time = time.time()
        self.metrics["total_executions"] += 1
        
        # Get function spec
        spec = self.registry.get(function_name)
        if not spec:
            self.metrics["failed_executions"] += 1
            result = FunctionResult(
                name=function_name,
                status=ExecutionStatus.NOT_FOUND,
                error=f"Function '{function_name}' not found in registry"
            )
            self._record_result(result)
            return result
        
        # Validate parameters
        is_valid, error_msg = spec.validate_parameters(parameters)
        if not is_valid:
            self.metrics["validation_errors"] += 1
            result = FunctionResult(
                name=function_name,
                status=ExecutionStatus.VALIDATION_ERROR,
                error=error_msg
            )
            self._record_result(result)
            return result
        
        # Check if implementation exists
        if not spec.implementation:
            self.metrics["failed_executions"] += 1
            result = FunctionResult(
                name=function_name,
                status=ExecutionStatus.FAILURE,
                error=f"Function '{function_name}' has no implementation"
            )
            self._record_result(result)
            return result
        
        # Determine timeout
        exec_timeout = timeout or spec.timeout_seconds or self.default_timeout
        
        # Execute with retries
        max_retries = min(spec.retry_count, self.max_retries)
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # Log execution attempt
                if self.enable_logging:
                    logger.info(
                        f"Executing function '{function_name}' "
                        f"(attempt {retry_count + 1}/{max_retries + 1})"
                    )
                
                # Execute with timeout
                result_data = await self._execute_with_timeout(
                    spec.implementation,
                    parameters,
                    exec_timeout
                )
                
                # Success!
                execution_time = time.time() - start_time
                self.metrics["successful_executions"] += 1
                self.metrics["total_execution_time"] += execution_time
                self.metrics["avg_execution_time"] = (
                    self.metrics["total_execution_time"] / 
                    self.metrics["total_executions"]
                )
                
                # Update registry usage stats
                self.registry.update_usage(function_name, True, execution_time)
                
                result = FunctionResult(
                    name=function_name,
                    status=ExecutionStatus.SUCCESS,
                    result=result_data,
                    execution_time=execution_time,
                    retry_count=retry_count,
                    metadata={"context": context} if context else {}
                )
                
                self._record_result(result)
                return result
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {exec_timeout}s"
                self.metrics["timeout_executions"] += 1
                
                if retry_count < max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** retry_count))
                    retry_count += 1
                    continue
                
                # Final timeout
                execution_time = time.time() - start_time
                self.registry.update_usage(function_name, False, execution_time)
                
                result = FunctionResult(
                    name=function_name,
                    status=ExecutionStatus.TIMEOUT,
                    error=last_error,
                    execution_time=execution_time,
                    retry_count=retry_count
                )
                self._record_result(result)
                return result
                
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                
                if self.enable_logging:
                    logger.error(
                        f"Function '{function_name}' failed: {last_error}\n"
                        f"Traceback: {traceback.format_exc()}"
                    )
                
                if retry_count < max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** retry_count))
                    retry_count += 1
                    continue
                
                # Final failure
                execution_time = time.time() - start_time
                self.metrics["failed_executions"] += 1
                self.registry.update_usage(function_name, False, execution_time)
                
                result = FunctionResult(
                    name=function_name,
                    status=ExecutionStatus.FAILURE,
                    error=last_error,
                    execution_time=execution_time,
                    retry_count=retry_count
                )
                self._record_result(result)
                return result
        
        # Should not reach here
        return FunctionResult(
            name=function_name,
            status=ExecutionStatus.FAILURE,
            error="Unexpected execution path"
        )
    
    async def execute_batch(
        self,
        executions: List[Dict[str, Any]],
        parallel: bool = True,
        max_parallel: int = 5
    ) -> List[FunctionResult]:
        """
        Execute multiple functions.
        
        Args:
            executions: List of dicts with 'name' and 'parameters'
            parallel: Whether to execute in parallel
            max_parallel: Maximum parallel executions
            
        Returns:
            List of FunctionResults
        """
        if not executions:
            return []
        
        if parallel:
            # Execute in parallel with concurrency limit
            semaphore = asyncio.Semaphore(max_parallel)
            
            async def execute_with_semaphore(exec_data):
                async with semaphore:
                    return await self.execute(
                        exec_data["name"],
                        exec_data.get("parameters", {}),
                        exec_data.get("timeout"),
                        exec_data.get("context")
                    )
            
            tasks = [execute_with_semaphore(e) for e in executions]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to FunctionResults
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(FunctionResult(
                        name=executions[i]["name"],
                        status=ExecutionStatus.FAILURE,
                        error=str(result)
                    ))
                else:
                    final_results.append(result)
            
            return final_results
        else:
            # Execute sequentially
            results = []
            for exec_data in executions:
                result = await self.execute(
                    exec_data["name"],
                    exec_data.get("parameters", {}),
                    exec_data.get("timeout"),
                    exec_data.get("context")
                )
                results.append(result)
            
            return results
    
    async def _execute_with_timeout(
        self,
        func: Callable,
        parameters: Dict[str, Any],
        timeout: float
    ) -> Any:
        """Execute function with timeout"""
        # Check if function is async
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(
                func(**parameters),
                timeout=timeout
            )
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, func, **parameters),
                timeout=timeout
            )
    
    def _record_result(self, result: FunctionResult):
        """Record execution result in history"""
        self.execution_history.append(result)
        
        # Trim history if needed
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_executions"] / self.metrics["total_executions"]
                if self.metrics["total_executions"] > 0 else 0
            ),
            "avg_retry_count": (
                sum(r.retry_count for r in self.execution_history[-100:]) / 
                min(100, len(self.execution_history))
                if self.execution_history else 0
            )
        }
    
    def get_history(
        self,
        function_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        limit: int = 100
    ) -> List[FunctionResult]:
        """
        Get execution history.
        
        Args:
            function_name: Filter by function name
            status: Filter by status
            limit: Maximum results to return
            
        Returns:
            List of FunctionResults
        """
        history = self.execution_history
        
        if function_name:
            history = [r for r in history if r.name == function_name]
        
        if status:
            history = [r for r in history if r.status == status]
        
        return history[-limit:]
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_executions": 0,
            "validation_errors": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0
        }




