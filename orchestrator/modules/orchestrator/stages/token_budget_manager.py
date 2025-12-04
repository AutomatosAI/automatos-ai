"""
Token Budget Manager
====================

Phase 1: Centralized Token Budget Management

Manages token allocation, tracking, and enforcement across:
- Workflow level (total budget)
- Task level (per-task limits)
- Subtask level (per-subtask limits)

Features:
- Real-time token tracking
- Budget enforcement with alerts
- Reserve allocation for critical operations
- Cost estimation and monitoring
- Budget recommendations based on complexity

Author: Automatos AI - Phase 1 Implementation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class BudgetStatus(Enum):
    """Budget status indicators"""
    HEALTHY = "healthy"  # < 70% used
    WARNING = "warning"  # 70-90% used
    CRITICAL = "critical"  # 90-100% used
    EXCEEDED = "exceeded"  # > 100% used


@dataclass
class TokenAllocation:
    """Token allocation for a specific entity (workflow, task, or subtask)"""
    
    entity_id: str
    entity_type: str  # "workflow", "task", "subtask"
    allocated_tokens: int
    used_tokens: int = 0
    reserved_tokens: int = 0
    
    @property
    def available_tokens(self) -> int:
        """Available tokens (allocated - used - reserved)"""
        return max(0, self.allocated_tokens - self.used_tokens - self.reserved_tokens)
    
    @property
    def usage_percent(self) -> float:
        """Usage percentage"""
        if self.allocated_tokens == 0:
            return 0.0
        return (self.used_tokens / self.allocated_tokens) * 100
    
    @property
    def status(self) -> BudgetStatus:
        """Budget status"""
        usage = self.usage_percent
        if usage >= 100:
            return BudgetStatus.EXCEEDED
        elif usage >= 90:
            return BudgetStatus.CRITICAL
        elif usage >= 70:
            return BudgetStatus.WARNING
        else:
            return BudgetStatus.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "allocated_tokens": self.allocated_tokens,
            "used_tokens": self.used_tokens,
            "reserved_tokens": self.reserved_tokens,
            "available_tokens": self.available_tokens,
            "usage_percent": self.usage_percent,
            "status": self.status.value
        }


@dataclass
class BudgetSummary:
    """Summary of budget status"""
    
    total_allocated: int
    total_used: int
    total_reserved: int
    total_available: int
    usage_percent: float
    status: BudgetStatus
    
    # Breakdown by entity type
    workflow_allocations: List[TokenAllocation] = field(default_factory=list)
    task_allocations: List[TokenAllocation] = field(default_factory=list)
    subtask_allocations: List[TokenAllocation] = field(default_factory=list)
    
    # Cost tracking
    estimated_cost: float = 0.0
    model: str = "gpt-4"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_allocated": self.total_allocated,
            "total_used": self.total_used,
            "total_reserved": self.total_reserved,
            "total_available": self.total_available,
            "usage_percent": self.usage_percent,
            "status": self.status.value,
            "estimated_cost": self.estimated_cost,
            "model": self.model,
            "workflow_count": len(self.workflow_allocations),
            "task_count": len(self.task_allocations),
            "subtask_count": len(self.subtask_allocations)
        }


class TokenBudgetManager:
    """
    Centralized token budget management
    
    Responsibilities:
    1. Allocate tokens to workflows, tasks, and subtasks
    2. Track token usage in real-time
    3. Enforce budget limits with alerts
    4. Provide budget recommendations
    5. Support reserve allocation for critical operations
    """
    
    def __init__(self, workflow_id: Optional[str] = None):
        """
        Initialize token budget manager
        
        Args:
            workflow_id: Optional workflow ID for scoped management
        """
        self.workflow_id = workflow_id
        self.allocations: Dict[str, TokenAllocation] = {}
        self.usage_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        
        from config import orchestrator_config
        self.config = orchestrator_config
        
        logger.info(f"TokenBudgetManager initialized (workflow: {workflow_id or 'global'})")
    
    def allocate_workflow_budget(
        self,
        workflow_id: str,
        total_budget: Optional[int] = None,
        reserve_percent: Optional[int] = None
    ) -> TokenAllocation:
        """
        Allocate token budget for a workflow
        
        Args:
            workflow_id: Workflow identifier
            total_budget: Total tokens to allocate (uses config default if None)
            reserve_percent: Percentage to reserve (uses config default if None)
        
        Returns:
            TokenAllocation for the workflow
        """
        if total_budget is None:
            total_budget = self.config.TOKEN_BUDGET_DEFAULT
        
        if reserve_percent is None:
            reserve_percent = self.config.TOKEN_RESERVE_PERCENT
        
        # Calculate reserve
        reserved_tokens = int(total_budget * (reserve_percent / 100))
        
        allocation = TokenAllocation(
            entity_id=workflow_id,
            entity_type="workflow",
            allocated_tokens=total_budget,
            reserved_tokens=reserved_tokens
        )
        
        self.allocations[workflow_id] = allocation
        
        logger.info(f"💰 Allocated workflow budget: {total_budget:,} tokens "
                   f"(reserve: {reserved_tokens:,} tokens, {reserve_percent}%)")
        
        return allocation
    
    def allocate_task_budget(
        self,
        task_id: str,
        workflow_id: str,
        num_subtasks: int,
        task_complexity: float = 0.5
    ) -> Tuple[TokenAllocation, int]:
        """
        Allocate token budget for a task and its subtasks
        
        Args:
            task_id: Task identifier
            workflow_id: Parent workflow identifier
            num_subtasks: Number of subtasks
            task_complexity: Complexity score (0.0-1.0)
        
        Returns:
            Tuple of (task_allocation, tokens_per_subtask)
        """
        # Get workflow allocation
        workflow_allocation = self.allocations.get(workflow_id)
        if not workflow_allocation:
            logger.warning(f"No workflow allocation found for {workflow_id}, creating default")
            workflow_allocation = self.allocate_workflow_budget(workflow_id)
        
        # Calculate task budget based on available tokens and complexity
        available = workflow_allocation.available_tokens
        
        # Use complexity to adjust allocation (more complex = more tokens)
        complexity_multiplier = 0.5 + (task_complexity * 0.5)  # 0.5-1.0 range
        
        # Don't exceed max task budget
        max_task_budget = min(
            self.config.TOKEN_BUDGET_MAX_TASK,
            available
        )
        
        task_budget = int(max_task_budget * complexity_multiplier)
        
        # Create task allocation
        task_allocation = TokenAllocation(
            entity_id=task_id,
            entity_type="task",
            allocated_tokens=task_budget
        )
        
        self.allocations[task_id] = task_allocation
        
        # Calculate per-subtask budget
        tokens_per_subtask = min(
            task_budget // num_subtasks if num_subtasks > 0 else task_budget,
            self.config.TOKEN_BUDGET_MAX_SUBTASK
        )
        
        logger.info(f"💰 Allocated task budget: {task_budget:,} tokens "
                   f"({tokens_per_subtask:,} per subtask × {num_subtasks} subtasks)")
        
        return task_allocation, tokens_per_subtask
    
    def allocate_subtask_budget(
        self,
        subtask_id: str,
        task_id: str,
        tokens: Optional[int] = None
    ) -> TokenAllocation:
        """
        Allocate token budget for a subtask
        
        Args:
            subtask_id: Subtask identifier
            task_id: Parent task identifier
            tokens: Tokens to allocate (uses default if None)
        
        Returns:
            TokenAllocation for the subtask
        """
        if tokens is None:
            tokens = self.config.TOKEN_BUDGET_MAX_SUBTASK
        
        # Ensure we don't exceed task budget
        task_allocation = self.allocations.get(task_id)
        if task_allocation:
            tokens = min(tokens, task_allocation.available_tokens)
        
        allocation = TokenAllocation(
            entity_id=subtask_id,
            entity_type="subtask",
            allocated_tokens=tokens
        )
        
        self.allocations[subtask_id] = allocation
        
        logger.debug(f"Allocated subtask budget: {tokens:,} tokens for {subtask_id}")
        
        return allocation
    
    def record_usage(
        self,
        entity_id: str,
        tokens_used: int,
        operation: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record token usage for an entity
        
        Args:
            entity_id: Entity identifier (workflow, task, or subtask)
            tokens_used: Number of tokens used
            operation: Description of the operation
            metadata: Additional metadata
        
        Returns:
            True if usage recorded successfully, False if budget exceeded
        """
        allocation = self.allocations.get(entity_id)
        if not allocation:
            logger.warning(f"No allocation found for {entity_id}, creating default")
            allocation = TokenAllocation(
                entity_id=entity_id,
                entity_type="subtask",
                allocated_tokens=self.config.TOKEN_BUDGET_MAX_SUBTASK
            )
            self.allocations[entity_id] = allocation
        
        # Record usage
        previous_used = allocation.used_tokens
        allocation.used_tokens += tokens_used
        
        # Log usage
        usage_record = {
            "entity_id": entity_id,
            "entity_type": allocation.entity_type,
            "tokens_used": tokens_used,
            "total_used": allocation.used_tokens,
            "allocated": allocation.allocated_tokens,
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.usage_history.append(usage_record)
        
        # Check for budget issues
        status_changed = self._check_and_alert(allocation, previous_used)
        
        logger.debug(f"🎟️  Recorded {tokens_used:,} tokens for {entity_id} "
                    f"(total: {allocation.used_tokens:,}/{allocation.allocated_tokens:,}, "
                    f"{allocation.usage_percent:.1f}%)")
        
        # Update parent allocations
        if allocation.entity_type == "subtask":
            self._propagate_usage_to_parent(entity_id, tokens_used)
        
        return allocation.status != BudgetStatus.EXCEEDED
    
    def _propagate_usage_to_parent(self, entity_id: str, tokens_used: int):
        """Propagate usage from subtask to task and workflow"""
        # Find parent task (assuming entity_id format: workflow_id.task_id.subtask_id)
        parts = entity_id.split('.')
        if len(parts) >= 2:
            task_id = '.'.join(parts[:-1])
            task_allocation = self.allocations.get(task_id)
            if task_allocation:
                task_allocation.used_tokens += tokens_used
        
        if len(parts) >= 3:
            workflow_id = parts[0]
            workflow_allocation = self.allocations.get(workflow_id)
            if workflow_allocation:
                workflow_allocation.used_tokens += tokens_used
    
    def _check_and_alert(self, allocation: TokenAllocation, previous_used: int) -> bool:
        """
        Check budget status and generate alerts if needed
        
        Returns:
            True if status changed
        """
        # Get current and previous status
        current_status = allocation.status
        
        # Calculate previous status
        previous_percent = (previous_used / allocation.allocated_tokens * 100) if allocation.allocated_tokens > 0 else 0
        if previous_percent >= 100:
            previous_status = BudgetStatus.EXCEEDED
        elif previous_percent >= 90:
            previous_status = BudgetStatus.CRITICAL
        elif previous_percent >= 70:
            previous_status = BudgetStatus.WARNING
        else:
            previous_status = BudgetStatus.HEALTHY
        
        # Check if status changed
        if current_status != previous_status:
            alert = {
                "entity_id": allocation.entity_id,
                "entity_type": allocation.entity_type,
                "status": current_status.value,
                "previous_status": previous_status.value,
                "usage_percent": allocation.usage_percent,
                "used_tokens": allocation.used_tokens,
                "allocated_tokens": allocation.allocated_tokens,
                "timestamp": datetime.now().isoformat()
            }
            self.alerts.append(alert)
            
            # Log appropriately
            if current_status == BudgetStatus.EXCEEDED:
                logger.error(f"🚨 BUDGET EXCEEDED: {allocation.entity_id} "
                           f"({allocation.used_tokens:,}/{allocation.allocated_tokens:,} tokens, "
                           f"{allocation.usage_percent:.1f}%)")
            elif current_status == BudgetStatus.CRITICAL:
                logger.warning(f"⚠️  BUDGET CRITICAL: {allocation.entity_id} "
                             f"({allocation.used_tokens:,}/{allocation.allocated_tokens:,} tokens, "
                             f"{allocation.usage_percent:.1f}%)")
            elif current_status == BudgetStatus.WARNING:
                logger.info(f"⚡ BUDGET WARNING: {allocation.entity_id} "
                          f"({allocation.used_tokens:,}/{allocation.allocated_tokens:,} tokens, "
                          f"{allocation.usage_percent:.1f}%)")
            
            return True
        
        return False
    
    def get_allocation(self, entity_id: str) -> Optional[TokenAllocation]:
        """Get allocation for an entity"""
        return self.allocations.get(entity_id)
    
    def get_summary(self) -> BudgetSummary:
        """
        Get summary of all budget allocations
        
        Returns:
            BudgetSummary with complete status
        """
        # Separate allocations by type
        workflow_allocations = [a for a in self.allocations.values() if a.entity_type == "workflow"]
        task_allocations = [a for a in self.allocations.values() if a.entity_type == "task"]
        subtask_allocations = [a for a in self.allocations.values() if a.entity_type == "subtask"]
        
        # Calculate totals from workflow allocations (to avoid double counting)
        if workflow_allocations:
            total_allocated = sum(a.allocated_tokens for a in workflow_allocations)
            total_used = sum(a.used_tokens for a in workflow_allocations)
            total_reserved = sum(a.reserved_tokens for a in workflow_allocations)
        else:
            total_allocated = sum(a.allocated_tokens for a in self.allocations.values())
            total_used = sum(a.used_tokens for a in self.allocations.values())
            total_reserved = sum(a.reserved_tokens for a in self.allocations.values())
        
        total_available = total_allocated - total_used - total_reserved
        usage_percent = (total_used / total_allocated * 100) if total_allocated > 0 else 0
        
        # Determine overall status
        if usage_percent >= 100:
            status = BudgetStatus.EXCEEDED
        elif usage_percent >= 90:
            status = BudgetStatus.CRITICAL
        elif usage_percent >= 70:
            status = BudgetStatus.WARNING
        else:
            status = BudgetStatus.HEALTHY
        
        # Estimate cost
        estimated_cost = self.config.estimate_cost(total_used)
        
        return BudgetSummary(
            total_allocated=total_allocated,
            total_used=total_used,
            total_reserved=total_reserved,
            total_available=total_available,
            usage_percent=usage_percent,
            status=status,
            workflow_allocations=workflow_allocations,
            task_allocations=task_allocations,
            subtask_allocations=subtask_allocations,
            estimated_cost=estimated_cost
        )
    
    def recommend_budget(
        self,
        num_subtasks: int,
        complexity_score: float,
        user_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Recommend token budget based on task characteristics
        
        Args:
            num_subtasks: Number of subtasks
            complexity_score: Complexity score (0.0-1.0)
            user_constraints: Optional user constraints
        
        Returns:
            Budget recommendations
        """
        if user_constraints is None:
            user_constraints = self.config.get_user_constraints()
        
        # Base tokens per subtask (scaled by complexity)
        base_tokens_per_subtask = int(
            self.config.TOKEN_BUDGET_MAX_SUBTASK * (0.5 + complexity_score * 0.5)
        )
        
        # Total for subtasks
        subtasks_total = base_tokens_per_subtask * num_subtasks
        
        # Reserve for aggregation/coordination (10-20% depending on complexity)
        reserve_percent = 10 + int(complexity_score * 10)
        reserve_tokens = int(subtasks_total * (reserve_percent / 100))
        
        # Total recommended
        total_recommended = subtasks_total + reserve_tokens
        
        # Check against constraints
        max_tokens = user_constraints.get('max_tokens', self.config.TOKEN_BUDGET_DEFAULT)
        if total_recommended > max_tokens:
            # Scale down
            scale_factor = max_tokens / total_recommended
            total_recommended = max_tokens
            subtasks_total = int(subtasks_total * scale_factor)
            base_tokens_per_subtask = subtasks_total // num_subtasks if num_subtasks > 0 else subtasks_total
            reserve_tokens = total_recommended - subtasks_total
        
        # Estimate cost
        estimated_cost = self.config.estimate_cost(total_recommended)
        max_cost = user_constraints.get('max_cost', self.config.MAX_COST_USD)
        
        within_budget = estimated_cost <= max_cost
        
        return {
            "total_tokens": total_recommended,
            "tokens_per_subtask": base_tokens_per_subtask,
            "num_subtasks": num_subtasks,
            "reserve_tokens": reserve_tokens,
            "reserve_percent": reserve_percent,
            "estimated_cost": estimated_cost,
            "max_cost": max_cost,
            "within_budget": within_budget,
            "recommendations": {
                "workflow": total_recommended,
                "per_subtask": base_tokens_per_subtask,
                "reserve": reserve_tokens
            }
        }
    
    def can_afford(self, entity_id: str, tokens_needed: int) -> bool:
        """
        Check if an entity can afford a token expenditure
        
        Args:
            entity_id: Entity identifier
            tokens_needed: Tokens needed
        
        Returns:
            True if within budget
        """
        allocation = self.allocations.get(entity_id)
        if not allocation:
            return False
        
        return allocation.available_tokens >= tokens_needed
    
    def get_alerts(self, clear: bool = False) -> List[Dict[str, Any]]:
        """
        Get budget alerts
        
        Args:
            clear: Clear alerts after retrieving
        
        Returns:
            List of alerts
        """
        alerts = self.alerts.copy()
        if clear:
            self.alerts.clear()
        return alerts
    
    def get_usage_history(
        self,
        entity_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get usage history
        
        Args:
            entity_id: Filter by entity (optional)
            limit: Limit number of records (optional)
        
        Returns:
            Usage history records
        """
        history = self.usage_history
        
        if entity_id:
            history = [h for h in history if h['entity_id'] == entity_id]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def reset(self):
        """Reset all allocations and history"""
        self.allocations.clear()
        self.usage_history.clear()
        self.alerts.clear()
        logger.info("Token budget manager reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manager state to dictionary"""
        summary = self.get_summary()
        return {
            "workflow_id": self.workflow_id,
            "summary": summary.to_dict(),
            "allocations": {k: v.to_dict() for k, v in self.allocations.items()},
            "alerts_count": len(self.alerts),
            "usage_records": len(self.usage_history)
        }


# Global manager instance
_global_manager: Optional[TokenBudgetManager] = None


def get_token_budget_manager(workflow_id: Optional[str] = None) -> TokenBudgetManager:
    """
    Get token budget manager instance
    
    Args:
        workflow_id: Optional workflow ID for scoped management
    
    Returns:
        TokenBudgetManager instance
    """
    if workflow_id:
        # Create scoped manager
        return TokenBudgetManager(workflow_id)
    else:
        # Use global manager
        global _global_manager
        if _global_manager is None:
            _global_manager = TokenBudgetManager()
        return _global_manager


# Export
__all__ = [
    'TokenBudgetManager',
    'TokenAllocation',
    'BudgetSummary',
    'BudgetStatus',
    'get_token_budget_manager'
]
