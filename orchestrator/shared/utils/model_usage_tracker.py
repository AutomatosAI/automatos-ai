"""
Model Usage Tracker (PRD-15)
=============================

Utilities for tracking LLM model usage, costs, and statistics
during workflow execution.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from models import Agent, LLMModel

logger = logging.getLogger(__name__)


class ModelUsageTracker:
    """
    Tracks model usage during workflow execution.
    Records input/output tokens, costs, and model information.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.usage_records: List[Dict[str, Any]] = []
    
    def record_usage(
        self,
        agent_id: int,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        execution_time_ms: float = 0
    ) -> Dict[str, Any]:
        """
        Record model usage for an agent execution.
        
        Args:
            agent_id: ID of the agent
            model_id: Model identifier (e.g. 'gpt-4')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            execution_time_ms: Execution time in milliseconds
            
        Returns:
            Usage record with calculated cost
        """
        try:
            # Get model pricing information
            model = self.db.query(LLMModel).filter(
                LLMModel.model_id == model_id
            ).first()
            
            if not model:
                logger.warning(f"Model {model_id} not found in registry, using default pricing")
                input_cost_per_1k = 0.03  # Default GPT-4 pricing
                output_cost_per_1k = 0.06
            else:
                input_cost_per_1k = model.input_cost_per_1k_tokens or 0
                output_cost_per_1k = model.output_cost_per_1k_tokens or 0
            
            # Calculate costs
            input_cost = (input_tokens / 1000) * input_cost_per_1k
            output_cost = (output_tokens / 1000) * output_cost_per_1k
            total_cost = input_cost + output_cost
            
            # Create usage record
            usage_record = {
                "agent_id": agent_id,
                "model_id": model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_cost, 6),
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.usage_records.append(usage_record)
            
            # Update agent's model usage stats
            self._update_agent_stats(agent_id, usage_record)
            
            logger.info(
                f"📊 Model usage recorded: Agent {agent_id} | Model {model_id} | "
                f"Tokens: {input_tokens}→{output_tokens} | Cost: ${total_cost:.6f}"
            )
            
            return usage_record
            
        except Exception as e:
            logger.error(f"Error recording model usage: {e}", exc_info=True)
            # Return basic record even if pricing lookup fails
            return {
                "agent_id": agent_id,
                "model_id": model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "error": str(e)
            }
    
    def _update_agent_stats(self, agent_id: int, usage_record: Dict[str, Any]):
        """Update agent's cumulative model usage statistics"""
        try:
            agent = self.db.query(Agent).filter(Agent.id == agent_id).first()
            if not agent:
                return
            
            # Initialize stats if not present
            if not agent.model_usage_stats:
                agent.model_usage_stats = {
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "total_requests": 0,
                    "avg_tokens_per_request": 0,
                    "last_used_at": None
                }
            
            # Update cumulative stats
            stats = agent.model_usage_stats
            total_tokens = usage_record.get("total_tokens", 0)
            total_cost = usage_record.get("total_cost", 0)
            
            stats["total_tokens"] += total_tokens
            stats["total_cost"] = round(stats["total_cost"] + total_cost, 6)
            stats["total_requests"] += 1
            stats["avg_tokens_per_request"] = int(
                stats["total_tokens"] / stats["total_requests"]
            )
            stats["last_used_at"] = datetime.utcnow().isoformat()
            
            # Use flag_modified to ensure SQLAlchemy tracks the JSON update
            from sqlalchemy.orm import attributes
            attributes.flag_modified(agent, "model_usage_stats")
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating agent stats: {e}", exc_info=True)
            self.db.rollback()
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get summary of all recorded usage in this session.
        
        Returns:
            Summary dict with totals and breakdown
        """
        if not self.usage_records:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "models_used": []
            }
        
        total_tokens = sum(r.get("total_tokens", 0) for r in self.usage_records)
        total_cost = sum(r.get("total_cost", 0) for r in self.usage_records)
        
        # Get unique models
        models_used = list(set(r["model_id"] for r in self.usage_records))
        
        # Model breakdown
        model_breakdown = {}
        for record in self.usage_records:
            model_id = record["model_id"]
            if model_id not in model_breakdown:
                model_breakdown[model_id] = {
                    "requests": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0
                }
            
            model_breakdown[model_id]["requests"] += 1
            model_breakdown[model_id]["total_tokens"] += record.get("total_tokens", 0)
            model_breakdown[model_id]["total_cost"] = round(
                model_breakdown[model_id]["total_cost"] + record.get("total_cost", 0),
                6
            )
        
        return {
            "total_requests": len(self.usage_records),
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "models_used": models_used,
            "model_breakdown": model_breakdown,
            "records": self.usage_records
        }
    
    def get_records(self) -> List[Dict[str, Any]]:
        """Get all usage records"""
        return self.usage_records.copy()

