"""
Intelligent Agent Selector
==========================

Implements Stage 2 of the 9-stage workflow: Agent Selection.
Uses a 4-dimensional scoring system to select the optimal agent for a given subtask.

Scoring Dimensions:
1. Semantic Match (40%): How well agent skills match requirements (Vector Similarity)
2. Performance (30%): Historical success rate and quality scores
3. Workload (20%): Current load balancing (Inverse proportional)
4. Cost Efficiency (10%): Token usage vs budget

"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from core.llm.semantic_skill_matcher import SemanticSkillMatcher
# PHASE 2: Import ContextOptimizer from search module
try:
    from modules.search.optimization.context_optimizer import ContextOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    class ContextOptimizer: pass

logger = logging.getLogger(__name__)

@dataclass
class AgentScore:
    agent_id: int
    agent_name: str
    total_score: float
    breakdown: Dict[str, float]
    match_details: Dict[str, Any]

class IntelligentAgentSelector:
    """
    Selects the best agent for a subtask using multi-dimensional scoring.
    """
    
    def __init__(self, skill_matcher: SemanticSkillMatcher):
        self.skill_matcher = skill_matcher
        self.logger = logging.getLogger(__name__)
        
        # Weights for the 4 dimensions
        self.weights = {
            "semantic": 0.4,
            "performance": 0.3,
            "workload": 0.2,
            "cost": 0.1
        }
        
        # Initialize optimizer if available
        if OPTIMIZER_AVAILABLE:
            self.optimizer = ContextOptimizer()
        else:
            self.optimizer = None
        
    async def select_best_agent(
        self,
        subtask_description: str,
        required_skills: List[str],
        available_agents: List[Dict[str, Any]],
        agent_performance_data: Dict[str, Any],
        current_workloads: Dict[int, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Select the optimal agent for the subtask.
        
        Args:
            subtask_description: Description of the task
            required_skills: List of required skills/tools
            available_agents: List of agent dictionaries
            agent_performance_data: Historical performance metrics
            current_workloads: Dict of agent_id -> current task count
            
        Returns:
            Selected agent dictionary with 'selection_score' and 'selection_reason' added,
            or None if no suitable agent found.
        """
        self.logger.info(f"🎯 STAGE 2: Selecting agent for: {subtask_description[:50]}...")
        
        scored_agents = []
        
        # 1. Calculate scores for all candidates
        for agent in available_agents:
            score = await self._calculate_agent_score(
                agent,
                subtask_description,
                required_skills,
                agent_performance_data.get(agent["name"], {}),
                current_workloads.get(agent["id"], 0)
            )
            
            if score.total_score > 0.0:  # Only consider viable candidates
                scored_agents.append(score)
        
        # 2. Sort by total score
        scored_agents.sort(key=lambda x: x.total_score, reverse=True)
        
        # 3. Select best match
        if not scored_agents:
            self.logger.warning("⚠️  No suitable agent found for subtask")
            return None
            
        best_match = scored_agents[0]
        selected_agent = next(a for a in available_agents if a["id"] == best_match.agent_id)
        
        # Add selection metadata
        selected_agent["selection_score"] = best_match.total_score
        selected_agent["selection_reason"] = self._generate_selection_reason(best_match)
        selected_agent["score_breakdown"] = best_match.breakdown
        
        self.logger.info(
            f"✅ Selected Agent: {selected_agent['name']} "
            f"(Score: {best_match.total_score:.2f})"
        )
        self.logger.debug(f"   Breakdown: {best_match.breakdown}")
        
        return selected_agent

    async def _calculate_agent_score(
        self,
        agent: Dict[str, Any],
        task_description: str,
        required_skills: List[str],
        performance_stats: Dict[str, Any],
        current_workload: int
    ) -> AgentScore:
        """Calculate the 4-dimensional score for a single agent."""
        
        # 1. Semantic Match Score (0.0 - 1.0)
        # We check if the agent has the required skills using semantic matching
        semantic_score = 0.0
        match_details = {}
        
        if required_skills:
            # Check each required skill against agent capabilities
            skill_scores = []
            for skill in required_skills:
                # Combine agent skills and tools for matching
                agent_capabilities = agent.get("skills", []) + agent.get("tools", [])
                
                # Use SemanticSkillMatcher
                is_match, similarity, detail = await self.skill_matcher.match_skill(
                    skill, 
                    agent_capabilities
                )
                skill_scores.append(similarity if is_match else 0.0)
                match_details[skill] = {"match": is_match, "similarity": similarity, "detail": detail}
            
            # Average score across all required skills
            semantic_score = sum(skill_scores) / len(skill_scores) if skill_scores else 0.0
        else:
            # If no specific skills required, match against task description
            # This is a fallback if decomposition didn't list skills
            agent_desc = f"{agent.get('name')} {agent.get('role')} {' '.join(agent.get('skills', []))}"
            semantic_score = await self.skill_matcher.compute_similarity(task_description, agent_desc)
            
        # 2. Performance Score (0.0 - 1.0)
        # Based on historical success rate and quality scores
        hist_success = performance_stats.get("success_rate", 0.8) # Default to 0.8 for new agents
        hist_quality = performance_stats.get("avg_quality", 0.7)
        performance_score = (hist_success * 0.6) + (hist_quality * 0.4)
        
        # 3. Workload Score (0.0 - 1.0)
        # Inverse of current load. Assuming max load of 5 tasks per agent.
        MAX_LOAD = 5
        workload_score = max(0.0, (MAX_LOAD - current_workload) / MAX_LOAD)
        
        # 4. Cost Efficiency Score (0.0 - 1.0)
        # Based on historical token usage vs average. 
        # For now, we'll use a simplified heuristic based on model type if available, or default.
        model = agent.get("model", "gpt-4")
        if "gpt-3.5" in model or "flash" in model:
            cost_score = 0.9
        elif "gpt-4" in model:
            cost_score = 0.6
        else:
            cost_score = 0.7
            
        # PHASE 2: Apply User's Resource Optimization Logic
        if self.optimizer:
            # Calculate optimizations based on total available agents (proxy for resources)
            num_agents = len(performance_stats) if performance_stats else 1
            # resource_limit is abstract here, assuming 1.0 (full capacity)
            opt_metrics = self.optimizer.optimize_agent_resources(num_agents=num_agents, resource_limit=1.0)
            
            # Apply boosts
            performance_score = min(1.0, performance_score + opt_metrics["performance_boost"])
            # Efficiency gain improves cost score
            cost_score = min(1.0, cost_score + opt_metrics["efficiency_gain"] * 0.5)
            
            # Log optimization
            # self.logger.debug(f"   Optimization: Perf +{opt_metrics['performance_boost']:.2f}, Cost +{opt_metrics['efficiency_gain']*0.5:.2f}")
            
        # Calculate Weighted Total
        total_score = (
            (semantic_score * self.weights["semantic"]) +
            (performance_score * self.weights["performance"]) +
            (workload_score * self.weights["workload"]) +
            (cost_score * self.weights["cost"])
        )
        
        return AgentScore(
            agent_id=agent["id"],
            agent_name=agent["name"],
            total_score=total_score,
            breakdown={
                "semantic": semantic_score,
                "performance": performance_score,
                "workload": workload_score,
                "cost": cost_score
            },
            match_details=match_details
        )

    def _generate_selection_reason(self, score: AgentScore) -> str:
        """Generate a human-readable reason for selection."""
        reasons = []
        
        if score.breakdown["semantic"] > 0.8:
            reasons.append("High skill match")
        elif score.breakdown["semantic"] > 0.6:
            reasons.append("Good skill match")
            
        if score.breakdown["performance"] > 0.8:
            reasons.append("Proven track record")
            
        if score.breakdown["workload"] > 0.8:
            reasons.append("Available immediately")
            
        return ", ".join(reasons) if reasons else "Best available option"
