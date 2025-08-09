
"""
Multi-Agent Collaborative Reasoning Engine
==========================================

Implementation of collaborative reasoning for multi-agent systems with:
- Consensus-based decision making
- Conflict resolution mechanisms  
- Reasoning quality assessment
- Multi-agent coordination protocols

Based on the mathematical foundation:
- Score(C) = Σ w_i * Agreement(A_i, A_j) 
- R* = arg min_R Conflict(R, C)
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Import models
from models import Task, Agent
from database import get_db

logger = logging.getLogger(__name__)

class CollaborativeReasoningEngine:
    """
    Advanced collaborative reasoning engine for multi-agent systems
    """
    
    def __init__(self):
        self.reasoning_history = []
        self.agent_performance = {}
        self.consensus_threshold = 0.7
        self.conflict_resolution_strategies = [
            "majority_vote",
            "weighted_consensus", 
            "expert_override",
            "iterative_refinement"
        ]
        
        logger.info("Collaborative Reasoning Engine initialized")
    
    async def collaborative_reasoning(
        self,
        db: Session,
        task_id: int,
        user_id: int,
        agents: List[str],
        reasoning_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform collaborative reasoning across multiple agents
        
        Args:
            db: Database session
            task_id: Task identifier
            user_id: User identifier  
            agents: List of agent identifiers
            reasoning_context: Optional context for reasoning
            
        Returns:
            Dict containing reasoning results and consensus data
        """
        start_time = time.time()
        
        try:
            # Retrieve task
            db_task = db.query(Task).filter(
                and_(Task.id == task_id, Task.owner_id == user_id)
            ).first()
            
            if not db_task:
                raise ValueError(f"Task {task_id} not found")
            
            # Generate individual agent reasoning
            reasoning_outputs = await self._generate_agent_reasoning(
                db, db_task, agents, reasoning_context
            )
            
            # Calculate agreement scores between agents
            agreement_matrix = self._calculate_agreement_matrix(reasoning_outputs)
            
            # Compute consensus score: Score(C) = Σ w_i * Agreement(A_i, A_j)
            consensus_score = self._calculate_consensus_score(
                reasoning_outputs, agreement_matrix
            )
            
            # Identify conflicts and apply resolution
            conflicts = self._identify_conflicts(reasoning_outputs, self.consensus_threshold)
            resolution_result = await self._resolve_conflicts(
                conflicts, reasoning_outputs, agreement_matrix
            )
            
            # Generate final reasoning result
            final_reasoning = {
                "agents": agents,
                "individual_outputs": reasoning_outputs,
                "agreement_matrix": agreement_matrix.tolist(),
                "consensus_score": float(consensus_score),
                "conflicts": conflicts,
                "resolution": resolution_result,
                "final_decision": resolution_result.get("decision"),
                "confidence": resolution_result.get("confidence", consensus_score),
                "reasoning_time": time.time() - start_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store reasoning results in task
            db_task.reasoning = final_reasoning
            db_task.consensus_score = consensus_score
            db.commit()
            
            # Update reasoning history
            self.reasoning_history.append({
                "task_id": task_id,
                "consensus_score": consensus_score,
                "num_agents": len(agents),
                "conflicts_resolved": len(conflicts),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(
                f"Collaborative reasoning completed for task {task_id}: "
                f"consensus={consensus_score:.3f}, conflicts={len(conflicts)}"
            )
            
            return final_reasoning
            
        except Exception as e:
            logger.error(f"Failed collaborative reasoning for task {task_id}: {e}")
            raise
    
    async def _generate_agent_reasoning(
        self,
        db: Session,
        task: Task,
        agents: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate individual reasoning outputs for each agent"""
        
        reasoning_outputs = []
        
        # Create thread pool for parallel agent reasoning
        with ThreadPoolExecutor(max_workers=min(len(agents), 4)) as executor:
            # Submit reasoning tasks for each agent
            future_to_agent = {
                executor.submit(self._agent_reasoning_task, agent, task, context): agent
                for agent in agents
            }
            
            # Collect results
            for future in future_to_agent:
                agent = future_to_agent[future]
                try:
                    reasoning_result = future.result(timeout=30)  # 30 second timeout
                    reasoning_outputs.append(reasoning_result)
                except Exception as e:
                    logger.error(f"Agent {agent} reasoning failed: {e}")
                    # Add fallback reasoning
                    reasoning_outputs.append({
                        "agent": agent,
                        "output": f"Fallback reasoning for {task.description}",
                        "confidence": 0.3,
                        "reasoning_type": "fallback",
                        "error": str(e)
                    })
        
        return reasoning_outputs
    
    def _agent_reasoning_task(
        self,
        agent: str,
        task: Task,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Individual agent reasoning task (simulation for now)"""
        
        # Simulate different agent reasoning approaches
        agent_strategies = {
            "analytical": self._analytical_reasoning,
            "creative": self._creative_reasoning,
            "systematic": self._systematic_reasoning,
            "intuitive": self._intuitive_reasoning
        }
        
        # Determine agent strategy based on agent name/type
        if "analytical" in agent.lower():
            strategy = "analytical"
        elif "creative" in agent.lower():
            strategy = "creative"
        elif "systematic" in agent.lower():
            strategy = "systematic"
        else:
            strategy = "intuitive"
        
        # Execute reasoning strategy
        reasoning_func = agent_strategies.get(strategy, self._analytical_reasoning)
        
        try:
            reasoning_output = reasoning_func(agent, task, context)
            
            # Add performance tracking
            if agent in self.agent_performance:
                self.agent_performance[agent]["reasoning_count"] += 1
            else:
                self.agent_performance[agent] = {"reasoning_count": 1, "success_rate": 1.0}
            
            return reasoning_output
            
        except Exception as e:
            # Update performance tracking
            if agent in self.agent_performance:
                self.agent_performance[agent]["success_rate"] *= 0.9
            
            raise e
    
    def _analytical_reasoning(
        self,
        agent: str,
        task: Task,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analytical reasoning approach"""
        
        # Analyze task complexity
        task_content = f"{task.title} {task.description or ''}"
        complexity_score = min(len(task_content.split()) / 50.0, 1.0)
        
        # Generate analytical output
        reasoning_steps = [
            "1. Problem decomposition and analysis",
            "2. Evidence gathering and evaluation", 
            "3. Logical inference and conclusion",
            f"4. Confidence assessment based on complexity ({complexity_score:.2f})"
        ]
        
        confidence = 0.6 + (0.3 * (1 - complexity_score))  # Higher confidence for simpler tasks
        
        return {
            "agent": agent,
            "output": f"Analytical reasoning for '{task.title}': {'; '.join(reasoning_steps)}",
            "confidence": confidence,
            "reasoning_type": "analytical",
            "complexity_score": complexity_score,
            "steps": reasoning_steps
        }
    
    def _creative_reasoning(
        self,
        agent: str,
        task: Task,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Creative reasoning approach"""
        
        # Generate creative alternatives
        alternatives = [
            "Novel approach through lateral thinking",
            "Unconventional solution pathways",
            "Creative synthesis of existing ideas",
            "Innovation through constraint relaxation"
        ]
        
        # Creativity boost based on task uniqueness
        task_uniqueness = len(set(task.title.lower().split())) / 10.0
        creativity_score = min(task_uniqueness, 1.0)
        
        confidence = 0.5 + (0.4 * creativity_score)
        
        return {
            "agent": agent,
            "output": f"Creative reasoning for '{task.title}': {'; '.join(alternatives)}",
            "confidence": confidence,
            "reasoning_type": "creative",
            "creativity_score": creativity_score,
            "alternatives": alternatives
        }
    
    def _systematic_reasoning(
        self,
        agent: str,
        task: Task,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Systematic reasoning approach"""
        
        # Follow systematic methodology
        methodology = [
            "Requirements analysis",
            "Solution architecture design",
            "Implementation planning",
            "Risk assessment and mitigation",
            "Quality assurance framework"
        ]
        
        # Systematic confidence based on completeness
        completeness_score = min(len(task.description or "") / 200.0, 1.0)
        confidence = 0.7 + (0.2 * completeness_score)
        
        return {
            "agent": agent,
            "output": f"Systematic reasoning for '{task.title}': {'; '.join(methodology)}",
            "confidence": confidence,
            "reasoning_type": "systematic",
            "completeness_score": completeness_score,
            "methodology": methodology
        }
    
    def _intuitive_reasoning(
        self,
        agent: str,
        task: Task,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Intuitive reasoning approach"""
        
        # Generate intuitive insights
        insights = [
            "Pattern recognition from similar cases",
            "Gut feeling based on experience", 
            "Holistic understanding of context",
            "Emergent solution discovery"
        ]
        
        # Intuitive confidence varies with experience
        experience_factor = np.random.uniform(0.4, 0.8)  # Simulate varying experience
        confidence = 0.5 + (0.3 * experience_factor)
        
        return {
            "agent": agent,
            "output": f"Intuitive reasoning for '{task.title}': {'; '.join(insights)}",
            "confidence": confidence,
            "reasoning_type": "intuitive",
            "experience_factor": experience_factor,
            "insights": insights
        }
    
    def _calculate_agreement_matrix(self, reasoning_outputs: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate agreement matrix between agent reasoning outputs"""
        
        n_agents = len(reasoning_outputs)
        agreement_matrix = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    # Calculate agreement based on confidence similarity and reasoning type
                    conf_i = reasoning_outputs[i]["confidence"]
                    conf_j = reasoning_outputs[j]["confidence"]
                    type_i = reasoning_outputs[i]["reasoning_type"]
                    type_j = reasoning_outputs[j]["reasoning_type"]
                    
                    # Confidence agreement (0-1)
                    conf_agreement = 1 - abs(conf_i - conf_j)
                    
                    # Type agreement bonus
                    type_agreement = 0.2 if type_i == type_j else 0.0
                    
                    # Output similarity (simplified text similarity)
                    output_i = reasoning_outputs[i]["output"]
                    output_j = reasoning_outputs[j]["output"]
                    output_agreement = self._text_similarity(output_i, output_j)
                    
                    # Combined agreement score
                    agreement_matrix[i, j] = (
                        0.4 * conf_agreement + 
                        0.3 * output_agreement + 
                        0.3 * (1.0 if type_i == type_j else 0.5)
                    )
        
        return agreement_matrix
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on common words"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_consensus_score(
        self,
        reasoning_outputs: List[Dict[str, Any]],
        agreement_matrix: np.ndarray
    ) -> float:
        """Calculate overall consensus score: Score(C) = Σ w_i * Agreement(A_i, A_j)"""
        
        n_agents = len(reasoning_outputs)
        if n_agents == 0:
            return 0.0
        
        # Weight agents by their confidence
        weights = np.array([output["confidence"] for output in reasoning_outputs])
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Calculate weighted consensus
        consensus_score = 0.0
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    consensus_score += weights[i] * weights[j] * agreement_matrix[i, j]
        
        # Normalize by number of agent pairs
        if n_agents > 1:
            consensus_score = consensus_score / (n_agents * (n_agents - 1))
        
        return consensus_score
    
    def _identify_conflicts(
        self,
        reasoning_outputs: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Identify reasoning conflicts below consensus threshold"""
        
        conflicts = []
        
        for i, output_i in enumerate(reasoning_outputs):
            for j, output_j in enumerate(reasoning_outputs[i+1:], i+1):
                # Check confidence difference
                conf_diff = abs(output_i["confidence"] - output_j["confidence"])
                
                # Check reasoning type compatibility
                type_compatible = output_i["reasoning_type"] == output_j["reasoning_type"]
                
                # Check output similarity
                output_sim = self._text_similarity(output_i["output"], output_j["output"])
                
                # Identify conflict if agreement is below threshold
                if conf_diff > 0.3 or (not type_compatible and output_sim < threshold):
                    conflicts.append({
                        "agents": [output_i["agent"], output_j["agent"]],
                        "confidence_difference": conf_diff,
                        "type_mismatch": not type_compatible,
                        "output_similarity": output_sim,
                        "severity": "high" if conf_diff > 0.5 else "medium"
                    })
        
        return conflicts
    
    async def _resolve_conflicts(
        self,
        conflicts: List[Dict[str, Any]],
        reasoning_outputs: List[Dict[str, Any]],
        agreement_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Resolve conflicts using various strategies: R* = arg min_R Conflict(R, C)"""
        
        if not conflicts:
            # No conflicts - use weighted average
            confidences = [output["confidence"] for output in reasoning_outputs]
            avg_confidence = np.mean(confidences)
            
            # Generate consensus decision
            decision = self._generate_consensus_decision(reasoning_outputs)
            
            return {
                "strategy": "no_conflicts",
                "decision": decision,
                "confidence": avg_confidence,
                "resolution_time": 0.0
            }
        
        # Choose resolution strategy based on conflict severity
        high_severity_conflicts = [c for c in conflicts if c["severity"] == "high"]
        
        if len(high_severity_conflicts) > len(conflicts) / 2:
            strategy = "iterative_refinement"
        elif len(reasoning_outputs) > 3:
            strategy = "majority_vote"
        else:
            strategy = "weighted_consensus"
        
        # Apply selected resolution strategy
        if strategy == "majority_vote":
            return await self._majority_vote_resolution(reasoning_outputs, conflicts)
        elif strategy == "weighted_consensus":
            return await self._weighted_consensus_resolution(reasoning_outputs, agreement_matrix)
        elif strategy == "expert_override":
            return await self._expert_override_resolution(reasoning_outputs, conflicts)
        else:  # iterative_refinement
            return await self._iterative_refinement_resolution(reasoning_outputs, conflicts)
    
    async def _majority_vote_resolution(
        self,
        reasoning_outputs: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Resolve conflicts using majority vote"""
        
        # Group by reasoning type
        type_groups = {}
        for output in reasoning_outputs:
            reasoning_type = output["reasoning_type"]
            if reasoning_type not in type_groups:
                type_groups[reasoning_type] = []
            type_groups[reasoning_type].append(output)
        
        # Find majority type
        majority_type = max(type_groups.keys(), key=lambda k: len(type_groups[k]))
        majority_outputs = type_groups[majority_type]
        
        # Calculate majority consensus
        avg_confidence = np.mean([output["confidence"] for output in majority_outputs])
        decision = self._generate_consensus_decision(majority_outputs)
        
        return {
            "strategy": "majority_vote",
            "decision": decision,
            "confidence": avg_confidence,
            "majority_type": majority_type,
            "majority_size": len(majority_outputs),
            "total_agents": len(reasoning_outputs)
        }
    
    async def _weighted_consensus_resolution(
        self,
        reasoning_outputs: List[Dict[str, Any]],
        agreement_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Resolve conflicts using weighted consensus"""
        
        # Weight by confidence and agreement
        confidences = np.array([output["confidence"] for output in reasoning_outputs])
        agreement_scores = np.mean(agreement_matrix, axis=1)  # Average agreement per agent
        
        # Combined weights
        weights = confidences * agreement_scores
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted confidence
        weighted_confidence = np.sum(weights * confidences)
        
        # Generate weighted decision
        decision = self._generate_weighted_decision(reasoning_outputs, weights)
        
        return {
            "strategy": "weighted_consensus",
            "decision": decision,
            "confidence": weighted_confidence,
            "weights": weights.tolist(),
            "agreement_scores": agreement_scores.tolist()
        }
    
    async def _expert_override_resolution(
        self,
        reasoning_outputs: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Resolve conflicts by deferring to highest confidence agent"""
        
        # Find agent with highest confidence
        best_agent_idx = np.argmax([output["confidence"] for output in reasoning_outputs])
        best_output = reasoning_outputs[best_agent_idx]
        
        return {
            "strategy": "expert_override", 
            "decision": best_output["output"],
            "confidence": best_output["confidence"],
            "expert_agent": best_output["agent"],
            "expert_type": best_output["reasoning_type"]
        }
    
    async def _iterative_refinement_resolution(
        self,
        reasoning_outputs: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Resolve conflicts through iterative refinement"""
        
        # Simulate iterative refinement process
        refined_outputs = []
        
        for output in reasoning_outputs:
            # Adjust confidence based on conflicts involving this agent
            agent_conflicts = [c for c in conflicts if output["agent"] in c["agents"]]
            confidence_penalty = len(agent_conflicts) * 0.1
            
            refined_confidence = max(0.1, output["confidence"] - confidence_penalty)
            
            refined_outputs.append({
                **output,
                "confidence": refined_confidence,
                "refinement_penalty": confidence_penalty
            })
        
        # Generate refined decision
        avg_confidence = np.mean([output["confidence"] for output in refined_outputs])
        decision = self._generate_consensus_decision(refined_outputs)
        
        return {
            "strategy": "iterative_refinement",
            "decision": decision,
            "confidence": avg_confidence,
            "refinement_iterations": 1,
            "refined_outputs": refined_outputs
        }
    
    def _generate_consensus_decision(self, reasoning_outputs: List[Dict[str, Any]]) -> str:
        """Generate consensus decision from multiple reasoning outputs"""
        
        if not reasoning_outputs:
            return "No decision available"
        
        # Combine reasoning types and approaches
        types = [output["reasoning_type"] for output in reasoning_outputs]
        type_counts = {t: types.count(t) for t in set(types)}
        dominant_type = max(type_counts.keys(), key=lambda k: type_counts[k])
        
        # Generate decision based on dominant reasoning type
        if dominant_type == "analytical":
            decision = "Proceed with analytical approach based on systematic analysis"
        elif dominant_type == "creative":
            decision = "Explore creative alternatives and innovative solutions"  
        elif dominant_type == "systematic":
            decision = "Follow systematic methodology with structured implementation"
        else:  # intuitive
            decision = "Trust intuitive insights while validating with additional data"
        
        # Add confidence qualifier
        avg_confidence = np.mean([output["confidence"] for output in reasoning_outputs])
        if avg_confidence > 0.8:
            qualifier = "with high confidence"
        elif avg_confidence > 0.6:
            qualifier = "with moderate confidence"
        else:
            qualifier = "with cautious approach"
        
        return f"{decision} ({qualifier})"
    
    def _generate_weighted_decision(
        self,
        reasoning_outputs: List[Dict[str, Any]],
        weights: np.ndarray
    ) -> str:
        """Generate decision using weighted combination of reasoning outputs"""
        
        # Find outputs with highest weights
        top_indices = np.argsort(weights)[-2:]  # Top 2 weighted outputs
        top_outputs = [reasoning_outputs[i] for i in top_indices]
        
        # Combine top reasoning approaches
        decision_parts = []
        for i, output in enumerate(top_outputs):
            weight = weights[top_indices[i]]
            if weight > 0.3:  # Only include significant contributions
                decision_parts.append(
                    f"{output['reasoning_type']} approach (weight: {weight:.2f})"
                )
        
        if decision_parts:
            decision = f"Combined strategy using: {', '.join(decision_parts)}"
        else:
            decision = self._generate_consensus_decision(reasoning_outputs)
        
        return decision
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reasoning statistics"""
        
        if not self.reasoning_history:
            return {"message": "No reasoning history available"}
        
        # Aggregate statistics
        total_sessions = len(self.reasoning_history)
        avg_consensus = np.mean([r["consensus_score"] for r in self.reasoning_history])
        avg_agents = np.mean([r["num_agents"] for r in self.reasoning_history])
        total_conflicts = sum([r["conflicts_resolved"] for r in self.reasoning_history])
        
        # Agent performance statistics
        agent_stats = {
            agent: {
                "reasoning_count": stats["reasoning_count"],
                "success_rate": stats["success_rate"]
            }
            for agent, stats in self.agent_performance.items()
        }
        
        return {
            "total_reasoning_sessions": total_sessions,
            "average_consensus_score": avg_consensus,
            "average_agents_per_session": avg_agents,
            "total_conflicts_resolved": total_conflicts,
            "agent_performance": agent_stats,
            "consensus_threshold": self.consensus_threshold,
            "available_strategies": self.conflict_resolution_strategies
        }
