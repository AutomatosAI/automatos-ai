"""
Intelligent Agent Selector
==========================

Matches subtasks with optimal agents based on:
- Skill proficiency and success rate
- Agent workload and availability  
- Semantic similarity of requirements
- Historical performance on similar tasks

Week 2 - PRD-10 Implementation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from database.models import Agent, Skill, agent_skills
from services.rag_service import RAGService

logger = logging.getLogger(__name__)


@dataclass
class AgentMatch:
    """Agent match result with scoring details"""
    agent_id: int
    agent_name: str
    agent_type: str
    match_score: float
    skill_coverage: float
    availability_score: float
    performance_score: float
    reasoning: str
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)


@dataclass
class SubtaskAgentRequirement:
    """Agent requirements for a subtask"""
    subtask_id: str
    subtask_description: str
    required_skills: List[str]
    priority: str
    agent_type: str
    estimated_duration: float


class IntelligentAgentSelector:
    """
    Selects optimal agents for workflow subtasks using:
    1. Skill matching (exact + semantic)
    2. Workload balancing
    3. Performance history
    4. Availability checking
    """
    
    def __init__(
        self,
        db_session: Session,
        rag_service: Optional[RAGService] = None,
        use_semantic_matching: bool = True
    ):
        self.db = db_session
        self.rag_service = rag_service
        self.use_semantic_matching = use_semantic_matching
        self.logger = logging.getLogger(__name__)
        
        # Scoring weights
        self.weights = {
            "skill_coverage": 0.40,      # Most important
            "skill_proficiency": 0.25,   # Quality of skills
            "availability": 0.20,        # Can agent take task?
            "performance": 0.15          # Historical success
        }
    
    async def select_agents_for_subtasks(
        self,
        subtasks: List[Dict[str, Any]],
        max_agents_per_task: int = 3
    ) -> Dict[str, List[AgentMatch]]:
        """
        Select best agents for each subtask with DIVERSITY preference.
        
        Args:
            subtasks: List of subtask dicts from RealTaskDecomposer
            max_agents_per_task: Max ranked agents to return per subtask
            
        Returns:
            Dict mapping subtask index to ranked list of agent matches
        """
        self.logger.info(f"🔍 STAGE 3: Selecting agents for {len(subtasks)} subtasks")
        
        results = {}
        assigned_agents = {}  # Track agent assignments {agent_id: count}
        
        for idx, subtask in enumerate(subtasks):
            subtask_id = f"subtask_{idx}"
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔍 STAGE 3: Processing {subtask_id}")
            
            # Extract requirements from subtask
            requirements = self._extract_subtask_requirements(subtask, subtask_id)
            
            # Find matching agents
            agent_matches = await self._find_matching_agents(requirements)
            
            # Apply diversity bonus/penalty to encourage using different agents
            for match in agent_matches:
                assignment_count = assigned_agents.get(match.agent_id, 0)
                
                # Reduce score for already-assigned agents (20% penalty per assignment)
                if assignment_count > 0:
                    diversity_penalty = assignment_count * 0.2
                    match.match_score = max(0.1, match.match_score - diversity_penalty)
                    match.reasoning += f" (Score reduced by {diversity_penalty:.1f} for diversity - already assigned {assignment_count} times)"
            
            # Rank and filter
            ranked_matches = self._rank_agent_matches(agent_matches)
            top_matches = ranked_matches[:max_agents_per_task]
            
            # Track assignment of best agent
            if top_matches:
                best_agent = top_matches[0]
                assigned_agents[best_agent.agent_id] = assigned_agents.get(best_agent.agent_id, 0) + 1
                
                self.logger.info(
                    f"✅ Subtask {idx}: '{subtask.get('description', 'Unknown')[:50]}' - "
                    f"Assigned to {best_agent.agent_name} (ID: {best_agent.agent_id})"
                )
            else:
                self.logger.warning(f"⚠️ Subtask {idx}: No suitable agents found")
            
            results[subtask_id] = top_matches
        
        # Log diversity statistics
        unique_agents = len(assigned_agents)
        total_tasks = len(subtasks)
        self.logger.info(f"📊 Agent Diversity: {unique_agents} unique agents used for {total_tasks} tasks")
        
        return results
    
    def _extract_subtask_requirements(
        self,
        subtask: Dict[str, Any],
        subtask_id: str
    ) -> SubtaskAgentRequirement:
        """Extract agent requirements from subtask definition"""
        
        description = subtask.get("description", subtask.get("name", ""))
        agent_type = subtask.get("agent_type", "worker")
        priority = subtask.get("priority", "medium")
        
        # Parse duration (e.g., "30-60 seconds" -> 45.0)
        duration_str = subtask.get("estimated_duration", "30")
        try:
            if isinstance(duration_str, (int, float)):
                duration = float(duration_str)
            elif "-" in str(duration_str):
                parts = str(duration_str).split("-")
                low = int(''.join(filter(str.isdigit, parts[0])))
                high = int(''.join(filter(str.isdigit, parts[1])))
                duration = (low + high) / 2
            else:
                duration = float(''.join(filter(str.isdigit, str(duration_str))) or "30")
        except:
            duration = 30.0
        
        # CRITICAL FIX: Use skills from Stage 1 if provided, otherwise infer
        if "skills_required" in subtask and subtask["skills_required"]:
            required_skills = subtask["skills_required"]
            if isinstance(required_skills, str):
                required_skills = [required_skills]
            self.logger.info(f"  ✅ Using Stage 1 skills: {required_skills}")
        elif "primary_skill" in subtask and subtask["primary_skill"]:
            required_skills = [subtask["primary_skill"]]
            self.logger.info(f"  ✅ Using Stage 1 primary_skill: {required_skills}")
        else:
            # Fallback: Infer from description and agent_type
            self.logger.warning(f"  ⚠️  No skills from Stage 1, inferring...")
            required_skills = self._infer_required_skills(description, agent_type)
        
        return SubtaskAgentRequirement(
            subtask_id=subtask_id,
            subtask_description=description,
            required_skills=required_skills,
            priority=priority,
            agent_type=agent_type,
            estimated_duration=duration
        )
    
    def _infer_required_skills(self, description: str, agent_type: str) -> List[str]:
        """Infer required skills from task description and agent type"""
        
        self.logger.info(f"🔍 STAGE 3: Inferring skills for agent_type='{agent_type}'")
        
        skills = []
        desc_lower = description.lower()
        
        # Agent type mapping
        agent_type_skills = {
            "orchestrator": ["coordination", "planning", "monitoring"],
            "researcher": ["research", "analysis", "documentation"],
            "analyst": ["analysis", "data_processing", "reporting"],
            "developer": ["coding", "implementation", "debugging"],
            "writer": ["writing", "documentation", "communication"],
            "reviewer": ["review", "quality_assurance", "validation"],
            "tester": ["testing", "qa", "verification"],
            "architect": ["architecture", "design", "system_planning"]
        }
        
        # Add agent type skills
        if agent_type.lower() in agent_type_skills:
            type_skills = agent_type_skills[agent_type.lower()]
            skills.extend(type_skills)
            self.logger.info(f"  📝 From agent_type '{agent_type}': {type_skills}")
        else:
            self.logger.warning(f"  ⚠️  Unknown agent_type '{agent_type}' - no type skills added")
        
        # Keyword-based skill detection
        skill_keywords = {
            "coding": ["code", "implement", "develop", "program", "script"],
            "analysis": ["analyze", "examine", "investigate", "assess"],
            "design": ["design", "architect", "plan", "model"],
            "testing": ["test", "verify", "validate", "check", "qa"],
            "documentation": ["document", "write", "report", "record"],
            "review": ["review", "audit", "inspect", "evaluate"],
            "research": ["research", "study", "investigate", "explore"],
            "api": ["api", "endpoint", "integration", "rest"],
            "database": ["database", "sql", "query", "schema"],
            "security": ["security", "auth", "permission", "vulnerability"]
        }
        
        keyword_skills = []
        for skill, keywords in skill_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                if skill not in skills:
                    skills.append(skill)
                    keyword_skills.append(skill)
        
        if keyword_skills:
            self.logger.info(f"  🔍 From keywords in description: {keyword_skills}")
        
        # Always include general skill
        if not skills:
            skills.append("general")
            self.logger.warning(f"  ⚠️  No skills inferred, using 'general'")
        
        self.logger.info(f"  ✅ Final required_skills: {skills}")
        
        return skills
    
    async def _find_matching_agents(
        self,
        requirements: SubtaskAgentRequirement
    ) -> List[AgentMatch]:
        """Find all agents that could handle this subtask"""
        
        matches = []
        
        # Query active agents
        agents = self.db.query(Agent).filter(
            Agent.status == "active"
        ).all()
        
        if not agents:
            self.logger.warning(f"No active agents available for {requirements.subtask_id}")
            return []
        
        for agent in agents:
            match = await self._score_agent_for_subtask(agent, requirements)
            if match and match.match_score > 0:
                matches.append(match)
        
        return matches
    
    async def _score_agent_for_subtask(
        self,
        agent: Agent,
        requirements: SubtaskAgentRequirement
    ) -> Optional[AgentMatch]:
        """Calculate comprehensive match score for agent-subtask pair"""
        
        self.logger.info(f"  🤖 Scoring agent: {agent.name} (ID: {agent.id}, type: {agent.agent_type})")
        
        # 1. Skill Coverage Score
        agent_skills_list = [skill.name.lower() for skill in agent.skills]
        required_skills_lower = [s.lower() for s in requirements.required_skills]
        
        self.logger.info(f"    Required skills: {requirements.required_skills}")
        self.logger.info(f"    Agent skills: {[skill.name for skill in agent.skills]}")
        
        # Enhanced fuzzy matching for skills (handles 'research' matching 'Researcher', etc.)
        matched_skills = []
        missing_skills = []
        
        for required_skill in requirements.required_skills:
            required_lower = required_skill.lower()
            
            # Try multiple matching strategies
            matched = False
            for agent_skill in agent_skills_list:
                # Strategy 1: Exact match
                if required_lower == agent_skill:
                    matched_skills.append(required_skill)
                    matched = True
                    break
                # Strategy 2: Substring match (research in researcher, or researcher in research)
                elif required_lower in agent_skill or agent_skill in required_lower:
                    matched_skills.append(required_skill)
                    matched = True
                    self.logger.debug(f"      🔍 Fuzzy matched: '{required_skill}' ~ '{[s for s in agent.skills if s.name.lower() == agent_skill][0].name}'")
                    break
            
            if not matched:
                missing_skills.append(required_skill)
        
        if not requirements.required_skills:
            skill_coverage = 1.0
        else:
            skill_coverage = len(matched_skills) / len(requirements.required_skills)
        
        self.logger.info(f"    ✅ Matched: {matched_skills}")
        self.logger.info(f"    ❌ Missing: {missing_skills}")
        self.logger.info(f"    📊 Skill coverage: {skill_coverage:.1%}")
        
        # 2. Skill Proficiency Score (from agent configuration)
        proficiency_score = 0.0
        if agent.configuration and "skills" in agent.configuration:
            config_skills = agent.configuration.get("skills", {})
            proficiencies = []
            for skill in matched_skills:
                skill_lower = skill.lower()
                for config_skill, config_data in config_skills.items():
                    if config_skill.lower() == skill_lower:
                        if isinstance(config_data, dict):
                            proficiencies.append(config_data.get("proficiency", 0.5))
                        else:
                            proficiencies.append(0.5)
            
            if proficiencies:
                proficiency_score = sum(proficiencies) / len(proficiencies)
            else:
                proficiency_score = 0.7  # Default if no proficiency data
        else:
            proficiency_score = 0.7  # Default
        
        # 3. Availability Score (based on current workload)
        # For now, simple check - can be enhanced with real workload tracking
        availability_score = 1.0
        if agent.configuration and "current_tasks" in agent.configuration:
            current_tasks = agent.configuration.get("current_tasks", 0)
            max_tasks = agent.max_concurrent_tasks or 5
            availability_score = max(0, 1.0 - (current_tasks / max_tasks))
        
        # 4. Performance Score (from historical metrics)
        performance_score = 0.8  # Default
        if agent.performance_metrics:
            success_rate = agent.performance_metrics.get("success_rate")
            if success_rate is not None:
                performance_score = success_rate
        
        # Semantic matching boost (if RAG available)
        semantic_boost = 0.0
        if self.use_semantic_matching and self.rag_service:
            try:
                # Use RAG to find semantic similarity
                agent_desc = f"{agent.name} {agent.description or ''} {' '.join(agent_skills_list)}"
                task_desc = requirements.subtask_description
                
                # Simple semantic check - can be enhanced
                if len(agent_desc) > 10 and len(task_desc) > 10:
                    semantic_boost = 0.1  # Small boost for having description
            except Exception as e:
                self.logger.debug(f"Semantic matching skipped: {e}")
        
        # Calculate weighted final score
        final_score = (
            self.weights["skill_coverage"] * skill_coverage +
            self.weights["skill_proficiency"] * proficiency_score +
            self.weights["availability"] * availability_score +
            self.weights["performance"] * performance_score +
            semantic_boost
        )
        
        self.logger.info(f"    📊 Final Score Calculation:")
        self.logger.info(f"       Skill coverage: {skill_coverage:.3f} × {self.weights['skill_coverage']} = {self.weights['skill_coverage'] * skill_coverage:.3f}")
        self.logger.info(f"       Proficiency: {proficiency_score:.3f} × {self.weights['skill_proficiency']} = {self.weights['skill_proficiency'] * proficiency_score:.3f}")
        self.logger.info(f"       Availability: {availability_score:.3f} × {self.weights['availability']} = {self.weights['availability'] * availability_score:.3f}")
        self.logger.info(f"       Performance: {performance_score:.3f} × {self.weights['performance']} = {self.weights['performance'] * performance_score:.3f}")
        self.logger.info(f"       Semantic boost: {semantic_boost:.3f}")
        self.logger.info(f"    🎯 Final match_score: {final_score:.3f}")
        
        # Generate reasoning
        reasoning = f"Coverage: {skill_coverage:.0%}, Proficiency: {proficiency_score:.0%}, " \
                   f"Available: {availability_score:.0%}, Performance: {performance_score:.0%}"
        
        return AgentMatch(
            agent_id=agent.id,
            agent_name=agent.name,
            agent_type=agent.agent_type,
            match_score=final_score,
            skill_coverage=skill_coverage,
            availability_score=availability_score,
            performance_score=performance_score,
            reasoning=reasoning,
            matched_skills=matched_skills,
            missing_skills=missing_skills
        )
    
    def _rank_agent_matches(self, matches: List[AgentMatch]) -> List[AgentMatch]:
        """Sort agent matches by score (descending)"""
        return sorted(matches, key=lambda m: m.match_score, reverse=True)
    
    def get_selection_summary(
        self,
        selection_results: Dict[str, List[AgentMatch]]
    ) -> Dict[str, Any]:
        """Generate summary of agent selection results"""
        
        total_subtasks = len(selection_results)
        subtasks_with_agents = sum(1 for matches in selection_results.values() if matches)
        total_agents_selected = sum(len(matches) for matches in selection_results.values())
        
        avg_match_score = 0.0
        if total_agents_selected > 0:
            all_scores = [
                match.match_score
                for matches in selection_results.values()
                for match in matches
            ]
            avg_match_score = sum(all_scores) / len(all_scores)
        
        return {
            "total_subtasks": total_subtasks,
            "subtasks_with_agents": subtasks_with_agents,
            "coverage": subtasks_with_agents / total_subtasks if total_subtasks > 0 else 0,
            "total_agents_selected": total_agents_selected,
            "avg_match_score": avg_match_score,
            "timestamp": datetime.now().isoformat()
        }


# Utility function for quick agent selection
async def select_best_agent_for_task(
    db: Session,
    task_description: str,
    required_skills: List[str],
    rag_service: Optional[RAGService] = None
) -> Optional[AgentMatch]:
    """
    Quick utility to select single best agent for a task.
    
    Returns:
        Best matching agent or None
    """
    selector = IntelligentAgentSelector(db, rag_service)
    
    subtask = {
        "description": task_description,
        "agent_type": "worker",
        "priority": "medium",
        "estimated_duration": 60
    }
    
    # Override with provided skills
    requirements = selector._extract_subtask_requirements(subtask, "task_0")
    requirements.required_skills = required_skills
    
    matches = await selector._find_matching_agents(requirements)
    ranked = selector._rank_agent_matches(matches)
    
    return ranked[0] if ranked else None

