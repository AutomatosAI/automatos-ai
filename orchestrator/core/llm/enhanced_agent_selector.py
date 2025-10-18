"""
Enhanced LLM Agent Selector - Intelligent task grouping and agent assignment
====

IMPROVED VERSION with:
1. Validation of multi-agent constraint
2. Intelligent fallback distribution
3. Better error handling

Uses LLM reasoning to:
1. Analyze if multiple tasks can be handled by one agent
2. Select the most appropriate agent based on actual requirements
3. Consider agent workload and efficiency
4. ENFORCE multi-agent distribution for workflows with 5+ tasks
"""

import json
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from core.llm import OrchestratorLLM
from database.models import Agent
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class GroupedAgentAssignment:
    """Agent assignment for a group of tasks"""
    agent_id: int
    agent_name: str
    agent_type: str
    task_ids: List[str]
    reasoning: str
    confidence: float
    efficiency_gain: str


class EnhancedLLMAgentSelector:
    """
    LLM-based agent selection with intelligent task grouping.
    NOW WITH VALIDATION to enforce multi-agent distribution.
    """
    
    def __init__(self, db_session: Session, llm: Optional[OrchestratorLLM] = None):
        self.db = db_session
        self.llm = llm or OrchestratorLLM(temperature=0.3)  # Lower temp for better reasoning
        self.logger = logger
        
    async def select_agents_with_grouping(
        self,
        subtasks: List[Dict[str, Any]],
        workflow_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List]:
        """
        Use LLM to intelligently group tasks and assign agents.
        IMPROVED: Now validates multi-agent constraint.
        """
        self.logger.info("🔍 STARTING LLM AGENT SELECTION (IMPROVED)")
        self.logger.info(f"  📋 Subtasks to assign: {len(subtasks)}")
        self.logger.info(f"  🎯 Workflow context provided: {workflow_context is not None}")
        
        # Get available agents
        self.logger.info("  🤖 Querying available agents...")
        agents = self.db.query(Agent).filter(Agent.status == "active").all()
        self.logger.info(f"  ✅ Found {len(agents)} active agents")
        
        if not agents:
            self.logger.error("  ❌ NO ACTIVE AGENTS FOUND IN DATABASE!")
            return {}
        
        # Log agent details
        for agent in agents[:5]:  # Show first 5
            self.logger.debug(f"    Agent: {agent.name} (ID: {agent.id}, Type: {agent.agent_type})")
        
        agent_info = self._format_agent_info(agents)
        self.logger.info(f"  📝 Formatted agent info: {len(agent_info)} chars")
        
        # Build the reasoning prompt (STRENGTHENED)
        self.logger.info("  🔨 Building LLM prompt...")
        prompt = self._build_grouping_prompt(subtasks, agent_info, workflow_context)
        prompt_str = str(prompt)
        self.logger.info(f"  ✅ Prompt built: {len(prompt_str)} chars")
        self.logger.debug(f"  First 500 chars of prompt: {prompt_str[:500]}...")
        
        # Get LLM decision
        self.logger.info("  🧠 Calling LLM for intelligent selection...")
        system_context = """You are an expert workflow orchestrator. Your goal is to:
1. ALWAYS use multiple agents for workflows with 5+ tasks
2. Group related tasks that can be handled by the same agent
3. Match agents to tasks based on their actual skills and expertise
4. Consider efficiency and avoid unnecessary handoffs between agents
5. CRITICAL: Single-agent solutions will be REJECTED"""
        
        try:
            self.logger.info("  ⏳ Awaiting LLM response...")
            
            # Add timeout to prevent infinite hanging
            import asyncio
            try:
                response = await asyncio.wait_for(
                    self.llm.generate_with_reasoning(
                        prompt=prompt + "\n\n**CRITICAL: Your response MUST be valid JSON only. Do NOT include any text before or after the JSON. Start with { and end with }.**",
                        context={
                            "system_instructions": system_context,
                            "response_format": "json"
                        }
                    ),
                    timeout=30.0  # 30 second timeout
                )
                self.logger.info(f"  ✅ LLM responded! Response type: {type(response)}")
                self.logger.info(f"  📊 Response has content: {hasattr(response, 'content')}")
                if hasattr(response, 'content'):
                    content_str = str(response.content)
                    self.logger.info(f"  📏 Response content length: {len(content_str)} chars")
                    self.logger.debug(f"  First 200 chars: {content_str[:200]}...")
            except asyncio.TimeoutError:
                self.logger.error(f"  ❌ LLM CALL TIMED OUT after 30 seconds!")
                return self._improved_fallback_assignment(subtasks, agents)
            
        except Exception as e:
            self.logger.error(f"  ❌ LLM CALL FAILED: {type(e).__name__}: {str(e)}")
            self.logger.error(f"  📍 Full traceback:", exc_info=True)
            return self._improved_fallback_assignment(subtasks, agents)
        
        # Parse the response WITH VALIDATION
        self.logger.info("  🔄 Parsing and validating LLM response...")
        assignments = self._parse_and_validate_llm_response(response, subtasks, agents)
        self.logger.info(f"  ✅ Parsing complete: {len(assignments)} assignments made")
        
        # Log assignment summary
        unique_agents = set()
        for subtask_id, matches in assignments.items():
            if matches:
                for match in matches:
                    unique_agents.add(match.agent_id)
                    self.logger.debug(f"    {subtask_id} → Agent {match.agent_name}")
        
        self.logger.info(f"  📊 FINAL SUMMARY: {len(unique_agents)} unique agents for {len(subtasks)} tasks")
        
        return assignments
        
    def _format_agent_info(self, agents: List[Agent]) -> str:
        """Format agent information for the LLM"""
        agent_lines = []
        for agent in agents:
            skills = [s.name for s in agent.skills] if agent.skills else []
            perf = agent.performance_metrics.get("success_rate", 0) if agent.performance_metrics else 0
            agent_lines.append(
                f"- {agent.name} (ID: {agent.id}, Type: {agent.agent_type}): "
                f"Skills: {skills}, Success Rate: {perf:.0%}"
            )
        return "\n".join(agent_lines)
    
    def _build_grouping_prompt(
        self,
        subtasks: List[Dict[str, Any]],
        agent_info: str,
        workflow_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build the prompt for intelligent agent selection (STRENGTHENED)"""
        
        # Format subtasks
        task_descriptions = []
        for i, task in enumerate(subtasks):
            task_descriptions.append(
                f"{i+1}. {task.get('description', 'No description')}"
                f" (Priority: {task.get('priority', 'medium')}, "
                f"Est. time: {task.get('estimated_duration', 'unknown')})"
            )
        
        workflow_desc = workflow_context.get("description", "No description") if workflow_context else "No context"
        
        # STRENGTHENED PROMPT with clearer multi-agent requirement
        prompt = f"""
WORKFLOW TASK: {workflow_desc}

SUBTASKS TO ASSIGN:
{chr(10).join(task_descriptions)}

AVAILABLE AGENTS:
{agent_info}

🚨 CRITICAL REQUIREMENT 🚨
{'=' * 60}
For workflows with 5 or more tasks, you MUST:
1. Use AT LEAST 2 DIFFERENT agents
2. Distribute tasks across multiple specialized agents
3. Create 2-4 task groups (NOT one big group)

Single-agent solutions will be AUTOMATICALLY REJECTED.
{'=' * 60}

INTELLIGENT ANALYSIS REQUIRED:

1. TASK RELATIONSHIP ANALYSIS:
   Examine the subtasks and identify natural groupings based on:
   - Sequential dependencies (tasks that build on each other)
   - Shared context (tasks working with the same information)
   - Similar skill requirements (tasks needing similar expertise)
   - Logical workflow phases (preparation → execution → finalization)

2. AGENT CAPABILITY MATCHING:
   For each task group, select agents based on:
   - Core competencies that align with task requirements
   - Proven success rates in similar work
   - Ability to handle the full scope of grouped tasks
   - Avoid over-specialization when a generalist would suffice

3. OPTIMIZATION PRINCIPLES:
   - **MANDATORY**: Use AT LEAST 2 different agents for {len(subtasks)} tasks
   - Match tasks to agents by their SPECIALIZED skills
   - Create 2-4 task groups - DO NOT put all tasks in one group
   - Each agent handles tasks matching their CORE expertise
   - Balance workload: 2-4 tasks per agent maximum

EXAMPLE GOOD DISTRIBUTION (for 7 tasks):
✅ Group 1: Tasks 0,1,6 → Agent: Technical Writer Pro (documentation tasks)
✅ Group 2: Tasks 2,3 → Agent: Code Analyzer (analysis tasks)
✅ Group 3: Tasks 4,5 → Agent: Researcher (research tasks)
Result: 3 agents, balanced workload

EXAMPLE BAD DISTRIBUTION (for 7 tasks):
❌ Group 1: Tasks 0,1,2,3,4,5,6 → Agent: Technical Writer Pro
Result: 1 agent doing everything - REJECTED!

OUTPUT FORMAT (JSON):
{{
  "reasoning": "Your analysis of task relationships and agent selection logic. EXPLAIN why you chose multiple agents.",
  "task_groups": [
    {{
      "group_name": "Research and Documentation",
      "task_indices": [0, 1, 6],
      "selected_agent": {{
        "id": 96,
        "name": "Technical Writer Pro",
        "reasoning": "Specialized in documentation tasks"
      }}
    }},
    {{
      "group_name": "Technical Analysis",
      "task_indices": [2, 3],
      "selected_agent": {{
        "id": 52,
        "name": "Code Analyzer Agent",
        "reasoning": "Best for analyzing architecture"
      }}
    }}
  ],
  "efficiency_metrics": {{
    "total_agents": 2,
    "tasks_per_agent": 3.5,
    "estimated_handoffs": 1
  }}
}}

⚠️  REMINDER: You MUST use at least 2 agents for {len(subtasks)} tasks. Single-agent solutions will be rejected.

Provide your analysis and selection:"""
        
        return prompt
    
    def _parse_and_validate_llm_response(
        self,
        response,  # LLMResponse object
        subtasks: List[Dict[str, Any]],
        agents: List[Agent]
    ) -> Dict[str, List]:
        """
        Parse LLM response AND validate multi-agent constraint.
        NEW: Rejects single-agent solutions for large workflows.
        """
        
        self.logger.info("  🔍 Starting response parsing...")
        
        try:
            # Get content from LLMResponse object
            if hasattr(response, 'content'):
                content = response.content
                self.logger.info(f"    Response.content type: {type(content)}")
            else:
                content = str(response)
                self.logger.info(f"    Response (no .content attr) type: {type(content)}")
            
            # CRITICAL: Handle both dict and str types
            if isinstance(content, dict):
                # Already parsed by orchestrator_llm's _parse_reasoning_response
                data = content
                self.logger.info("    ✅ Using pre-parsed dict as data")
                self.logger.info(f"    Dict keys: {list(data.keys())}")
            else:
                # It's a string, need to extract and parse JSON
                response_text = str(content)
                self.logger.info(f"    Extracted text length: {len(response_text)} chars")
                
                # Extract JSON from response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                self.logger.info(f"    JSON boundaries: start={json_start}, end={json_end}")
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    self.logger.info(f"    Attempting to parse JSON: {len(json_str)} chars")
                    data = json.loads(json_str)
                    self.logger.info(f"    ✅ JSON parsed successfully! Keys: {list(data.keys())}")
                else:
                    raise ValueError("No JSON found in response")
            
            # Convert to assignment format
            assignments = {}
            
            for group in data.get("task_groups", []):
                agent_info = group.get("selected_agent", {})
                agent_id = agent_info.get("id")
                
                if not agent_id:
                    continue
                
                # Find the agent
                agent = next((a for a in agents if a.id == agent_id), None)
                if not agent:
                    continue
                
                # Assign to all tasks in the group
                for task_idx in group.get("task_indices", []):
                    if task_idx < len(subtasks):
                        subtask_id = f"subtask_{task_idx}"
                        
                        # Create assignment
                        from core.intelligent_agent_selector import AgentMatch
                        match = AgentMatch(
                            agent_id=agent.id,
                            agent_name=agent.name,
                            agent_type=agent.agent_type,
                            match_score=0.9,  # High score for LLM selection
                            skill_coverage=0.85,
                            availability_score=1.0,
                            performance_score=0.85,
                            reasoning=agent_info.get("reasoning", "LLM selected"),
                            matched_skills=[],
                            missing_skills=[]
                        )
                        
                        assignments[subtask_id] = [match]
            
            # Check for unassigned tasks and assign them to a fallback agent
            assigned_indices = set()
            for group in data.get("task_groups", []):
                assigned_indices.update(group.get("task_indices", []))
            
            unassigned = [i for i in range(len(subtasks)) if i not in assigned_indices]
            
            if unassigned:
                self.logger.warning(f"⚠️ LLM missed {len(unassigned)} tasks: {unassigned}")
                # Assign to first available agent as fallback
                fallback_agent = agents[0] if agents else None
                if fallback_agent:
                    from core.intelligent_agent_selector import AgentMatch
                    for idx in unassigned:
                        subtask_id = f"subtask_{idx}"
                        match = AgentMatch(
                            agent_id=fallback_agent.id,
                            agent_name=fallback_agent.name,
                            agent_type=fallback_agent.agent_type,
                            match_score=0.7,
                            skill_coverage=0.7,
                            availability_score=1.0,
                            performance_score=0.75,
                            reasoning="Fallback assignment for unassigned task",
                            matched_skills=[],
                            missing_skills=[]
                        )
                        assignments[subtask_id] = [match]
                        self.logger.info(f"  ✅ Assigned missed task {idx} to {fallback_agent.name}")
            
            # 🆕 VALIDATION: Enforce multi-agent constraint for large workflows
            if len(subtasks) >= 5:
                unique_agents = len(set(
                    match.agent_id 
                    for task_assignments in assignments.values() 
                    if task_assignments
                    for match in task_assignments
                ))
                
                if unique_agents < 2:
                    self.logger.warning(
                        f"⚠️ VALIDATION FAILED: LLM used only {unique_agents} agent(s) for {len(subtasks)} tasks"
                    )
                    self.logger.warning("🔄 Rejecting LLM solution, using improved fallback with forced distribution...")
                    return self._improved_fallback_assignment(subtasks, agents)
                else:
                    self.logger.info(f"✅ VALIDATION PASSED: {unique_agents} agents for {len(subtasks)} tasks")
            
            # Log the grouping
            self.logger.info(f"🎯 LLM Agent Selection Complete:")
            self.logger.info(f"  Reasoning: {data.get('reasoning', 'No reasoning provided')}")
            if "efficiency_metrics" in data:
                metrics = data["efficiency_metrics"]
                self.logger.info(f"  Efficiency: {metrics.get('total_agents')} agents for {len(subtasks)} tasks")
                self.logger.info(f"  Tasks per agent: {metrics.get('tasks_per_agent', 0):.1f}")
            self.logger.info(f"  Total assignments created: {len(assignments)}/{len(subtasks)}")
            
            return assignments
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {type(e).__name__}: {str(e)}", exc_info=True)
            
            # Safe preview: always convert to str first
            try:
                if hasattr(response, 'content'):
                    content_preview = str(response.content)[:500]
                else:
                    content_preview = str(response)[:500]
                self.logger.debug(f"Response was: {content_preview}")
            except:
                self.logger.debug("Could not preview response")
            
            # Fallback to improved assignment
            return self._improved_fallback_assignment(subtasks, agents)
    
    def _improved_fallback_assignment(
        self,
        subtasks: List[Dict[str, Any]],
        agents: List[Agent]
    ) -> Dict[str, List]:
        """
        IMPROVED fallback assignment with intelligent multi-agent distribution.
        NEW: Uses task keywords to match agent types and distributes across multiple agents.
        """
        
        self.logger.info("🔄 Using IMPROVED fallback assignment (multi-agent)")
        
        assignments = {}
        
        # Group agents by type/specialization
        researchers = [a for a in agents if 'research' in a.agent_type.lower() or 'research' in a.name.lower()]
        analysts = [a for a in agents if 'analyst' in a.agent_type.lower() or 'analyz' in a.name.lower()]
        writers = [a for a in agents if 'writer' in a.agent_type.lower() or 'writer' in a.name.lower() or 'documentation' in a.agent_type.lower()]
        developers = [a for a in agents if 'developer' in a.agent_type.lower() or 'code' in a.agent_type.lower()]
        reviewers = [a for a in agents if 'review' in a.agent_type.lower() or 'quality' in a.agent_type.lower()]
        
        self.logger.info(f"  Available by type: {len(researchers)} researchers, {len(analysts)} analysts, {len(writers)} writers, {len(developers)} developers, {len(reviewers)} reviewers")
        
        # Distribute tasks by keyword matching
        from core.intelligent_agent_selector import AgentMatch
        
        used_agents = set()
        
        for i, task in enumerate(subtasks):
            desc = task.get('description', '').lower()
            agent = None
            reasoning = ""
            
            # Match task to agent type by keywords
            if any(kw in desc for kw in ['research', 'collect', 'gather', 'find', 'identify', 'locate', 'discover']):
                agent = researchers[0] if researchers else None
                reasoning = "Matched keywords: research/collect/gather"
            elif any(kw in desc for kw in ['analyze', 'extract', 'examine', 'study', 'evaluate', 'assess']):
                agent = analysts[0] if analysts else None
                reasoning = "Matched keywords: analyze/extract/examine"
            elif any(kw in desc for kw in ['write', 'draft', 'document', 'compose', 'create', 'compile']):
                agent = writers[0] if writers else None
                reasoning = "Matched keywords: write/draft/document"
            elif any(kw in desc for kw in ['code', 'implement', 'develop', 'program', 'build']):
                agent = developers[0] if developers else None
                reasoning = "Matched keywords: code/implement/develop"
            elif any(kw in desc for kw in ['review', 'check', 'validate', 'verify', 'proofread']):
                agent = reviewers[0] if reviewers else None
                reasoning = "Matched keywords: review/check/validate"
            
            # If no match, use round-robin distribution
            if not agent:
                agent = agents[i % len(agents)]
                reasoning = f"Round-robin distribution (task {i} % {len(agents)} agents)"
            
            # Track that we've used this agent
            used_agents.add(agent.id)
            
            subtask_id = f"subtask_{i}"
            match = AgentMatch(
                agent_id=agent.id,
                agent_name=agent.name,
                agent_type=agent.agent_type,
                match_score=0.75,
                skill_coverage=0.75,
                availability_score=1.0,
                performance_score=0.75,
                reasoning=f"Fallback: {reasoning}",
                matched_skills=[],
                missing_skills=[]
            )
            assignments[subtask_id] = [match]
            self.logger.debug(f"    Task {i} → {agent.name} ({reasoning})")
        
        self.logger.info(f"✅ Fallback assigned {len(subtasks)} tasks to {len(used_agents)} unique agents")
        if assignments:
            agent_names = [match[0].agent_name for match in list(assignments.values())[:3] if match]
            self.logger.info(f"  Agents used: {', '.join(agent_names)}")
        
        return assignments
