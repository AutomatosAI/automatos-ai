"""
LLM Agent Selector - Reasoning-based agent selection
=====================================================

PRD-16: Stage 3 - LLM-driven agent selection with function calling.
Replaces algorithmic selection with reasoning-based decisions.

This implements the Software 3.0 pattern:
1. LLM analyzes the situation
2. LLM calls functions to gather information  
3. LLM reasons about the information
4. LLM makes an informed decision
5. LLM explains its reasoning
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

# Import existing models and infrastructure
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from database.models import Agent, WorkflowExecution
from core.llm import (
    OrchestratorLLM,
    FunctionRegistry,
    FunctionExecutor,
    FunctionSpec,
    FunctionParameter,
    FunctionCategory
)

logger = logging.getLogger(__name__)


@dataclass
class AgentMatch:
    """Agent match with reasoning"""
    agent_id: int
    agent_name: str
    agent_type: str
    match_score: float
    reasoning: str
    skills_matched: List[str] = field(default_factory=list)
    performance_score: float = 0.0
    availability_score: float = 0.0
    collaboration_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "match_score": self.match_score,
            "reasoning": self.reasoning,
            "skills_matched": self.skills_matched,
            "performance_score": self.performance_score,
            "availability_score": self.availability_score,
            "collaboration_score": self.collaboration_score
        }


@dataclass
class AgentSelectionResult:
    """Complete result of agent selection with reasoning"""
    agent_id: int
    agent_name: str
    reasoning: str
    confidence: float
    alternatives_considered: List[int]
    risk_factors: List[str]
    function_calls_made: List[str]
    selection_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMAgentSelector:
    """
    LLM-driven agent selection with reasoning.
    Reference: Software 3.0 - Function calling as cognitive extension
    """
    
    def __init__(
        self,
        db_session: Session,
        llm: Optional[OrchestratorLLM] = None,
        use_caching: bool = True,
        cache_ttl_seconds: int = 300
    ):
        """
        Initialize LLM agent selector.
        
        Args:
            db_session: Database session
            llm: Orchestrator LLM instance
            use_caching: Whether to cache selections
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.db = db_session
        self.llm = llm or OrchestratorLLM(temperature=0.7)
        self.use_caching = use_caching
        self.cache_ttl = cache_ttl_seconds
        
        # Initialize function registry and executor
        self.registry = FunctionRegistry()
        self.executor = FunctionExecutor(self.registry)
        
        # Register agent selection functions
        self._register_functions()
        
        # Cache for recent selections
        self.selection_cache: Dict[str, AgentSelectionResult] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        logger.info("LLMAgentSelector initialized with function calling support")
    
    def _register_functions(self):
        """Register all functions for agent selection"""
        
        # 1. Query available agents
        self.registry.register(FunctionSpec(
            name="query_available_agents",
            description="Query database for agents matching criteria. Returns agents with their capabilities, current status, and recent performance.",
            category=FunctionCategory.QUERY,
            parameters=[
                FunctionParameter(
                    name="skills",
                    type="array",
                    description="Required skills (e.g., ['research', 'analysis'])",
                    required=True,
                    items={"type": "string"}
                ),
                FunctionParameter(
                    name="min_proficiency",
                    type="number",
                    description="Minimum skill proficiency level",
                    required=False,
                    default=0.6,
                    minimum=0,
                    maximum=1
                ),
                FunctionParameter(
                    name="status",
                    type="string",
                    description="Agent availability status",
                    required=False,
                    default="available",
                    enum=["available", "busy", "offline", "any"]
                ),
                FunctionParameter(
                    name="max_workload",
                    type="number",
                    description="Maximum acceptable current workload",
                    required=False,
                    default=0.8,
                    minimum=0,
                    maximum=1
                ),
                FunctionParameter(
                    name="agent_type",
                    type="string",
                    description="Optional agent type filter",
                    required=False
                )
            ],
            implementation=self._query_agents
        ))
        
        # 2. Get agent performance history
        self.registry.register(FunctionSpec(
            name="get_agent_performance_history",
            description="Get detailed performance metrics for a specific agent on similar tasks. Returns success rate, quality scores, execution times, and recent failures.",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="agent_id",
                    type="number",
                    description="Database ID of the agent",
                    required=True
                ),
                FunctionParameter(
                    name="task_type",
                    type="string",
                    description="Type of task to filter history",
                    required=False
                ),
                FunctionParameter(
                    name="time_window_days",
                    type="number",
                    description="Number of days to look back",
                    required=False,
                    default=30,
                    minimum=1,
                    maximum=90
                ),
                FunctionParameter(
                    name="include_failures",
                    type="boolean",
                    description="Include failed task details",
                    required=False,
                    default=True
                )
            ],
            implementation=self._get_performance_history
        ))
        
        # 3. Analyze task requirements
        self.registry.register(FunctionSpec(
            name="analyze_task_requirements",
            description="Deep analysis of subtask to understand true requirements beyond stated skills. Uses NLP to extract implicit requirements.",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="subtask_description",
                    type="string",
                    description="Full subtask description",
                    required=True
                ),
                FunctionParameter(
                    name="task_context",
                    type="object",
                    description="Additional context about the task",
                    required=False,
                    properties={}
                ),
                FunctionParameter(
                    name="priority",
                    type="string",
                    description="Task priority level",
                    required=False,
                    enum=["high", "medium", "low"]
                )
            ],
            implementation=self._analyze_requirements
        ))
        
        # 4. Check agent availability
        self.registry.register(FunctionSpec(
            name="check_agent_availability",
            description="Check real-time availability and current workload for specific agents. Returns detailed status including current tasks, estimated completion time.",
            category=FunctionCategory.MONITORING,
            parameters=[
                FunctionParameter(
                    name="agent_ids",
                    type="array",
                    description="List of agent IDs to check",
                    required=True,
                    items={"type": "number"}
                ),
                FunctionParameter(
                    name="include_queue",
                    type="boolean",
                    description="Include queued tasks in response",
                    required=False,
                    default=True
                )
            ],
            implementation=self._check_availability
        ))
        
        # 5. Get agent collaboration history
        self.registry.register(FunctionSpec(
            name="get_agent_collaboration_history",
            description="Check if agents have successfully collaborated on past tasks. Returns synergy scores and collaboration patterns.",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="agent_id",
                    type="number",
                    description="Primary agent ID",
                    required=True
                ),
                FunctionParameter(
                    name="potential_collaborators",
                    type="array",
                    description="Other agents in the workflow",
                    required=False,
                    items={"type": "number"}
                )
            ],
            implementation=self._get_collaboration_history
        ))
        
        # 6. Compare agents
        self.registry.register(FunctionSpec(
            name="compare_agents",
            description="Direct comparison of multiple agents across various dimensions. Returns structured comparison data.",
            category=FunctionCategory.ANALYSIS,
            parameters=[
                FunctionParameter(
                    name="agent_ids",
                    type="array",
                    description="Agents to compare",
                    required=True,
                    items={"type": "number"}
                ),
                FunctionParameter(
                    name="comparison_criteria",
                    type="array",
                    description="Criteria to compare",
                    required=False,
                    default=["performance", "reliability", "quality"],
                    items={"type": "string", "enum": ["performance", "reliability", "speed", "quality", "cost"]}
                )
            ],
            implementation=self._compare_agents
        ))
    
    async def select_agent_with_reasoning(
        self,
        subtask: Dict[str, Any],
        workflow_context: Dict[str, Any]
    ) -> AgentSelectionResult:
        """
        LLM reasons through agent selection using functions.
        
        This implements the core Software 3.0 pattern:
        1. LLM analyzes the situation
        2. LLM calls functions to gather information
        3. LLM reasons about the information
        4. LLM makes an informed decision
        5. LLM explains its reasoning
        """
        import asyncio
        start_time = asyncio.get_event_loop().time()
        
        # Check cache first
        cache_key = self._get_cache_key(subtask)
        if self.use_caching and cache_key in self.selection_cache:
            cached = self.selection_cache[cache_key]
            if datetime.now() - self.cache_timestamps[cache_key] < timedelta(seconds=self.cache_ttl):
                logger.info(f"Using cached agent selection for {subtask.get('subtask_id')}")
                return cached
        
        # Build selection prompt
        prompt = self._build_selection_prompt(subtask, workflow_context)
        
        # Get available functions for agent selection
        functions = self.registry.get_for_prompt(
            categories=[
                FunctionCategory.QUERY,
                FunctionCategory.ANALYSIS,
                FunctionCategory.MONITORING
            ]
        )
        
        # Execute LLM reasoning with function calling
        response = await self.llm.generate_with_functions(
            prompt=prompt,
            functions=functions,
            function_executor=self._execute_function,
            max_function_calls=10,
            context={"subtask": subtask, "workflow": workflow_context}
        )
        
        # Parse response
        try:
            # Extract structured response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
            else:
                # Fallback parsing
                result_data = {
                    "selected_agent_id": 0,
                    "selected_agent_name": "Unknown",
                    "reasoning": response.content,
                    "confidence": response.confidence,
                    "alternatives_considered": [],
                    "risk_factors": [],
                    "function_calls_made": [fc["name"] for fc in response.function_calls]
                }
            
            selection_result = AgentSelectionResult(
                agent_id=result_data.get("selected_agent_id", 0),
                agent_name=result_data.get("selected_agent_name", "Unknown"),
                reasoning=result_data.get("reasoning", response.reasoning or ""),
                confidence=float(result_data.get("confidence", response.confidence)),
                alternatives_considered=result_data.get("alternatives_considered", []),
                risk_factors=result_data.get("risk_factors", []),
                function_calls_made=result_data.get("function_calls_made", []),
                selection_time=asyncio.get_event_loop().time() - start_time,
                metadata={
                    "tokens_used": response.tokens_used,
                    "cost": response.cost,
                    "model": self.llm.model
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback to first available agent
            agents = await self._query_agents({"skills": subtask.get("skills_required", [])})
            if agents["matching_agents"]:
                first_agent = agents["matching_agents"][0]
                selection_result = AgentSelectionResult(
                    agent_id=first_agent["agent_id"],
                    agent_name=first_agent["name"],
                    reasoning=f"Fallback selection due to parsing error: {e}",
                    confidence=0.3,
                    alternatives_considered=[],
                    risk_factors=["LLM response parsing failed"],
                    function_calls_made=[],
                    selection_time=asyncio.get_event_loop().time() - start_time
                )
            else:
                raise ValueError("No agents available and LLM parsing failed")
        
        # Log the decision
        logger.info(f"🤖 AGENT SELECTED: {selection_result.agent_name}")
        logger.info(f"  📊 Confidence: {selection_result.confidence:.2%}")
        logger.info(f"  🧠 Reasoning: {selection_result.reasoning[:200]}...")
        logger.info(f"  🔧 Functions used: {', '.join(selection_result.function_calls_made)}")
        
        # Cache the result
        if self.use_caching:
            self.selection_cache[cache_key] = selection_result
            self.cache_timestamps[cache_key] = datetime.now()
        
        return selection_result
    
    def _build_selection_prompt(
        self,
        subtask: Dict[str, Any],
        workflow_context: Dict[str, Any]
    ) -> str:
        """Build the agent selection prompt"""
        
        # Format failed attempts if any
        failed_attempts = workflow_context.get("failed_attempts", [])
        failures_text = ""
        if failed_attempts:
            failures_text = "PAST FAILURES:\n"
            for attempt in failed_attempts[-3:]:  # Show last 3 failures
                failures_text += f"- Agent {attempt.get('agent_name')}: {attempt.get('error')}\n"
        
        prompt = f"""
You are selecting an agent for this subtask in a multi-agent workflow.

SUBTASK DETAILS:
- ID: {subtask.get('subtask_id')}
- Description: {subtask.get('description')}
- Required Skills: {subtask.get('skills_required', [])}
- Agent Type Suggested: {subtask.get('agent_type')}
- Priority: {subtask.get('priority', 'medium')}
- Dependencies: {subtask.get('dependencies', [])}
- Estimated Duration: {subtask.get('estimated_duration', 'unknown')}

WORKFLOW CONTEXT:
- Workflow ID: {workflow_context.get('workflow_id')}
- Total Subtasks: {workflow_context.get('total_subtasks', 0)}
- Completed Subtasks: {len(workflow_context.get('completed_subtasks', []))}
- Failed Attempts: {len(failed_attempts)}
- Time Remaining: {workflow_context.get('time_remaining', 'unlimited')}
- Other Agents Selected: {workflow_context.get('selected_agents', [])}

{failures_text}

YOUR TASK:
Select the OPTIMAL agent for this subtask using the following process:

1. UNDERSTAND THE REQUIREMENTS:
   - Call analyze_task_requirements() to get deep analysis
   - What skills are truly needed?
   - What makes this task unique?
   - Are there implicit requirements?

2. FIND CANDIDATES:
   - Call query_available_agents() to find matching agents
   - Consider both exact and related skills
   - Don't be too restrictive initially

3. EVALUATE PERFORMANCE:
   - For top candidates, call get_agent_performance_history()
   - Look at success rates, quality scores, similar tasks
   - Consider recent performance trends
   - Review failure patterns

4. CHECK AVAILABILITY:
   - Call check_agent_availability() for top candidates
   - Consider current workload
   - Estimate wait time if busy

5. CONSIDER COLLABORATION:
   - Call get_agent_collaboration_history() if applicable
   - Has this agent worked well with others in this workflow?
   - Are there synergy opportunities?

6. COMPARE & DECIDE:
   - If multiple good options, call compare_agents()
   - Weigh: skill match, performance, availability, cost
   - Consider workflow context (failures, time pressure)
   - Think about risk vs. reward

7. MAKE SELECTION:
   - Choose ONE agent
   - Explain your reasoning clearly
   - Rate your confidence (0-1)
   - Note any risks or concerns

DECISION CRITERIA (in order of importance):
1. Skill Match: Does agent have required skills?
2. Reliability: History of success on similar tasks?
3. Quality: Produces high-quality outputs?
4. Availability: Can start soon?
5. Collaboration: Works well with other selected agents?
6. Cost: Reasonable token/time usage?

SPECIAL CONSIDERATIONS:
- If priority=high: Favor reliability over cost
- If previous failures: Avoid agents that failed before
- If time_remaining low: Favor fastest reliable agent
- If complex task: Favor agents with highest quality scores

Think step-by-step. Use functions to gather data. Make an informed decision.

Provide your response in this structure:
{{
    "selected_agent_id": <integer>,
    "selected_agent_name": "<string>",
    "reasoning": "<detailed explanation>",
    "confidence": <0-1>,
    "alternatives_considered": [<agent_ids>],
    "risk_factors": ["<list of concerns if any>"],
    "function_calls_made": [<list of functions you called>]
}}
"""
        return prompt
    
    async def _execute_function(
        self,
        function_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute function calls made by the LLM"""
        logger.info(f"  🔧 LLM calling function: {function_name}({parameters})")
        
        # Get function spec
        spec = self.registry.get(function_name)
        if not spec or not spec.implementation:
            return {"error": f"Function {function_name} not found or not implemented"}
        
        # Execute function
        result = await self.executor.execute(function_name, parameters)
        
        if result.status.value == "success":
            return result.result
        else:
            return {"error": result.error}
    
    async def _query_agents(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query agents from database"""
        skills = parameters.get('skills', [])
        min_proficiency = parameters.get('min_proficiency', 0.6)
        status = parameters.get('status', 'available')
        max_workload = parameters.get('max_workload', 0.8)
        agent_type = parameters.get('agent_type')
        
        # Build query
        query = self.db.query(Agent)
        
        # Filter by status if not 'any'
        if status != 'any':
            query = query.filter(Agent.status == status)
        
        # Filter by agent type if provided
        if agent_type:
            query = query.filter(Agent.agent_type == agent_type)
        
        # Note: workload filtering would require additional field
        # For now, we'll use performance_metrics
        
        agents = query.all()
        
        # Filter by skills (manual because skills are in JSON)
        matching_agents = []
        for agent in agents:
            # Get agent capabilities
            capabilities = agent.configuration or {}
            agent_skills = capabilities.get('skills', [])
            
            # Calculate skill match
            skill_matches = 0
            for required_skill in skills:
                for agent_skill in agent_skills:
                    # Fuzzy match (case-insensitive, substring)
                    if (required_skill.lower() in str(agent_skill).lower() or
                        str(agent_skill).lower() in required_skill.lower()):
                        skill_matches += 1
                        break
            
            skill_coverage = skill_matches / len(skills) if skills else 1.0
            
            if skill_coverage >= min_proficiency:
                # Get performance metrics
                perf_metrics = agent.performance_metrics or {}
                
                matching_agents.append({
                    'agent_id': agent.id,
                    'name': agent.name,
                    'agent_type': agent.agent_type,
                    'skills': agent_skills,
                    'skill_coverage': skill_coverage,
                    'status': agent.status,
                    'current_workload': perf_metrics.get('current_workload', 0.5),
                    'avg_success_rate': perf_metrics.get('success_rate', 0.7),
                    'total_tasks_completed': perf_metrics.get('tasks_completed', 0)
                })
        
        return {
            'matching_agents': matching_agents,
            'total_found': len(matching_agents),
            'query_criteria': {
                'skills': skills,
                'min_proficiency': min_proficiency,
                'status': status,
                'max_workload': max_workload
            }
        }
    
    async def _get_performance_history(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get agent performance history"""
        agent_id = int(parameters['agent_id'])
        time_window_days = parameters.get('time_window_days', 30)
        include_failures = parameters.get('include_failures', True)
        
        # Get agent
        agent = self.db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            return {'error': f'Agent {agent_id} not found'}
        
        # Get recent executions for this agent
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        
        executions = self.db.query(WorkflowExecution).filter(
            WorkflowExecution.agent_id == agent_id,
            WorkflowExecution.created_at >= cutoff_date
        ).all()
        
        # Calculate metrics
        total_tasks = len(executions)
        successful_tasks = sum(1 for e in executions if e.status == 'completed')
        failed_tasks = sum(1 for e in executions if e.status == 'failed')
        
        # Get performance metrics from agent
        perf_metrics = agent.performance_metrics or {}
        
        return {
            'agent_id': agent_id,
            'agent_name': agent.name,
            'time_window_days': time_window_days,
            'metrics': {
                'success_rate': successful_tasks / total_tasks if total_tasks > 0 else perf_metrics.get('success_rate', 0.7),
                'avg_quality_score': perf_metrics.get('quality_score', 0.75),
                'total_tasks': total_tasks,
                'successful_tasks': successful_tasks,
                'failed_tasks': failed_tasks,
                'avg_execution_time_seconds': perf_metrics.get('avg_execution_time', 120),
                'recent_trend': 'stable'  # Would calculate from real data
            },
            'recent_failures': [] if not include_failures else [
                {
                    'execution_id': e.id,
                    'error': e.error_message,
                    'timestamp': e.created_at.isoformat()
                }
                for e in executions if e.status == 'failed'
            ][:5]  # Limit to 5 most recent failures
        }
    
    async def _analyze_requirements(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task requirements"""
        description = parameters['subtask_description']
        task_context = parameters.get('task_context', {})
        priority = parameters.get('priority', 'medium')
        
        # Keyword-based analysis (could be enhanced with NLP)
        keywords = {
            'research': ['research', 'find', 'search', 'investigate', 'explore', 'discover', 'analyze'],
            'analysis': ['analyze', 'examine', 'evaluate', 'assess', 'compare', 'review', 'study'],
            'writing': ['write', 'document', 'create', 'draft', 'compose', 'author', 'generate'],
            'coding': ['code', 'implement', 'develop', 'program', 'build', 'debug', 'test'],
            'design': ['design', 'architect', 'plan', 'structure', 'model', 'blueprint'],
            'data': ['data', 'database', 'query', 'extract', 'transform', 'process'],
            'communication': ['communicate', 'coordinate', 'collaborate', 'discuss', 'present'],
            'optimization': ['optimize', 'improve', 'enhance', 'refine', 'tune', 'performance']
        }
        
        detected_skills = []
        confidence_scores = {}
        
        desc_lower = description.lower()
        for skill, triggers in keywords.items():
            matches = sum(1 for trigger in triggers if trigger in desc_lower)
            if matches > 0:
                detected_skills.append(skill)
                # More matches = higher confidence
                confidence_scores[skill] = min(0.95, 0.6 + (matches * 0.1))
        
        # Estimate complexity
        complexity = 'low'
        if len(detected_skills) > 3:
            complexity = 'high'
        elif len(detected_skills) > 1:
            complexity = 'medium'
        
        # Estimate duration based on complexity
        duration_map = {'low': 60, 'medium': 180, 'high': 300}
        estimated_duration = duration_map[complexity]
        
        return {
            'detected_skills': detected_skills,
            'confidence_scores': confidence_scores,
            'complexity_estimate': complexity,
            'estimated_duration_seconds': estimated_duration,
            'implicit_requirements': [
                "attention to detail" if "analyze" in desc_lower else None,
                "creativity" if "design" in desc_lower or "create" in desc_lower else None,
                "technical expertise" if "implement" in desc_lower or "code" in desc_lower else None
            ],
            'priority': priority
        }
    
    async def _check_availability(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check agent availability"""
        agent_ids = parameters['agent_ids']
        include_queue = parameters.get('include_queue', True)
        
        agents = self.db.query(Agent).filter(Agent.id.in_(agent_ids)).all()
        
        availability = {}
        for agent in agents:
            # Get performance metrics for workload
            perf_metrics = agent.performance_metrics or {}
            current_workload = perf_metrics.get('current_workload', 0.5)
            
            availability[agent.id] = {
                'agent_name': agent.name,
                'status': agent.status,
                'current_workload': current_workload,
                'can_start_immediately': agent.status == 'active' and current_workload < 0.8,
                'estimated_wait_time_seconds': 0 if agent.status == 'active' else 300,
                'queued_tasks': [] if not include_queue else []  # Would fetch from queue
            }
        
        return availability
    
    async def _get_collaboration_history(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get agent collaboration history"""
        agent_id = int(parameters['agent_id'])
        potential_collaborators = parameters.get('potential_collaborators', [])
        
        # This would query actual collaboration data
        # For now, return simulated data
        return {
            'agent_id': agent_id,
            'collaboration_score': 0.85,
            'past_collaborations': [
                {
                    'collaborator_id': cid,
                    'success_rate': 0.9,
                    'synergy_score': 0.85
                }
                for cid in potential_collaborators[:2]  # Simulate some history
            ],
            'synergy_patterns': [
                "Effective knowledge sharing",
                "Complementary skills"
            ] if potential_collaborators else []
        }
    
    async def _compare_agents(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple agents"""
        agent_ids = parameters['agent_ids']
        criteria = parameters.get('comparison_criteria', ['performance', 'reliability', 'quality'])
        
        agents = self.db.query(Agent).filter(Agent.id.in_(agent_ids)).all()
        
        comparison = {
            'agents': {},
            'criteria': criteria,
            'recommendation': None
        }
        
        best_score = 0
        best_agent = None
        
        for agent in agents:
            perf_metrics = agent.performance_metrics or {}
            
            agent_scores = {
                'performance': perf_metrics.get('success_rate', 0.7),
                'reliability': perf_metrics.get('reliability', 0.75),
                'quality': perf_metrics.get('quality_score', 0.7),
                'speed': perf_metrics.get('speed_score', 0.7),
                'cost': 1.0 - perf_metrics.get('cost_factor', 0.3)  # Lower cost is better
            }
            
            comparison['agents'][agent.id] = {
                'name': agent.name,
                **{c: agent_scores.get(c, 0.5) for c in criteria}
            }
            
            # Calculate overall score
            overall_score = sum(agent_scores.get(c, 0.5) for c in criteria) / len(criteria)
            if overall_score > best_score:
                best_score = overall_score
                best_agent = agent
        
        if best_agent:
            comparison['recommendation'] = {
                'agent_id': best_agent.id,
                'agent_name': best_agent.name,
                'reason': f'Highest overall score: {best_score:.2f}',
                'score': best_score
            }
        
        return comparison
    
    def _get_cache_key(self, subtask: Dict[str, Any]) -> str:
        """Generate cache key for subtask"""
        import hashlib
        key_parts = [
            subtask.get('subtask_id', ''),
            str(subtask.get('skills_required', [])),
            subtask.get('priority', 'medium')
        ]
        key_str = '|'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def select_agents_for_subtasks(
        self,
        subtasks: List[Dict[str, Any]],
        workflow_context: Optional[Dict[str, Any]] = None,
        max_agents_per_task: int = 3
    ) -> Dict[str, List[AgentMatch]]:
        """
        Select agents for multiple subtasks.
        
        This is a wrapper that calls select_agent_with_reasoning for each subtask
        and formats the results for compatibility with existing code.
        """
        results = {}
        workflow_ctx = workflow_context or {
            "workflow_id": "unknown",
            "total_subtasks": len(subtasks),
            "completed_subtasks": [],
            "selected_agents": []
        }
        
        for idx, subtask in enumerate(subtasks):
            subtask_id = subtask.get('subtask_id', f'subtask_{idx}')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 LLM AGENT SELECTION: Processing {subtask_id}")
            
            try:
                # Get LLM selection
                selection = await self.select_agent_with_reasoning(subtask, workflow_ctx)
                
                # Convert to AgentMatch format for compatibility
                agent_match = AgentMatch(
                    agent_id=selection.agent_id,
                    agent_name=selection.agent_name,
                    agent_type="llm_selected",
                    match_score=selection.confidence,
                    reasoning=selection.reasoning,
                    skills_matched=subtask.get('skills_required', []),
                    performance_score=0.8,  # Would come from function calls
                    availability_score=0.9,  # Would come from function calls
                    collaboration_score=0.85  # Would come from function calls
                )
                
                results[subtask_id] = [agent_match]  # Single best agent
                
                # Update workflow context for next selection
                workflow_ctx["selected_agents"].append({
                    "subtask_id": subtask_id,
                    "agent_id": selection.agent_id,
                    "agent_name": selection.agent_name
                })
                
            except Exception as e:
                logger.error(f"LLM agent selection failed for {subtask_id}: {e}")
                # Fallback to empty list
                results[subtask_id] = []
        
        return results




