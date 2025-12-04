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
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

# Import existing models and infrastructure
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models import Agent, WorkflowExecution
from core.llm import (
    OrchestratorLLM,
    FunctionRegistry,
    FunctionExecutor,
    FunctionSpec,
    FunctionParameter,
    FunctionCategory
)
from core.llm.semantic_skill_matcher import SemanticSkillMatcher, get_skill_matcher

logger = logging.getLogger(__name__)


def _normalize_tags(raw_tags: Optional[Any]) -> List[str]:
    """Normalize raw tag inputs into a deduplicated list of lowercase-trimmed strings."""
    if raw_tags is None:
        return []
    items: List[str] = []
    if isinstance(raw_tags, str):
        items = [segment.strip() for segment in raw_tags.split(',')]
    elif isinstance(raw_tags, (list, tuple, set)):
        for value in raw_tags:
            if isinstance(value, str):
                items.extend([segment.strip() for segment in value.split(',')])
    else:
        return []

    seen = set()
    normalized: List[str] = []
    for item in items:
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


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
        
        # Initialize semantic skill matcher for intelligent matching
        # Initialize semantic skill matcher for intelligent matching
        self.skill_matcher = get_skill_matcher()
        logger.info("✅ Semantic skill matcher initialized (using embeddings for fuzzy matching)")
        
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
        # Reduced max_function_calls from 10 to 5 to prevent context overflow
        response = await self.llm.generate_with_functions(
            prompt=prompt,
            functions=functions,
            function_executor=self._execute_function,
            max_function_calls=5,  # Reduced to prevent 8K token limit
            context={"subtask": subtask, "workflow": workflow_context}
        )
        
        # Parse response
        try:
            # Extract structured response - try multiple methods
            import re
            result_data = None
            
            # Method 1: Check if response.content is already a dict (happens with some LLM clients)
            if isinstance(response.content, dict):
                result_data = response.content
                logger.info("✅ Response content is already a dict")
            # Method 2: Try to parse entire response as JSON string
            elif isinstance(response.content, str):
                try:
                    result_data = json.loads(response.content)
                    logger.info("✅ Parsed response as direct JSON")
                except json.JSONDecodeError:
                    pass
            
            # Method 3: Extract JSON from markdown code blocks (only for strings)
            if not result_data and isinstance(response.content, str):
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.content, re.DOTALL)
                if code_block_match:
                    try:
                        result_data = json.loads(code_block_match.group(1))
                        logger.info("✅ Parsed response from code block")
                    except json.JSONDecodeError:
                        pass
            
            # Method 4: Find any JSON object in the response (only for strings)
            if not result_data and isinstance(response.content, str):
                json_match = re.search(r'\{[^{]*"selected_agent_id"[^}]*\}', response.content, re.DOTALL)
                if json_match:
                    try:
                        result_data = json.loads(json_match.group())
                        logger.info("✅ Parsed response from regex match")
                    except json.JSONDecodeError:
                        pass
            
            # Method 4: Extract from function calls if agent was mentioned
            if not result_data and response.function_calls:
                # Try to find agent selection from function call context
                for fc in response.function_calls:
                    if fc.get("name") in ["get_agent_performance_history", "check_agent_availability"]:
                        params = fc.get("parameters", {})
                        if "agent_id" in params:
                            agent_id = params["agent_id"]
                            if isinstance(agent_id, list):
                                agent_id = agent_id[0]
                            
                            # Get agent info
                            agent = self.db.query(Agent).filter(Agent.id == agent_id).first()
                            if agent:
                                result_data = {
                                    "selected_agent_id": agent.id,
                                    "selected_agent_name": agent.name,
                                    "reasoning": f"Inferred from function calls: {fc['name']}",
                                    "confidence": 0.6
                                }
                                logger.info(f"✅ Inferred agent {agent.id} from function calls")
                                break
            
            # Fallback: Create minimal structure
            if not result_data:
                logger.warning("⚠️ Could not parse JSON, using fallback structure")
                # Handle different response.content types
                content_str = str(response.content)[:500] if response.content else "No content"
                result_data = {
                    "selected_agent_id": 0,
                    "selected_agent_name": "Unknown",
                    "reasoning": content_str,  # First 500 chars
                    "confidence": response.confidence if hasattr(response, 'confidence') else 0.3,
                    "alternatives_considered": [],
                    "risk_factors": ["Response parsing required fallback"],
                    "function_calls_made": [fc.get("name", "unknown") for fc in response.function_calls] if response.function_calls else []
                }
            
            # Validate required fields and provide defaults
            result_data.setdefault("selected_agent_id", 0)
            result_data.setdefault("selected_agent_name", "Unknown")
            result_data.setdefault("reasoning", "No reasoning provided")
            result_data.setdefault("confidence", 0.5)
            result_data.setdefault("alternatives_considered", [])
            result_data.setdefault("risk_factors", [])
            result_data.setdefault("function_calls_made", [])
            
            selection_result = AgentSelectionResult(
                agent_id=int(result_data["selected_agent_id"]),
                agent_name=result_data["selected_agent_name"],
                reasoning=result_data["reasoning"],
                confidence=float(result_data["confidence"]),
                alternatives_considered=result_data["alternatives_considered"],
                risk_factors=result_data["risk_factors"],
                function_calls_made=result_data["function_calls_made"],
                selection_time=asyncio.get_event_loop().time() - start_time,
                metadata={
                    "tokens_used": response.tokens_used if hasattr(response, 'tokens_used') else 0,
                    "cost": response.cost if hasattr(response, 'cost') else 0,
                    "model": self.llm.model if hasattr(self.llm, 'model') else "unknown"
                }
            )
            
            # If we got agent_id 0 or "Unknown", try fallback
            if selection_result.agent_id == 0 or selection_result.agent_name == "Unknown":
                logger.warning("⚠️ Parsed response has invalid agent, attempting fallback")
                raise ValueError("Invalid agent selection: agent_id is 0 or Unknown")
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.info(f"Response content (first 500 chars): {response.content[:500]}")
            
            # Fallback to first available agent
            agents = await self._query_agents({
                "skills": subtask.get("skills_required", []),
                "tags": subtask.get("tags", [])
            })
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
                logger.info(f"✅ Fallback: Using agent {first_agent['agent_id']} ({first_agent['name']})")
            else:
                raise ValueError(f"No agents available and LLM parsing failed: {e}")
        
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
- Tags: {subtask.get('tags', [])}
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
        tags = parameters.get('tags', [])
        min_proficiency = parameters.get('min_proficiency', 0.6)
        status = parameters.get('status', 'available')
        max_workload = parameters.get('max_workload', 0.8)
        agent_type = parameters.get('agent_type')
        
        if isinstance(skills, str):
            skills = [skills]
        if isinstance(tags, str):
            tags = [tags]
        skills = [skill for skill in skills if isinstance(skill, str) and skill.strip()]
        tags = [tag for tag in tags if isinstance(tag, str) and tag.strip()]
        requirements = skills + tags
        
        # Build query
        query = self.db.query(Agent)
        
        # Filter by status if not 'any'
        # FIXED: Treat 'available' as 'active' since that's the actual status in DB
        if status == 'available':
            query = query.filter(Agent.status == 'active')
        elif status != 'any':
            query = query.filter(Agent.status == status)
        
        # Filter by agent type if provided
        # FIXED: More lenient matching - check if requested type is IN agent_type or vice versa
        if agent_type and agent_type != 'any':
            # Try exact match first
            exact_query = query.filter(Agent.agent_type == agent_type)
            if exact_query.count() > 0:
                query = exact_query
            # If no exact match, skip agent_type filter and rely on skill matching
            # This allows 'document_generator' to match 'custom' type agents with PDF skills
        
        # Note: workload filtering would require additional field
        # For now, we'll use performance_metrics
        
        # Get agents WITH skills eagerly loaded (FIXED: don't double-query!)
        from sqlalchemy.orm import joinedload
        agents = query.options(joinedload(Agent.skills)).all()
        
        logger.info(f"🔍 _query_agents: Found {len(agents)} agents from database")
        logger.info(f"   Filters: skills={skills}, tags={tags}, agent_type={agent_type}, status={status}")
        
        matching_agents = []
        for agent in agents:
            # Get agent skills AND their tools
            agent_skill_list = []
            agent_tools = []
            agent_tags = _normalize_tags(getattr(agent, 'tags', None))
            if agent.skills:
                for skill in agent.skills:
                    agent_skill_list.append(skill.name)
                    # Extract tool names from tools_schema
                    if skill.tools_schema:
                        try:
                            import json
                            schema = json.loads(skill.tools_schema) if isinstance(skill.tools_schema, str) else skill.tools_schema
                            # FIXED: Check both formats - {'tools': [...]} and direct list
                            if isinstance(schema, dict) and 'tools' in schema:
                                tools = [tool.get('name', '') for tool in schema['tools'] if tool.get('name')]
                            elif isinstance(schema, list):
                                # Old format: list of tool specs with 'function' key
                                tools = [tool.get('function', {}).get('name', '') for tool in schema if tool.get('function')]
                            else:
                                tools = []
                            agent_tools.extend(tools)
                        except Exception as e:
                            logger.warning(f"Failed to parse tools_schema for skill {skill.name}: {e}")
                            pass
            
            logger.info(f"   Agent {agent.id} ({agent.name}): skills={agent_skill_list}, tags={agent_tags}, tools={agent_tools}")
            
            # Calculate skill match using SEMANTIC MATCHING (not dumb substring matching!)
            skill_matches = 0
            matched_by = []
            
            # If NO skills required, match all agents (100% coverage)
            if not requirements:
                skill_matches = 1
                skill_coverage = 1.0
                matched_by.append("no_requirements")
            else:
                # Use semantic matching with embeddings!
                # This intelligently matches "writing" → "writer", "author", "documentation"
                skill_coverage, match_details = await self.skill_matcher.compute_skill_coverage(
                    required_skills=requirements,
                    agent_skills=agent_skill_list,
                    agent_tools=agent_tools,
                    agent_tags=agent_tags
                )
                
                # Extract match info
                skill_matches = sum(1 for m in match_details if m["matched"])
                for match in match_details:
                    if match["matched"]:
                        match_type = match["match_type"]  # 'skill' or 'tool'
                        capability = match["agent_capability"]
                        similarity = match["similarity"]
                        matched_by.append(f"{match_type}:{capability}(sim={similarity:.2f})")
                        logger.debug(
                            f"      ✅ Semantic match: '{match['required']}' → '{capability}' "
                            f"(similarity={similarity:.3f})"
                        )
            
            # CRITICAL FIX: If agent has NO skills, give it a HIGH baseline score based on agent_type/description
            # This allows the LLM to select agents based on their type/description even without explicit skill mappings
            if len(agent_skill_list) == 0:
                # Agent has no skills mapped - use agent_type/description as fallback
                # Give it a HIGH baseline score so LLM seriously considers it
                skill_coverage = 0.8  # HIGH score - let LLM judge based on type/description
                matched_by.append(f"agent_type:{agent.agent_type}")
                matched_by.append(f"description_based_match")
                logger.info(f"      → Agent has NO skills, using HIGH baseline (0.8) for LLM semantic matching")
                logger.info(f"         agent_type={agent.agent_type}, description={agent.description[:100] if agent.description else 'N/A'}...")
            
            logger.info(f"      → skill_coverage={skill_coverage:.2f}, min_proficiency={min_proficiency}, matched={skill_coverage >= min_proficiency}")
            
            # RELAXED FILTER: Include agents even with low skill_coverage if they have no skills
            # Let the LLM decide if agent_type/description is a good match
            include_agent = skill_coverage >= min_proficiency or len(agent_skill_list) == 0
            
            if include_agent:
                # Get performance metrics
                perf_metrics = agent.performance_metrics or {}
                
                matching_agents.append({
                    'agent_id': agent.id,
                    'name': agent.name,
                    'agent_type': agent.agent_type,
                    'description': agent.description or '',  # CRITICAL: Include description for LLM reasoning
                    'skills': agent_skill_list,
                    'tools': agent_tools,
                    'tags': agent_tags,
                    'matched_by': matched_by,
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
                'tags': tags,
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
            WorkflowExecution.started_at >= cutoff_date
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
            str(subtask.get('tags', [])),
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
        
        # Merge workflow_context with defaults to ensure required keys exist
        default_ctx = {
            "workflow_id": "unknown",
            "total_subtasks": len(subtasks),
            "completed_subtasks": [],
            "selected_agents": []
        }
        
        if workflow_context:
            # Start with defaults, then override with provided values
            workflow_ctx = {**default_ctx, **workflow_context}
            # Ensure selected_agents is always a list (in case it was provided as None)
            if "selected_agents" not in workflow_ctx or workflow_ctx["selected_agents"] is None:
                workflow_ctx["selected_agents"] = []
        else:
            workflow_ctx = default_ctx
        
        # BATCH PROCESSING - NO LOOPS!
        logger.info(f"🚀 BATCH MODE: Processing ALL {len(subtasks)} subtasks at once")
        
        # Step 1: Query ALL agents ONCE
        agents_result = await self._query_agents({"status": "available"})
        all_agents = agents_result.get("matching_agents", [])
        logger.info(f"  📊 Found {len(all_agents)} available agents")
        
        # Step 2: Batch semantic scoring for ALL subtasks
        import asyncio
        scoring_tasks = []
        for idx, subtask in enumerate(subtasks):
            subtask_id = subtask.get('subtask_id', f'subtask_{idx}')
            task = self._score_agents_for_subtask(subtask, all_agents, subtask_id)
            scoring_tasks.append((subtask_id, task))
        
        # Run all scoring in parallel
        logger.info(f"  🧠 Computing semantic scores for {len(scoring_tasks)} subtasks...")
        scored_results = await asyncio.gather(*[task for _, task in scoring_tasks])
        
        # Map scored agents back to subtask IDs
        subtask_scores = {}
        for (subtask_id, _), scores in zip(scoring_tasks, scored_results):
            subtask_scores[subtask_id] = scores
        
        logger.info(f"  ✅ Semantic scoring complete")
        
        # Step 3: DIRECT ASSIGNMENT - Use best semantic match (NO LLM!)
        logger.info(f"  ⚡ FAST MODE: Using semantic scores directly (NO LLM)")
        
        for idx, (subtask_id, scores) in enumerate(subtask_scores.items()):
            normalized_id = f"subtask_{idx}"
            if scores:
                best = scores[0]  # Already sorted by score (highest first)
                agent_match = AgentMatch(
                    agent_id=best['agent_id'],
                    agent_name=best['name'],
                    agent_type=best.get('agent_type', 'general'),
                    match_score=best['score'],
                    reasoning=f"Semantic match: {best['score']:.0%} skill coverage ({len(best.get('skills_matched', []))} skills matched)",
                    skills_matched=best.get('skills_matched', []),
                    performance_score=0.8,
                    availability_score=0.9,
                    collaboration_score=0.85
                )
                # Always store using normalized execution key
                results[normalized_id] = [agent_match]
                # Preserve original key if different (for analytics/logging)
                if subtask_id != normalized_id:
                    results[subtask_id] = [agent_match]
                logger.info(f"    ✓ {normalized_id}: Agent {best['agent_id']} ({best['name']}) - {best['score']:.0%} match")
            else:
                logger.warning(f"    ⚠️  {normalized_id}: No suitable agents found")
                results[normalized_id] = []
                if subtask_id != normalized_id:
                    results[subtask_id] = []
        
        logger.info(f"  ✅ Assigned {len(results)} agents using pure semantic matching")
        
        return results
    
    async def _score_agents_for_subtask(
        self,
        subtask: Dict[str, Any],
        all_agents: List[Dict[str, Any]],
        subtask_id: str
    ) -> List[Dict[str, Any]]:
        """Score all agents for a single subtask using semantic matching"""
        
        # Handle both field names and ensure it's a list
        required_skills = subtask.get('skills_required', subtask.get('primary_skill', []))
        if isinstance(required_skills, str):
            required_skills = [required_skills]
        elif not required_skills:
            required_skills = []
        
        required_tags = subtask.get('tags', [])
        if isinstance(required_tags, str):
            required_tags = [required_tags]
        elif not required_tags:
            required_tags = []
        
        required_type = subtask.get('agent_type')
        
        requirements = [req for req in required_skills + required_tags if isinstance(req, str) and req.strip()]
            
        if not requirements:
            logger.info(f"⚠️ Subtask {subtask_id} arrived with no required requirements (skills/tags). Will rely on agent_type={required_type}")
        
        # Log what we're matching
        logger.info(
            f"🔍 Matching subtask {subtask_id}: skills={required_skills}, tags={required_tags}, "
            f"desc='{subtask.get('description', '')[:100]}'"
        )
        
        if not requirements and not required_type:
            # No requirements - return all agents with default score
            return [{
                'agent_id': agent['agent_id'],
                'name': agent['name'],
                'agent_type': agent.get('agent_type', 'general'),
                'score': 0.5,
                'skills_matched': []
            } for agent in all_agents[:5]]  # Top 5
        
        # Score each agent
        scored = []
        for agent in all_agents:
            agent_skills = agent.get('skills', [])
            agent_tools = agent.get('tools', [])
            agent_tags = agent.get('tags', [])
            
            # Log agent capabilities for first few agents
            if len(scored) < 3:
                logger.info(
                    f"  🤖 Agent {agent['agent_id']} ({agent['name']}): "
                    f"skills={agent_skills[:3] if agent_skills else 'none'}, "
                    f"tags={agent_tags[:3] if agent_tags else 'none'}, "
                    f"tools={len(agent_tools)} tools"
                )
            
            # Semantic skill coverage
            coverage, match_details = await self.skill_matcher.compute_skill_coverage(
                required_skills=requirements,
                agent_skills=agent_skills,
                agent_tools=agent_tools,
                agent_tags=agent_tags
            )
            if match_details:
                summary = ", ".join(
                    f"{m['required']}→{m['agent_capability']}({m['similarity']:.2f})" if m['matched']
                    else f"{m['required']}→∅({m['similarity']:.2f})"
                    for m in match_details
                )
                logger.info(f"     📊 Match details: {summary}")
            else:
                logger.info("     📊 Match details: (none)")
            
            # Agent type match
            type_match = bool(required_type and agent.get('agent_type') == required_type)
            
            # How many skills did we actually match?
            matched_skills = [d['required'] for d in match_details if d['matched']]
            matched_count = len(matched_skills)
            
            # Build a composite score used only for ordering.
            # Start with semantic coverage, then add adjustments for type/skill alignment.
            sort_score = coverage
            
            if requirements:
                if matched_count == 0:
                    # Hard-penalize agents that meet none of the requested skills
                    sort_score -= 0.4
                else:
                    # Small boost proportional to skill coverage
                    sort_score += min(0.2, matched_count * 0.05)
            
            if required_type:
                sort_score += 0.2 if type_match else -0.2
            else:
                # Light boost for specialists even when no type stated
                sort_score += 0.05 if agent.get('agent_type') != 'general' else 0.0
            
            scored.append({
                'agent_id': agent['agent_id'],
                'name': agent['name'],
                'agent_type': agent.get('agent_type', 'general'),
                'score': coverage,
                'skills_matched': matched_skills,
                'tags': agent_tags,
                'coverage': coverage,
                'matched_skill_count': matched_count,
                'type_match': type_match,
                'sort_score': sort_score
            })
        
        # Sort by score and return top matches
        scored.sort(
            key=lambda x: (
                x.get('matched_skill_count', 0) > 0,
                x.get('type_match', False),
                x.get('sort_score', 0.0),
                x.get('coverage', 0.0)
            ),
            reverse=True
        )
        
        # ALWAYS return at least the top agent (no empty lists!)
        if scored:
            return scored[:5]  # Top 5 agents
        elif all_agents:
            # Fallback: return first available agent with low score
            return [{
                'agent_id': all_agents[0]['agent_id'],
                'name': all_agents[0]['name'],
                'agent_type': all_agents[0].get('agent_type', 'general'),
                'score': 0.2,
                'skills_matched': [],
                'tags': all_agents[0].get('tags', []),
                'coverage': 0.0
            }]
        else:
            return []  # Only return empty if NO agents exist at all
    
    def _build_batch_assignment_prompt(
        self,
        subtasks: List[Dict[str, Any]],
        subtask_scores: Dict[str, List[Dict[str, Any]]],
        workflow_context: Dict[str, Any]
    ) -> str:
        """Build prompt for single LLM call with ALL subtask data"""
        
        prompt = f"""You have {len(subtasks)} subtasks to assign to agents. Based on semantic skill matching, here are the top agent candidates for each subtask.

**Your Task:** Assign the BEST agent to each subtask. Consider:
- Semantic skill coverage scores
- Agent specialization and type
- Load balancing (don't overload one agent)
- Workflow context

**Subtasks and Top Candidates:**

"""
        
        for idx, subtask in enumerate(subtasks):
            subtask_id = subtask.get('subtask_id', f'subtask_{idx}')
            scores = subtask_scores.get(subtask_id, [])
            
            prompt += f"\n{'='*60}\n"
            prompt += f"**Subtask {idx+1}:** {subtask.get('description', 'No description')[:100]}\n"
            prompt += f"- Required Skills: {', '.join(subtask.get('skills_required', []))}\n"
            prompt += f"- Tags: {', '.join(subtask.get('tags', []))}\n"
            prompt += f"- Agent Type: {subtask.get('agent_type', 'any')}\n"
            prompt += f"- Priority: {subtask.get('priority', 'medium')}\n\n"
            
            prompt += "**Top Agent Candidates:**\n"
            for i, agent in enumerate(scores[:3], 1):
                prompt += f"{i}. Agent {agent['agent_id']} ({agent['name']})\n"
                prompt += f"   - Type: {agent['agent_type']}\n"
                prompt += f"   - Score: {agent['score']:.0%}\n"
                prompt += f"   - Skills Matched: {', '.join(agent.get('skills_matched', []))}\n"
            
            if not scores:
                prompt += "⚠️ No strong candidates found\n"
        
        prompt += f"\n\n**Response Format (JSON):**\n"
        prompt += "```json\n{\n"
        for idx, subtask in enumerate(subtasks):
            subtask_id = subtask.get('subtask_id', f'subtask_{idx}')
            prompt += f'  "{subtask_id}": {{"agent_id": <id>, "reasoning": "<why>"}}'
            if idx < len(subtasks) - 1:
                prompt += ","
            prompt += "\n"
        prompt += "}\n```\n"
        
        return prompt
    
    def _parse_batch_assignments(
        self,
        llm_response: str,
        subtask_scores: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Parse LLM's batch assignment response"""
        import json
        import re
        
        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if json_match:
                assignments_raw = json.loads(json_match.group(1))
            else:
                # Try direct JSON parse
                assignments_raw = json.loads(llm_response)
            
            # Validate and enrich assignments
            assignments = {}
            for subtask_id, assignment in assignments_raw.items():
                agent_id = assignment.get('agent_id')
                reasoning = assignment.get('reasoning', 'LLM assignment')
                
                # Find agent details from scores
                scores = subtask_scores.get(subtask_id, [])
                agent_data = next((a for a in scores if a['agent_id'] == agent_id), None)
                
                if agent_data:
                    assignments[subtask_id] = {
                        'agent_id': agent_id,
                        'agent_name': agent_data['name'],
                        'agent_type': agent_data.get('agent_type'),
                        'score': agent_data['score'],
                        'reasoning': reasoning,
                        'skills_matched': agent_data.get('skills_matched', [])
                    }
                else:
                    # Use first available agent if LLM picked invalid ID
                    if scores:
                        best = scores[0]
                        assignments[subtask_id] = {
                            'agent_id': best['agent_id'],
                            'agent_name': best['name'],
                            'agent_type': best.get('agent_type'),
                            'score': best['score'],
                            'reasoning': f"Fallback to best match (LLM picked invalid ID)",
                            'skills_matched': best.get('skills_matched', [])
                        }
            
            return assignments
                
        except Exception as e:
            logger.error(f"Failed to parse LLM batch response: {e}")
            logger.error(f"Response was: {llm_response[:500]}")
            
            # Fallback: Use best semantic match for each subtask
            assignments = {}
            for subtask_id, scores in subtask_scores.items():
                if scores:
                    best = scores[0]
                    assignments[subtask_id] = {
                        'agent_id': best['agent_id'],
                        'agent_name': best['name'],
                        'agent_type': best.get('agent_type'),
                        'score': best['score'],
                        'reasoning': 'Best semantic match (LLM parse failed)',
                        'skills_matched': best.get('skills_matched', [])
                    }
            return assignments




