"""
Auto Brain — The Progressive Complexity Assessor
====================================================

PRD-68: Progressive Complexity Routing (Atom → Organism).

Auto receives every message and determines its complexity level:
  - ATOM: Direct response (greetings, chitchat) — no tools, no memory
  - MOLECULE: Single tool calls without deep memory
  - CELL: Needs memory + tools + reasoning
  - ORGAN: Multi-agent coordination
  - ORGANISM: Full PRD-59 Neural Swarm pipelines

3-Tier Assessment:
  Tier 1: Redis cache lookup (<5ms, free)
  Tier 2: Regex fast-paths (<5ms, free)
  Tier 3: LLM classification (~200ms, ~$0.001)

The ComplexityAssessment flows through the existing wiring:
  api/chat.py → service.py → integration.py → smart_orchestrator.py
where needs_memory and tool_hints drive downstream behavior.
"""

import logging
import re
import json
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Complexity levels (Progressive Complexity Model PRD-68)
# ---------------------------------------------------------------------------

class Complexity(str, Enum):
    """Task complexity on the Atom → Organism scale."""
    ATOM = "atom"            # Simple: greetings, factual, chitchat (<200 tokens)
    MOLECULE = "molecule"    # Needs a tool or specific agent skill (~1K tokens)
    CELL = "cell"            # Needs memory + tool + reasoning (~3K tokens)
    ORGAN = "organ"          # Multi-agent coordination (~6K tokens)
    ORGANISM = "organism"    # Enterprise pipeline, learning + feedback (~12K tokens)


class Action(str, Enum):
    """What Auto should do with this request."""
    RESPOND = "respond"      # Auto responds directly (no delegation)
    DELEGATE = "delegate"    # Route to a single sub-agent
    WORKFLOW = "workflow"    # Trigger multi-agent workflow


@dataclass
class ComplexityAssessment:
    """Result of Auto's complexity assessment."""
    complexity: Complexity
    action: Action
    reasoning: str
    target_agent_id: Optional[int] = None
    target_agent_name: Optional[str] = None
    matched_tools: List[str] = field(default_factory=list)
    confidence: float = 0.0
    # PRD-68: Fields consumed by smart_orchestrator.py
    needs_memory: bool = False
    tool_hints: List[str] = field(default_factory=list)
    needs_multi_agent: bool = False

    def to_dict(self):
        return {
            "complexity": self.complexity.value,
            "action": self.action.value,
            "reasoning": self.reasoning,
            "tool_hints": self.tool_hints,
            "needs_memory": self.needs_memory,
            "needs_multi_agent": self.needs_multi_agent,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Tier 2: Fast Heuristic Patterns
# ---------------------------------------------------------------------------

# Must be the ENTIRE message (with optional punctuation).
# "hello" → atom.  "hello can you create an image" → NOT atom.
_ATOM_PATTERNS = [
    # Greetings (with optional name: "hi auto", "morning auto", "hey there")
    r"^(hi|hello|hey|howdy|yo|sup)(\s+\w+)?[\s!?.,:]*$",
    r"^(good\s+)?(morning|afternoon|evening|night)(\s+\w+)?[\s!?.,:]*$",
    r"^(g'day|hiya|heya|oi|ello|mornin)(\s+\w+)?[\s!?.,:]*$",
    # Informal greetings and check-ins
    r"^what'?s\s+up[\s!?.,:]*$",
    r"^how'?s\s+it\s+going[\s!?.,:]*$",
    r"^how\s+are\s+(you|things|ya)[\s!?.,:]*$",
    r"^how'?s\s+everything[\s!?.,:]*$",
    r"^long\s+time\s+no\s+see[\s!?.,:]*$",
    # Thanks / bye / acknowledgements
    r"^(thanks|thank you|thx|ty|cheers)(\s+\w+)?[\s!?.,:]*$",
    r"^(bye|goodbye|see ya|later|cya|see you)(\s+\w+)?[\s!?.,:]*$",
    r"^(ok|okay|yes|no|sure|cool|nice|great|awesome|perfect|got it|alright|grand|brilliant)[\s!?.,:]*$",
    # Identity questions
    r"^(what|who)\s+(are|is)\s+(you|automatos|auto)[\s!?.]*$",
    r"^what\s+can\s+you\s+do[\s!?.]*$",
    # Simple chitchat (no tools needed)
    r"^tell\s+me\s+a\s+joke[\s!?.,:]*$",
    r"^what\s+(time|day)\s+is\s+it[\s!?.,:]*$",
    r"^(lol|haha|lmao|rofl|ha+)[\s!?.,:]*$",
]

_PLATFORM_KEYWORDS = {
    "platform_list_agents": [
        "list my agents", "what agents do i have", "show my agents",
        "how many agents do i have", "show me my agents",
    ],
    "platform_list_recipes": [
        "list my recipes", "what recipes do i have", "show my recipes",
        "list my workflows", "show my workflows", "how many recipes",
        "how many workflows",
    ],
    "platform_get_llm_usage": [
        "token usage", "llm usage", "how much have i spent",
        "my api cost", "my token spend", "how many tokens",
        "show my usage", "show my spending",
    ],
    "platform_list_documents": [
        "list my documents", "what documents do i have",
        "show my documents", "how many documents do i have",
        "what files have i uploaded", "show my uploaded files",
    ],
    "platform_get_workspace_info": [
        "workspace info", "my workspace info",
        "tell me about my workspace", "show workspace details",
    ],
    "platform_list_connected_apps": [
        "what apps are connected", "show my integrations",
        "list my connected apps", "list my integrations",
        "what integrations do i have",
    ],
    "platform_list_tools": [
        "what tools", "list my tools", "available tools",
        "what can i use", "show my tools", "what integrations",
        "composio tools", "connected tools",
    ],
    "platform_list_llms": [
        "what models", "available models", "list llms", "list models",
        "what llms", "cheapest model", "show models",
        "openrouter models",
    ],
    "platform_list_datasources": [
        "what data", "data sources", "what databases", "list datasources",
        "what documents", "rag sources", "nl2sql", "queryable databases",
        "what repos are indexed",
    ],
    "platform_query_data": [
        "query the database", "query data", "how many users",
        "what's our mrr", "current revenue", "latest metrics",
        "users signed up", "average order value", "show me the data",
        "run a query", "ask the database", "check the numbers",
        "how many customers", "total sales", "query my database",
    ],
    "platform_workspace_stats": [
        "workspace stats", "platform stats", "usage stats",
        "how many queries", "agent activity", "what's being used",
        "show stats", "show usage",
    ],
    "platform_execute_recipe": [
        "run the recipe", "execute recipe", "trigger recipe",
        "run automation", "start recipe",
    ],
    "platform_get_recipe_execution": [
        "recipe status", "execution status", "recipe result",
        "did the recipe run", "check recipe",
    ],
    "platform_get_system_health": [
        "system health", "platform health", "system status",
        "check health", "health check", "is everything working",
    ],
    "platform_delete_document": [
        "delete document", "remove document",
        "delete from knowledge base",
    ],
    "platform_reprocess_document": [
        "reprocess document", "re-embed document", "reindex document",
        "regenerate chunks", "rebuild embeddings",
    ],
    "platform_delete_recipe": [
        "delete recipe", "remove recipe", "delete automation",
    ],
    "platform_get_activity_feed": [
        "recent activity", "activity feed", "what's been happening",
        "show activity", "what has been running", "activity log",
    ],
    # Memory search
    "platform_search_memory": [
        "search memory", "what do you remember", "recall",
        "do you remember", "what's in memory", "memory search",
        "what do you know about me", "check memory",
        "stored memories", "my preferences", "what have you learned",
    ],
    # Chat history search
    "platform_search_chat_history": [
        "search chat", "search conversations", "find in chat",
        "what did we talk about", "what did i say about",
        "chat history search", "find conversation",
        "did i mention", "when did i say", "previous chat",
        "search my chats", "look through chats",
    ],
    # PRD-73: Monitoring tools
    "platform_query_loki_logs": [
        "search logs", "query logs", "check logs", "show logs",
        "error logs", "find in logs", "log search", "loki",
        "what errors happened", "any errors in logs",
        "backend logs", "worker logs", "service logs",
    ],
    "platform_query_prometheus": [
        "check metrics", "prometheus", "service health",
        "error rate", "response time", "latency",
        "database connections", "redis memory", "redis usage",
        "postgres health", "db health", "are services up",
        "p95", "uptime", "prometheus metrics",
    ],
    "platform_get_alerts": [
        "firing alerts", "active alerts", "any alerts",
        "infrastructure alerts", "check alerts", "alert history",
        "monitoring alerts", "system alerts", "what's broken",
        "any issues", "infrastructure issues",
    ],
    # Marketplace discovery & workspace inventory (PRD-71)
    "platform_browse_marketplace_agents": [
        "marketplace agents", "browse agents", "search agents",
        "available agents", "find an agent", "hire agent",
        "show me agents", "agent catalog", "team members",
    ],
    "platform_browse_marketplace_plugins": [
        "marketplace plugins", "browse plugins", "search plugins",
        "available plugins", "find a plugin",
    ],
    "platform_browse_marketplace_skills": [
        "marketplace skills", "browse skills", "search skills",
        "available skills", "find a skill",
    ],
    "platform_list_workspace_plugins": [
        "my plugins", "workspace plugins", "enabled plugins",
        "installed plugins", "what plugins do i have",
    ],
    "platform_list_workspace_skills": [
        "my skills", "workspace skills", "enabled skills",
        "installed skills", "what skills do i have",
    ],
    "platform_list_workspace_models": [
        "my models", "workspace models", "installed models",
        "what models do i have", "enabled models",
    ],
    "platform_install_plugin": [
        "install plugin", "enable plugin", "add plugin",
        "activate plugin",
    ],
    "platform_install_skill": [
        "install skill", "enable skill", "add skill",
        "activate skill",
    ],
    "platform_install_model": [
        "install model", "enable model", "add model",
        "activate model",
    ],
    # Agent assignment (PRD-71)
    "platform_assign_tool_to_agent": [
        "assign tool to agent", "add tool to agent",
        "give agent a tool", "connect tool to agent",
    ],
    "platform_assign_skill_to_agent": [
        "assign skill to agent", "add skill to agent",
        "give agent a skill", "attach skill to agent",
    ],
    "platform_assign_plugin_to_agent": [
        "assign plugin to agent", "add plugin to agent",
        "give agent a plugin", "attach plugin to agent",
    ],
    "platform_configure_agent_heartbeat": [
        "configure heartbeat", "configure heartbeats", "set heartbeat",
        "agent heartbeat", "heartbeat schedule", "enable heartbeat",
        "disable heartbeat", "heartbeat interval", "set active hours",
        "heartbeat config",
    ],
    "platform_create_agent": [
        "create agent", "create an agent", "build agent", "build an agent",
        "make agent", "make an agent", "set up agent", "setup agent",
        "new agent", "create a new agent", "build me an agent",
    ],
    "platform_update_agent": [
        "update agent", "modify agent", "change agent", "edit agent",
        "configure agent", "reconfigure agent",
    ],
    # Workspace tools (file I/O, exec, git)
    "workspace_read_file": [
        "read file", "show file", "open file", "cat file",
        "view source", "read the code", "show me the code",
    ],
    "workspace_write_file": [
        "write file", "create file", "save file", "update file",
        "edit file", "modify file", "fix the code",
    ],
    "workspace_list_dir": [
        "list files", "list directory", "show directory", "project structure",
        "what files are there", "ls", "show files",
    ],
    "workspace_grep": [
        "search code", "find in code", "grep", "search for",
        "where is the function", "find definition",
    ],
    "workspace_exec": [
        "run tests", "run command", "execute command", "npm test",
        "pytest", "run the build", "run linter",
    ],
    "workspace_git": [
        "git status", "git diff", "git commit", "git push",
        "git log", "git pull", "check git", "commit changes",
        "push changes",
    ],
    # PRD-76: Agent Reports
    "platform_submit_report": [
        "submit report", "file report", "create report",
        "write report", "status report", "save report",
    ],
    "platform_get_latest_report": [
        "latest report", "agent report", "get report",
        "read report", "show report", "what did the agent find",
        "sentinel report", "agent standup", "last report",
    ],
    # PRD-72: Board Tasks
    "platform_create_task": [
        "create a task", "add a task", "raise a task", "new task",
        "create board task", "add to board", "create a bug report",
        "create sub-task", "create follow-up task", "task for",
        "raise a bug", "create work item",
    ],
    "platform_list_tasks": [
        "list tasks", "show tasks", "what tasks", "board tasks",
        "tasks on the board", "my tasks", "show the board",
        "what's in progress", "in progress tasks", "assigned tasks",
        "inbox tasks", "review tasks", "tasks for agent",
        "what's assigned to",
    ],
    "platform_board_summary": [
        "board summary", "how's the board", "board overview",
        "daily standup", "standup summary", "how many tasks",
        "task summary", "busiest agent", "who's busiest",
        "any failed tasks", "task stats", "board status",
        "how are we doing", "team status", "workload",
    ],
    "platform_get_task": [
        "show task", "get task", "task details", "task info",
        "what's task", "status of task", "details for task",
    ],
    "platform_assign_task": [
        "assign task", "give task to", "assign to agent",
        "hand off task", "delegate task", "reassign task",
    ],
    "platform_update_task_status": [
        "move task", "start task", "run task now",
        "mark task as", "complete task", "finish task",
        "task to done", "task to in progress", "trigger task",
        "execute task", "kick off task",
    ],
    # PRD-77: Agent Self-Scheduling
    "platform_schedule_task": [
        "schedule a task", "schedule follow-up", "remind me",
        "check again later", "schedule for tomorrow", "set up recurring",
        "schedule weekly", "schedule daily", "create a scheduled task",
        "follow up on this", "come back to this",
    ],
    "platform_list_scheduled_tasks": [
        "scheduled tasks", "what tasks are scheduled", "show scheduled",
        "list scheduled tasks", "pending tasks", "my scheduled tasks",
        "upcoming tasks",
    ],
    "platform_cancel_scheduled_task": [
        "cancel scheduled task", "stop scheduled task",
        "remove scheduled task", "cancel recurring task",
    ],
    # PRD-77: Memory Browsing
    "platform_browse_memories": [
        "browse memories", "show all memories", "list memories",
        "what has been remembered", "view memories",
        "memory explorer", "show memory contents",
    ],
    "platform_delete_memory": [
        "delete memory", "remove memory", "forget this",
        "erase memory", "clear memory",
    ],
    # PRD-108: Shared Mission Field
    "platform_field_query": [
        "query field", "search field", "what did other agents find",
        "shared context", "field results", "mission findings",
        "what's in the field", "check the field",
    ],
    "platform_field_inject": [
        "share finding", "inject into field", "add to field",
        "share with team", "publish finding", "contribute to field",
    ],
    "platform_field_stability": [
        "field stability", "how converged", "field status",
        "is the field stable", "convergence check",
    ],
    # Blog Widget
    "platform_publish_blog_post": [
        "blog", "publish post", "blog post", "write article",
        "publish article", "write blog", "create blog post",
        "publish blog", "write a post",
    ],
    "platform_list_blog_posts": [
        "list blog posts", "show blog posts", "what blog posts",
        "published articles", "blog articles", "my blog",
        "check blog", "existing posts",
    ],
    "platform_get_blog_post": [
        "read blog post", "get blog post", "show blog draft",
        "fetch article", "read article", "blog content",
        "show post content", "get draft",
    ],
    "platform_update_blog_post": [
        "update blog post", "edit blog post", "revise article",
        "improve draft", "edit article", "update post",
        "change blog post", "set cover image",
    ],
    # PRD-82A: Missions
    "platform_create_mission": [
        "launch a mission", "start a mission", "create a mission",
        "run a mission", "new mission", "kick off a mission",
        "deep research", "multi-agent research", "research mission",
        "launch mission to", "start mission to",
    ],
    "platform_list_missions": [
        "list missions", "show missions", "what missions",
        "running missions", "mission status", "my missions",
        "any missions", "active missions", "completed missions",
    ],
    "platform_get_mission": [
        "show mission", "get mission", "mission details",
        "mission info", "how is mission", "check mission",
    ],
    # PRD-121: HARNESS Self-Optimizing Loop
    "platform_harness_status": [
        "harness status", "optimization status", "org health",
        "how is the team performing", "team performance",
        "harness state", "optimization loop",
    ],
    "platform_harness_trigger": [
        "run harness", "trigger harness", "optimize now",
        "run optimization", "trigger optimization",
        "optimize the team", "tune the team",
    ],
    "platform_harness_history": [
        "harness history", "optimization history", "past optimizations",
        "harness runs", "optimization runs", "what has harness done",
        "harness changelog",
    ],
    # PRD-126: Business Knowledge Graph
    "platform_query_graph": [
        "how does", "connected to", "relationship between", "relate to",
        "end to end", "flow", "process for", "overview of", "map of",
        "what connects", "trace from", "path between",
    ],
    "platform_graph_impact": [
        "what if we change", "impact of", "what breaks", "affects",
        "downstream", "upstream", "dependencies of", "depends on",
        "consequences of", "ripple effect",
    ],
    "platform_graph_communities": [
        "what areas", "departments", "clusters", "domains",
        "groups of", "categories of", "themes in",
    ],
    "platform_graph_neighbors": [
        "related to", "associated with", "linked to", "touches", "involves",
    ],
    "platform_graph_stats": [
        "graph health", "knowledge coverage", "how complete",
        "graph size", "how connected",
    ],
}

_atom_re = [re.compile(p, re.IGNORECASE) for p in _ATOM_PATTERNS]

_MEMORY_PATTERN = re.compile(
    r"\b(do you remember|recall when|my name is|last time we|"
    r"previously we discussed|earlier (i|we|you) said|what did (i|we|you) (say|tell|ask))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# AutoBrain (The Assessor)
# ---------------------------------------------------------------------------

class AutoBrain:
    """
    PRD-68: 3-Tier Progressive Complexity Assessor.

    Evaluates every incoming request to determine the required execution depth
    (Atom → Organism), bypassing heavy tools and memory for simple requests.

    Tier 1: Redis cache (<5ms)
    Tier 2: Regex heuristics (<5ms)
    Tier 3: LLM classification (~200ms, configurable model via system settings)
    """

    def __init__(self, db: Session, workspace_id: str):
        self._db = db
        self._workspace_id = workspace_id
        self._redis = None
        try:
            from core.redis.client import get_redis_client
            self._redis = get_redis_client()
        except Exception:
            logger.debug("[AutoBrain] Redis not available, cache disabled")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def assess(
        self,
        message: str,
        conversation_length: int = 0,
    ) -> ComplexityAssessment:
        """Run the 3-Tier complexity assessment."""
        if not message or not message.strip():
            return ComplexityAssessment(
                complexity=Complexity.ATOM, action=Action.RESPOND,
                reasoning="Empty message", confidence=1.0,
            )

        msg_lower = message.lower().strip()

        # ── Tier 1: Redis cache lookup (<5ms) ──
        cached = self._cache_lookup(msg_lower)
        if cached:
            return cached

        # ── Tier 2: Regex fast-paths (FREE, <5ms) ──
        heur = self._run_fast_heuristics(msg_lower)
        if heur:
            self._cache_store(msg_lower, heur)
            return heur

        # ── Tier 3: LLM classification (~200ms) ──
        llm_result = await self._llm_classify(message, conversation_length)
        self._cache_store(msg_lower, llm_result)
        return llm_result

    # ------------------------------------------------------------------
    # Tier 2: Fast heuristics
    # ------------------------------------------------------------------

    def _run_fast_heuristics(self, msg_lower: str) -> Optional[ComplexityAssessment]:
        # ATOM: Pure chitchat
        """
        Run fast, rule-based heuristics to classify a lowercased user message into a ComplexityAssessment.
        
        Checks for three fast paths (in order): ATOM (chitchat/greetings), MOLECULE (platform/tool queries), and CELL (explicit memory recall). If a heuristic matches, returns a pre-built ComplexityAssessment with a suggested action, confidence, and any matched tool hints; if no heuristic matches, returns None.
        
        Parameters:
            msg_lower (str): The normalized, lowercase user message to classify.
        
        Returns:
            Optional[ComplexityAssessment]: A ComplexityAssessment when a fast-path heuristic matches, or `None` if no heuristic applies.
        """
        if self._is_atom(msg_lower):
            assessment = ComplexityAssessment(
                complexity=Complexity.ATOM, action=Action.RESPOND,
                reasoning="Greeting or chitchat", confidence=0.95,
                needs_memory=False, tool_hints=[], needs_multi_agent=False,
            )
            logger.info(
                "[AutoBrain] assessed",
                extra={
                    "tier": 2, "complexity": "atom", "action": "respond",
                    "confidence": 0.95, "latency_ms": 0, "cache_hit": False,
                    "workspace_id": self._workspace_id,
                },
            )
            return assessment

        # MOLECULE: Platform queries
        platform_tool = self._match_platform_query(msg_lower)
        if platform_tool:
            assessment = ComplexityAssessment(
                complexity=Complexity.MOLECULE, action=Action.RESPOND,
                reasoning=f"Platform query ({platform_tool})",
                matched_tools=[platform_tool], tool_hints=["platform"],
                confidence=0.90, needs_memory=False, needs_multi_agent=False,
            )
            logger.info(
                "[AutoBrain] assessed",
                extra={
                    "tier": 2, "complexity": "molecule", "action": "respond",
                    "confidence": 0.90, "latency_ms": 0, "cache_hit": False,
                    "workspace_id": self._workspace_id,
                },
            )
            return assessment

        # CELL: Memory recall
        if self._is_memory_recall(msg_lower):
            assessment = ComplexityAssessment(
                complexity=Complexity.CELL, action=Action.RESPOND,
                reasoning="Explicit memory recall", confidence=0.85,
                needs_memory=True, tool_hints=[], needs_multi_agent=False,
            )
            logger.info(
                "[AutoBrain] assessed",
                extra={
                    "tier": 2, "complexity": "cell", "action": "respond",
                    "confidence": 0.85, "latency_ms": 0, "cache_hit": False,
                    "workspace_id": self._workspace_id,
                },
            )
            return assessment

        return None

    # ------------------------------------------------------------------
    # Tier 3: LLM classification
    # ------------------------------------------------------------------

    async def _llm_classify(
        self, message: str, conversation_length: int
    ) -> ComplexityAssessment:
        """
        Classify an incoming message's execution complexity and routing using a lightweight LLM.
        
        Sends a structured classification prompt to an LLM to determine complexity (atom|molecule|cell|organ|organism),
        recommended action (respond|delegate|workflow), tool hints, memory and multi-agent needs, and a short reasoning
        string. If LLM classification fails, returns an ATOM/RESPOND fallback assessment with reduced confidence.
        
        Parameters:
            message (str): The user message to classify.
            conversation_length (int): The current conversation turn count (used to provide context to the classifier).
        
        Returns:
            ComplexityAssessment: An assessment containing complexity, action, reasoning, confidence, needs_memory,
            tool_hints, and needs_multi_agent. On LLM failure this will be an ATOM/RESPOND assessment with lower confidence.
        """
        logger.info("[AutoBrain] Tier 3 LLM classifying: '%s'", message[:80])
        t0 = time.monotonic()

        prompt = f"""You are a message complexity classifier for an AI platform.

Analyze the user's message step by step, then classify it.

Message: "{message}"
Conversation turn: {conversation_length}

## Reasoning Steps (think through each):

1. **Intent**: What is the user asking for? (greeting, question, action, complex task)
2. **Tool need**: Does this require external data or actions? (database, email, search, file ops)
3. **Memory need**: Does this reference past conversations or user preferences?
4. **Coordination**: How many systems need to work together?

## Classification levels:

- **atom**: Greetings, chitchat, opinions, simple factual questions, jokes, acknowledgements. NO tools needed. This is the most common category — when in doubt, choose atom.
- **molecule**: Needs ONE tool or action. "Send email", "search docs", "check Jira", "list my agents".
- **cell**: Needs tools + memory/context. "Reply to that email we discussed", "update the report from last week".
- **organ**: Multiple agents coordinating. "Research this bug, plan a fix, open a PR".
- **organism**: Enterprise multi-step pipeline. Rare.

## Examples:

- "Morning Auto" → atom (greeting)
- "How are you?" → atom (chitchat)
- "What's the weather like?" → atom (conversational)
- "Tell me about yourself" → atom (identity question)
- "Send an email to John" → molecule (email tool)
- "What agents do I have?" → molecule (platform query)
- "Search my docs for the Q4 report" → molecule (search tool)
- "Remember last week's meeting? Update those notes" → cell (memory + action)

**Default bias: atom.** Most messages are simpler than they look.

Return ONLY valid JSON:
{{
  "complexity": "atom|molecule|cell|organ|organism",
  "action": "respond|delegate|workflow",
  "tool_hints": [],
  "needs_memory": false,
  "needs_multi_agent": false,
  "reasoning": "one sentence"
}}

action mapping: "respond" for atom, "delegate" for molecule/cell, "workflow" for organ/organism.
tool_hints: short domain keywords like "email", "github", "jira", "code", "database", "platform". Use "platform" when the user wants to create/list/manage agents, skills, plugins, recipes, or workspace resources. Empty for atom."""

        try:
            from core.llm import create_llm_manager

            llm = create_llm_manager(service_name="complexity_assessor")
            response = await llm.generate_response(
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON block
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
                assessment = ComplexityAssessment(
                    complexity=Complexity(data.get("complexity", "atom").lower()),
                    action=Action(data.get("action", "respond").lower()),
                    reasoning=data.get("reasoning", "LLM classified"),
                    confidence=0.85,
                    needs_memory=data.get("needs_memory", False),
                    tool_hints=data.get("tool_hints", []),
                    needs_multi_agent=data.get("needs_multi_agent", False),
                )
                logger.info(
                    "[AutoBrain] assessed",
                    extra={
                        "tier": 3,
                        "complexity": assessment.complexity.value,
                        "action": assessment.action.value,
                        "confidence": assessment.confidence,
                        "latency_ms": elapsed_ms,
                        "cache_hit": False,
                        "workspace_id": self._workspace_id,
                    },
                )
                return assessment
        except Exception:
            logger.exception("[AutoBrain] Tier 3 LLM classification failed, falling back to ATOM")

        # Fallback: treat as MOLECULE / RESPOND so tools remain available.
        # Rationale: a wrong MOLECULE only adds tool schemas to the context
        # (model still decides via tool_choice=auto). A wrong ATOM strips
        # tools entirely, making the agent unable to fulfil action requests.
        return ComplexityAssessment(
            complexity=Complexity.MOLECULE, action=Action.RESPOND,
            reasoning="LLM classification failed — defaulting to MOLECULE (tools available)",
            confidence=0.50, needs_memory=False, tool_hints=[],
            needs_multi_agent=False,
        )

    # ------------------------------------------------------------------
    # Redis cache (Tier 1)
    # ------------------------------------------------------------------

    def _cache_lookup(self, msg_lower: str) -> Optional[ComplexityAssessment]:
        """
        Look up a previously stored ComplexityAssessment for a lowercased message in Redis and return it if present.
        
        Parameters:
            msg_lower (str): The input message normalized to lowercase used to construct the cache key.
        
        Returns:
            ComplexityAssessment or None: The reconstructed assessment from cache with "` (cached)`" appended to the reasoning if found; returns `None` when Redis is unavailable, the key is missing, or lookup/parsing fails.
        """
        if not self._redis:
            return None
        try:
            cache_key = self._make_cache_key(msg_lower)
            raw = self._redis.get(cache_key)
            if raw:
                data = json.loads(raw)
                logger.info(
                    "[AutoBrain] assessed",
                    extra={
                        "tier": 1, "complexity": data.get("complexity", "?"),
                        "confidence": data.get("confidence", 0.90),
                        "latency_ms": 0, "cache_hit": True,
                        "workspace_id": self._workspace_id,
                    },
                )
                return ComplexityAssessment(
                    complexity=Complexity(data["complexity"]),
                    action=Action(data["action"]),
                    reasoning=data.get("reasoning", "cached") + " (cached)",
                    confidence=data.get("confidence", 0.90),
                    needs_memory=data.get("needs_memory", False),
                    tool_hints=data.get("tool_hints", []),
                    needs_multi_agent=data.get("needs_multi_agent", False),
                )
        except Exception:
            logger.debug("[AutoBrain] Cache lookup failed")
        return None

    def _cache_store(self, msg_lower: str, assessment: ComplexityAssessment) -> None:
        if not self._redis:
            return
        try:
            cache_key = self._make_cache_key(msg_lower)
            from config import config
            ttl = int(config.COMPLEXITY_CACHE_TTL_HOURS or 24) * 3600
            self._redis.setex(cache_key, ttl, json.dumps(assessment.to_dict()))
        except Exception:
            logger.debug("[AutoBrain] Cache store failed, non-critical")

    def _make_cache_key(self, msg_lower: str) -> str:
        h = hashlib.sha256(msg_lower.encode()).hexdigest()[:16]
        return f"complexity:{self._workspace_id}:{h}"

    # ------------------------------------------------------------------
    # Pattern matchers (Tier 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_atom(msg_lower: str) -> bool:
        for pattern in _atom_re:
            if pattern.match(msg_lower):
                return True
        return False

    @staticmethod
    def _match_platform_query(msg_lower: str) -> Optional[str]:
        for tool_name, phrases in _PLATFORM_KEYWORDS.items():
            for phrase in phrases:
                # Word-boundary match to avoid false triggers on substrings
                if re.search(r'\b' + re.escape(phrase) + r'\b', msg_lower):
                    return tool_name

        # Catch-all: if the message explicitly mentions a platform_* action name
        # (e.g. "use platform_configure_agent_heartbeat"), treat as platform query.
        platform_match = re.search(r'\bplatform_[a-z_]+\b', msg_lower)
        if platform_match:
            return platform_match.group(0)

        return None

    @staticmethod
    def _has_action_keywords(message: str) -> bool:
        """Cheap scan for action-oriented keywords. Used only as Tier 3 fallback."""
        msg = message.lower()
        return any(kw in msg for kw in (
            "send", "email", "search", "find", "create", "open",
            "run", "fetch", "query", "calendar", "schedule",
            "deploy", "build", "delete", "update", "upload",
            "download", "generate", "analyze", "report",
        ))

    @staticmethod
    def _is_memory_recall(msg_lower: str) -> bool:
        return bool(_MEMORY_PATTERN.search(msg_lower))
