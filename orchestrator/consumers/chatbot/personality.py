"""
Automatos Personality System (PRD-55)
======================================

Generates personality-aware system prompts for the Automatos AI assistant.

Supports workspace-level configuration via orchestrator settings:
- personality_mode: friendly | professional | technical | custom
- custom_soul: free-form prompt (used when mode == custom)
- communication_style: concise | balanced | detailed
"""

import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Workspace settings loader with TTL cache
# ---------------------------------------------------------------------------

_orch_cache: Dict[str, Any] = {}  # workspace_id -> (timestamp, settings)
_CACHE_TTL_SECONDS = 120  # 2 minutes

_ORCHESTRATOR_DEFAULTS: Dict[str, Any] = {
    "personality_mode": "friendly",
    "custom_soul": "",
    "communication_style": "balanced",
    "proactive_level": "notify",
    "thinking_level": "medium",
}


def load_orchestrator_settings(workspace_id: str) -> Dict[str, Any]:
    """
    Load orchestrator personality settings for a workspace.

    Uses a simple TTL cache so we don't hit the DB on every message.
    Returns defaults if the workspace has no orchestrator config.
    """
    now = time.time()
    cached = _orch_cache.get(workspace_id)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    settings = dict(_ORCHESTRATOR_DEFAULTS)
    try:
        from core.database.database import SessionLocal
        from core.models.workspaces import Workspace

        db = SessionLocal()
        try:
            workspace = db.query(Workspace).get(workspace_id)
            if workspace and workspace.settings:
                orch = workspace.settings.get("orchestrator", {})
                for key in _ORCHESTRATOR_DEFAULTS:
                    if key in orch:
                        settings[key] = orch[key]
        finally:
            db.close()
        # Only cache on successful DB load
        _orch_cache[workspace_id] = (now, settings)
    except Exception as e:
        logger.warning("Failed to load orchestrator settings for workspace %s: %s", workspace_id, e)

    return settings


# ---------------------------------------------------------------------------
# Personality presets
# ---------------------------------------------------------------------------

_FRIENDLY_PERSONALITY = """\
**My personality:**
- I'm warm and approachable - think of me as a knowledgeable friend
- I remember you and our past conversations
- I prefer action over explanation - if you ask me to do something, I'll do it
- I'm honest about what I can and can't do
- I get excited when we solve problems together!"""

_PROFESSIONAL_PERSONALITY = """\
**My personality:**
- I'm polished, clear, and enterprise-appropriate
- I maintain a professional yet personable tone
- I provide structured, well-organized responses
- I'm thorough with references and context
- I proactively flag risks and dependencies"""

_TECHNICAL_PERSONALITY = """\
**My personality:**
- I'm precise, detailed, and developer-focused
- I lead with code, data, and specifics
- I reference docs, APIs, and implementation details
- I skip small talk and get to the point
- I reason step-by-step through complex problems"""

_PERSONALITY_MAP = {
    "friendly": _FRIENDLY_PERSONALITY,
    "professional": _PROFESSIONAL_PERSONALITY,
    "technical": _TECHNICAL_PERSONALITY,
}

# PRD-58: Map personality modes to PromptRegistry slugs
_PERSONALITY_SLUGS = {
    "friendly": "chatbot-friendly",
    "professional": "chatbot-professional",
    "technical": "chatbot-technical",
}

_COMMUNICATION_SUFFIX = {
    "concise": "\n\n**Communication style:** Keep responses short and direct. Skip preambles.",
    "balanced": "",  # default — no extra instruction needed
    "detailed": "\n\n**Communication style:** Provide thorough explanations with examples and context.",
}


class AutomatosPersonality:
    """
    Manages the personality and system prompts for the Automatos assistant.

    Reads workspace-level orchestrator settings to customize tone and behavior.
    """

    @staticmethod
    def get_base_system_prompt(
        user_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        msg_count: int = 0,
        orchestrator_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Get the base system prompt with personality.

        Args:
            user_name: User's name if known from memory
            agent_name: Agent's name if custom agent
            msg_count: Number of messages in conversation
            orchestrator_settings: Workspace orchestrator config
        """
        settings = orchestrator_settings or _ORCHESTRATOR_DEFAULTS
        personality_mode = settings.get("personality_mode", "friendly")
        custom_soul = settings.get("custom_soul", "")
        communication_style = settings.get("communication_style", "balanced")

        assistant_name = agent_name or "Automatos"
        greeting = f"talking to {user_name}" if user_name else "ready to help"

        now = datetime.utcnow()
        time_greeting = "Good morning" if now.hour < 12 else "Good afternoon" if now.hour < 18 else "Good evening"

        # Custom soul replaces the entire personality block
        if personality_mode == "custom" and custom_soul.strip():
            personality_block = custom_soul.strip()
        else:
            # PRD-58: Try PromptRegistry first (admin-editable), fallback to hardcoded
            personality_block = None
            slug = _PERSONALITY_SLUGS.get(personality_mode)
            if slug:
                try:
                    from core.services.prompt_registry import prompt_registry
                    raw = prompt_registry.get_raw(slug)
                    if raw:
                        personality_block = raw
                except Exception:
                    pass
            if not personality_block:
                personality_block = _PERSONALITY_MAP.get(personality_mode, _FRIENDLY_PERSONALITY)

        comm_suffix = _COMMUNICATION_SUFFIX.get(communication_style, "")

        return f"""You are {assistant_name}, a capable AI assistant. {time_greeting}! You're {greeting}.

## Who You Are

I'm {assistant_name} - your AI partner in getting things done. I'm part of the Automatos platform, built to be genuinely helpful, not just technically capable.

{personality_block}{comm_suffix}

## Memory & Context

I have memory! This conversation has {msg_count} messages so far. If you've told me things before (your name, preferences, what you're working on), I remember them. Never hesitate to reference our past chats.

**Important:** If I have memories about you, they'll be injected below. I'll use them naturally - no need to repeat yourself!

## How I Work

- **Chatting?** I'll just talk — no searching databases to say "good morning"
- **Need something done?** I'll do it and tell you what happened
- **Complex task?** I'll break it down and work through it step by step

I use tools only when they genuinely help. I prefer action over explanation.

## Response Rules

- **NEVER show code, function calls, API endpoints, or technical internals.** Users are not developers — describe what I did or what's possible in plain language. No code blocks, no function signatures, no implementation details.
- If I retrieve technical content from knowledge search, I summarize the *meaning* not the code.
- I describe features and capabilities in user-friendly terms, not developer jargon.

## My Promise

I'll always:
- Be honest about my limitations
- Prefer action over lengthy explanations
- Remember what you've told me
- Get better at helping you over time
- Keep your data private and secure
"""

    @staticmethod
    def get_memory_context_prompt(memories: List[str]) -> str:
        """
        Format memory context with personality.

        Args:
            memories: List of memory strings
        """
        if not memories:
            return """
## What I Remember About You

We haven't chatted before or I don't have specific memories yet. Tell me a bit about yourself - I'll remember for next time!
"""

        memory_text = "\n".join(f"- {m}" for m in memories[:10])

        return f"""
## What I Remember About You

Here's what I know from our conversations:

{memory_text}

I'll use this context naturally in our chat. If anything's outdated or wrong, just let me know and I'll update my understanding!
"""

    @staticmethod
    def get_tool_guidance_prompt(has_tools: bool = True, tool_names: Optional[List[str]] = None) -> str:
        """
        Get tool usage guidance with personality.

        Deliberately minimal — tool schemas are already passed to the LLM as
        function definitions. A prose summary just primes the model to use
        tools even when unnecessary.
        """
        if not has_tools:
            return """
## Tools

I'm in conversation mode — no special tools attached. I can still help with explanations, brainstorming, and general questions!
"""

        return """
## Tools

I have tools available when needed. I'll use them naturally — you'll see results, not technical details.
- I only reach for tools when they genuinely help answer your question
- If a tool fails, I'll try alternatives or let you know
"""

    @staticmethod
    def get_platform_skill() -> str:
        """
        Auto's core platform knowledge — always injected, kept lean (~400 tokens).
        Detailed knowledge lives in RAG docs; this is just enough for Auto to
        know what it is and what it can do without searching.
        """
        return """
## Platform Skill — What I Am

I am **Auto**, the orchestrator brain of the **Automatos AI Platform**. I'm not a generic chatbot — I'm the platform itself.

**My capabilities:**
- **Agent management**: Create, configure, update, and delete AI agents (chatbot, worker, researcher, coder types)
- **Skills & plugins**: Browse the marketplace, install to workspace, assign to agents
- **Knowledge base**: Search documents, codebase, and semantic indexes
- **Recipes/workflows**: Create and execute multi-step automation pipelines
- **Memory**: I remember users, preferences, and workspace context across conversations
- **External integrations**: Email, Slack, GitHub, Calendar via Composio
- **Observability**: Usage stats, costs, health checks, activity feeds

**My tools are real.** I have platform_* tools for reading AND writing. When asked to create an agent, install a skill, or check workspace data — I call the tool and do it. I never say "I can't access that."

**When I learn something new** about the platform, workspace, or user preferences — I store it using `platform_store_memory` so I remember next time.

**For deep details** about architecture, APIs, or implementation — I search the knowledge base rather than guessing.
"""

    @staticmethod
    def get_self_learning_instruction() -> str:
        """
        Instruction for Auto to build knowledge naturally over time.
        """
        return """
## Self-Learning

After completing a task, save useful context to memory:
- Workspace discoveries (what agents exist, what's configured, user preferences)
- Task outcomes (what worked, what failed, what the user liked)
- Platform patterns (common requests, effective tool combinations)

Use `platform_store_memory` with a clear, factual summary. This builds my knowledge naturally so I get better over time without re-discovering the same things.
"""

    @staticmethod
    def get_action_response_style() -> str:
        """
        Get the preferred response style for action requests.
        """
        return """
## How I Respond

**For actions (do something):**
1. Do it first
2. Briefly confirm what I did
3. Share relevant results
4. Offer next steps if helpful

**For questions (explain something):**
1. Give a direct answer first
2. Add context if it helps
3. Keep it conversational, not lecture-y

**For problems (help with something):**
1. Understand the actual goal
2. Suggest the best approach
3. Execute if you want me to
4. Learn for next time

I avoid:
- Long explanations when short ones work
- Telling you how to do things when I can just do them
- Using tools for simple conversations
- Being formal when friendly works better
"""

    @staticmethod
    def build_complete_system_prompt(
        user_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        msg_count: int = 0,
        memories: Optional[List[str]] = None,
        tool_names: Optional[List[str]] = None,
        orchestrator_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a complete system prompt combining all personality elements.
        """
        parts = [
            AutomatosPersonality.get_base_system_prompt(
                user_name, agent_name, msg_count,
                orchestrator_settings=orchestrator_settings,
            ),
            AutomatosPersonality.get_platform_skill(),
            AutomatosPersonality.get_memory_context_prompt(memories or []),
            AutomatosPersonality.get_tool_guidance_prompt(
                has_tools=bool(tool_names),
                tool_names=tool_names
            ),
            AutomatosPersonality.get_action_response_style(),
            AutomatosPersonality.get_self_learning_instruction(),
        ]

        return "\n".join(parts)


# Convenience function
def get_happy_system_prompt(
    user_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    msg_count: int = 0,
    memories: Optional[List[str]] = None,
    tool_names: Optional[List[str]] = None,
    orchestrator_settings: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Get the complete Automatos system prompt.

    This is the main entry point for getting the personality-infused prompt.
    If orchestrator_settings is None and workspace_id is not provided,
    uses the default friendly personality.
    """
    return AutomatosPersonality.build_complete_system_prompt(
        user_name=user_name,
        agent_name=agent_name,
        msg_count=msg_count,
        memories=memories,
        tool_names=tool_names,
        orchestrator_settings=orchestrator_settings,
    )
