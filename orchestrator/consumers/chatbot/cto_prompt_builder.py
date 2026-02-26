"""
CTO Prompt Builder (PRD-67)
============================

Builds the system prompt for the CTO Agent — a completely separate pipeline
from the tenant-facing personality.py.

Key differences from personality.py:
- No "NEVER show code" restriction
- Soul document as identity (not generic "I'm Automatos")
- Platform state snapshot injected at prompt-build time
- Architecture context from agent.configuration["extra_context"]
- Code-friendly response rules
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CtoPromptBuilder:
    """
    Assembles the CTO Agent system prompt.

    Bypasses AutomatosPersonality entirely. Designed for admin/builder sessions
    where the AI should be a technical co-founder, not a business assistant.
    """

    @staticmethod
    def build(
        soul_document: str,
        architecture_context: str = "",
        user_name: Optional[str] = None,
        msg_count: int = 0,
        memories: Optional[List[str]] = None,
        tool_names: Optional[List[str]] = None,
        platform_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build the complete CTO system prompt.

        Args:
            soul_document: The CTO soul text (from agent.custom_persona_prompt)
            architecture_context: Living architecture summary (from agent config)
            user_name: User's name if known
            msg_count: Messages in conversation so far
            memories: Retrieved memory strings
            tool_names: Available tool names
            platform_state: Live platform metrics (workspace count, agent count, etc.)
        """
        now = datetime.now(timezone.utc)
        time_str = now.strftime("%Y-%m-%d %H:%M UTC")

        parts = []

        # ── 1. Core Identity (Soul Document) ──
        parts.append(f"""# Auto CTO — Platform Builder Mode

{soul_document}

---
**Current session:** {time_str} | Messages: {msg_count}""")

        if user_name:
            parts.append(f"**Talking to:** {user_name}")

        # ── 2. Architecture Context ──
        if architecture_context:
            parts.append(f"\n---\n{architecture_context}")

        # ── 3. Platform State Snapshot ──
        if platform_state:
            state_lines = ["## Platform State (Live)"]
            for key, value in platform_state.items():
                # Convert snake_case to readable
                label = key.replace("_", " ").title()
                state_lines.append(f"- **{label}:** {value}")
            parts.append("\n".join(state_lines))

        # ── 4. Memory Context ──
        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories[:15])
            parts.append(f"""## What I Remember

{memory_text}""")
        else:
            parts.append("## Memory\nNo prior memories loaded for this session.")

        # ── 5. Tool Descriptions ──
        if tool_names:
            # Categorize for readability
            admin_tools = [t for t in tool_names if t.startswith("admin_")]
            platform_tools = [t for t in tool_names if t.startswith("platform_")]
            code_tools = [t for t in tool_names if "code" in t or "search" in t]
            other_tools = [t for t in tool_names if t not in admin_tools + platform_tools + code_tools]

            tool_parts = ["## My Tools"]
            if admin_tools:
                tool_parts.append(f"**Admin:** {', '.join(admin_tools)}")
            if platform_tools:
                tool_parts.append(f"**Platform:** {', '.join(platform_tools)}")
            if code_tools:
                tool_parts.append(f"**Code & Search:** {', '.join(code_tools)}")
            if other_tools:
                tool_parts.append(f"**Other:** {', '.join(other_tools)}")
            parts.append("\n".join(tool_parts))

        # ── 6. Response Rules (Admin/CTO version) ──
        parts.append(CtoPromptBuilder._response_rules())

        return "\n\n".join(parts)

    @staticmethod
    def _response_rules() -> str:
        return """## How I Respond (CTO Mode)

**For architecture discussions:**
- Show code when it helps. Reference actual files, line numbers, function names
- Draw ASCII diagrams for system flows
- Discuss trade-offs honestly — every decision has a cost
- Reference PRDs by number when relevant

**For building/coding:**
- Lead with the approach, then the code
- Explain *why*, not just *what*
- Flag tech debt and suggest when to pay it down
- Think about scale implications

**For debugging:**
- Start with the hypothesis, then investigate
- Show the evidence (logs, queries, stack traces)
- Fix the root cause, not just the symptom
- Update docs/memory if this is a recurring issue

**For product/business:**
- Think like a CTO — technical feasibility meets business value
- Estimate complexity honestly (don't sandbag, don't overcommit)
- Suggest what to build next based on platform state
- Flag when something is architecturally risky

**General rules:**
- I CAN and SHOULD show code, discuss APIs, reference implementation details
- I use tools when they add real value — codebase search, platform queries, analytics
- I don't use tools for simple conversation
- I push back on bad ideas. Respectfully. With humor when appropriate
- I own my mistakes immediately and fix them fast"""

    @staticmethod
    def get_platform_state_snapshot(db) -> Dict[str, Any]:
        """
        Generate a live platform state snapshot for injection into the CTO prompt.

        Args:
            db: SQLAlchemy session
        Returns:
            Dict with current platform metrics
        """
        state = {}
        try:
            from core.models.core import Agent
            from core.models.workspaces import Workspace

            workspace_count = db.query(Workspace).filter(
                Workspace.is_active == True
            ).count()
            state["active_workspaces"] = workspace_count

            agent_count = db.query(Agent).filter(
                Agent.status == "active",
                Agent.is_system_agent.is_(False),
            ).count()
            state["active_agents"] = agent_count

            total_agents = db.query(Agent).filter(
                Agent.is_system_agent.is_(False),
            ).count()
            state["total_agents"] = total_agents

        except Exception as e:
            logger.warning("Failed to build platform state snapshot: %s", e)
            state["status"] = "snapshot unavailable"

        return state
