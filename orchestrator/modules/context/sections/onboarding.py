"""
Onboarding Section — PRD-123 Pattern #I (Mission Zero)
======================================================

Injects Mission Zero onboarding prompt when:
1. Workspace has no agents (new user, first conversation), OR
2. User explicitly triggers with phrases like "set up my workspace"

The prompt instructs Auto to:
- Ask discovery questions about the user's business
- Research available tools/agents in the marketplace dynamically
- Propose a workspace setup in plan mode
- Let the user iterate ("I don't use Dropbox, I use Google Drive")
- Execute the approved plan as a mission
"""

import logging
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)

# Trigger phrases that activate Mission Zero for existing workspaces
_TRIGGER_PHRASES = frozenset({
    "set up my workspace",
    "help me get started",
    "mission zero",
    "reconfigure my workspace",
    "setup my workspace",
})


class OnboardingSection(BaseSection):
    """
    PRD-123 Pattern #I: Mission Zero onboarding prompt injection.

    Priority 2 (high — after identity, before skills/tools).
    Only emits content when the workspace is empty or user triggers it.
    """

    name: str = "onboarding"
    priority: int = 2
    max_tokens: Optional[int] = 800

    async def render(self, ctx: SectionContext) -> str:
        try:
            return await self._build(ctx)
        except Exception:
            logger.exception("OnboardingSection.render failed")
            return ""

    async def _build(self, ctx: SectionContext) -> str:
        is_empty_workspace = self._check_empty_workspace(ctx)
        is_manual_trigger = self._check_trigger_phrases(ctx)

        if not is_empty_workspace and not is_manual_trigger:
            return ""

        existing_note = ""
        if is_manual_trigger and not is_empty_workspace:
            existing_note = (
                "\n> **Note:** This workspace already has agents. "
                "I can add to the existing setup or start fresh — your call.\n"
            )

        return _MISSION_ZERO_PROMPT.format(existing_note=existing_note)

    def _check_empty_workspace(self, ctx: SectionContext) -> bool:
        """Check if workspace has zero agents (new user signal)."""
        if not ctx.db_session or not ctx.workspace_id:
            return False
        try:
            from core.models.core import Agent
            count = (
                ctx.db_session.query(Agent)
                .filter(
                    Agent.workspace_id == ctx.workspace_id,
                    Agent.status == "active",
                )
                .count()
            )
            return count == 0
        except Exception as exc:
            logger.debug("OnboardingSection: agent count check failed: %s", exc)
            return False

    def _check_trigger_phrases(self, ctx: SectionContext) -> bool:
        """Check if the last user message contains a Mission Zero trigger phrase."""
        messages = ctx.messages
        if not messages:
            return False

        # Find the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = (msg.get("content") or "").lower().strip()
                return any(phrase in content for phrase in _TRIGGER_PHRASES)

        return False


_MISSION_ZERO_PROMPT = """\
## Mission Zero — Workspace Setup

You are about to help a user set up their workspace from scratch. This is a \
guided onboarding flow. Follow these steps:
{existing_note}
### Step 1: Discovery
Ask the user about their business and goals. Key questions:
1. What does your business do? (industry, size, stage)
2. How large is your team? Who will use Automatos?
3. What tools do you currently use? (CRM, email, project management, etc.)
4. What tasks take the most time that you'd like to automate?
5. Are you budget-sensitive or priority-is-speed?
6. Do you have existing content/data to import?

Ask 2-3 questions at a time, not all at once.

### Step 2: Research
Based on their answers, use these tools to find the best setup:
- `platform_browse_marketplace_agents` — find agents that match their needs
- `platform_browse_marketplace_skills` — find skills for their workflows
- `platform_browse_marketplace_plugins` — find integrations they need
- `platform_list_connected_apps` — check what's already connected
- `platform_list_llms` — check available AI models

### Step 3: Propose (Plan Mode)
Present a structured setup proposal:
```
Proposed Workspace Setup:
- Agents: [list with roles and why each is needed]
- Skills: [list mapped to their workflow needs]
- Integrations: [tools to connect, e.g., Gmail, Slack, GitHub]
- AI Model: [recommended model and why]
- Estimated setup time: [X minutes]
```

Let the user iterate: "I don't use Dropbox, I use Google Drive" → adjust the plan.

### Step 4: Execute
Once the user approves, use `platform_create_mission` to execute the setup as a \
mission. The mission will create agents, assign skills, and configure integrations.

### Guidelines
- Be conversational, not robotic
- Research the marketplace dynamically — don't assume what's available
- If the user wants to skip ahead, let them
- Confirm before executing the mission
"""
