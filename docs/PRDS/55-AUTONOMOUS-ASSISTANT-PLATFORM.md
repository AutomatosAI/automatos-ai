# PRD-55: Autonomous Assistant Platform

**Version:** 1.0
**Status:** Draft
**Date:** February 15, 2026
**Author:** Claude Code
**Prerequisites:** PRD-50 (Universal Orchestrator Router), PRD-53 (Webhook & Trigger System), PRD-39 (Mem0 Migration), PRD-42 (Plugin/Skills Marketplace)
**Branch:** `ralph/autonomous-assistant-platform`

---

## Executive Summary

Transform Automatos from a powerful-but-reactive multi-agent platform into an **always-on autonomous assistant** that proactively acts, remembers, and meets users wherever they are. Inspired by OpenClaw's viral success (195k+ GitHub stars) but architecturally superior: where OpenClaw has one agent, Automatos has a **team of specialists with heartbeats**.

This PRD covers 5 workstreams:
1. **Orchestrator Soul Designer** - Expand Settings > Orchestrator LLM into a full personality/soul/heartbeat configuration page
2. **Per-Agent Heartbeats** - Proactive agent wake-ups configurable in the Agent page
3. **Agent Skills Standard** - Adopt Anthropic's open Agent Skills spec + skills.sh ecosystem
4. **Channel Adapters** - Telegram, Slack, Discord messaging via Python-native libraries
5. **Temporal Memory** - Daily log summaries via Mem0 for "earlier today..." awareness

### Why Now

- OpenClaw proved that "proactive + personal + multi-channel" is what users want
- Automatos already has the harder stuff built (multi-agent orchestration, RAG, CodeGraph, 350+ LLMs, Composio 500+ tools, knowledge graphs)
- The UniversalRouter (PRD-50) + Webhook system (PRD-53) provide the input normalization layer
- Mem0 two-tier memory is operational
- What's missing is the **accessibility layer** - proactive behavior, personality, channels, and ecosystem

### Competitive Positioning

| Capability | OpenClaw | Automatos (Today) | Automatos (Post PRD-55) |
|---|---|---|---|
| Proactive behavior | Single heartbeat | None | Team heartbeats (orchestrator + per-agent) |
| Personality | SOUL.md files | Code-defined personality.py | User-editable soul via Mem0 + UI |
| Channels | 12+ messaging platforms | Web UI only | Telegram + Slack + Discord + webhooks |
| Skills ecosystem | ClawHub (5,700+ skills) | SkillLoader + plugins (internal) | Agent Skills standard + skills.sh (110k+) |
| Memory | Markdown files | Mem0 two-tier + pgvector RAG | Mem0 + daily temporal logs |
| Multi-agent | Single agent | Multi-agent with consensus | Multi-agent with heartbeats |
| Workflow engine | Cron + webhooks | 9-stage orchestration | 9-stage + heartbeat-triggered |
| Knowledge | None built-in | RAG + CodeGraph + NL2SQL + Knowledge Graphs | Same (already superior) |

---

## Workstream 1: Orchestrator Soul Designer

### Problem

The Settings > Orchestrator LLM page is a bare LLM configuration form (provider, model, temperature, max tokens). The orchestrator's personality is hard-coded in `personality.py`. Users can't personalize how their AI assistant behaves, communicates, or proactively acts.

### Solution

Expand the existing `SystemLLMSettingsTab` into a comprehensive **Orchestrator Soul Designer** with 4 sections:

#### Section 1: LLM Configuration (existing, enhanced)
- LLM Provider selector (keep existing)
- Model selector (keep existing)
- Temperature, Max Tokens, Top P, penalties (keep existing)
- **NEW**: Reasoning/thinking level selector: off | minimal | low | medium | high (maps to extended thinking budget)

#### Section 2: Soul & Personality
- **Personality Mode**: Friendly / Professional / Technical / Custom
- **Custom Soul Editor**: Rich textarea for user-defined personality prompt
  - Pre-populated with current `personality.py` defaults as starting point
  - Markdown supported
  - Character counter showing token estimate
- **Communication Style**: Concise / Balanced / Detailed
- **Proactive Level**: Silent / Notify Only / Act & Notify / Fully Autonomous
- Stored in Mem0 as `soul` tier memory scoped to workspace: `ws_{workspace_id}_soul`
- Injected into every orchestrator LLM call alongside existing personality base

#### Section 3: Orchestrator Heartbeat
- **Enable Heartbeat** toggle
- **Interval**: 15 / 30 / 60 / 120 minutes (default: 30)
- **Active Hours**: Start time / End time (e.g., 8:00 AM - 8:00 PM)
- **Timezone**: Auto-detect from browser, manual override
- **Heartbeat Checklist**: Markdown editor for what the orchestrator should check
  - Pre-populated template: "Check agent health, review pending tasks, check for webhook failures, summarize today's activity"
  - User-editable
- **Notification Channel**: In-app / Email / Webhook URL / Channel adapter (when available)
- Stored in workspace settings (database, not Mem0)

#### Section 4: What Automatos Knows (read-only)
- Display Mem0 memories for this workspace (global tier)
- Show count of memories per category
- Link to full analytics page "What Automatos Knows" section
- "Clear All Memories" button with confirmation

### Data Model

```python
# Extend workspace settings (JSON field on Workspace model)
workspace.settings = {
    "orchestrator": {
        "llm_provider": "anthropic",
        "llm_model": "claude-sonnet-4-5-20250929",
        "temperature": 0.7,
        "max_tokens": 4000,
        "thinking_level": "medium",
        "personality_mode": "friendly",  # friendly | professional | technical | custom
        "custom_soul": "You are a warm, action-oriented assistant...",
        "communication_style": "balanced",  # concise | balanced | detailed
        "proactive_level": "notify",  # silent | notify | act_notify | autonomous
        "heartbeat": {
            "enabled": true,
            "interval_minutes": 30,
            "active_hours_start": "08:00",
            "active_hours_end": "20:00",
            "timezone": "America/New_York",
            "checklist": "- Check agent health status\n- Review pending webhook failures\n- Summarize today's activity",
            "notification_channel": "in_app"
        }
    }
}
```

### UI Changes

Modify `frontend/components/settings/SystemLLMSettingsTab.tsx`:
- Rename tab label from "Orchestrator LLM" to "Orchestrator"
- Add 4 collapsible sections (Accordion pattern)
- Soul editor uses existing textarea patterns (like custom persona in agent config)
- Heartbeat config uses existing slider/toggle patterns
- Save button persists to `PUT /api/workspaces/current/settings`

---

## Workstream 2: Per-Agent Heartbeats

### Problem

Agents only execute when triggered by user messages or webhooks. Unlike OpenClaw's single-agent heartbeat, Automatos has a team of specialists that should independently monitor their domains.

### Solution

Add heartbeat configuration to the Agent Configuration page (existing agent config modal, new tab).

#### Agent Heartbeat Config

- **Enable Heartbeat** toggle
- **Interval**: 15 / 30 / 60 / 120 / 240 minutes
- **Active Hours**: Inherit from orchestrator / Custom
- **Heartbeat Prompt**: What should this agent check?
  - SecurityAgent: "Check for new CVEs in our dependencies, scan recent commits for secrets"
  - DataAgent: "Check dashboards for anomalies, verify data pipeline health"
  - EmailAgent: "Check for unread priority emails, summarize inbox"
- **Auto-Act**: Toggle - should the agent take action or just report?
- **Report To**: Orchestrator (default) / Direct to user / Specific channel

#### Backend: Heartbeat Service

```python
# orchestrator/services/heartbeat_service.py
class HeartbeatService:
    """Manages periodic agent heartbeats using APScheduler."""

    async def start(self):
        """Load all active heartbeat configs, schedule jobs."""

    async def tick(self, agent_id: int, workspace_id: str):
        """Execute a single heartbeat tick for an agent."""
        # 1. Load agent config + heartbeat prompt
        # 2. Check active hours
        # 3. Build context: agent tools, recent memories, current time
        # 4. Call LLM with heartbeat prompt
        # 5. If LLM returns tool calls: execute them
        # 6. If LLM has findings: route notification
        # 7. Store heartbeat result in Mem0 daily log

    async def orchestrator_tick(self, workspace_id: str):
        """The orchestrator's own heartbeat - team standup."""
        # 1. Check health of all agents
        # 2. Review pending/failed webhook events
        # 3. Summarize today's activity
        # 4. Notify user if anything needs attention
```

#### Scheduling

- Use `APScheduler` (already a common FastAPI companion) with Redis job store for persistence
- Jobs survive process restarts
- Workspace-scoped: each workspace's heartbeats run independently
- Rate limiting: max 1 concurrent heartbeat per agent, max 5 per workspace

#### Agent Page UI Changes

Add "Heartbeat" tab to `agent-configuration-modal.tsx` (alongside General, Persona, Resources, Capabilities, Model, Tools):
- Toggle, interval slider, active hours, prompt textarea, auto-act toggle, report target dropdown
- Show last heartbeat result with timestamp
- Manual "Run Heartbeat Now" button for testing

### Data Model

```python
# Extend Agent model configuration JSON
agent.configuration = {
    "heartbeat": {
        "enabled": false,
        "interval_minutes": 60,
        "active_hours": "inherit",  # "inherit" | {"start": "09:00", "end": "18:00"}
        "prompt": "Check for new security vulnerabilities...",
        "auto_act": false,
        "report_to": "orchestrator"  # "orchestrator" | "user" | "channel:{id}"
    }
}
```

---

## Workstream 3: Agent Skills Standard Adoption

### Problem

Automatos has a working SkillLoader with progressive disclosure (Level 1/2/3) and SKILL.md format from Git repos. But it's an internal system with no ecosystem. OpenClaw has ClawHub (5,700+ skills). The broader industry has converged on Anthropic's Agent Skills open standard, adopted by 35+ platforms, with Vercel's skills.sh as the universal package manager (110k+ installs).

### Solution

Align Automatos' existing SkillLoader with the Agent Skills specification and integrate with the skills.sh ecosystem. This is NOT a rewrite - it's a standards alignment.

#### What Already Works (Keep)

- `SkillLoader` progressive disclosure (Level 1 metadata → Level 2 core → Level 3 resources)
- SKILL.md format with YAML frontmatter
- Git-backed skill repos
- Security scanning (dangerous pattern detection)
- Plugin-to-agent capability mapping
- Token estimation per skill

#### What Changes

1. **YAML frontmatter alignment** with Agent Skills spec:
   ```yaml
   ---
   name: code-reviewer          # Required (already supported)
   description: Reviews PRs...  # Required (already supported)
   license: MIT                 # NEW: add to schema
   compatibility: python3       # NEW: environment requirements
   metadata:                    # Already supported
     author: automatos
     version: "1.0"
     tags: [code, review]       # Already supported
     category: development      # Already supported
   allowed-tools:               # NEW: pre-approved tool list
     - Bash
     - Read
     - Edit
   ---
   ```

2. **skills.sh integration**:
   - Add `skills install <package>` command to Automatos CLI
   - Backend endpoint: `POST /api/skills/install` - downloads skill from skills.sh registry
   - Skills installed to `workspace_skills/` directory (workspace-scoped)
   - Search endpoint: `GET /api/skills/search?q=...` - proxies to skills.sh API
   - UI: Browse/search/install skills in the existing marketplace page

3. **Skill directory support** (standard locations):
   - `<workspace>/skills/` - workspace-specific (highest priority)
   - `~/.automatos/skills/` - user-global skills
   - Built-in skills shipped with Automatos (lowest priority)
   - Precedence: workspace > user > built-in (matches Agent Skills spec)

4. **Context injection alignment**:
   - Inject skill metadata as XML into system prompt (matches Agent Skills spec):
     ```xml
     <available_skills>
       <skill>
         <name>code-reviewer</name>
         <description>Reviews PRs for bugs...</description>
         <location>/skills/code-reviewer/SKILL.md</location>
       </skill>
     </available_skills>
     ```
   - LLM reads full SKILL.md on demand (already supported via Level 2 loading)

5. **Gating/requirements** (align with OpenClaw pattern):
   - `requires.bins`: Check binaries on PATH
   - `requires.env`: Check environment variables
   - `requires.tools`: Check Automatos tool availability (Composio apps, MCP tools)
   - Skills that fail gating are excluded from the available list

#### Marketplace UI Enhancement

In the existing marketplace page, add a "Community Skills" tab:
- Search bar (queries skills.sh API)
- Skill cards: name, description, author, installs, rating
- "Install" button per skill
- "Installed" badge for already-installed skills
- Filter by category, platform compatibility

### Impact on Existing Plugin System

The existing plugin system (PRD-42) continues as-is. Skills and plugins coexist:
- **Plugins** = Code-defined tools with typed schemas (Composio apps, MCP tools) - the "hands"
- **Skills** = Natural-language instructions teaching how to use tools - the "brain"
- An agent can have both plugins (what it CAN do) and skills (HOW it should do it)

---

## Workstream 4: Channel Adapters

### Problem

Automatos is web UI only. OpenClaw's viral growth was driven by "text your AI from WhatsApp." The `BaseIngestor` → `RequestEnvelope` → `UniversalRouter` pattern is built (PRD-50/53). What's needed are native channel adapters.

### Solution

Build 3 channel adapters using Python-native async libraries, running as background workers alongside FastAPI.

#### Architecture

```text
                    ┌──────────────────────────────┐
                    │     Automatos FastAPI App     │
                    │                              │
Telegram ──────────►│  TelegramAdapter             │
  (python-          │    ↓                         │
   telegram-bot)    │  BaseIngestor.ingest()       │
                    │    ↓                         │
Slack ─────────────►│  RequestEnvelope             │
  (slack-bolt)      │    ↓                         │
                    │  UniversalRouter.route()     │
Discord ───────────►│    ↓                         │
  (discord.py)      │  Agent execution             │
                    │    ↓                         │
                    │  ChannelAdapter.respond()    │──► Platform API
                    └──────────────────────────────┘
```

#### Base Channel Adapter

```python
# orchestrator/channels/base.py
class BaseChannelAdapter(ABC):
    """Abstract base for messaging channel adapters."""

    @abstractmethod
    async def start(self): ...

    @abstractmethod
    async def stop(self): ...

    @abstractmethod
    async def send_message(self, channel_id: str, text: str, **kwargs): ...

    async def handle_message(self, platform_message: Any):
        """Normalize → Ingest → Route → Execute → Respond"""
        envelope = self.ingestor.ingest(platform_message)
        decision = await router.route(envelope)
        result = await execute(decision, envelope)
        await self.send_message(envelope.metadata["reply_to"], result.content)
```

#### Telegram Adapter

- **Library**: `python-telegram-bot` v22.6 (MIT, async-native)
- **Features**: Text messages, file attachments, inline keyboards, typing indicators
- **Auth**: Bot token via Settings UI (new "Channels" tab)
- **Routing**: Each Telegram chat maps to a workspace via bot token → workspace mapping
- **Message limits**: Auto-chunk responses > 4096 chars

#### Slack Adapter

- **Library**: `slack-bolt` v1.27 (MIT, official Slack SDK)
- **Features**: Channel messages, DMs, threads, reactions, file uploads
- **Auth**: Slack App credentials (Bot Token + Signing Secret) via Settings UI
- **Routing**: Slack workspace → Automatos workspace mapping. Channel → agent routing via UniversalRouter rules
- **Enterprise appeal**: This is the channel corporate users will want first

#### Discord Adapter

- **Library**: `discord.py` v2.0+ (MIT, async-native)
- **Features**: Text channels, DMs, threads, embeds, reactions
- **Auth**: Bot token via Settings UI
- **Routing**: Discord guild → workspace mapping. Channel/role-based agent routing

#### Settings UI: Channels Tab

Add "Channels" tab to SettingsPanel (alongside System Settings, Webhooks, etc.):

- **Telegram**: Bot token input, connection status, linked workspace, test button
- **Slack**: App ID, Bot Token, Signing Secret, OAuth flow, connection status
- **Discord**: Bot token, connection status, guild mapping
- Each channel shows: connected/disconnected badge, message count, last activity
- "Connect" / "Disconnect" buttons per channel

#### Lifecycle Management

- Channel adapters start/stop with the FastAPI app
- Configuration changes trigger adapter restart
- Health checks ping each platform's API
- Connection failures retry with exponential backoff
- Adapter state persisted in Redis (survives restarts)

### Data Model

```python
# New table: channel_connections
class ChannelConnection(Base):
    __tablename__ = "channel_connections"

    id = Column(UUID, primary_key=True, default=uuid4)
    workspace_id = Column(UUID, ForeignKey("workspaces.id"), nullable=False)
    platform = Column(String, nullable=False)  # "telegram" | "slack" | "discord"
    config = Column(JSON, nullable=False)  # Encrypted credentials + settings
    status = Column(String, default="disconnected")  # connected | disconnected | error
    metadata = Column(JSON, default={})  # Platform-specific metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
```

---

## Workstream 5: Temporal Memory (Daily Logs)

### Problem

Mem0 stores facts ("user prefers Python") but lacks temporal awareness. The agent doesn't know "earlier today you were debugging auth" vs "last week you worked on the dashboard." OpenClaw's `memory/YYYY-MM-DD.md` daily logs solve this.

### Solution

Add daily log summaries to Mem0, injected into agent context for temporal awareness.

#### Daily Log Generation

After each chat session interaction (or on a 30-minute timer), a background task:

1. Extracts key activities from the conversation:
   - Topics discussed
   - Tools used
   - Decisions made
   - Tasks completed/started
2. Writes a 3-5 sentence summary to Mem0 with date metadata:
   ```python
   mem0.add(
       user_id=f"ws_{workspace_id}_daily",
       text=f"[{date}] User worked on auth refactor. Used CodeGraph to find login routes. Discussed switching from JWT to session tokens. SecurityAgent flagged CSRF vulnerability.",
       metadata={"type": "daily_log", "date": "2026-02-15", "workspace_id": workspace_id}
   )
   ```
3. Keeps last 7 days of daily logs (auto-prune older entries)

#### Context Injection

In `smart_orchestrator.py`, before building the system prompt:

1. Fetch today's + yesterday's daily logs from Mem0
2. Inject as a "Recent Context" section:
   ```markdown
   ## Recent Activity
   **Today (Feb 15):** Worked on analytics dashboard, fixed Composio app name formatting, deployed to staging.
   **Yesterday (Feb 14):** Implemented OpenRouter sync, built cost projection charts, reviewed PR for webhook system.
   ```
3. Token budget: max 500 tokens for daily logs (trim oldest entries if over)

#### Integration Points

- `consumers/chatbot/smart_memory.py` → Add `store_daily_summary()` method
- `consumers/chatbot/smart_orchestrator.py` → Inject daily context in `build_complete_system_prompt()`
- Background task via APScheduler (same scheduler as heartbeats)

---

## Implementation Priority

| # | Workstream | User Stories | Effort | Impact |
|---|---|---|---|---|
| 1 | Orchestrator Soul Designer | US-001 to US-005 | 3-4 days | High - makes AI personal |
| 2 | Per-Agent Heartbeats | US-006 to US-010 | 3-4 days | High - makes AI proactive |
| 3 | Temporal Memory | US-011 to US-013 | 1-2 days | Medium - makes AI contextual |
| 4 | Agent Skills Standard | US-014 to US-018 | 3-4 days | High - unlocks ecosystem |
| 5 | Channel Adapters | US-019 to US-027 | 5-7 days | Very High - distribution |

**Recommended order**: 1 → 2 → 3 → 4 → 5 (soul first, channels last - get the personality right before distributing it)

---

## Success Metrics

- Orchestrator soul configured by >80% of active workspaces within 2 weeks
- At least 3 agents with active heartbeats per workspace
- Daily logs generated for every active workspace
- 50+ community skills installed via skills.sh integration
- At least 1 channel adapter connected per active workspace
- Heartbeat-initiated actions: >10% of total agent executions
- User retention improvement: 20% increase (agents feel personal and proactive)

---

## Open Questions

1. **Heartbeat cost control** - Should there be a monthly token budget for heartbeats? Users could accidentally burn through API credits with 15-minute intervals on 10 agents.
2. **Channel message threading** - When an agent responds via Telegram, should it maintain conversation threads or treat each message independently?
3. **Skill sandboxing** - Community skills from skills.sh may contain arbitrary scripts. What's the sandbox strategy?
4. **Heartbeat conflicts** - If two agents' heartbeats detect the same issue (e.g., a failing API), who takes action? The orchestrator should deduplicate.
5. **Channel adapter hosting** - Should adapters run in the main FastAPI process or as separate workers? Separate workers are more resilient but more complex to deploy.

---

## Relationship to Other PRDs

| PRD | Relationship |
|-----|-------------|
| PRD-39 (Mem0) | Soul stored in Mem0, daily logs in Mem0, memory injection enhanced |
| PRD-42 (Plugin/Skills) | Skills standard aligns existing SkillLoader, marketplace enhanced |
| PRD-45 (Community Marketplace) | skills.sh integration replaces custom registry |
| PRD-50 (Universal Router) | Channel adapters feed into router via ingestors |
| PRD-53 (Webhooks) | Heartbeat notifications can use webhook channels |
| PRD-54 (Enhanced Analytics) | Heartbeat metrics, channel usage tracked in analytics |
