# PRD-71: Unified Skills Architecture

**Version:** 1.0
**Status:** Draft — Awaiting Review
**Priority:** P1 — High
**Author:** Gar Kavanagh + Auto CTO
**Created:** 2026-03-04
**Updated:** 2026-03-04
**Dependencies:** PRD-22 (Skills / Git-Backed Skill Loading), PRD-42 (Plugin Marketplace), PRD-02 (Agent Factory)
**Branch:** `fix/pentest-remediation-70` (will move to dedicated branch before implementation)

---

## Executive Summary

PRD-22 introduced Skills — Git-backed SKILL.md files that inject methodology into agent system prompts. PRD-42 introduced Plugins — zip packages uploaded to a Marketplace with security scanning and admin approval. Both systems do the same thing: teach agents HOW to do tasks by injecting content into system prompts. Having two systems creates bugs, confusion, and maintenance overhead.

**This PRD unifies them into one system.**

### The Problem in One Sentence

An agent with a "workspace-git" skill AND an "email" plugin silently loses ALL skill content because the codebase has a mutual exclusion bug: `if has_plugins → skip skills`.

### What Changes

| Today | After PRD-71 |
|-------|-------------|
| Two concepts: Skills + Plugins | One concept: **Skills** |
| Two distribution channels: Git import + Marketplace upload | One channel: **Marketplace** (admin uploads, security scans, user enables) |
| Three injection points: agent_factory, chatbot/service, recipe_executor | One injection point: **Agent Factory** (at activation time) |
| Runtime keyword matching to select which assigned skills to load | One rule: **Load ALL assigned skills** |
| Two security standards: 8 patterns (skills) vs 42 + LLM (plugins) | One standard: **42 patterns + LLM** for everything |
| Skills and plugins mutually exclusive | Both always load |

### What Stays the Same

- `MarketplacePlugin` model — still the distribution/governance/approval layer
- `PluginSecurityScan` model — security scanning at upload time
- `PluginContentCache` — S3/Redis caching for plugin content
- `SkillLoader` — progressive disclosure engine (used by agent_factory to load SKILL.md content)
- All existing Skills API endpoints (browse, assign, list)
- All existing Admin Plugin API endpoints (upload, approve, reject, scan)
- Marketplace UI — users browse and enable skills
- Capabilities tab — users assign skills to agents

---

## 1. Problem Statement

### 1.1 Mutual Exclusion Bug (P0)

**Severity:** Production bug — silent data loss

Three files contain identical mutual exclusion logic:

```
agent_factory.py:2174-2201
chatbot/service.py:1179-1213
recipe_executor.py:426-470
```

Pattern in all three:
```python
has_plugins = False
plugin_rows = plugin_svc.get_assigned_plugins(agent.id)
if plugin_rows:
    has_plugins = True
    # load plugin content...

if not has_plugins:
    # load skills...
```

**Impact:** Any agent with at least one plugin assigned gets zero skill content. The user assigns skills, sees them in the UI, but the agent never receives them. No error, no warning, no log message.

### 1.2 Three Divergent Injection Points

Skills are loaded independently in three places, each with different behavior:

| File | How it loads skills | Bug? |
|------|-------------------|------|
| `agent_factory.py` | Queries `agent.skills` relationship, runs keyword matching + context window budgeting, loads via SkillLoader | Over-engineered selection logic |
| `chatbot/service.py` | Queries `Skill` table by name, reads `skill.content["tools_schema"]` | **Bug:** `content` column doesn't exist on Skill model |
| `recipe_executor.py` | Queries `agent.skills` relationship, loads via SkillLoader | Duplicates agent_factory logic |

Having three injection points means three places to maintain, three places where bugs can hide, and three places that drift apart over time.

### 1.3 Over-Engineered Runtime Selection

The current agent_factory contains three methods that run at every agent activation:

- `_select_relevant_skills()` — scores each assigned skill against the task using keyword matching
- `_calculate_max_skills()` — determines how many skills fit in the context window
- `_estimate_skill_tokens()` — estimates token count per skill using a hardcoded lookup table

This means: the user assigns skills to an agent, then at runtime the system second-guesses that decision and may not load some of them. It's like hiring a Java developer and then quizzing them on Java every morning to decide if they should use Java today.

**The fix is simple:** Load ALL assigned skills. If there are too many, warn at assignment time (Phase 4), not at runtime.

### 1.4 Two Security Standards

| Content Type | Scanner | Patterns | LLM Analysis |
|-------------|---------|----------|--------------|
| Skills (Git import via PRD-22) | `scan_for_dangerous_patterns()` in skill_loader.py | 8 regex patterns | No |
| Plugins (Marketplace via PRD-42) | `static_scan()` + `llm_security_scan()` in plugin_security_scanner.py | 42 patterns (code + network + filesystem + prompt injection) | Yes (Claude Haiku) |

Same type of content (text injected into agent prompts), wildly different security treatment. A prompt injection attack that would be caught in a plugin sails through the skill scanner.

### 1.5 Recipe Prompts Are Too Long

The Bug Fixer recipe is 220+ lines of step-by-step instructions that essentially "program" the LLM because there's no clean way to give agents reusable methodology. With skills properly working, the recipe says WHAT to do and the skills provide HOW.

---

## 2. Goals

| ID | Goal | Success Metric |
|----|------|---------------|
| G1 | Eliminate mutual exclusion bug | Agent with skills AND plugins gets ALL content in system prompt |
| G2 | Single injection point | Only `agent_factory.py` loads skills — down from 3 files |
| G3 | Load ALL assigned skills | No runtime keyword matching or selection logic |
| G4 | Unified security scanning | All skills scanned with 42 patterns + LLM before reaching marketplace |
| G5 | Skills distributed via Marketplace | Admin uploads, security scans, user enables — one pipeline |
| G6 | Token budget awareness | Warn users at assignment time if combined skill tokens are high |

---

## 3. Architecture Overview

### 3.1 Current Data Flow (Broken)

```
Admin uploads plugin zip → S3 → Security scan (42 patterns + LLM) → Approve → Marketplace
Admin imports git repo   → Clone → Security scan (8 patterns only)  → Auto-active → Skills table

User assigns plugins to agent → agent_factory loads plugins → SKIPS ALL SKILLS
User assigns skills to agent  → agent_factory loads skills → Only if NO plugins assigned
```

### 3.2 Target Data Flow

```
                                    ┌─────────────────────┐
Admin uploads zip ──────────────────►│                     │
                                    │  Marketplace        │
Admin imports git repo ─────────────►│  (Security Scan:    │
                                    │   42 patterns + LLM)│
                                    └────────┬────────────┘
                                             │ Approve
                                             ▼
                                    ┌─────────────────────┐
                                    │  SkillMaterializer   │
                                    │  SKILL.md → Skill    │
                                    │  DB records          │
                                    └────────┬────────────┘
                                             │
                                             ▼
                                    ┌─────────────────────┐
                                    │  Skills Table        │
                                    │  (prompt_template,   │
                                    │   tools_schema,      │
                                    │   package_slug)      │
                                    └────────┬────────────┘
                                             │
                               User assigns skills to agent
                                             │
                                             ▼
                                    ┌─────────────────────┐
                                    │  Agent Factory       │
                                    │  _build_agent_       │
                                    │  system_prompt()     │
                                    │                      │
                                    │  1. Load ALL skills  │
                                    │  2. Load plugin ctx  │
                                    │  3. No filtering     │
                                    └────────┬────────────┘
                                             │
                                 System prompt with skills baked in
                                             │
                              ┌──────────────┼──────────────┐
                              ▼              ▼              ▼
                          Chatbot      Recipe Executor   Workflow
                        (uses pre-    (uses pre-built   (uses pre-
                         built prompt)  prompt)          built prompt)
```

### 3.3 Key Design Decisions

**D1: No new `SkillInjectionService`.** The previous attempt created a new service called from all 3 injection points. That's wrong — it preserves the 3-injection-point problem. Instead, fix `agent_factory._build_agent_system_prompt()` directly and remove skill loading from the other two files.

**D2: No `origin_type` column.** A skill is a skill regardless of whether it came from a git import, marketplace upload, or seed data. The `package_slug` column is enough to trace marketplace origin when needed.

**D3: Plugins still load via PluginContextService.** We're not deleting the plugin loading code from agent_factory — we're removing the mutual exclusion. Both skills AND plugins load. Over time, plugins will be migrated to skills via the materializer, but backward compatibility is maintained.

**D4: No runtime selection logic.** If a user assigns 5 skills, the agent gets all 5. Token budget warnings happen at assignment time (UI), not at runtime.

---

## 4. User Journeys

### J1: Admin Publishes a Skill Package

```
1. Admin uploads zip containing SKILL.md files
   → POST /api/admin/plugins/upload
   → System runs security scan (42 patterns + LLM analysis)
   → Blocked content never reaches marketplace

2. Admin reviews scan results, approves or rejects
   → POST /api/admin/plugins/{id}/approve

3. On approval, SkillMaterializer extracts SKILL.md files:
   → Parses YAML frontmatter + markdown body
   → Creates Skill DB records (name, description, prompt_template, tools_schema)
   → Links back via package_slug = plugin.slug

4. Skills appear in Marketplace for users to browse
```

### J2: User Enables and Assigns Skills

```
1. User browses Marketplace → finds "bug-fixer" skill package
2. User enables it for their workspace
   → POST /api/workspaces/{id}/plugins

3. Skills from that package appear in Capabilities tab
4. User assigns skills to their Bug Fixer agent
   → Capabilities UI / POST /api/agents/{id}/skills

5. UI shows estimated token cost per skill
6. If total exceeds threshold → yellow warning (not a blocker):
   "This agent's skills use ~18K tokens. Consider removing unused skills."
```

### J3: Agent Uses Skills at Runtime

```
1. User starts a chat / recipe triggers a step / workflow runs
2. AgentFactory.activate_agent() → _build_agent_system_prompt()
3. Loads ALL assigned skills:
   a. Query agent_skills junction → get all active Skill records
   b. Load each skill's content via SkillLoader (progressive disclosure Level 2)
   c. Fall back to skill.prompt_template or skill.description
   d. Extract tools_schema from each skill
   e. Append as "## Your Specialized Skills" section
4. ALSO loads plugin context via PluginContextService (backwards compat)
5. System prompt with all skills + plugins baked in
6. Chatbot/recipe/workflow use the pre-built prompt — no re-querying
```

---

## 5. Implementation Phases

### Phase 1: Fix Agent Factory — Single Injection Point

**Goal:** Eliminate the mutual exclusion bug. One place loads skills.

#### 1.1 Modify `agent_factory.py` — `_build_agent_system_prompt()`

**What to delete:**
- `_select_relevant_skills()` method (~90 lines) — keyword matching, scoring, fallback
- `_calculate_max_skills()` method (~30 lines) — runtime context window budgeting
- `_estimate_skill_tokens()` method (~10 lines) — hardcoded token lookup
- `enable_smart_skill_selection` parameter from method signature
- `has_plugins` / `if not has_plugins` branching (~95 lines)

**What to replace it with:**
```python
# 1. Always load ALL assigned skills (no mutual exclusion, no selection)
if getattr(agent, 'skills', None):
    sections.append("\n## Your Specialized Skills\n")
    loader = get_skill_loader(db) if db else None

    for skill in agent.skills:
        if not skill.is_active:
            continue

        # Load content via SkillLoader (Level 2 progressive disclosure)
        content = None
        if loader:
            try:
                content = loader.load_skill_core(skill.name, db=db)
            except Exception as e:
                self.logger.warning(f"Failed to load core content for '{skill.name}': {e}")

        if not content or not content.strip():
            content = skill.prompt_template or skill.description or ""

        if content:
            sections.append(f"### {skill.name}\n{content}")

        # Extract tool schemas
        if skill.tools_schema and isinstance(skill.tools_schema, dict):
            for tool_def in skill.tools_schema.get("tools", []):
                skill_tool_schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool_def.get("name"),
                        "description": tool_def.get("description", ""),
                        "parameters": tool_def.get("parameters", {}),
                    }
                })

# 2. Also load plugin context (backwards compat — no mutual exclusion)
try:
    from core.services.plugin_context_service import PluginContextService
    plugin_svc = PluginContextService(db) if db else None
    if plugin_svc:
        plugin_rows = plugin_svc.get_assigned_plugins(agent.id)
        if plugin_rows:
            tier1 = plugin_svc.build_tier1_summary(plugin_rows)
            if tier1:
                sections.append(tier1)
            tier2 = plugin_svc.build_tier2_content_sync(plugin_rows, task_context=task_context)
            if tier2:
                sections.append(tier2)
except Exception as e:
    self.logger.warning(f"Failed to load plugins for agent {agent.id}: {e}")
```

**File:** `orchestrator/modules/agents/factory/agent_factory.py`

#### 1.2 Modify `chatbot/service.py` — `_load_agent_context()`

**What to delete:** The entire skill/plugin loading block (lines ~1179-1226):
- `has_plugins` flag
- Plugin loading via `PluginContextService`
- Skill querying via `Skill` model
- Tool schema extraction from `skill.content` (the buggy line)

**What to replace it with:** Return empty `extra_context` and empty `skill_tools`. The agent's system prompt (built by agent_factory) already contains all skill + plugin content. The chatbot doesn't need to duplicate it.

```python
# Skills and plugins are already baked into the agent's system prompt
# by agent_factory._build_agent_system_prompt() at activation time.
# No need to re-query here.
extra_context = ""
tool_schemas = []
```

The persona and description loading stays — those are used by the chatbot for identity injection separate from skills.

**File:** `orchestrator/consumers/chatbot/service.py`

#### 1.3 Modify `recipe_executor.py` — `_build_system_prompt()`

**What to delete:** The entire skill/plugin loading block (lines ~426-470):
- `has_plugins` flag
- Plugin loading via `PluginContextService`
- Skill loading via `SkillLoader`

**What to replace it with:** Same approach — the recipe executor should use the agent's pre-built system prompt from agent_factory rather than building its own. If it must build a prompt (for cases where agent_factory wasn't called first), include skills loading identical to agent_factory's new logic (no mutual exclusion, load all).

**File:** `orchestrator/api/recipe_executor.py`

#### 1.4 Delete dead code

- `api/community_skills.py` — router not registered, references `scan_content()` which doesn't exist

---

### Phase 2: Plugin-to-Skill Materialization

**Goal:** When admin approves a marketplace plugin, its SKILL.md files become Skill DB records. At runtime, the unified skill loading path handles everything.

#### 2.1 Create `SkillMaterializer` service

**New file:** `orchestrator/core/services/skill_materializer.py`

Called by the `approve_plugin` endpoint after approval succeeds:

1. Fetch plugin's files from S3 (already uploaded and scanned)
2. Find all `SKILL.md` files in the package
3. Parse YAML frontmatter + markdown body (reuse `parse_yaml_frontmatter` from skill_loader.py)
4. For each SKILL.md, create or update a `Skill` record:
   - `name` = from frontmatter
   - `description` = from frontmatter
   - `prompt_template` = markdown body
   - `tools_schema` = from frontmatter `tools` key
   - `package_slug` = plugin.slug (new column — traces marketplace origin)
   - `workspace_id` = from the plugin's context or a system workspace
   - `is_active` = True
5. Store created skill IDs on `plugin.materialized_skill_ids` (new JSONB column)

Security scanning already happened at upload time — no re-scanning during materialization.

#### 2.2 Modify `admin_plugins.py` — approve endpoint

After `plugin.approval_status = "approved"` and before `db.commit()`, call:

```python
from core.services.skill_materializer import SkillMaterializer
materializer = SkillMaterializer(db)
materialized = materializer.materialize_plugin(plugin)
# materializer sets plugin.materialized_skill_ids internally
```

#### 2.3 Modify `agent_plugins.py` — auto-assign materialized skills

When `update_agent_plugins` (PUT endpoint) assigns plugins to an agent, also create `agent_skills` junction entries for each plugin's materialized skills. This ensures the agent gets skill content through the normal skill loading path in agent_factory.

#### 2.4 DB Migration

```sql
ALTER TABLE skills ADD COLUMN package_slug VARCHAR(100);
ALTER TABLE marketplace_plugins ADD COLUMN materialized_skill_ids JSONB DEFAULT '[]';
```

No `origin_type` column. `package_slug` is enough.

---

### Phase 3: Unified Security Scanning

**Goal:** All skills — regardless of import method — get the same 42-pattern + LLM security treatment. Scanning happens at import/upload time only, never at runtime.

#### 3.1 Add `quick_scan()` to `plugin_security_scanner.py`

A synchronous, single-content scanner using all 42 patterns from `ALL_PATTERN_GROUPS`. Returns a list of findings. No LLM call (that happens during the full marketplace scan).

```python
def quick_scan(content: str, file_path: str = "<inline>") -> List[StaticFinding]:
    """Synchronous scan of a single content string against all 42 patterns."""
```

Used by SkillLoader when git repos are imported via the admin git import flow.

#### 3.2 Modify `skill_loader.py` — upgrade scanner

Replace the 8 hardcoded `DANGEROUS_PATTERNS` in `SkillLoaderConfig` and the `scan_for_dangerous_patterns()` function with a call to `quick_scan()`.

**Current behavior (weak):** Skills that match 8 patterns are silently skipped during indexing.
**New behavior:** Skills that match any of the 42 patterns get `is_active = False` + audit log entry explaining why.

This only runs during git import/re-index — never at runtime.

---

### Phase 4: Token Budget Warning at Assignment Time

**Goal:** Help users make informed decisions about how many skills to assign. This is a UI/API enhancement, not a runtime blocker.

#### 4.1 Backend: Add token estimates to skill API responses

When listing skills for an agent, include estimated token counts:

```json
{
    "skills": [
        {"name": "bug-fixer", "estimated_tokens": 2400},
        {"name": "workspace-git", "estimated_tokens": 1800}
    ],
    "total_estimated_tokens": 4200,
    "warning": null
}
```

When total exceeds threshold (e.g., 15K tokens):
```json
{
    "total_estimated_tokens": 18200,
    "warning": "This agent's skills use ~18K tokens. Consider removing unused skills for faster responses."
}
```

Token estimation: `len(prompt_template) / 4` (rough chars-to-tokens approximation).

#### 4.2 Frontend: Show warning in Capabilities tab

Yellow warning banner when total exceeds threshold. Not a blocker — informational only.

---

### Phase 5: Example Skills (Prove the Pattern)

**Goal:** Create real SKILL.md files that demonstrate skills replacing verbose recipe prompts.

#### Skills to create:

| Skill | Purpose | Used By |
|-------|---------|---------|
| `workspace-git` | Git operations: branch naming (`fix/{issue_key}`), commit format, PR creation | Bug Fixer, QA Engineer |
| `jira-admin` | Jira: read/create/transition tickets, priority mapping, comment format | Bug Fixer, QA Engineer |
| `bug-fixer` | Methodology: ticket → code → failing test → fix → verify → PR | Bug Fixer recipe |
| `qa-engineer` | Test running, failure classification (P0-P3), report format | QA Engineer recipe |

These live in a **separate skills repository** (not embedded in the platform codebase). Admins import via git import or upload as a marketplace package.

#### Recipe Transformation Example

**Before — Bug Fixer recipe (220 lines):**
```
## Step A: Set up the workspace
1. workspace_git operation="checkout" args="-b fix/{input.issue_key}"
## Step B: Find the relevant code
1. Read the scratchpad for ticket details...
2. workspace_grep to search for keywords...
[... 200 more lines of step-by-step instructions ...]
```

**After — Bug Fixer recipe (~5 lines):**
```
Fix the bug described in Jira ticket {input.issue_key} in the automatos-ai repo.
Read the ticket, find the code, write a failing test, fix the bug, run tests,
and open a draft PR.
```

The agent's assigned skills (bug-fixer, workspace-git, jira-admin) provide the HOW.

---

## 6. Files Changed

| File | Action | What Changes |
|------|--------|-------------|
| `modules/agents/factory/agent_factory.py` | MODIFY | Remove mutual exclusion, load ALL skills + plugins, delete selection logic (3 methods) |
| `consumers/chatbot/service.py` | MODIFY | Remove independent skill/plugin loading from `_load_agent_context()` |
| `api/recipe_executor.py` | MODIFY | Remove independent skill/plugin loading from `_build_system_prompt()` |
| `core/services/skill_materializer.py` | CREATE | Convert approved plugin SKILL.md files into Skill DB records |
| `core/services/plugin_security_scanner.py` | MODIFY | Add `quick_scan()` synchronous single-content scanner |
| `modules/agents/services/skill_loader.py` | MODIFY | Replace 8-pattern scanner with `quick_scan()` (42 patterns) |
| `api/admin_plugins.py` | MODIFY | Call SkillMaterializer after plugin approval |
| `api/agent_plugins.py` | MODIFY | Auto-assign materialized skills when plugin assigned to agent |
| `core/models/core.py` | MODIFY | Add `package_slug` column to Skill model |
| `core/models/marketplace_plugins.py` | MODIFY | Add `materialized_skill_ids` column to MarketplacePlugin model |
| Alembic migration | CREATE | Add `package_slug` and `materialized_skill_ids` columns |
| `api/community_skills.py` | DELETE | Dead code — router not registered, references non-existent function |

## 7. What Gets Deleted

| Code | File | Why |
|------|------|-----|
| `_select_relevant_skills()` | agent_factory.py | Replaced by "load all assigned" |
| `_calculate_max_skills()` | agent_factory.py | Runtime budgeting → assignment-time warning instead |
| `_estimate_skill_tokens()` | agent_factory.py | Moved to API response (Phase 4) |
| `enable_smart_skill_selection` param | agent_factory.py | No longer needed |
| `has_plugins` / `if not has_plugins` branching | agent_factory.py, chatbot/service.py, recipe_executor.py | The mutual exclusion bug |
| Skill/plugin loading block | chatbot/service.py:1179-1226 | Duplicated what agent_factory does |
| Skill/plugin loading block | recipe_executor.py:426-470 | Duplicated what agent_factory does |
| `DANGEROUS_PATTERNS` (8 patterns) | skill_loader.py | Replaced by `quick_scan()` (42 patterns) |
| `scan_for_dangerous_patterns()` | skill_loader.py | Replaced by `quick_scan()` |
| `api/community_skills.py` | entire file | Dead code |

---

## 8. What Does NOT Change

- `MarketplacePlugin` model (except adding `materialized_skill_ids`)
- `PluginSecurityScan` model
- `PluginContentCache` (S3/Redis caching)
- `SkillLoader` class (progressive disclosure engine — still used by agent_factory)
- All existing Skills API endpoints
- All existing Admin Plugin API endpoints
- Marketplace UI
- Capabilities tab UI (except adding token warning)
- Agent activation flow in agent_factory (just the skill/plugin section changes)
- `PluginContextService` (still loads plugin content — backwards compat)

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Removing skill loading from chatbot/service breaks skill_tools | Tool schemas from skills stop working in chatbot | Verify agent_factory's skill_tool_schemas are passed through AgentRuntime to chatbot |
| recipe_executor expects its own system prompt | Recipe steps may not have agent_factory-built prompt available | Keep skill loading in recipe_executor but use the same logic as agent_factory (no mutual exclusion) |
| Large skill content bloats system prompts | Slower responses, higher costs | Phase 4 token warning; future: skill summarization |
| SkillMaterializer fails silently | Approved plugins have no materialized skills | Log errors, return materialization status in approve response |

---

## 10. Testing Strategy

### Unit Tests

1. `agent_factory._build_agent_system_prompt()` loads ALL assigned skills (no filtering)
2. `agent_factory._build_agent_system_prompt()` loads skills AND plugin content together (mutual exclusion bug gone)
3. `SkillMaterializer.materialize_plugin()` creates correct Skill records from SKILL.md files
4. `quick_scan()` detects all 42 pattern categories
5. `quick_scan()` returns empty findings for clean content
6. Token estimation returns reasonable values for known skill sizes

### Integration Tests

1. Agent with 3 skills assigned → system prompt contains all 3 skill sections
2. Agent with skills AND plugins → both present in system prompt
3. Plugin approval → SKILL.md materialized → Skill records created → available for assignment
4. Git import → skill scanned with 42 patterns → flagged skills get `is_active = False`
5. Plugin assigned to agent → materialized skills auto-assigned via agent_skills junction

### Manual Validation

1. Create Bug Fixer agent with bug-fixer + workspace-git + jira-admin skills
2. Assign an email plugin to the same agent
3. Start a chat → verify system prompt contains BOTH skill content and plugin content
4. Run simplified recipe prompt → verify agent follows methodology from skills
5. Assign >15K tokens of skills → verify warning appears in Capabilities tab

---

## 11. Future Extensions (Not In Scope)

- **User-uploaded skills** — once security flow is proven, allow workspace users to upload
- **Inline SKILL.md editor** — create/edit skills in the UI
- **Skill versioning** — pin to specific version, rollback
- **Skill usage analytics** — which skills get loaded, how often, token cost
- **Frontend unification** — merge skills/plugins tabs into single "Skills" view
- **Skill summarization** — for agents with many skills, generate condensed summaries to save tokens

---

## 12. Open Questions

1. **recipe_executor prompt source:** Does recipe_executor always have access to an agent_factory-built prompt, or does it sometimes build from scratch? If from scratch, it needs its own skill loading (matching agent_factory logic, not duplicating old behavior).

2. **Workspace ID for materialized skills:** When SkillMaterializer creates Skill records from a marketplace plugin, which `workspace_id` should they get? Options: (a) a system/global workspace, (b) created per-workspace when a user enables the plugin, (c) nullable for marketplace skills.

3. **Plugin content vs skill content:** After materialization, should agent_factory load BOTH the materialized skills AND the plugin's PluginContextService content? Or only the skills? Loading both would double the content. Recommendation: load skills only; stop loading plugin context once a plugin has been fully materialized.

4. **Migration of existing plugins:** Should we run SkillMaterializer retroactively on already-approved plugins? Or only for newly approved ones going forward?
