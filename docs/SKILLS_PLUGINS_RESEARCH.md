# Skills & Plugins — Current State Research

Research on the existing skill and plugin systems in Automatos AI Platform. What exists, how it works, where the gaps are.

---

## Two Separate Systems

There are **two distinct systems** that serve overlapping purposes:

| | Skills (PRD-22) | Plugins (PRD-42) |
|---|---|---|
| **Source** | Git repos, DB seeds, skills.sh | Zip upload, GitHub import |
| **Storage** | Filesystem (`~/.automatos/skills/`) + DB | S3 (`plugins/{slug}/{version}/`) + DB |
| **Format** | SKILL.md with YAML frontmatter | manifest.json + code/content files |
| **Security** | 8 regex patterns | 2-stage: 42 regex patterns + LLM analysis |
| **Approval** | None (auto-activate) | Full workflow: pending, approved, rejected |
| **Context injection** | Direct prompt injection + tools_schema | Two-tier: DB summary + S3 content via Redis cache |

**Critical design rule:** If an agent has plugins assigned, **all skills are silently skipped**. They never co-exist on the same agent at runtime.

---

## 1. Plugin System (PRD-42)

### Pipeline: Upload, Scan, Approve, Enable, Assign, Inject

**Stage 1 — Upload** (`POST /api/admin/plugins/upload`)
- Admin uploads `.zip` (max 10 MB)
- `PluginUploadService` validates: max 500 files, 10 MB/file, 50 MB total, 100x compression ratio (zip bomb defense)
- `manifest.json` must exist at root with `slug`, `name`, `version`
- All text files read into dict for scanning; binary files skipped

**Stage 2 — Security Scan** (`plugin_security_scanner.py`)
- **Static scan**: 4 pattern categories:
  - Dangerous code (13 patterns): subprocess, eval, os.system, ctypes, pickle, etc.
  - Network access (9 patterns): requests, urllib, socket, aiohttp, httpx
  - Filesystem writes/deletes (8 patterns): open(w/a), rmtree, unlink, makedirs
  - Prompt injection (12 patterns): instruction override, jailbreak, exfiltration, identity rewriting
- If any **critical** static finding: auto-blocked, LLM scan skipped
- **LLM scan**: All files concatenated, sent to Claude Haiku for security audit
  - risk_score < 20: `safe`
  - risk_score 20-69: `review_required`
  - risk_score >= 70: `blocked`
  - LLM failure: `review_required` (graceful fallback)
- Results stored in `PluginSecurityScan` table

**Stage 3 — S3 Upload + DB Record**
- Files extracted to S3 under `plugins/{slug}/{version}/`
- `MarketplacePlugin` record created with `approval_status="pending"`
- Auto-categorisation runs keyword matching (26 category-keyword mappings)
- `PluginSyncHistory` audit record created

**Stage 4 — Admin Approval**
- `POST /api/admin/plugins/{id}/approve` or `/reject`
- GitHub imports auto-approve if scan verdict is `safe`

**Stage 5 — Workspace Enable** (`workspace_plugins.py`)
- `POST /api/workspaces/{id}/plugins` — only approved + active plugins can be enabled
- `DELETE` cascades removal to all agent assignments in that workspace

**Stage 6 — Agent Assignment** (`agent_plugins.py`)
- `PUT /api/agents/{id}/plugins` — full replacement, priority from list order
- Validates every plugin is enabled for the agent's workspace

**Stage 7 — Context Injection** (`plugin_context_service.py` + `plugin_cache.py`)
- **Tier 1** (always loaded): DB-field summary — name, slug, description, tags, counts. ~200 tokens/plugin
- **Tier 2** (top 2 relevant): Full SKILL.md + tool schemas from S3 via Redis cache. ~2000 tokens/plugin
- Relevance scoring: keyword matching against task context (name, slug, description words, tags at 2x weight)
- Three consumers call this: AgentFactory, chatbot service, recipe executor

### Plugin Models

| Model | Table | Purpose |
|---|---|---|
| `PluginCategory` | `plugin_categories` | Hierarchical categories (self-referential) |
| `MarketplacePlugin` | `marketplace_plugins` | Core plugin record — metadata, S3 path, security, approval, usage |
| `PluginSecurityScan` | `plugin_security_scans` | Static + LLM scan results |
| `PluginSyncHistory` | `plugin_sync_history` | Audit log (upload, approve, reject, deactivate) |
| `WorkspaceEnabledPlugin` | `workspace_enabled_plugins` | Junction: workspace to plugin |
| `AgentAssignedPlugin` | `agent_assigned_plugins` | Junction: agent to plugin (with priority) |

### Plugin API Endpoints

**Admin** (`/api/admin/plugins/`):
- `POST /upload` — Upload zip
- `POST /import-github` — Import from GitHub URL
- `POST /{id}/approve` | `POST /{id}/reject`
- `GET /{id}/scan` — View scan results
- `POST /{id}/deactivate` — Deactivate + cascade remove assignments
- `GET /pending` — Paginated pending list
- `DELETE /{id}` — Hard delete (S3 + DB)
- `POST /backfill-categories` — Re-run auto-categorisation

**Public** (`/api/marketplace/plugins/`):
- `GET /` — Browse approved+active with filters, search, sorting
- `GET /categories`
- `GET /{id}` — Detail with manifest + content items
- `GET /{id}/content` — Raw file content

**Workspace** (`/api/workspaces/{id}/plugins/`):
- `GET /` — List enabled
- `POST /` — Enable a plugin
- `DELETE /{plugin_id}` — Disable + cascade

**Agent** (`/api/agents/{id}/`):
- `GET /plugins` — List assigned
- `PUT /plugins` — Replace all assignments
- `GET /assembled-context` — Preview full system prompt with plugins

### Plugin Config (`config.py`)

```
MARKETPLACE_S3_BUCKET     = "automatos-marketplace"
MARKETPLACE_LOCAL_DIR     = None  (uses ~/.automatos/marketplace/)
PLUGIN_MAX_UPLOAD_SIZE_MB = 10
PLUGIN_LLM_SCAN_MODEL     = "claude-haiku-4-20250414"
PLUGIN_CACHE_TTL_SECONDS   = 3600
```

---

## 2. Skill System (PRD-22)

### Three Layers of Skills

1. **Seed skills** — 32 hardcoded skills across 4 categories (development, security, infrastructure, analytics). DB-only, `implementation` field is pseudo-code (not executable)
2. **Git-backed skills** — Cloned from repos, SKILL.md format, progressive 3-level disclosure
3. **Community skills** — From `skills.sh` marketplace. **Dead code** — router not registered in main.py, `scan_content()` method doesn't exist on the scanner

### Git-Backed Skill Flow

**Import** (`POST /api/v1/skills/sources/git`)
- Admin-only (PRD-70). Rate-limited per workspace
- URL validated via `git_sanitizer` (HTTPS only, domain allowlist, no embedded credentials)
- Shallow clone (depth 50), 5-minute timeout
- Discovers all `SKILL.md` files via `rglob`

**Indexing** (`skill_loader.py: _index_repository()`)
- Parses YAML frontmatter (name, description required)
- Runs `scan_for_dangerous_patterns()` — 8 regex patterns only:
  - `__import__`, `eval(`, `exec(`, `compile(`, `os.system`, `os.popen`, `rm -rf`, `DROP TABLE`
- Dangerous skills are **skipped** during indexing (not blocked, just ignored)
- Creates `Skill` + `SkillFile` DB records
- File classification by type:
  - `SKILL.md` = core (level 2)
  - Other `.md` = resource (level 3)
  - `scripts/` = script (level 3)
  - `examples/` or `data/` = example (level 3)

**Progressive 3-Level Disclosure**
- **Level 1 — Metadata** (~50 tokens): YAML frontmatter dict. Cached in memory (1000 entries)
- **Level 2 — Core** (~2000 tokens): Markdown body from `prompt_template` column or SKILL.md file. Cached in memory (100 entries)
- **Level 3 — Resources** (on-demand): Specific resource files. LRU cache (50 entries)

**Prompt Injection** (`agent_factory.py: _build_agent_system_prompt()`)
- Only runs if agent has **no plugins assigned**
- `_select_relevant_skills()` scores skills by keyword matching against task context
- For each selected skill: loads core content, appends to system prompt
- Extracts `tools_schema` from skill's `content` JSON, converts to OpenAI function calling format

### Skill Models

| Model | Table | Purpose |
|---|---|---|
| `Skill` | `skills` | Core record — name, description, category, prompt_template, tools_schema, git fields, tags |
| `SkillFile` | `skill_files` | Individual files within a skill (path, type, load_level, size, token estimate) |
| `SkillSource` | `skill_sources` | Git repo tracking — URL, branch, commit SHA, sync status. **No workspace_id** |
| `SkillVersion` | `skill_versions` | Version history with commit SHAs and changelogs |
| `SkillAuditLog` | `skill_audit_logs` | Action audit trail |
| `agent_skills` | `agent_skills` | Junction table: agent to skill (M2M) |

### Skill API Endpoints (`/api/v1/skills/`)

**Source Management:**
- `POST /sources/git` — Import repo (admin-only, rate-limited)
- `GET /sources` — List with filters
- `POST /sources/{id}/update` — Pull and re-index
- `POST /sources/{id}/rollback?commit_sha=X`
- `DELETE /sources/{id}` — Soft-delete

**Skill CRUD:**
- `GET /` — List with filtering (category, source, search, tags)
- `GET /{id}` — Detail with file list
- `GET /{id}/content?level=N` — Progressive disclosure
- `DELETE /{id}` — Soft-delete + remove agent assignments

**Agent Assignment:**
- `GET /agents/{agent_id}/skills` — List assigned
- `POST /agents/{agent_id}/skills` — Assign (add or replace)
- `DELETE /agents/{agent_id}/skills` — Remove

**Recommendation:**
- `POST /recommend` — Lexical keyword matching (not semantic)

### Other Skill Components

**Semantic Skill Matcher** (`core/llm/semantic_skill_matcher.py`)
- Embedding-based cosine similarity matching
- **Exists but NOT used** by skills API or agent_factory
- Only used by the routing system for agent selection

**Git Sanitizer** (`core/security/git_sanitizer.py`)
- HTTPS-only, domain allowlist, blocks embedded credentials
- Blocks dangerous git flags (`--upload-pack`, `-c`, `--config`)
- `build_git_clone_cmd()` uses `--` separator to prevent argument injection
- **Only applied to clone** — `_git_pull` and `_git_checkout` bypass it

---

## 3. Chatbot Pipeline — Where Skills/Plugins Fit

### Message Flow

```
User message
  -> AutoBrain.assess()           — complexity classification (ATOM/MOLECULE/CELL/ORGANISM)
  -> StreamingChatService         — central orchestration
      -> _load_agent_context()    — THE ONLY PLACE skills/plugins load
      -> SmartChatIntegration     — personality, memory, tool filtering
          -> IntentClassifier     — 9 intents, tool routing decisions
          -> SmartToolRouter      — filters tools by intent
      -> LLM call
      -> Tool execution loop     — ToolRouter.execute_and_format()
```

### The ATOM Path

When AutoBrain classifies a message as ATOM (greetings, simple questions):
- **Everything is bypassed**: no tools, no memory, no SmartChatIntegration, no skills, no plugins
- Minimal system prompt, straight to LLM
- This is the fastest path

### Intent Classification (9 Intents)

| Intent | Tools? | Memory? |
|---|---|---|
| GREETING | No | No |
| CHITCHAT | No | No |
| MEMORY_RECALL | No | Yes |
| FACTUAL (default) | No | Yes |
| DATA_QUERY | Yes | No |
| SEARCH | Yes | No |
| EXTERNAL_ACTION | Yes | No |
| CREATION | Yes | No |
| MULTI_STEP | Yes | Maybe |

Default is FACTUAL with `requires_tools=false`. Tools only activate with clear signal.

### Tool Execution Gates

| Layer | Gate | Effect |
|---|---|---|
| AutoBrain | ATOM classification | Bypasses entire pipeline: no tools, no memory |
| AutoBrain | `tool_hints` | Overrides intent-based tool routing |
| IntentClassifier | `requires_tools=false` | SmartToolRouter returns empty tools |
| SmartToolRouter | Category filtering | Limits to intent-relevant tools (max 15) |
| SmartOrchestrator | Platform tools always-on | `platform_*` tools bypass intent filtering |
| ToolRegistry | `validate_tool_access()` | Agent-level tool access control |
| Capability Filter | Action eligibility | Blocks Composio actions not matching intent |
| Tool loop | Max 10 iterations | Hard limit on tool call rounds |
| Tool loop | Per-tool retry (3 max) | Prevents infinite retry on one tool |
| Tool loop | Seen-call dedup | Skips identical tool+args combos |

### Where Skills/Plugins Appear in the Pipeline

Only in `service.py: _load_agent_context()`:
1. Check for assigned plugins first
2. If plugins exist: build tier 1 + tier 2 context, skip skills entirely
3. If no plugins: load skills from DB, inject `prompt_template` + `tools_schema`

No other pipeline component (AutoBrain, IntentClassifier, SmartOrchestrator, ToolRouter) has any awareness of skills or plugins.

---

## 4. Key Files Reference

| Component | Path |
|---|---|
| Plugin upload service | `orchestrator/core/services/plugin_upload_service.py` |
| Plugin security scanner | `orchestrator/core/services/plugin_security_scanner.py` |
| Plugin context service | `orchestrator/core/services/plugin_context_service.py` |
| Plugin cache (Redis) | `orchestrator/core/services/plugin_cache.py` |
| Plugin S3 service | `orchestrator/core/services/marketplace_s3.py` |
| Plugin models | `orchestrator/core/models/marketplace_plugins.py` |
| Admin plugin API | `orchestrator/api/admin_plugins.py` |
| Public marketplace API | `orchestrator/api/marketplace_plugins.py` |
| Workspace plugin API | `orchestrator/api/workspace_plugins.py` |
| Agent plugin API | `orchestrator/api/agent_plugins.py` |
| Skill loader | `orchestrator/modules/agents/services/skill_loader.py` |
| Skill models | `orchestrator/core/models/core.py` (lines 280-426) |
| Skills API | `orchestrator/api/skills.py` |
| Community skills (dead) | `orchestrator/api/community_skills.py` |
| Semantic matcher (unused) | `orchestrator/core/llm/semantic_skill_matcher.py` |
| Git sanitizer | `orchestrator/core/security/git_sanitizer.py` |
| Seed skills | `orchestrator/core/seeds/seed_skills.py` |
| Agent factory (prompt) | `orchestrator/modules/agents/factory/agent_factory.py` |
| Chatbot service | `orchestrator/consumers/chatbot/service.py` |
| Smart orchestrator | `orchestrator/consumers/chatbot/smart_orchestrator.py` |
| Intent classifier | `orchestrator/consumers/chatbot/intent_classifier.py` |
| AutoBrain | `orchestrator/consumers/chatbot/auto.py` |
| Tool router | `orchestrator/modules/tools/tool_router.py` |
| Config | `orchestrator/config.py` (lines 275-283) |

---

## 5. Observed Gaps (No Recommendations — Just Observations)

### Plugin System
1. `PluginContentCache.invalidate_plugin()` exists but is **never called** — not on delete, deactivate, or update. Only TTL expiration clears stale cache
2. Once Redis `_redis_available = False`, it **never recovers** — requires process restart
3. No re-scan endpoint — can't re-scan a plugin without re-uploading
4. No reactivate endpoint — deactivated plugins can't be turned back on
5. `enable_count` managed via manual increment/decrement — could drift from actual junction table count
6. `slug` is unique (not `slug + version`) — only one version of a plugin can exist at a time
7. LLM scan sends all files concatenated in one message — no truncation for large plugins near 50 MB limit
8. `s3_bucket` column defaults to `"automatos-marketplace"` while runtime uses `config.MARKETPLACE_S3_BUCKET` — could drift
9. Hard delete doesn't call `invalidate_plugin()` — stale cache persists until TTL
10. Hardcoded `"gpt-4"` fallback in assembled-context endpoint

### Skill System
1. Security scanning has **8 regex patterns** vs plugins' **42 patterns + LLM analysis** — no prompt injection detection for skills
2. Skills with dangerous patterns are silently **skipped** during indexing — no notification, no blocking, no audit
3. `_git_pull` and `_git_checkout` **bypass git_sanitizer** — only clone is sanitized
4. `get_skill_script_path()` has **no path traversal protection** — `../` in script_name could escape skill directory
5. `SkillSource` has **no workspace_id** — sources are global, not per-workspace
6. Semantic matcher exists but is **not used** by skills API or agent factory
7. Community skills router is **not registered** in main.py — dead code
8. `community_skills.py` calls `scanner.scan_content()` which **doesn't exist** — would throw at runtime
9. `apply_skills()` references `SKILL_PROMPTS` dict that **doesn't exist** — dead code
10. Commit SHA not validated in `_git_checkout` — raw string passed to shell

### Cross-System
1. Skills and plugins are **mutually exclusive** at runtime — if agent has plugins, all skills are silently skipped
2. No unified "content" abstraction — two separate pipelines, models, APIs, storage mechanisms
3. `max_plugins=2` for tier-2 loading is hardcoded — not configurable per agent or workspace
4. Relevance scoring in both systems uses simple keyword matching — semantic matcher exists but unused
5. No way to create a skill/plugin without git or zip — no inline creation from markdown content
6. No skill templates or starter content — must write from scratch
