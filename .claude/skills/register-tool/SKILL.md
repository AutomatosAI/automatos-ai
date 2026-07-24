---
name: register-tool
description: Register a new platform_* tool using the canonical 3-file pattern (platform_actions.py + platform_executor.py + auto.py). Use when adding any new tool that agents should be able to call.
disable-model-invocation: true
---

# Register Platform Tool

Slash command: `/register-tool <tool_name>` (e.g. `/register-tool platform_export_workspace`)

The canonical pattern for adding a new platform tool to Automatos. Documented in `prd71-tools.md` memory and used for every tool since 2026-03-08.

## The 3-file pattern

Every platform_* tool requires exactly three changes:

1. **`orchestrator/core/platform/platform_actions.py`** — `ActionDefinition` registration (schema)
2. **`orchestrator/core/platform/platform_executor.py`** — handler method + `_handlers` dict entry (execution)
3. **`orchestrator/consumers/chatbot/auto.py`** — Tier 2 keyword in `_PLATFORM_KEYWORDS` (routing)

If you skip any one of these, the tool is unwired and won't be callable.

## Workflow

### Step 1 — Confirm the tool doesn't already exist
```bash
grep -n "<tool_name>" orchestrator/core/platform/platform_actions.py orchestrator/core/platform/platform_executor.py
```
If matches exist, **stop** — you're extending, not registering.

### Step 2 — Decide what the tool does
Capture these before writing code:
- **Inputs**: what parameters does the LLM need to provide?
- **Output**: structured dict the LLM can read (success/data/error envelope)
- **Side effects**: DB writes? S3 uploads? External API calls?
- **Permissions**: workspace-scoped? user-scoped? Mission-scoped?
- **Tier 2 keywords**: what user phrases should route to this tool? (3–6 short keywords)

### Step 3 — Add the ActionDefinition (file 1 of 3)
Edit `platform_actions.py` and append a new `ActionDefinition`. Use existing entries as templates (look at `platform_submit_report` or `platform_install_skill` — both are good models).

Required fields:
- `name="<tool_name>"`
- `description="..."` — one sentence, what the tool does (LLM reads this)
- `parameters=ActionParameters(...)` — JSON schema for inputs
- `executor_handler="<handler_method_name>"` — must match step 4

### Step 4 — Add the executor handler (file 2 of 3)
Edit `platform_executor.py`:

1. Add a handler method on `PlatformActionExecutor`:
   ```python
   async def <handler_method_name>(self, params: dict, context: ExecutionContext) -> dict:
       # 1. Validate inputs
       # 2. Do the work (DB query, API call, etc.)
       # 3. Return {"success": True, "data": ...} or {"success": False, "error": "..."}
   ```
2. Add the dispatch entry to `_handlers`:
   ```python
   self._handlers["<tool_name>"] = self.<handler_method_name>
   ```

### Step 5 — Add the keyword routing (file 3 of 3)
Edit `auto.py` and add the tool to `_PLATFORM_KEYWORDS`:
```python
"<tool_name>": ["keyword1", "keyword2", "phrase three"],
```

These are matched against user messages for Tier 2 routing. Keep them short and high-signal.

### Step 6 — Verify
Run these three checks (no test yet — these are wiring checks):
```bash
python -c "from core.platform.platform_actions import ACTIONS; print('<tool_name>' in [a.name for a in ACTIONS])"
python -c "from core.platform.platform_executor import PlatformActionExecutor; e = PlatformActionExecutor(); print('<tool_name>' in e._handlers)"
grep -n '<tool_name>' orchestrator/consumers/chatbot/auto.py
```
All three must show the tool. If any is missing, the tool is unwired.

### Step 7 — Tell the user where to test
Per memory: never local-build. Push to a branch, let Railway deploy, then test against the live URL with a real agent or in the chatbot UI.

## Anti-patterns
- Adding the executor but forgetting `_handlers` entry → silent NotImplementedError
- Adding keywords too broad → routes false positives to your tool
- Returning unstructured strings instead of `{"success": ..., "data": ...}` — agents can't parse it
- Editing `platform_actions.py` without bumping any docs / SKILL.md that references the tool list

## References
- Memory: `prd71-tools.md` — the original 19 tools and how the pattern emerged
- Memory: `tool-plumbing-architecture.md` — how `platform_*` is dispatched in `unified_executor.py`
- Memory: `agentfactory-rewrite.md` — why `get_tools_for_agent()` is the single tool source
