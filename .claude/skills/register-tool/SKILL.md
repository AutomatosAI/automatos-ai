---
name: register-tool
description: Register a new platform_* action as an actions_<domain>.py / handlers_<domain>.py pair in orchestrator/modules/tools/discovery/. Use when adding any new tool that agents should be able to call.
disable-model-invocation: true
---

# Register Platform Tool

Slash command: `/register-tool <tool_name>` (e.g. `/register-tool platform_export_workspace`)

`orchestrator/modules/tools/README.md`, section "Adding a platform action", is the current pattern: a definition in `discovery/actions_<domain>.py`, its handler in `discovery/handlers_<domain>.py`, and a test under `modules/tools/tests/`. Read it first and follow an existing pair such as `actions_blog.py` / `handlers_blog.py`.

## Before writing code

- `grep -rn "<tool_name>" orchestrator/modules/tools/discovery/` — a match means you are extending, not registering. Prefer extending an existing action over adding a near-duplicate.
- Settle the contract: the inputs, the result shape (`{"success": True, "data": ...}` or `{"success": False, "error": ...}`), the side effects, and the permission level (`requires_confirmation=True` for anything destructive).

## The description is the contract

The model picks and calls the action from its description and parameter schema alone. Say what it does, when to use it and when not to, what each parameter means and what it does not return; every enum value must be one the handler accepts. Write it plainly: no MUST/ALWAYS boosters, and no worked examples or call transcripts in the text.

## Verify

CI is the gate; nothing runs on this machine. The test goes under `modules/tools/tests/`; push and let CI run it.
