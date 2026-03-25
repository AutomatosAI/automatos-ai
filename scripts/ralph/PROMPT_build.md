# Build Mode

Implement ONE task from the plan, validate, commit, exit.

## Phase 0: Orient

Study with subagents:
- @CLAUDE.md (how to build/test)
- @scripts/ralph/IMPLEMENTATION_PLAN.md (current state — tasks + key references + tool reference + quality bar)
- @scripts/ralph/prd.json (acceptance criteria for each story)

### Key References

- **GOLD STANDARD SKILLS** (read BOTH before writing ANY skill):
  - `/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills/sentinel/SKILL.md` — monitoring/devops pattern
  - `/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills/scout/SKILL.md` — sales/outreach pattern
- **Skills repo**: `/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills/` — all skills live here (NOT in automatos-ai)
- **Seed script**: `scripts/seed_agent_catalog.py` — parses SKILL.md frontmatter, populates DB
- **Skill injection**: `orchestrator/modules/context/sections/skills.py` — how skill content becomes system prompt
- **Platform tools**: `orchestrator/modules/tools/discovery/platform_actions.py` — all registered platform_* tools
- **Workspace tools**: `orchestrator/modules/tools/discovery/workspace_actions.py` — all workspace_* tools
- **Agent model**: `orchestrator/core/models/core.py` — Agent fields (persona, skills, model_config)

### Check for completion

```bash
grep -c "^\- \[ \]" scripts/ralph/IMPLEMENTATION_PLAN.md || echo 0
```

- If 0: Run validation -> commit -> output **RALPH_COMPLETE** -> exit
- If > 0: Continue to Phase 1

## Phase 1: Implement

1. **Study the plan** — Choose the FIRST unchecked task from @scripts/ralph/IMPLEMENTATION_PLAN.md
2. **Read prd.json** — Find the matching US-XXX story and follow its acceptance criteria exactly
3. **Read reference skills** — ALWAYS read sentinel/SKILL.md and scout/SKILL.md before writing skills
4. **Read existing skill** — Before rewriting a skill, read the current version to understand what exists
5. **Implement** — ONE task only. Implement completely — no placeholders or stubs
6. **Validate** — Check skill quality and format

### Skill Writing Rules (CRITICAL)

- Skills live in `/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills/{slug}/SKILL.md`
- Frontmatter format: name, description, version, tags, category: agent-role, tools: [{name, description}]
- Body MUST have: identity paragraph, numbered workflow with JSON tool call blocks, output format template, "What NOT To Do" section
- Tool calls use ```json blocks with realistic parameters — copy the style from Sentinel
- Every tool name MUST be a real Automatos tool (see IMPLEMENTATION_PLAN.md Tool Reference)
- 60-100 lines per skill — dense, actionable, no filler
- No references to external platforms (Cursor, OpenClaw, Qwen, etc.)
- Agent identity should be Automatos-specific: "You are the X for the Automatos platform" or "You are the workspace's X"
- Composio tools use: `composio_execute` with `action` and relevant app params

### Validation

For seed script changes (US-001):
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai && python3 scripts/seed_agent_catalog.py --dry-run 2>&1 | tail -5
```

For skill rewrites (US-002 through US-010):
```bash
# Check skills were written
for slug in [LIST_FROM_STORY]; do
  test -f /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills/$slug/SKILL.md && echo "OK: $slug" || echo "MISSING: $slug"
done

# Check frontmatter has tools: with name/description format
for slug in [LIST_FROM_STORY]; do
  grep -c "  - name:" /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills/$slug/SKILL.md && echo "OK: $slug has tools" || echo "BAD: $slug missing tool format"
done

# Check body has workflow section
for slug in [LIST_FROM_STORY]; do
  grep -c "## Workflow\|## workflow" /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills/$slug/SKILL.md && echo "OK: $slug" || echo "BAD: $slug missing workflow"
done
```

For final validation (US-011):
```bash
python3 scripts/seed_agent_catalog.py --dry-run 2>&1 | tail -5
find /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills -name "SKILL.md" | wc -l
```

Note: Pre-existing errors may exist. Only check for NEW errors introduced by your changes.

## Phase 2: Update & Learn

**Update scripts/ralph/IMPLEMENTATION_PLAN.md:**
- Mark completed task `- [x] Completed`
- Add any discovered bugs or issues

**Update scripts/ralph/progress.txt:**
- Log what was completed this iteration

## Phase 3: Commit & Exit

For skill rewrites, commit to BOTH repos:
```bash
# Commit skills to skills repo
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills && git add -A && git commit -m "feat(skills): [category] rewrite [N] skills to Automatos quality"

# Commit plan updates to main repo
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai && git add -A && git commit -m "chore(skills): update plan — [description]"
```

For seed script changes (US-001), commit to main repo only:
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai && git add -A && git commit -m "fix(skills): [description]"
```

Check remaining:
```bash
grep -c "^\- \[ \]" scripts/ralph/IMPLEMENTATION_PLAN.md || echo 0
```

- If > 0: Say "X tasks remaining" and EXIT
- If = 0: Output **RALPH_COMPLETE**

## Guardrails

99999. Read sentinel/SKILL.md and scout/SKILL.md EVERY iteration before writing skills.
999999. Every tool name must be a real Automatos platform/workspace/composio tool.
9999999. Implement functionality completely. No placeholders or stubs. No generic workflows.
99999999. Keep @scripts/ralph/IMPLEMENTATION_PLAN.md current with learnings.
999999999. Skills go in the SKILLS REPO, not in automatos-ai.
9999999999. ONE task per iteration. Validation MUST pass. Never output RALPH_COMPLETE if tasks remain.
