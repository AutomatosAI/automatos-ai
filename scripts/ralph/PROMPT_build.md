# Build Mode

Implement ONE task from the plan, validate, commit, exit.

## Phase 0: Orient

Study with subagents:
- @CLAUDE.md (how to build/test)
- @docs/PRDS/72-ACTIVITY-COMMAND-CENTRE.md (full requirements)
- @scripts/ralph/IMPLEMENTATION_PLAN.md (current state)
- @scripts/ralph/prd.json (acceptance criteria for each story)

### Key References

- **Design system**: `frontend/app/globals.css` — glass-card, card-glow, stage-*, log-entry-*, semantic colour tokens
- **Shared components**: `frontend/components/shared/` — PageHeader, StatsBar, FilterTabs, SearchInput
- **Hook pattern**: `frontend/hooks/use-workflow-api.ts` — React Query key factory + useQuery + useMutation with toast
- **Backend pattern**: `orchestrator/api/execution_history.py` — FastAPI router + get_request_context_hybrid() auth
- **Existing heartbeat API**: `orchestrator/api/heartbeat.py` — NOT registered in main.py yet
- **Existing workflows**: `frontend/components/workflows/` — RecipesTab, ActiveWorkflowsPanel, etc.
- **Analytics page** (design reference): `frontend/components/analytics/analytics-page.tsx`

### Check for completion

```bash
grep -c "^\- \[ \]" scripts/ralph/IMPLEMENTATION_PLAN.md || echo 0
```

- If 0: Run validation → commit → output **RALPH_COMPLETE** → exit
- If > 0: Continue to Phase 1

## Phase 1: Implement

1. **Study the plan** — Choose the FIRST unchecked task from @scripts/ralph/IMPLEMENTATION_PLAN.md
2. **Read prd.json** — Find the matching US-XXX story in @scripts/ralph/prd.json and follow its acceptance criteria exactly
3. **Search first** — Don't assume not implemented. Check if the component/endpoint already exists
4. **Read existing code** — Before creating a new file, read the reference files listed in the story notes and follow existing patterns
5. **Implement** — ONE task only. Implement completely — no placeholders or stubs
6. **Validate** — Run typecheck. All acceptance criteria must be met

### Design Rules (CRITICAL)

- NO hardcoded hex colours — use `hsl(var(--primary))`, `hsl(var(--success))`, `hsl(var(--info))`, etc.
- All cards use `glass-card` class from globals.css
- All panels use `glass-panel` class
- Status badges use semantic CSS variables from PRD Section 1
- Follow typography scale from PRD Section 2
- Use `framer-motion` for entrance animations
- Use existing shared components (PageHeader, StatsBar, FilterTabs) — don't rebuild them
- Backend endpoints use `get_request_context_hybrid()` for auth + workspace isolation
- Hook files follow `use-{feature}-api.ts` naming with query key factories

### Validation

For frontend changes:
```bash
cd frontend && npx tsc --noEmit 2>&1 | head -30
```

For backend changes:
```bash
cd orchestrator && python -c "import api.main" 2>&1
```

If validation cannot run (e.g., no node_modules), verify via grep that all imports resolve.

## Phase 2: Update & Learn

**Update scripts/ralph/IMPLEMENTATION_PLAN.md:**
- Mark completed task `- [x] Completed`
- Add any discovered bugs or issues
- Note new tasks discovered during implementation

**Update CLAUDE.md** (if you learned something new):
- Add correct commands or patterns discovered
- Keep it brief and operational

## Phase 3: Commit & Exit

```bash
git add -A && git commit -m "feat(activity): [description of what was implemented]"
```

Check remaining:
```bash
grep -c "^\- \[ \]" scripts/ralph/IMPLEMENTATION_PLAN.md || echo 0
```

- If > 0: Say "X tasks remaining" and EXIT
- If = 0: Output **RALPH_COMPLETE**

## Guardrails

99999. Capture the why — tests and implementation importance.
999999. Single sources of truth, no migrations/adapters.
9999999. Implement functionality completely. No placeholders or stubs.
99999999. Keep @scripts/ralph/IMPLEMENTATION_PLAN.md current with learnings.
999999999. For any bugs you notice, resolve them or document them even if unrelated.
9999999999. ONE task per iteration. Search before implementing. Validation MUST pass. Never output RALPH_COMPLETE if tasks remain.
