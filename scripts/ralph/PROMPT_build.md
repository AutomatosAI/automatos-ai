# Build Mode

Implement ONE task from the plan, validate, commit, exit.

## Phase 0: Orient

Study with subagents:
- @CLAUDE.md (how to build/test)
- @scripts/ralph/IMPLEMENTATION_PLAN.md (current state — tasks + key references)
- @scripts/ralph/prd.json (acceptance criteria for each story)

### Key References

- **Chat component**: `frontend/components/chatbot/chat.tsx` — main chat with quick links (~line 726), Code button (~line 947/1079), MultimodalInput (~line 1060)
- **Chat input**: `frontend/components/chatbot/multimodal-input.tsx` — textarea + toolbar, handleSubmit, sendMessage callback
- **Mission store**: `frontend/stores/mission-store.ts` — isMissionMode, activePlanningMissionId, setMissionMode, planModifications, taskFeedback
- **Mission hooks**: `frontend/hooks/use-missions-api.ts` — useCreateMission, useApproveMission, useRejectMission, useMission, useMissions, etc.
- **Mission types**: `frontend/types/missions.ts` — MissionResponse, RunState, TaskState, TERMINAL_RUN_STATES, RUN_STATE_CONFIG, computeMissionStats
- **Mission components**: `frontend/components/missions/` — MissionStatusBadge, MissionDAGCanvas, MissionDetailPage, MissionTaskNode, MissionActivityFeed, TaskInspector, HumanReviewPanel, MissionList, MissionCard
- **UX Spec**: `docs/UX/MISSION-CONTROL-UX-SPEC.md` — design decisions and interaction patterns
- **Backend API**: POST /api/missions {goal, config?} → MissionResponse. GET /api/missions → paginated list. GET /api/missions/{id} → detail with tasks + events. POST .../approve, reject, review, pause, resume, cancel.

### Check for completion

```bash
grep -c "^\- \[ \]" scripts/ralph/IMPLEMENTATION_PLAN.md || echo 0
```

- If 0: Run validation → commit → output **RALPH_COMPLETE** → exit
- If > 0: Continue to Phase 1

## Phase 1: Implement

1. **Study the plan** — Choose the FIRST unchecked task from @scripts/ralph/IMPLEMENTATION_PLAN.md
2. **Read prd.json** — Find the matching US-XXX story in @scripts/ralph/prd.json and follow its acceptance criteria exactly
3. **Search first** — Don't assume not implemented. Check if the component/service already exists
4. **Read existing code** — Before creating or editing a file, read the files listed in the story notes and the Key References above to follow existing patterns
5. **Implement** — ONE task only. Implement completely — no placeholders or stubs
6. **Validate** — Run typecheck. All acceptance criteria must be met

### Architecture Rules (CRITICAL)

- React Query v4 — use `isLoading` NOT `isPending` for loading states
- Zustand for UI state (mission mode, selections), React Query for server state (fetching/mutations)
- shadcn/ui components (Button, Skeleton, Input, Textarea, ScrollArea, etc.)
- Lucide React icons EXCLUSIVELY — no other icon libraries
- Dark surfaces, orange accents (#f97316 / orange-500)
- Glass morphism: `backdrop-blur` + `bg-opacity` (e.g., `bg-card/50 backdrop-blur`)
- Framer Motion for animations (already imported in chat.tsx)
- `toast` from `'sonner'` for notifications
- `useRouter` from `'next/navigation'` (App Router, NOT pages router)
- Next.js strict route typing — use `as any` cast on `router.push()` for dynamic routes like `/missions/${id}`
- Follow immutable data patterns — return new objects, don't mutate
- BEFORE DELETING ANY CODE: grep EVERY file for callers

### Validation

For frontend TypeScript (check only mission/chat errors):
```bash
cd frontend && npx tsc --noEmit 2>&1 | grep -iE "mission|chat\.tsx|mission-created" | head -20
```

Note: There are ~781 pre-existing TS errors in unrelated files (workflow-service, permissions, context-service, etc.). Mission and chat components are clean. Only check for NEW errors introduced by your changes.

If no mission/chat errors appear, validation passes.

## Phase 2: Update & Learn

**Update scripts/ralph/IMPLEMENTATION_PLAN.md:**
- Mark completed task `- [x] Completed`
- Add any discovered bugs or issues
- Note new tasks discovered during implementation

**Update scripts/ralph/progress.txt:**
- Log what was completed this iteration

## Phase 3: Commit & Exit

```bash
git add -A && git commit -m "feat(missions): [description of what was implemented]"
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
