# PRD-82A Phase 6: Chat Mission Mode — Implementation Plan

## Overview
Add Mission mode toggle to chat input, create missions from chat, show inline mission cards, wire plan approval on mission detail page.

## Branch: ralph/82a-mission-chat-mode

---

## Tasks

- [x] US-001: Add Mission mode toggle button to chat quick links
- [x] US-002: Show mission mode indicator banner in chat
- [x] US-003: Intercept chat submit to create mission when in mission mode
- [x] US-004: Show inline mission card in chat after creation (new component)
- [x] US-005: Render mission card in chat when activePlanningMissionId is set
- [x] US-006: Add plan approval controls to mission detail page
- [x] US-007: Set DAG canvas to plan mode when awaiting_approval
- [x] US-008: Handle /mission slash command in chat input

---

## Key References

- **Chat component**: `frontend/components/chatbot/chat.tsx` — main chat with quick links at ~line 726, Code button at ~line 947/1079
- **Chat input**: `frontend/components/chatbot/multimodal-input.tsx` — textarea + toolbar
- **Mission store**: `frontend/stores/mission-store.ts` — isMissionMode, activePlanningMissionId, planModifications, taskFeedback
- **Mission hooks**: `frontend/hooks/use-missions-api.ts` — useCreateMission, useApproveMission, useRejectMission, useMission, etc.
- **Mission types**: `frontend/types/missions.ts` — MissionResponse, RunState, TaskState, etc.
- **Mission components**: `frontend/components/missions/` — MissionStatusBadge, MissionDAGCanvas, MissionDetailPage, etc.
- **UX Spec**: `docs/UX/MISSION-CONTROL-UX-SPEC.md` — design decisions and interaction patterns
- **Backend API**: POST /api/missions (create), approve, reject, review, pause, resume, cancel

## Architecture Notes

- React Query v4 — use `isLoading` not `isPending`
- Zustand store for UI state, React Query for server state
- shadcn/ui components (Button, Skeleton, etc.)
- Lucide React icons exclusively
- Dark surfaces, orange accents (#f97316 / orange-500)
- Glass morphism: backdrop-blur + bg opacity
- Framer Motion for animations
- toast from 'sonner' for notifications
- useRouter from 'next/navigation' (App Router)
- Next.js strict route typing — use `as any` cast on router.push() for dynamic routes

## Validation

For frontend TypeScript:
```bash
cd frontend && npx tsc --noEmit 2>&1 | grep -i "mission\|chat" | head -20
```

Note: There are ~781 pre-existing TS errors in unrelated files (workflow-service, permissions, etc.). Mission components are clean. Only check for NEW errors in mission/chat files.
