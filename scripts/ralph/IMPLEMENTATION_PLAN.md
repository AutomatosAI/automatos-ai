# Chat Mode Bar — Implementation Plan

## Overview
Replace navigation quick links with functional core modes (Code, Mission) and user-pinnable agent shortcuts.

## Branch: ralph/chat-mode-bar

---

## Tasks

- [ ] US-001: Create usePinnedAgents hook with localStorage
- [ ] US-002: Create ChatModeBar component with core modes
- [ ] US-003: Add pinned agent buttons to ChatModeBar
- [ ] US-004: Create PinAgentPicker dropdown for adding pins
- [ ] US-005: Replace quickLinks with ChatModeBar in chat.tsx
- [ ] US-006: Add Pin to Chat menu item on agent cards

---

## Key References

- **Chat component**: `frontend/components/chatbot/chat.tsx` — quick links at ~line 790, Code button at ~line 1010/1175, handleOpenCodeCanvas at ~line 82
- **Chat input**: `frontend/components/chatbot/multimodal-input.tsx` — textarea + toolbar
- **Agent selector**: `frontend/components/chatbot/agent-selector.tsx` — Agent interface, dropdown, handleAgentChange
- **Agent roster**: `frontend/components/agents/agent-roster.tsx` — agent cards with dropdown menus
- **Mission store**: `frontend/stores/mission-store.ts` — isMissionMode, setMissionMode
- **Workspace store**: check for useWorkspaceStore or workspace-provider for workspace ID
- **Agent hooks**: `frontend/hooks/use-agent-api.ts` — useAgents() hook
- **Existing pin/fav**: NONE — this is net new functionality

## Architecture Notes

- React Query v4 — use `isLoading` not `isPending`
- `use-local-storage-state` package is already installed for localStorage hooks
- shadcn/ui components: DropdownMenu, Button, etc.
- Lucide React icons exclusively (Code2, Target, Plus, Pin, PinOff, Check)
- Dark surfaces, orange accents (#f97316 / orange-500)
- Existing button styling in chat.tsx — match exactly
- Agent IDs are numbers (not UUIDs) — see Agent interface in agent-selector.tsx
- Max 6 pinned agents — enforce in hook
- SSR safety: typeof window !== 'undefined' checks (Next.js)

## Validation

```bash
cd frontend && npx tsc --noEmit 2>&1 | grep -iE "chat-mode|pin-agent|pinned|chat\.tsx|agent-roster" | head -20
```

Note: ~781 pre-existing TS errors in unrelated files. Only check for NEW errors.
