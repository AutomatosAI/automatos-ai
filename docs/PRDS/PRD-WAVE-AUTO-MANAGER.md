# Wave: Auto as Manager — Rollout Overview (PRD-224 → PRD-229)

> **Status:** Planning document — written 2026-08-27. All six PRDs are drafted and unscheduled; nothing is implemented. Companion review: https://claude.ai/code/artifact/f31677a8-f2cb-47fe-b7dd-f705d764418b
> Scope decisions already made (Gerard, 2026-08-27): no CLI-agent wrapping or file-hive mechanics; guardrails covered by existing Guardrails/Blueprints/SDLC; missions stay the large-task lane — this wave is about **line management, questions, doctrine, and legibility**.

## The six PRDs

| PRD | Title | What it delivers | Size | Schema | New routes |
|---|---|---|---|---|---|
| [224](./PRD-224-AUTO-TICKET-LANE.md) | The Ticket Lane | "Have my accountant agent do X" → assigned ticket, supervised, result reported back in-thread. ASSIGN lane + watches learn `board_task` + auto-watch on assign | M | none (enum-in-code) | none |
| [225](./PRD-225-AGENT-QUESTIONS-ASK-ME.md) | Agent Questions (ASK ME) | Free-text asks park work; Questions tab with blocked-cascade; answer→auto-resume; Telegram reply answers; ingress trust gate | L | extend `approval_grants` | 1 (answer) |
| [226](./PRD-226-AUTO-MANAGER-DOCTRINE.md) | Manager Doctrine | The management prompt: three lanes, delegate-don't-implement, reuse-before-create, 4-part dispatch contracts, board-as-ledger, narration | S | persona backfill only | none |
| [227](./PRD-227-BOARD-LIGHT-UP.md) | Board Light-Up | Agent moves push SSE + blocked/failed parity; mission narration into the launching thread; bell deep-link fixes | S | none | none |
| [228](./PRD-228-FLEET-STATE.md) | Fleet State | One read-model: per-agent current work/leases/asks/cost; `platform_fleet_status` for Auto; live fleet view | M | none (read-model) | 1 (fleet) |
| [229](./PRD-229-MID-RUN-CLARIFICATIONS.md) | Mid-Run Clarifications | `ask_orchestrator`: Auto answers routine questions from run context; the rest escalate into 225's queue; delete dead `inter_agent.py` | M | none | none |

## Dependencies

```
227 (wiring) ──────────────┐
226 (doctrine) ────────────┼──► 224 (ticket lane needs the lane doctrine + visible board)
                           │
225 (questions) ◄──────────┴─── 224 escalations prefer 225 once it exists (soft dep)
   │
   └──► 229 (clarifications escalate INTO 225's queue — hard dep)

228 (fleet) — independent; grounds 226's "awareness" and enriches 229's answering context (soft deps)
```

Hard dependency: **229 needs 225.** Everything else is soft — any order works, but the soft deps are why the suggested sequence below exists. 224's Wave-2 escalation target upgrades from "escalation card" to "question" when 225 lands; shipping 224 first is fine.

## Suggested rollout (for discussion — Gerard owns the schedule)

- **Batch 1 — make the floor visible and give Auto its voice: 227 + 226.** Small, no schema, no routes, immediately felt in every existing flow (missions narrate, agent moves live, bell works). Doctrine eval gate de-risks everything after.
- **Batch 2 — the centerpiece: 224.** The ticket lane, on a board that's now live and a persona that now manages. Demo moment: "have <agent> do X" → card appears, moves, reports back.
- **Batch 3 — the queue: 225.** Model + tab + Telegram. Second demo moment: answer Auto's question from the phone and watch the work resume.
- **Batch 4 — awareness: 228.** Fleet endpoint + tool + view ("how's the team doing?").
- **Batch 5 — the multiplier: 229.** Clarifications + the `inter_agent.py` decide-and-delete.

## Activation dials introduced (flag loudly at ship time — nothing lands dark)

| Dial | PRD | Default |
|---|---|---|
| `AUTO_TICKET_WATCH` (auto-supervise assigned tickets) | 224 | on |
| Channel `trigger_mode` (`strict\|communication_only\|allow_all`) | 225 | strict |
| `question_pending` quiet-hours behavior | 225 | open Q |
| Narration + `MISSION_NARRATION_TASK_CAP` | 227 | on / 8 |
| `CLARIFICATION_BUDGET` per run | 229 | 3 |

## Cross-cutting rules that bind every PRD here

- CI is the only gate; nothing runs locally. Gold sets and eval fixtures stay LOCAL (public repo).
- Route-manifest procedure for the two new routes (225, 228): `RouterSpec` → regenerate + commit `reports/route-manifest.json` with bumped count → `api-client.ts` parity → route-contract CI.
- Alembic single head (225's migration is the only schema change in the wave).
- No new tables (225 extends `approval_grants`), no duplicate hooks, delete superseded code in the same PR (229's Component D).
- Dead vocabulary is not resurrected: nothing builds on `RunState.AWAITING_HUMAN` / `TASK_HUMAN_*` (zero writers; retired surface).

## Open questions collected (need Gerard before the relevant batch)

1. **224:** default start mode for tickets (immediate vs heartbeat); auto-assign fallback when no agent named (proposal: ask); `AUTO_TICKET_WATCH` default on.
2. **225:** dismiss keeps subject blocked (proposal: yes); quiet-hours bypass for questions; Telegram target chat (workspace default vs dedicated questions channel).
3. **226:** persona tone (CTO vs floor-manager voice); narration as per-workspace dial; backfill policy for customized souls.
4. **227:** narration for non-chat-launched missions (proposal: yes, to Auto thread); task-line cap default.
5. **228:** fleet view location (Agents page vs Command Center); canonical cost source; period definition.
6. **229:** clarification budget default; parked-task partial output as draft (proposal: yes); **confirm delete of `inter_agent.py`**.
