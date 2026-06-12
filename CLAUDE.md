# CLAUDE.md — Automatos AI Platform

> Project-specific guidance for Claude. Complements the user's global `~/.claude/CLAUDE.md`.

## 1. Where this project is

Automatos is **at PRD 136+**, not greenfield. The platform has 109 SQLAlchemy tables, 103 API routers, 599 frontend TS files, and a Knowledge Graph mapping every code path. **Most "new" work isn't new** — it's rehousing, refactoring, consolidating, or extending what already exists.

Treat this codebase like a mature product, not a startup MVP.

---

## 2. Default mindset: REUSE over BUILD

Before writing any code for a PRD, work through this order:

1. **Search the graph first.** `graphify-out/graph.json`, `GRAPH_REPORT.md`, `db.json` — these exist for a reason. Use them.
2. **Search the codebase.** Component, hook, table, endpoint — does it already exist?
3. **Read the relevant memory files** at `~/.claude/projects/-Users-gkavanagh-Development-Automatos-AI-Platform-automatos-ai/memory/MEMORY.md` — many concepts (Playbooks, Missions, Tasks, tools, plumbing) are documented there.
4. **Then** decide: reuse, extend, refactor, or build new.

If you're about to write a new component/hook/table/endpoint, the burden is on you to justify *why* the existing one isn't enough.

---

## 3. PRD framing — what type is this?

Before estimating or implementing, classify the PRD:

| Type | Signal | Default action |
|---|---|---|
| **Rehouse / IA change** | "Move X from Y to Z", "rename", "redesign page" | Components, hooks, APIs already exist. Move/relabel, don't rebuild. |
| **Refactor / Consolidation** | "Unify X and Y", "deprecate", "single source of truth" | Pick the canonical path, migrate the others, **delete the losers**. |
| **Extension** | "Add X to existing Y" | New code is the increment, not the system. |
| **Net-new feature** | No precedent in the platform | Justify why it doesn't fit an existing pattern. Build small, prove the value, then expand. |

If the PRD is ambiguous, **ask before assuming**.

---

## 4. Clean-coding rules (Automatos-specific)

These are non-negotiable on this codebase:

- **No backward-compat shims.** When a path is replaced, the old one is deleted in the same PR. No "_legacy" suffix that lives forever. (See memory: `feedback-no-backward-compat.md`.)
- **No file hacks for DB data.** Personas, configs, agent definitions belong in the database, not loaded from files at runtime. (See memory: `feedback-no-file-hacks-for-db-data.md`.)
- **No `os.getenv()` outside `config.py`.** All env reads go through the canonical config module. Enforced across 86 files.
- **No hardcoded values.** Constants and config only.
- **No new tables when an existing one fits.** This is the #1 rule for PRDs that feel "new" — the table probably exists.
- **No new tools when an existing tool can be extended.** The 3-file platform-tool registration pattern is the canonical extension point. (See memory: `prd71-tools.md`.)
- **No duplicate hooks.** If `useDeliverables` exists, don't add `useDeliverablesV2`. Refactor the existing one.

---

## 5. Replace cleanly — delete what's superseded

When a PRD replaces an existing surface (component, route, table, hook, tool):

1. Build the replacement.
2. Migrate callers / data.
3. **Delete the original in the same PR.**
4. Remove orphan imports, unused files, dead routes.
5. Update memory files if the change affects documented architecture.

Do NOT leave both running "just in case." That's how 200+ tables and 103 routers happened. (See memory: `prd131-consolidation.md` for the cleanup we already had to do.)

**Exception:** If migration risk is high (production data, integrations), keep both *temporarily* with a documented sunset date. Default is delete.

---

## 6. Ask before assuming

Stop and ask the user when:

- The PRD says "build X" but X looks like it already exists. → "I see `<thing>` at `<path>`. Is the PRD adding to it or replacing it?"
- A table/route/component name is ambiguous. → "There are 3 things called `task_*`. Which one are we touching?"
- The instruction implies a new feature but a config dial would do. → "Is this meant to be a setting users toggle, or a new code path?"
- Scope creeps mid-implementation. → "We started with X but now seeing Y is involved. Stay narrow or expand?"

Asking costs 30 seconds. Wrong assumptions cost half a day.

---

## 7. Plan first, code second

For non-trivial PRDs (anything beyond a one-file edit):

1. **Map the existing surface** — what tables, routes, components, hooks are involved?
2. **Identify reuse candidates** — what can be moved/extended vs built?
3. **Surface the questions** — anything ambiguous gets raised before code is written.
4. **Then propose the plan** — wave-by-wave if it spans multiple surfaces.

Never go "[user request] → code". Always go "[user request] → research → plan → code".

---

## 8. Auto-memory — read before research, write after learning

The `MEMORY.md` index at `~/.claude/projects/-Users-gkavanagh-Development-Automatos-AI-Platform-automatos-ai/memory/` is project history. **Check it first for any topic that appears there** — the answer to many "how does X work?" questions is already written down.

After learning something new about the platform during a PRD, update memory:
- New architectural patterns → save
- Bug-fix root causes → save
- User-corrected mistakes (e.g. "Playbooks ≠ Missions") → save as feedback memory
- Project decisions and dates → save as project memory

Do NOT save what's already in the code (file paths, function names, recent commits). `git log` and grep are authoritative for those.

---

## 9. Graphify is your second brain

`graphify-out/graph.json` (18MB) and `GRAPH_REPORT.md` map the codebase's actual structure. For broad questions ("how does X flow?", "what calls Y?", "is Z dead code?"), **start with the graph** before grepping.

After significant changes (new files, renames, architecture shifts), suggest running `/graphify --update` to keep it fresh.

---

## 10. Canonical terms — do not drift

| Use | Do not use |
|---|---|
| **Playbook** | ~~Recipe~~ (legacy) |
| **Mission** | ~~Workflow~~, ~~Job~~ |
| **Task** | (canonical: `BoardTask`; mission sub-tasks are `OrchestrationTask`) |
| **Deliverable** | ~~Output~~, ~~Workspace file~~, ~~Artifact~~ (in user-facing copy) |
| **Knowledge Graph** | ~~Business Graph~~ |
| **Command Center** | ~~Activity~~ (renamed) |
| **Auto** | "the assistant" — Auto is a proper noun, a character |

Drift costs the user. Use the right word.

---

## 11. The "shiny new" trap

Sometimes a PRD genuinely is new. When it is:

- Build the smallest version that proves value.
- Reuse existing patterns (3-file tool registration, ActionRegistry, FilePreview, MarketplaceGrid layout, etc.).
- Don't introduce new dependencies if existing ones cover it.
- Don't invent UI patterns that aren't already in the design system.

New is a privilege earned by ruling out reuse, not a starting position.

---

**Last updated:** 2026-04-25

## 12. Descoping is the user's call — NEVER defer unilaterally

Do **not** carve scope out to "follow-on PRD", "Phase 2", "out of scope", "deferred to PRD-X", or "a separate effort" on your own initiative. That is **Gerard's decision, not yours.**

- If a piece is needed for the feature to **actually work end-to-end**, it **is** the work — build it now. "It works" means the user can use it for its stated purpose, not that the happy path compiles.
- If a piece is genuinely large and separable, **surface the choice and let Gerard decide** — present it as an open question, never as an already-settled deferral.
- When you catch yourself about to write "follow-on", "out of scope", "Phase 2", or "deferred", **stop**: either do it, or ask. Do not narrate a deferral as if it were decided.

Gerard writes the PRDs. Silently shrinking their scope wastes that work and erodes trust. Finish what the PRD set out to do.

---

**TL;DR:** This is a mature codebase. Reuse first, build last. Delete what you replace. Ask before assuming. Read the graph and memory before grepping. Don't reinvent what shipped in PRDs 1–135. **Don't defer scope — that's Gerard's call (§12).**
