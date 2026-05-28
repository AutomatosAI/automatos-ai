---
name: prd-classify
description: Classify an Automatos PRD into Rehouse / Refactor / Extension / Net-new before estimating, so the right amount of reuse vs build is planned. Use when starting any PRD on this codebase.
disable-model-invocation: true
---

# PRD Classifier

Slash command: `/prd-classify <PRD#>` or `/prd-classify` (then paste the PRD).

This skill enforces CLAUDE.md §3 ("REUSE over BUILD"). It surfaces existing surfaces in the codebase **before** estimating, so a PRD doesn't get costed as net-new when it's actually a relabel.

## Workflow

### 1. Read the PRD
Locate the PRD file in `docs/PRDs/` or wherever the user dropped it. Read it end-to-end. Note:
- The stated goal
- The user-facing artifacts mentioned (pages, components, tables, tools, agents)
- Any "build / create / add" language

### 2. Map existing surfaces (mandatory)
For each artifact mentioned in the PRD, check what already exists. Use parallel searches:

- **Frontend components**: `grep -rli "<artifact>" frontend/components/`
- **Hooks**: `find frontend/hooks -type f | xargs grep -li "<artifact>"`
- **API routers**: `grep -rl "@router\." orchestrator/api/ | xargs grep -l "<artifact>"`
- **DB tables**: `grep -rE "class \w+\(.*Base" orchestrator/modules/ orchestrator/core/ | grep -i "<artifact>"`
- **Platform tools**: read `orchestrator/core/platform/platform_actions.py`
- **Skills**: `ls ../automatos-skills/`
- **Knowledge graph**: query `graphify-out/graph.json` (if present) for related nodes

If any of these already exist, **list them with paths** before proceeding.

### 3. Classify
Pick exactly one classification:

| Type | Signal | Default action |
|---|---|---|
| **Rehouse / IA change** | "Move X from Y to Z", "rename", "redesign page" | Components/hooks/APIs exist. Move/relabel, don't rebuild. |
| **Refactor / Consolidation** | "Unify X and Y", "deprecate", "single source of truth" | Pick canonical path, migrate others, delete the losers. |
| **Extension** | "Add X to existing Y" | New code is the increment, not the system. |
| **Net-new feature** | No precedent in the platform | Justify why it doesn't fit an existing pattern. |

### 4. Surface ambiguity
If anything is unclear, **ask before estimating**. Examples:
- "PRD says 'build the Reports page' — I see `frontend/components/activity/activity-reports.tsx`. Is the PRD adding to it or replacing it?"
- "There are 3 things called `task_*` in the schema. Which one does this touch?"
- "The PRD reads like a config dial — should this be a setting, not a new code path?"

### 5. Report
Output the classification, the existing surfaces found, and a one-paragraph plan with the next concrete step. Do NOT start coding — `/prd-classify` is a research command.

## Output format

```
PRD-XXX classification: <Rehouse|Refactor|Extension|Net-new>

Existing surfaces:
- <path/to/file.tsx> — <one-line of what it does>
- <path/to/file.py> — <one-line>
- ...

Reuse plan:
- Reuse:   <thing>
- Extend:  <thing>
- Build:   <thing>   (justify why nothing fits)
- Delete:  <thing>   (what gets superseded — see CLAUDE.md §5)

Open questions for the user:
- <question 1>
- <question 2>

Next step: <one concrete action>
```

## Anti-patterns to avoid
- Starting with code before classification
- Assuming "build X" means "build from scratch" when X exists
- Skipping the graph search because grep felt faster
- Padding the "Build" column when "Extend" would do
