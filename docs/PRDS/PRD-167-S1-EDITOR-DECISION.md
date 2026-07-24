# PRD-167 S1 — Block Editor Library Decision Memo

**Status:** Decided — **Plate (`@udecode/plate`)**
**Date:** 2026-06-12
**Author:** PRD-167 implementation

The OSS research pass in the original session never ran (session limit). This is the
time-boxed spike the PRD called for: **BlockNote vs Plate vs Puck** against the hard
requirements. The decision is binding for S5; the loser deps never enter `package.json`.

## Requirements (from PRD-167 S1)

| # | Requirement | Why it matters |
|---|---|---|
| R1 | Custom **variable-chip inline node** | `{{user.name}}` chips must live *inside* flowing text, be atomic (non-editable interior), and round-trip to our block schema |
| R2 | **Controlled JSON output** | We own the storage format (`blocks` JSONB). The editor must be a controlled component emitting plain JSON we can map to/from our schema |
| R3 | **React 18** | Platform frontend is React 18 |
| R4 | **MIT / Apache license** | Hard gate. Automatos ships a **closed-source SaaS** — copyleft on the editing surface is unacceptable |
| R5 | **Table + image blocks** | Templates need tables (invoices, data) and logo/image blocks |

## Scored matrix

Scale: ✅ meets / ⚠️ partial-or-assembly / ❌ fails. License is a **gate** (a ❌ there is disqualifying regardless of other scores).

| Lib | R1 chips | R2 controlled JSON | R3 React 18 | **R4 license (GATE)** | R5 table+image | Verdict |
|---|---|---|---|---|---|---|
| **BlockNote** (TypeCellOS) | ✅ custom inline content | ✅ native block JSON | ✅ | ❌ **MPL-2.0 core + GPL-3.0 `xl-*`** | ⚠️ **table block is in the GPL-3.0 `@blocknote/xl-*` tier** | **DISQUALIFIED** |
| **Plate** (`@udecode/plate`) | ✅ inline void + `mention` plugin is exactly this pattern | ✅ Slate value is plain JSON, fully controlled | ✅ (18/19) | ✅ **MIT** | ✅ `plate-table`, `plate-media` (MIT) | **CHOSEN** |
| **Puck** (`@measured/puck`) | ❌ no inline text model — it composes *components*, not prose | ✅ JSON, but page-layout shaped | ✅ | ✅ MIT | ⚠️ via custom components | **WRONG CATEGORY** |

## Decision: Plate

**BlockNote is the most Notion-like out of the box and would have been the default
pick — but it fails the license gate.** Its core is **MPL-2.0** (file-level copyleft,
borderline-tolerable) and, critically, the **table block and several advanced blocks
ship in the `@blocknote/xl-*` packages under GPL-3.0**. Pulling a GPL-3.0 dependency
into a proprietary SaaS bundle to satisfy R5 (tables) is a non-starter. The PRD made
license a hard requirement precisely to catch this.

**Puck** is MIT but is a visual **page/layout builder** (drag-drop component
composition). It has no inline-text model, so R1 (variable chips inside flowing
paragraph text) can't be expressed cleanly. Wrong abstraction for a document editor.

**Plate** is **MIT**, built on Slate, and is purpose-built for exactly this:
- Variable chips → Slate **inline void elements**; Plate's `mention` plugin is this
  pattern already (atomic, non-editable interior, trigger-driven insertion).
- Controlled JSON → Slate's `value` is a plain serialisable JSON array; Plate is a
  fully controlled component. We map that value to/from our canonical `blocks` schema.
- Tables (`@udecode/plate-table`) and media/images (`@udecode/plate-media`) are MIT.
- React 18/19 supported.

## Architectural consequence (important)

We do **not** adopt Slate's node format as our storage format. We define our own
**canonical block schema** (`blocks` JSONB — heading/text/table/image/variable/
page-break/section) as the durable storage + render contract. Plate is only the
*editing surface*; it serialises to/from the canonical schema via a thin adapter
(`frontend/lib/templates/plate-adapter.ts`).

This keeps the **renderers (block→HTML→PDF, block→DOCX) and the whole backend
independent of the editor choice** — the editor is swappable, the schema and renderers
are not. It is also why S2/S3/S4/S6 carry no frontend-library dependency at all.

## Dependencies entering `package.json` (Plate only)

```
@udecode/plate            (MIT) — core + react
@udecode/plate-basic-elements, plate-basic-marks
@udecode/plate-table      (MIT) — table block
@udecode/plate-media      (MIT) — image/logo block
slate, slate-react, slate-history  (MIT) — peer engine
```

BlockNote and Puck packages are **not** added. Spike branch artifacts: none retained
(this memo is the artifact).

## Implementation note (this PR)

This PR ships a **dependency-free structured block editor** (`components/documents/blocks/`)
that emits the canonical block schema directly — no editor library is added to
`package.json` yet. Rationale:

1. The delivery constraint for this PR was "runs without dependencies" — adding 5+ Plate
   packages that can't be browser-verified in the build environment is a risk we defer.
2. The architectural decision above already makes the editor a swappable surface behind
   the canonical schema + `plate-adapter` seam. The structured editor authors variable
   chips as `{{path}}` tokens that parse to/from `VariableRun`s and resolve live in the
   preview pane — genuinely usable for the non-technical flow.
3. **Plate remains the chosen rich-editor target.** Dropping it in is purely additive:
   implement `plate-adapter.ts` (canonical ⇄ Slate value) and swap the editing surface;
   the backend, schema and renderers do not change.

So: Plate is the decided library; the dependency-free editor is v1's editing surface.
This is called out here rather than silently shipped.
