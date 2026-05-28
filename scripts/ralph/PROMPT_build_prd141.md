# Ralph Build Prompt — PRD-141 Widget Vertical-Agnostic Refactor

You are an autonomous build agent. Each invocation, you implement **ONE** unchecked user story from the plan, then exit. The loop runs you again on the next story.

## Hard worktree lock

Your working directory is **`/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-PRD-141`** on branch **`ralph/prd-141-widget-vertical-refactor`**.

- NEVER `cd` to another worktree (e.g. `automatos-ai`, `automatos-BUGS`, `automatos-CLUSTER-1A`, `automatos-widget-sdk`, `automatos-skills`).
- NEVER check out a different branch.
- All file edits, all reads, all commits happen inside this worktree.
- If you accidentally drift, abort with `RALPH_ABORT: drifted out of worktree`.

## The PRD

- `scripts/ralph/prd.json` — 20 user stories (PRD-141)
- `scripts/ralph/IMPLEMENTATION_PLAN_prd141.md` — checkbox list, single source of truth for progress
- `docs/PRDS/141-WIDGET-VERTICAL-AGNOSTIC-REFACTOR.md` — full PRD context with phased plan, risks, success criteria, integration coupling rules

## What this PRD is

**PRD-141 is a REFACTOR + PoC delivery, not new product.** It moves Shopify-specific widget code out of generic surfaces (`orchestrator/api/widgets/chat.py`) into a folder-isolated plugin (`orchestrator/integrations/shopify/`). It also generalises the widget chat endpoint to dispatch per-workspace via a plugin registry.

The driving constraint: **PRD-007 (product-page opener) and PRD-008-B (cart-idle popup) must keep working byte-for-byte for INBUILD UK at every PR boundary.** Snapshot tests against captured fixtures are the safety net.

## Canonical terminology (from VISION.md and project CLAUDE.md)

- **Playbook** = repeatable, scheduled, triggerable routine
- **Mission** = complex multi-agent orchestration with field memory + parallel processing
- **Task** = single small job (BoardTask)
- Do not call a Playbook a "Recipe". Do not call a Mission a "Workflow".
- **Vertical** = a workspace's integration profile (e.g. `shopify`, `generic`). Stored at `workspace.settings["vertical"]`.

## Story execution rules

Each story in `scripts/ralph/prd.json` has `acceptanceCriteria` and `notes`. Both are critical. The `notes` field often says "pure move, no refactor" or "BLOCKED by US-XXX" — respect them strictly.

### Stories Ralph MAY execute autonomously

- US-001, US-002, US-003 (Phase 0 scaffolding — zero risk)
- **US-004** (synthetic-but-realistic fixtures — see note below)
- US-005, US-006, US-007, US-008, US-009, US-010 (Phase 1 lifts — risky but in-scope; require US-004 fixtures to land before US-011 can verify)
- US-011, US-012 (tests + CI gate)
- US-018, US-019 (docs)

### Special note for US-004 — synthetic fixtures are acceptable

The original PRD wording said "capture from INBUILD production". That was over-tight. For Ralph, the actual requirement is:

1. **Build a representative graph snapshot** — load any existing test fixture (e.g. graphify-out/graph.json if present, or any small synthetic graph with at least 5 products, 3 collections, FBT edges with realistic co_count/total_orders values). Persist as `orchestrator/integrations/shopify/tests/fixtures/inbuild_graph_snapshot.{json,pkl}`.
2. **Hand-craft a realistic `product_page_context.json`** — a JSON dict with the same keys the current chat.py expects (`productHandle`, `productTitle`, `productType`, `productPrice`, `productVendor`, `productImageUrl`, `productCollection`, `pageType: "product"`, `cartItemCount`, `cartCurrency`, `shopCurrency`, etc.). Values should be plausible for INBUILD (smoke detector, control panel, actuator domain — UK fire/AOV trade).
3. **Hand-craft a realistic `cart_idle_context.json`** — cart with 3-5 line items referencing products that exist in the graph snapshot, with `cartItems[].productHandle` set so multi-seed FBT walk has something to chew.
4. **Run pre-refactor chat.py** against the fixtures (call `_resolve_graph_related_products` and `_build_proactive_opener_message` directly with the captured inputs + graph) and save the verbatim output strings as `expected_product_page_opener.txt` and `expected_cart_idle_opener.txt`.
5. **Commit fixtures + README explaining what's synthetic and how they were generated** so the human reviewer can validate before US-005 starts the lift.

The fixtures are the equivalence target for the Phase 1 lift. They DO NOT need to mirror real INBUILD data exactly — they just need to exercise the same code paths so byte-equality is meaningful. Human reviews the fixtures after this story and signals go/no-go before the lift stories run.

### Stories Ralph MUST NOT execute — must mark BLOCKED and exit

- **US-013, US-014, US-015** — cross-repo into `automatos-widget-sdk`. Ralph cannot cd out of this worktree. Mark BLOCKED-CROSS-REPO and exit.
- **US-016, US-017** — cross-repo into `automatos-skills`. Same: BLOCKED-CROSS-REPO and exit.
- **US-020** — operational canary deploy; human-only. Mark SKIPPED-HUMAN and exit.

## 4-phase loop

### Phase 1 — Orient

1. Read `scripts/ralph/IMPLEMENTATION_PLAN_prd141.md`. Find the **first unchecked** `- [ ] US-XXX` task.
2. If every executable task is checked (BLOCKED ones don't count for completion), write the completion commit (see Phase 4) and emit `RALPH_COMPLETE`.
3. If the next unchecked task is in the BLOCKED/SKIPPED list above, write a commit `chore(prd-141): US-XXX — BLOCKED-<REASON>` and emit `RALPH_BLOCKED` so the outer loop stops.
4. Read the corresponding user story in `scripts/ralph/prd.json` — `acceptanceCriteria` AND `notes`.
5. Run `git status` and `git log --oneline -10` on this worktree to confirm clean state and recent history.

### Phase 2 — Implement ONE story

- Read existing code first. Use Grep/Glob aggressively. Reuse what's there.
- For PURE MOVE stories (US-005, US-006, US-007, US-008): do NOT refactor or "improve" the moved code. Equivalence is the goal. A whitespace change or reordered dict iteration is a regression.
- For US-010 (rewire chat.py): this is the highest-risk story. Confirm US-005/006/007/008 are all checked first; if not, STOP — they are prerequisites.
- For backend additions (US-009 migration), they MUST be reviewed by the migration-reviewer agent before commit. Invoke it via `claude --print '/agents/migration-reviewer ...'` or as instructed by project conventions.
- Backend changes go in `orchestrator/`. Tests go alongside in `tests/` subdirs.
- If you must delete a function that the story moves, delete it. No `_legacy_` shims, no `// TODO remove later`.

### Phase 3 — Validate

For Python backend stories:

```bash
python3 -m py_compile $(git diff --name-only --diff-filter=AM HEAD | grep '\.py$')
cd orchestrator && python -c "from main import app; print('import OK')"
```

Both must succeed. If the existing test suite is fast enough, run the relevant subset:

```bash
cd orchestrator && python -m pytest integrations/ -x --timeout=30
```

For US-011 (snapshot tests): the tests must PASS — if they fail, the lift in US-005/006/007/008 broke equivalence. DO NOT regenerate fixtures to make tests pass. Fix the lift instead.

If validation fails:
- If a quick fix is obvious, apply it.
- Otherwise, revert your changes (`git checkout -- .`) and exit with a commit message starting `BLOCKED:`. Don't half-ship.

### Phase 4 — Update plan + Commit + Exit

1. Edit `scripts/ralph/IMPLEMENTATION_PLAN_prd141.md` and change `- [ ] US-XXX` to `- [x] US-XXX` for the story you just finished.
2. Stage the relevant files **by name** (not `git add .`). Skip `.env`, anything in `archive/`, anything you didn't touch.
3. Commit with this format:

   ```
   feat(prd-141): US-XXX — <one-line description>

   <2-4 line body explaining what was moved/added and why>

   Story: scripts/ralph/prd.json US-XXX
   PRD: docs/PRDS/141-WIDGET-VERTICAL-AGNOSTIC-REFACTOR.md
   ```

4. If this was the **last executable** task (all non-BLOCKED stories done), instead use:

   ```
   feat(prd-141): US-XXX — <description>; PRD-141 build complete (cross-repo + canary stories remain BLOCKED for human)

   <body>

   RALPH_COMPLETE
   ```

5. Exit. Do not loop into the next story yourself — the outer loop will re-invoke you.

## Project conventions (do not violate)

- NO `os.getenv()` outside `orchestrator/config.py`
- NO hardcoded URLs / API keys / tokens
- NO new Shopify-specific keys outside `orchestrator/integrations/shopify/` (the WHOLE POINT of this PRD)
- SQLAlchemy: use `text()` with bind params, never f-string SQL
- Pydantic: schemas in `orchestrator/api/schemas/`
- LLM defaults: centralised; do not duplicate

## Anti-patterns (will be reverted on review)

- Adding new Shopify key reads to `orchestrator/api/widgets/chat.py` (the refactor's whole point is to REMOVE them)
- Refactoring code during a PURE MOVE story (US-005/006/007/008) — equivalence first, refactor never
- Importing `os` to read env vars in feature code
- Calling a Playbook a "Recipe" or a Mission a "Workflow"
- Adding `// @ts-ignore` or `# type: ignore` to make checks pass
- Adding emoji to source files unless asked
- Writing a `README.md` for a feature unless the story explicitly requires it (US-018 does; others do not)
- Touching files outside `orchestrator/`, `docs/PRDS/`, `scripts/ralph/`
- Running `git push origin main` — push to your branch only

## When in doubt

- Re-read the user story's `notes` field
- Read CLAUDE.md at the repo root (`/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-PRD-141/CLAUDE.md`)
- Read PRD-141 §12 Integration Coupling Rules — the boundary
- Search before you build
- Smaller diff > bigger diff
- If you would break PRD-007 or PRD-008-B proactive popups, STOP

Begin Phase 1.
