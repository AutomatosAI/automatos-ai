# PRD-190: Phase 2 · Wave 1 · P2-09 — Deliverables: stop the client-facing artifact rotting

**Phase:** Phase 2 — Module Deep-Review remediation · **Wave 1** (resurrect the dead client-facing loops)
**Branch:** `feat/p2-w1-deliverables` · **Worktree:** `automatos-ai-p2w1-deliverables`
**Dependencies:** **PRD-185** (Wave 0 — observability & feed loops, `649482aa3`) — esp. **S12** which already shipped the `/deliverable-freshness` endpoint this PRD's metric feeds. **Relates to PRD-167** (template block editor — the schema/renderer/registry machinery this PRD repairs the edges of).
**Build size:** S (four surgical fixes; all half-built or primitive-exists) · **Risk:** Low (no rebuilds, no new tables, no new renderer, no migration)
**Source:** `reports/dossiers/deliverables-documents.md` §C (C-2/C-3/C-4), §J (J1, J2, J3, J5, J7); report id **P2-09**

---

## Overview

The Phase-2 review's one-line finding is **"good bones, open loops."** The deliverables plane is the sharpest instance of it: the block-schema / brand-kit / variable-resolver / registry / flywheel **fusion** (PRD-63/PRD-167/PRD-129) is genuinely well-built code — and then the last inch of every client-facing loop is left un-wired. This wave wires those last inches. It is **the output plane** — literally what the client is handed — so it is where "line two" of the North Star lives.

Judged against the **North Star** — *does this make Auto more autonomously capable and the agents' output higher-quality for clients?* — P2-09 is a direct hit on both halves:

- **Higher-quality client output:** the durable Deliverable link currently **404s after one hour** (C-2/J1), and a Deliverable with visible red `[[company.address]]` markers can still be **handed to a client** because the "refuse to finalise" promise was never enforced (C-4/J5).
- **More autonomous Auto:** the `generate_document` ToolSpec on the **non-chat autonomy lane** (missions / board / scheduled) omits `template_id` (C-3/J2) — so an agent that follows the platform's own discovery flow (`platform_get_template_schema` hands it an id and says "use before generate_document") **cannot pass the id it was just given**. Id-driven templates work in chat and nowhere else. That is precisely the unattended path the North Star cares about.

Plus one metric: **clean-render rate is a tracked number nowhere** (J3/J7). The freshness half of that story already shipped in Wave 0 (PRD-185 S12 `/deliverable-freshness`); this wave adds the **quality** half — persist per-generation clean-render + template-lane on the Deliverable so "how many documents rendered clean this week" stops being a vibe.

**No moat framing; no new capability.** Every story here activates a primitive that already exists in the code and closes a loop that is already 80% built.

**PILOT lens (locked):** the dossier's real-data pass found the block pipeline has **almost no production evidence** — the recent Deliverables window is 100% chat-sourced PNG slides, no block-rendered PDF/DOCX. That is a **cold-start / unproven** state, **not a defect, and not in scope to "drive usage."** What *is* in scope: link-rot, `[[unresolved]]`-reaching-clients, and the missing `template_id` are **real client-facing bugs that fire the moment a real branded document is generated** — this wave fixes them and adds the tests that hold them fixed, so the loop is correct when real traffic arrives. See `feedback-pilot-usage-not-quality-signal`.

---

## Findings & Scope (all `file:line` confirmed by grep against `649482aa3`; the review tree drifted from the dossier's pin — numbers below are re-verified)

| Finding | Issue (verified in code) | Fix | Story |
|---|---|---|---|
| **J1 / C-2** (F030) | `_build_result` overwrites the stable app path with the **raw presigned S3 URL** as `download_url` (`generation_service.py:641-644`), and the presign is minted `ExpiresIn=3600` (`generation_service.py:704`). The persisted Deliverable's link therefore **404s after one hour**. It flows into tool results (`agent_platform_tools.py:837`) and into the Deliverable via `preview_url=result.download_url` (`generation_service.py:218`). A re-mint endpoint **already exists** (`api/document_generation.py:528-596`, `serve_generated_file`: local-then-S3-redirect) but records don't point at it. | Stop overwriting `download_url` with the presign — persist the stable `/api/documents/generated/{filename}` path and let the existing re-mint endpoint own presigning on demand. Delete the `s3_url` overwrite. | **S1** |
| **J2 / C-3** (F031) | The registry `generate_document` ToolSpec declares only `title/format/data/template_name` — **no `template_id`** (`tool_registry.py:1185-1218`) — yet the handler already parses it, validates the UUID, and threads it into generation + registration (`agent_platform_tools.py:726, 768-774, 787, 799`); `platform_get_template_schema` hands the agent an id and says "use before generate_document" (`actions_documents.py:250-262`). The chatbot's *inline* schema **does** declare `template_id` (`agent_platform_tools.py:227-229`), so it works in chat only. The non-chat lane (missions/board/scheduled) can't pass the id — the coordinator producer already routes `template_id` through when given one (`coordinator_service.py:917-936`). | Add the `template_id` `ToolParameter` to the registry ToolSpec (reuse the inline schema's wording). One parameter; the handler is already ready. | **S2** |
| **J5 / C-4** | The renderers *return* `unresolved` (known-but-empty) and the resolver returns `unknown` (not-in-catalog), but both PDF and DOCX call sites only `logger.warning` and proceed to build the file (`generation_service.py:176-180`, `:319-323`). So a client can still receive a document with visible red `[[company.address]]` markers. The honesty primitive exists; the gate on top of it was never wired. | Thread `unresolved`/`unknown` off the render result up to `generate()` (extend `GeneratedDocument`), and **block finalisation** when either is non-empty — raise a typed `UnresolvedDeliverableError` (loud, not silent) rather than returning a rotted document. | **S3** |
| **J3 / J7** | Clean-render rate is unmeasured (`RenderedHtml.unresolved` / `ResolvedVariables.unknown` exist in-process but are discarded after logging). Freshness (J3) is **already shipped** — PRD-185 S12 added `/deliverable-freshness` (`analytics_real.py:530-570`, own-workspace, honest empties). The remaining gap is the **quality** number: nothing records whether a generation resolved clean, or whether it used a block template vs the legacy Jinja fallback. | Persist `render.unresolved_count` / `render.unknown_count` / `render.template_lane` on the Deliverable `extra` JSONB at registration (no new table — `extra` already carries `template_id`); aggregate a **clean-render rate** in the existing `get_stats`. Reuse the existing freshness tile — **do not** add a parallel surface. | **S4** |

*(Dossier J4 = the `repeat`/`each` block primitive and J6 = Plate editor adoption are **out of this PRD's scope** — P2-09's headline is the four edge-fixes above. J4/J6 are separately-scoped Wave-1/Wave-2 items and are Gerard's to sequence; this PRD does not defer them silently, it states them as not-this-report. See Open Questions.)*

---

## Stories (test-first — write the failing test, make it green, refactor)

> Four stories, each independently shippable. **Files:** are the confirmed edit sites; **Test:** names the failing test to write first; tests are **pure** (mock S3 / storage at the boundary — no AWS, no DB round-trip to S3, run in CI with no external service).

### S1 · Kill the link-rot (F030) — the client-facing artifact stops rotting after an hour · _dossier J1 · P2-09_

**Files:** `orchestrator/modules/documents/generation_service.py:641-644` (the `download_url = s3_url` overwrite in `_build_result`); the existing re-mint endpoint `orchestrator/api/document_generation.py:528-596` (`serve_generated_file` — no change needed, it already serves local-then-S3-redirect; assert its contract in test).
**Test:** `test_deliverable_download_url_is_stable_app_path` — generate a document with S3 upload **mocked to succeed**, and assert `result.download_url == "/api/documents/generated/{filename}"` (today: it is the `ExpiresIn=3600` presign). `test_download_url_survives_presign_expiry` — assert the persisted `download_url` contains **no** `X-Amz-Expires` / signature query params (i.e. it is not a raw presign that can expire).
**Notes:** The fix is *deletion*, not addition — remove the `if s3_url: download_url = s3_url` overwrite so `_build_result` keeps the stable `/api/documents/generated/{filename}` it already computed on line 641. Keep the S3 **upload** (persistence across ephemeral containers) — only stop persisting the *expiring link*; the re-mint endpoint (`serve_generated_file`) already mints a fresh presign on each request. This is the dossier's "cheap and half-built" fix (persist the app-relative path; let the re-mint endpoint own presigning). No `os.getenv`, no config change — `S3_DOCUMENTS_BUCKET` / `AWS_*` already live in `config.py:822-839`.

### S2 · `template_id` on the autonomy lane (F031) — non-chat agents can use templates · _dossier J2 · P2-09_

**Files:** `orchestrator/modules/tools/registry/tool_registry.py:1185-1218` (add the `template_id` `ToolParameter` to the `generate_document` ToolSpec `parameters` list). No handler change — `agent_platform_tools.py:725-799` already parses `template_id`, validates the UUID (`:768-774`), and threads it into `generate(...)` (`:787`) and `register_as_deliverable(...)` (`:799`).
**Test:** `test_generate_document_toolspec_exposes_template_id` — resolve the `generate_document` ToolSpec from the registry and assert a `template_id` parameter is present (string/uuid, `required=False`) — today it is absent, so a non-chat agent can't pass it. `test_toolspec_matches_inline_chat_schema_for_template_id` — assert the registry ToolSpec and the chatbot inline schema (`agent_platform_tools.py:227-229`) now **agree** on `template_id`, closing the chat-only gap.
**Notes:** Reuse the inline schema's description verbatim ("UUID of a specific template to fill (from platform_list_templates). Takes precedence over template_name."). This is a **trivial** one-parameter add that unblocks id-driven template generation on the missions/board/scheduled lane — the exact autonomy path the North Star cares about — and it makes the platform's own discovery flow (`platform_get_template_schema` → "use before generate_document") actually followable. Do **not** build a second generation path; the single handler already does the work.

### S3 · Enforce the unresolved gate — clients stop receiving `[[unresolved]]` documents · _dossier J5 · P2-09_

**Files:** `orchestrator/modules/documents/models.py:9-18` (extend `GeneratedDocument` with `unresolved: List[str]` / `unknown: List[str]`, `field(default_factory=list)`); `orchestrator/modules/documents/generation_service.py:169-181` (PDF `_render_block_html`) + `:307-325` (DOCX block path) — capture `rendered.unresolved` / `resolved.unknown` instead of discarding after log; `generation_service.py:106-159` (`generate()` — the finalisation gate). Add a typed `UnresolvedDeliverableError`.
**Test:** `test_unresolved_variable_blocks_finalisation` — generate against a block template whose `{{company.address}}` resolves empty and assert the generation **raises `UnresolvedDeliverableError`** (with the offending paths) rather than returning a `GeneratedDocument` — i.e. the document is **BLOCKED, not delivered**. `test_unknown_variable_blocks_finalisation` — an authoring-error `{{not_a_real.path}}` (unknown) is likewise blocked. `test_clean_render_passes` — a fully-resolved template returns a `GeneratedDocument` normally (no false positives).
**Notes:** The primitives already exist and are honest — `RenderedHtml.unresolved` (`blocks/html_renderer.py:36-38`), `RenderedDocx.unresolved` (`blocks/docx_renderer.py:45-47`), `ResolvedVariables.unresolved`/`unknown` (`variables/resolver.py:35-36`). The bug is that both call sites `logger.warning` and proceed (`generation_service.py:176-180`, `:319-323`). Replace the warning-and-proceed with **capture-and-gate**: thread the lists onto `GeneratedDocument`, and in `generate()` raise if non-empty. **No silent `except`**, no "render anyway with a flag" — a client-facing artifact with `[[…]]` markers must not be finalised. (The visible-red-marker path stays for *preview*; the gate is on *finalise*.) Fail loud, per house rules.

### S4 · Clean-render + template-lane metric — document quality becomes a tracked number · _dossier J3/J7 · P2-09_

**Files:** `orchestrator/modules/documents/generation_service.py:183-223` (`register_as_deliverable` — add render counts to the `extra` dict it already builds for `template_id`); `orchestrator/services/deliverable_service.py:156-303` (`register` already accepts + persists `extra` JSONB at `:172, :274`; `:759` passes it through) and `:510-528` (`get_stats` — add a clean-render aggregate). **No new endpoint** — the freshness tile already exists (`api/analytics_real.py:530-570`, PRD-185 S12).
**Test:** `test_clean_render_recorded_on_extra` — a clean generation registers with `extra["render"]["unresolved_count"] == 0` and `extra["render"]["template_lane"] in {"block","legacy"}`. `test_get_stats_reports_clean_render_rate` — with a mix of clean and (pre-gate, legacy-lane) generations recorded, `get_stats()` returns a `clean_render_rate` = clean ÷ total. Both **mock the DB view** at the boundary (pure).
**Notes:** This is the dossier's G-1 ("the single highest-value metric and nearly free — the data already exists in-process, just persist it on the Deliverable `extra` and aggregate") and J7. Ride on the existing `extra` JSONB — **no new table, no new column, no new tile** (§4/§5: reuse over build). Since S3 now blocks any *un*clean generation at finalise, the `unresolved_count` recorded here will be `0` for all *newly-finalised block* documents — so the metric's live value is (a) `template_lane` coverage (block vs the legacy Jinja fallback — tracks the PRD-167 migration off Jinja) and (b) a durable audit that clean-render is enforced. Surface it through the **existing** freshness/stats read-model that Wave 0 already exposes to workspace admins; do not add a parallel Command Center surface.

---

## Sequencing (mostly parallel-safe)

- **S3 → S4** is the only soft ordering worth noting: S4's `template_lane`/`unresolved_count` fields ride on the same `extra` dict and are cleanest to land once S3 has threaded the render result onto `GeneratedDocument`. They can still be built in parallel if `extra` field ownership is coordinated.
- **S1, S2 are fully independent** — disjoint files (`generation_service._build_result` / `tool_registry` ToolSpec), land in any order or parallel worktrees.
- The only shared file across stories is `generation_service.py` (S1 edits `_build_result`; S3 edits the render helpers + `generate()`; S4 edits `register_as_deliverable`) — these are **disjoint functions** in the same module; coordinate at the diff level, no logical conflict.
- No `config.py` change in any story (S3/AWS flags already exist) — nothing to coordinate there.

---

## Verification (CI is the only gate — no local runs)

Per current project convention (`feedback-no-local-servers`, tightened 2026-07-03): **do not run servers, builds, `next dev`, headless Chromium, `pytest`, `tsc`, or installs on the dev machine.** Write the code + **pure** tests and let **CI (the PR checks) verify.** Every new test must run with **no external service** — mock S3 / boto3 at the boundary (S1), resolve ToolSpecs from the in-process registry (S2), drive the resolver/renderer with in-memory block trees (S3), and mock the `v_workspace_outputs` read-model (S4). The **unresolved-gate test (S3) asserts a Deliverable with `[[unresolved]]` variables is BLOCKED — raises, not returns** — this is the load-bearing assertion of the wave and must be pure and deterministic. Commit, push, open a PR against `main`; **CI is the gate.**

---

## Conventions (non-negotiable — see `automatos-ai/CLAUDE.md`)

- No `os.getenv()` outside `config.py` — this wave adds **zero** new flags (`AWS_*` / `S3_DOCUMENTS_BUCKET` already at `config.py:822-839`).
- **No backward-compat shims — delete what you replace.** S1 is a *deletion* (the `download_url = s3_url` overwrite goes, no `_legacy` fallback kept). No parallel renderer, no second generation path — S2 extends the **one** existing handler; the `template_id`-on-the-autonomy-lane fix **reuses the existing template path** (dossier + house-rule requirement).
- **No new table where an existing one fits, no new tool where an existing one extends** — S4 rides on the `deliverables.extra` JSONB and the existing `get_stats`; S2 extends the existing `generate_document` ToolSpec (the 3-file platform-tool registration pattern), it does not register a new tool.
- Immutable patterns; small focused functions; comprehensive error handling; **no silent `except`** — S3 exists *because* of a warn-and-proceed; its fix must raise loud (`UnresolvedDeliverableError`), never swallow.
- Reuse the Wave-0 surface — the deliverable-freshness tile (`analytics_real.py:530-570`) is **done**; S4 feeds it, it does not rebuild it.
- Canonical vocab: **Deliverable** (the client-output word — used consistently; never Output/Artifact in user-facing copy), **Playbook** (not Recipe), **Knowledge Graph**, **Command Center**, **Auto**.
- Branch `feat/p2-w1-deliverables`; worktree `automatos-ai-p2w1-deliverables`; commit, push, open a PR; CI is the gate.

## Success metrics (the definition of "the artifact stops rotting")

- **The persisted Deliverable `download_url` is a stable app path** (`/api/documents/generated/{filename}`) that re-mints on demand — it **does not 404 after an hour** (S1). Dossier metric G-2 (durable-link liveness) moves from ~0%-after-1h toward ~100%.
- **A non-chat agent (mission/board/scheduled) can pass `template_id`** to `generate_document` and get id-driven template generation — parity with chat (S2).
- **A Deliverable with any `unresolved` or `unknown` variable is blocked at finalise** — a client can no longer receive a `[[company.address]]`-marked document (S3).
- **Clean-render rate + template-lane coverage are tracked numbers** on the Deliverable `extra`, aggregated in `get_stats` and readable through the existing workspace-admin surface (S4). Dossier metric G-1 moves from "unknown (not recorded)" to recorded.

## What this wave gates / relates to

- **Feeds PRD-185 S12's freshness/health surface** with the quality half of the deliverables story (clean-render), completing the "is Auto producing, and is it clean?" pair for the output plane.
- **Unblocks id-driven template use on the autonomy lane** — a precondition for any later work that has missions/playbooks emit branded Deliverables from named templates.
- **Does not gate, and is not gated by, the block `repeat`/`each` primitive (J4) or Plate editor adoption (J6)** — those are the deliverables plane's larger Wave-1/Wave-2 investments and are sequenced separately (Open Questions).

---

## Open Questions (Gerard's call — surfaced, not deferred · §12)

1. **J4 (loop/`repeat` block primitive) + J6 (Plate editor):** the dossier's two **larger** deliverables upgrades — the `repeat`/`each` block that closes the invoice/line-item gap (C-1) and migrates the 5 Jinja seeds onto blocks, and adopting Plate for the authoring surface (C-7). Both are **out of P2-09's four-fix scope** by design (this report is the edge-fixes). They are real Wave-1/Wave-2 work — **Gerard's to schedule as their own PRD(s)**, not this one's to absorb or silently drop. Flagged here so the cut is visible.
2. **S3 gate strictness — hard-block vs. allow-with-explicit-override:** S3 blocks finalisation on any `unresolved`/`unknown`. If a workspace legitimately wants a document with a *known-blank* optional field (e.g. no `company.address` on file yet), the strict gate would refuse it. Options: (a) **hard-block always** (simplest, safest — proposed default); (b) allow an explicit `allow_unresolved=True` caller opt-in that still records the count in S4's metric. Proposed: ship (a); add (b) only if a real workspace hits the wall. **Gerard's call.**
3. **S4 `unresolved_count` after S3:** because S3 blocks unclean *block-lane* finalisation, `unresolved_count` will be `0` on all newly-finalised block documents — so the metric's live signal is really *template-lane coverage* + a clean-render audit, not a varying unresolved rate. If a varying rate is wanted (e.g. to watch the legacy Jinja lane, which S3's block gate doesn't cover), that's a small extension. Noted so the metric's meaning isn't over-sold.

---

*Traceability: every story cites its dossier ref in `reports/dossiers/deliverables-documents.md` (§C defects C-2/C-3/C-4, §J upgrades J1/J2/J3/J5/J7) and report id **P2-09**. `file:line` refs were re-verified by grep against `origin/main` @ `649482aa3` — the dossier's pin (`77bc9c6d5`) had drifted; the numbers in the Findings table above are the confirmed current lines. North-Star framed (durable + clean client output; id-driven templates on the autonomy lane); PILOT lens applied (no-prod-evidence is cold-start, not a defect; link-rot / `[[unresolved]]` / missing `template_id` are real client-facing bugs); no moat framing.*
