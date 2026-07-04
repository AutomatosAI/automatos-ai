# Deliverables, documents & templates — module dossier

**Key:** `deliverables-documents` · **Tier:** standard · **Status:** live
**Pinned tree:** `p2-src` = origin/main @ `77bc9c6d5` (2026-07-03). All `file:line` refer to that tree.
**North Star:** does this make Auto + the agents more autonomous and their *client-facing output* higher-quality? This module is where "line two" of the North Star lives — it is literally what the client sees.
**Scope note:** Section F (enterprise bar) is written informationally; the adversarial-input / tenant-isolation / defensive-hardening lens is deliberately **excluded** (separate Opus pass).

---

## A. What it is

The client-facing **output plane**: the code that turns agent/mission/chat work into downloadable, branded artifacts. It has four folded parts:

1. **Document generation** (PRD-63/PRD-167) — an editor-independent **block schema** (`modules/documents/blocks/schema.py`) rendered to **PDF** (WeasyPrint), **DOCX** (python-docx, compiled from the same block tree) and **XLSX** (XlsxWriter), with a **workspace brand kit** (`brand_kit.py`) and a **strict variable resolver** (`variables/resolver.py`) that fills `{{user/company/brand/date/data.*}}` chips. A legacy Jinja-HTML / uploaded-`.docx` path still runs behind it, plus Gotenberg/LibreOffice DOCX→PDF conversion (`conversion_service.py`).
2. **Deliverables registry** (PRD-129/133b) — a single `deliverables` grid unified over `blog_posts` + `agent_reports` + ad-hoc artifacts via the `v_workspace_outputs` DB **view** (`services/deliverable_service.py`), surfaced at `/api/deliverables` and the `/deliverables` Gallery (three tabs: Outputs / Blogs / Templates).
3. **Agent reports** (PRD-76) — an agent-authored report lifecycle with 1–5 **grading**, execution metrics, and knowledge-flywheel re-ingest (`services/report_service.py`).
4. **Attachments** (PRD-127) — the *inbound* leg: ephemeral uploads → LLM content parts (image_url / extracted text) through the single `AttachmentResolver` (`modules/attachments/resolver.py`).

The agent-facing surface is three tools — `platform_list_templates`, `platform_get_template_schema`, `generate_document` (`modules/tools/discovery/actions_documents.py`, `modules/agents/services/agent_platform_tools.py:721`) — and a dependency-free block **Template Studio** in the frontend (`frontend/components/documents/blocks/`).

---

## B. What it does — real implementation + data path

### B.1 Generation dispatch (the one entry point)
`DocumentGenerationService.generate()` (`generation_service.py:106-159`) resolves a template (by id or name; defaults to a "Basic Report" for PDF), injects `title` into `data`, normalises LLM-shaped section keys (`heading/body`→`title/content`, `generation_service.py:433-467`), and dispatches to `generate_pdf` / `generate_docx` / `generate_xlsx`.

**PDF has three ordered render paths** (`generation_service.py:229-287`):
- **Path 1 — block template** (`template.blocks` present): `validate_blocks` → `render_document_html` (brand-aware, variable chips resolved) → WeasyPrint. The canonical path.
- **Path 2 — legacy `template_content`** (user-authored Jinja HTML): rendered through a **`SandboxedEnvironment`** (`generation_service.py:100`, PRD-156 S4 anti-SSTI) with the brand kit exposed as `{{brand.*}}`; charts embedded as base64.
- **Path 3 — no template**: `blocks_from_legacy(data)` maps the legacy `title/sections/metrics/highlights/recommendations` shape into blocks and renders through the brand path (`legacy_mapper.py:43-91`). This replaced the old hardcoded `#ff6b35`-orange fallback.

**DOCX** (`generation_service.py:293-361`): block templates compile **directly** to a `python-docx` Document from the same tree (`docx_renderer.py:204`, Q71 — no uploaded `.docx` needed); legacy uploaded `.docx` still renders via `docxtpl`.

**XLSX** (`generation_service.py:367-427`): straightforward `columns`/`rows` → XlsxWriter with header formatting and auto-fit. Note the header fill is a **hardcoded `#1a1a2e`** (`generation_service.py:397`) — XLSX does *not* consume the brand kit (only PDF/DOCX do).

### B.2 The block schema (the storage + render contract)
`schema.py` is a Pydantic tree with `extra="forbid"` on every model (`schema.py:33-35`) — a malformed block raises a field-level error, no silent coercion. Block types: `heading`, `text`, `table`, `image` (incl. `source="brand_logo"`), `variable`, `page_break`, `section` (nestable). Inline content is a discriminated union of `TextRun` (with marks) and `VariableRun` (`schema.py:65`). `SCHEMA_VERSION=1` is stored alongside blocks for future migration (`schema.py:27`).

### B.3 Variable resolution (honest-by-design)
`VariableResolver.resolve()` (`variables/resolver.py:150-160`) fetches the requesting user, the workspace `BusinessProfile`, and the brand kit, builds a pure context (`build_context`, `resolver.py:54-117`), and resolves paths. The catalog is **static, 23 paths** across `user/company/brand/date` (`variables/catalog.py:25-53`) plus a dynamic `data.*` namespace filled per-generation. **Policy** (`resolver.py:120-140`): a *known* path that resolves empty → `unresolved`; an *unknown* path → `unknown` (authoring error). The HTML renderer emits an unresolved chip as a **visible red `[[path]]` marker**, never a blank (`html_renderer.py:45-51`), and returns the `unresolved` list so a caller *could* refuse to finalise. (In practice the callers only **log** the unresolved list — `generation_service.py:176-180, 319-323` — they do not block the render; see §C.)

### B.4 Brand kit
Stored on `workspace.settings['brand_kit']` (no new table, `brand_kit.py:1-11`). `BrandKit` Pydantic model, hex validation on write, **lenient on read** (bad kit → neutral defaults, never crashes a render, `brand_kit.py:69-81`). Defaults are a neutral navy palette, explicitly *not* Automatos orange.

### B.5 Deliverables registry (the unification)
`DeliverableService.register()` (`deliverable_service.py:156-303`) is an idempotent upsert (`ON CONFLICT (workspace_id, file_path) WHERE deleted_at IS NULL`) that **refuses** `blog_post`/`report` artifact types (`deliverable_service.py:195-203`) — those are owned by BlogService/ReportService and surfaced through the `v_workspace_outputs` UNION view, killing the PRD-129 double-write. `list_deliverables`, `get_stats`, `soft_delete` (routes the UPDATE to the correct source table via a fixed allow-list, `deliverable_service.py:583-618`), and `apply_retention` (keep-N-per-agent for heartbeat noise) all read the view. All SQL is `text()` with bound params.

Generated documents are wired into the registry via `register_as_deliverable()` (`generation_service.py:183-223`) called from **two** producers: the agent tool path (`agent_platform_tools.py:793`) and the mission-synthesis path (`coordinator_service.py:930`). Each also fires **knowledge-flywheel ingest** so the document's markdown becomes retrievable knowledge (`agent_platform_tools.py:806-822`, `report_service.py:297-317`) — this is the F089 line (documents feeding the flywheel).

### B.6 Attachments (inbound)
`AttachmentResolver.resolve()` (`resolver.py:65-145`) is the single choke point that turns `attachment_ids` into LLM parts: vision-capability gate (`resolver.py:147-221`), ≤20 images (`resolver.py:32`), inline-base64 <500 KB else signed URL (`resolver.py:223-244`), a text budget for extracted documents (`resolver.py:246-276`), CI-enforced as the only site constructing `image_url` parts (`resolver.py:8`).

### B.7 Real data path — what actually got produced
Live prod Postgres (read-only, 2026-07-04 — `evidence/data/deliverables.md`, `evidence/data/census.md`):
- `deliverables` = **2,242 rows**, but **newest is 2026-06-16** and the recent window is **100% `chat`-sourced PNG slides** (the social-content engine's weekly/daily carousels). **No `document`/`report`/`blog` artifact appears in the recent window.**
- `agent_reports` = **3,845 rows**, current (newest 2026-07-03) — the report leg is alive; the block-document leg is not visibly exercised in prod data.
- Client-facing output **stopped 2026-06-16**, coinciding with daily content playbooks failing on **OpenRouter 402** while their board tasks still closed `done` (`evidence/real-data-inventory.md §3`). Net: **~2.5 weeks of zero client-facing artifacts, and nothing surfaced it.**

**The load-bearing observation:** the whole PRD-167 block/brand/variable stack is well-built code with **almost no production evidence that a real branded client document was ever generated through it** — the artifacts that exist are chat PNGs and agent-report markdown, not the block-rendered PDFs/DOCX this module's investment went into.

---

## C. Honest quality — how good is it *really*?

**Maturity: 3 / 5** (solid, honest architecture; unproven in production; three named defects still open; a real capability gap for the most common client document shape).

### What is genuinely good
- **Editor-independence is real and correct** (`schema.py:1-17`): one block tree → PDF + DOCX from the *same* source (`docx_renderer.py`, `html_renderer.py`). Swapping the editor doesn't touch the renderers. This is the right architecture and better than most in-house doc stacks.
- **The "no silent blank" variable policy** (`html_renderer.py:45-51`, `resolver.py:120-140`) is exactly the honesty discipline the July review kept asking for — an unresolved chip is *visible*, and the unresolved list is returned to the caller.
- **`extra="forbid"` everywhere** (`schema.py:35`) means malformed template bodies fail loud, not silently.
- **Brand kit removed the hardcoded orange** from every render path (`generation_service.py:85-88`, `brand_kit.py:27-33`) — a real closure of the July F031-family branding complaint.
- **Deliverables de-duplication is a clean consolidation** — the `v_workspace_outputs` view + `register()`-refuses-report/blog design (`deliverable_service.py:195-203`) genuinely killed the double-write drift that orphaned 29 reports (`report_service.py:287-290`).
- **SSRF discipline in both renderers** — WeasyPrint URL fetcher blocks non-public hosts and `file://` (`generation_service.py:31-56`); the DOCX image fetch refuses private hosts *and redirects* (`docx_renderer.py:88-120`). (Not scored here — hardening pass — but noted as evidence of care.)

### Concrete defects, with evidence

**C-1 — blocks v1 has no array/loop primitive; the most common client documents stay on Jinja. (design gap, CONFIRMED)**
The block schema has no repeat/iteration construct — `TableBlock.rows` is a *static* list (`schema.py:86-93`); there is no "for each line-item" block. The mapper's own docstring admits it: *"blocks v1 models static + scalar-variable content. The array-driven seeds (Invoice line-items, multi-section reports) are generated from their per-call data … templates a user authors in the editor use variable chips instead"* (`legacy_mapper.py:9-12`, `block_starters.py:9-11`). Consequence: the **five legacy starter templates** (Basic Report, Invoice, Executive Summary, Meeting Notes, Data Export — `seed_templates.py:18-206`) all carry `template_content`/`data_schema` and render on the **legacy Jinja path**, not blocks. Only two block-native starters exist (Branded Letter, Branded Report — `block_starters.py:71-100`), and neither iterates. **An invoice with N line items — the single most common branded business document — cannot be authored in the block editor.** This is the biggest North-Star gap: a client-quality invoice/lineitem report either falls back to hand-authored Jinja (which non-technical users can't touch) or to the LLM emitting a flat `data` blob rendered by `blocks_from_legacy`.

**C-2 — F030 NOT DONE: the durable deliverable link dies after one hour. (CONFIRMED, verbatim from residual map)**
`_build_result` overwrites the stable app path with the **raw presigned S3 URL** as `download_url` when upload succeeds (`generation_service.py:642-645`), and the presign is minted with **`ExpiresIn=3600`** (`generation_service.py:697-705`). The persisted deliverable's link therefore **404s after an hour**. A re-minting endpoint exists (`api/document_generation.py:559-596` redirects to a fresh presign) but deliverable records don't reference it. For a client-facing artifact trail this is the most damaging bug in the module — the thing the client is handed **rots**, and it flows into tool results and the widget via `result_formatter`. `git log` since 2026-07-01 on these files: zero commits (`phase0-residual-map.md:475-478`). The fix is cheap and half-built (persist the `/api/documents`-relative path; let the re-mint endpoint own presigning).

**C-3 — F031 NOT DONE on the registry lane: `generate_document` ToolSpec omits `template_id`. (CONFIRMED)**
The ActionRegistry `generate_document` ToolSpec declares only `title/format/data/template_name` — **no `template_id`** (`modules/tools/registry/tool_registry.py:1185-1218`) — yet the handler parses it (`agent_platform_tools.py:726, 767-787`) and `platform_get_template_schema` hands the agent an id and says "use before generate_document" (`actions_documents.py:246-263`). An agent that follows the discovery flow on the registry/`tool_router` lane cannot pass the id it was just given. **Nuance (verified):** the chatbot's separate *inline* schema **does** declare `template_id` (`agent_platform_tools.py:209-238`), so id-driven generation works in chat; the gap is specifically the registry/tool_router lane used by non-chat agents (missions/board/scheduled) — exactly the autonomy path the North Star cares about (`phase0-residual-map.md:541-544`).

**C-4 — the "refuse to finalise on unresolved" promise is not enforced.**
The renderer *returns* `unresolved`, but both call sites only `logger.warning` (`generation_service.py:176-180, 319-323`) and proceed to build the PDF/DOCX. So a client can still receive a document with visible red `[[company.address]]` markers in it. The honesty primitive exists; the gate on top of it was never wired. (Contrast the docstring's stated intent, `schema.py`/`html_renderer.py:13-16`.)

**C-5 — three parallel "data → markdown/blocks" mappers with drift risk.**
`_data_to_markdown` (`generation_service.py:516-623`, ~110 lines, for the widget/`result.content`), `blocks_from_legacy` (`legacy_mapper.py`, for no-template PDF), and the Jinja seed templates each interpret the same loose `data` shape differently. `_normalize_sections` (`generation_service.py:433-467`) is a pile of alias-guessing (`heading/body/header/text/name/description/…`) that exists precisely because the input contract is unspecified. This is fragile and a quality lottery depending on which shape the LLM emits.

**C-6 — XLSX ignores the brand kit; header colour hardcoded.**
`generate_xlsx` uses a fixed `#1a1a2e` header fill (`generation_service.py:397`) and never reads the brand kit — a branded data export won't match the branded PDF/DOCX. Minor, but it's a visible inconsistency in "what the client sees."

**C-7 — the frontend editor is a hand-rolled stopgap.**
`BlockEditor.tsx` is a dependency-free custom editor (lucide + shadcn primitives, manual up/down/indent, `frontend/components/documents/blocks/BlockEditor.tsx`) — Plate/BlockNote were deferred behind the schema seam (per PRD-167 memo; BlockNote failed the MIT gate). It works and it's honest, but it is a basic form-style block list, not a Notion-class authoring surface: no drag-drop reorder, no slash menu, no rich inline table editing. For a non-technical client building their own branded template, this is the weakest link in the experience.

### Verdict on quality
The **architecture** is a 4 — genuinely well-factored, honest, secure-minded. The **delivered capability** is a 3 because (a) the most common client document (line-item invoice/report) can't be authored in the new system, (b) the durable link rots (F030), (c) there is essentially **no production evidence** the block pipeline ever produced a real client document, and (d) two of three honesty affordances (unresolved-gate, template_id on the autonomy lane) aren't wired end-to-end. It is not *bad*; it is *unproven and incomplete for its stated job*.

---

## D. Competitive teardown

The right comparison set is **template-driven document-generation engines** (not "document AI" broadly). Automatos sits in an unusual spot: it built an in-house block schema + renderers rather than adopting one.

### D-1 — Carbone (open-source report/document generator) — beats Automatos on the exact gap
Carbone injects JSON into a template authored in **Word/Excel/LibreOffice/Google Docs** and outputs PDF/DOCX/XLSX/PPTX/ODT ([carbone.io](https://carbone.io/)). Crucially it has **native repetitions, nested loops, array sorting/filtering and aggregate functions** — the `{d.users[i]}…{d.users[i+1]}` loop pattern ([carbone.io/documentation/design/repetitions/with-arrays.html](https://carbone.io/documentation/design/repetitions/with-arrays.html), [dev.to cheat sheet](https://dev.to/carbone/cheat-sheet-for-carbone-2ikd)). That is **exactly C-1** — invoices, line-item reports, multi-section documents are Carbone's core use case, authored by a non-technical user in Word, no code. It's OSS/free self-hosted, with cloud tiers ($29–$595/mo) and self-hosted licences ($1,500–$2,940/yr) ([carbone.io/pricing.html](https://carbone.io/pricing.html)). **Where Automatos stands:** behind on templating power (no loops), but ahead on *agent integration* — Carbone is a rendering engine with no agent/tool story; Automatos already has `generate_document` wired into the agent loop and the brand kit + variable catalog as first-class concepts.

### D-2 — BlockNote (block editor, MPL-2.0) — beats Automatos on the *authoring surface*
BlockNote is a Notion-style block editor for React (on ProseMirror/Tiptap) with drag-drop blocks, nesting, slash menus, real-time collaboration (Yjs), and **built-in PDF / Word / ODT export** ([blocknotejs.org](https://www.blocknotejs.org/), [github.com/TypeCellOS/BlockNote](https://github.com/TypeCellOS/BlockNote)). Core is MPL-2.0 (usable closed-source); only the "XL" export/AI packages are GPL-3.0 — which is why it failed Automatos's MIT gate. **Where Automatos stands:** its hand-rolled `BlockEditor` (C-7) is a small fraction of BlockNote's UX; but BlockNote's export is a client-side convenience, whereas Automatos needs *server-side, brand-injected, variable-resolved, agent-invoked* rendering, which BlockNote doesn't provide.

### D-3 — Plate (Slate-based editor framework, MIT) — the editor Automatos actually planned for
Plate is a plugin rich-text framework on Slate + shadcn/ui with 50+ plugins, MCP/AI integration, collaboration, and an **MIT** core ([platejs.org](https://platejs.org/), [github.com/udecode/plate](https://github.com/udecode/plate)). It's the licence-clean choice PRD-167 deferred to behind the schema seam. **Where Automatos stands:** the seam is deliberately Plate-shaped; adopting Plate for the editor is a known, low-risk upgrade that leaves the schema/renderers intact.

### D-4 — Nutrient DWS / Document Engine (agent-native document infra) — beats Automatos on the *agent* story it thinks it owns
Nutrient (ex-PSPDFKit) ships an **MCP server** and a **universal agent skill** (Claude Code / Codex / Cursor / 35+ agents) that expose **DOCX-template-fill from a JSON model**, PDF↔DOCX conversion, OCR, redaction, digital signing, and merge — as agent tools ([nutrient.io/ai/infrastructure](https://www.nutrient.io/ai/infrastructure/), [github.com/PSPDFKit-labs/nutrient-agent-skill](https://github.com/PSPDFKit-labs/nutrient-agent-skill), [nutrient.io/api/processor-api](https://www.nutrient.io/api/processor-api/)). This is a superset of Automatos's `generate_document` for the *fill-a-template* case, plus OCR/redaction/signing Automatos entirely lacks. **Where Automatos stands:** behind on document-*processing* breadth (no OCR/redaction/sign), roughly level on template-fill, ahead only in that its generation is natively fused with its own brand kit + memory/flywheel rather than being an external API call.

### D-5 — python-docx-template / docxtpl (the library Automatos already uses on the legacy lane)
docxtpl **does** support loops over table rows (`{%tr for … %}`) and conditionals inside a Word template ([docxtpl.readthedocs.io](https://docxtpl.readthedocs.io/), [deepwiki](https://deepwiki.com/elapouya/python-docx-template/3.2-template-processing)). Notably, **Automatos's own legacy path already has loop capability via docxtpl** — the loop gap (C-1) is only in the *new block* system, which regressed on that dimension relative to the legacy engine it's meant to replace. That is worth stating plainly: the block rewrite is more honest and brand-aware but **less capable on iteration** than the docxtpl path it supersedes.

**Net competitive read:** Automatos's differentiation is *not* the renderer (Carbone/Nutrient are stronger) or the editor (BlockNote/Plate are stronger). It is the **fusion**: brand kit + variable catalog + agent tool + deliverables registry + flywheel re-ingest, all inside the platform, so Auto can generate a branded document unattended and it shows up in the client's gallery and becomes retrievable knowledge. That fusion is real and valuable — but it currently sits on a renderer that can't do line items and a link that expires in an hour.

---

## E. Build / extend / adopt / replace — verdict

**Verdict: EXTEND the schema/registry/agent fusion (keep it) + ADOPT Carbone's templating model for iteration + ADOPT Plate for the editor. Do NOT rewrite; do NOT replace the whole plane.**

Rationale against the §2 reuse bias:
- **Keep (the fusion is the value):** the block schema as the storage contract, the brand kit, the variable catalog + honest-unresolved policy, the `v_workspace_outputs` registry, `register_as_deliverable` + flywheel ingest, and the three agent tools. Nothing external gives you *brand-injected, variable-resolved, agent-invoked, gallery-registered, flywheel-ingested* generation as one path. This is earned in-house code.
- **The iteration gap (C-1) does not justify a rewrite** — it justifies **one new block primitive** (a `repeat`/`each` block binding a `data.*` array to a child sub-tree, mirroring Carbone's `[i]` loop semantics). That closes invoices/line-item reports and lets the five legacy Jinja seeds migrate onto blocks, retiring the legacy `template_content` lane per the PRD sunset note. **If the owner wants it faster than building the primitive:** wrap **Carbone (OSS, self-hosted, ~$0 or ≤$2,940/yr)** as an alternate renderer keyed off a Word/ODT template, and keep the block path for chip-driven simple docs — but that fragments the "one schema" story, so building the loop primitive is the cleaner call unless timeline forces the adopt.
- **The editor (C-7): ADOPT Plate** (MIT, seam already Plate-shaped — [platejs.org](https://platejs.org/)). This is the lowest-risk, highest-visible-quality upgrade in the module. Do not keep hand-rolling `BlockEditor`.
- **Document *processing* (OCR/redaction/sign): out of scope today, but if a vertical needs it, ADOPT Nutrient DWS** as an MCP/agent-skill rather than building it ([nutrient.io/ai/infrastructure](https://www.nutrient.io/ai/infrastructure/)). Flag, don't build.
- **Do NOT replace WeasyPrint yet** (see F) — it's correct for A4 branded docs at pilot volume; revisit only if generation goes high-throughput.

Rough effort: loop primitive + migrate 5 seeds ≈ medium; Plate adoption ≈ medium; F030 fix ≈ small; F031 fix ≈ trivial.

---

## F. Enterprise bar

*(Informational; adversarial-input / tenant-isolation / PII deferred to the dedicated Opus hardening pass.)*

- **Scale / latency — the WeasyPrint ceiling is real.** WeasyPrint "does not scale well — ok at one page, go get a coffee at 500" ([weasyprint.com comparison](https://weasyprint.com/2026/05/02/weasyprint-vs-others-which-pdf-generator-should-you-choose/), [speedata benchmark](https://news.speedata.de/2026/02/10/typesetting-benchmark/)). For pilot-scale branded 1–5 page docs it's fine; for high-volume or long documents (100s of pages) it becomes a latency and CPU problem. Generation is **synchronous inside the agent tool call** (`agent_platform_tools.py:781`) and PDF rendering is blocking (WeasyPrint isn't async; the `async def` doesn't offload it) — a slow render stalls the tool loop. If throughput ever matters, **Typst** (200–500 ms typical, serverless-friendly — [typst.app/blog](https://typst.app/blog/2025/automated-generation/)) or Gotenberg-as-a-service is the escape hatch.
- **Reliability / durability — currently failing (F030).** The durable artifact is a 1-hour URL; the "source of truth is S3" comment (`generation_service.py:6-10`) is undermined by persisting an expiring link rather than the object key. `agent_reports` content lives as workspace files read on demand (`report_service.py:477-485`) — reliable, but a `WorkspaceClient` failure yields `content: null` with an error field (honest, at least). Ephemeral-container design is handled for reports (DB row + workspace file) but **broken for generated documents** (expiring link).
- **Availability / degradation:** Gotenberg→LibreOffice→raise fallback for DOCX/XLSX→PDF is a sensible degradation ladder (`conversion_service.py:43-64`); optional deps (weasyprint/docxtpl/xlsxwriter/boto3) fail with actionable install messages, not opaque 500s. Brand-kit read is fail-soft (`brand_kit.py:79-81`).
- **Cost-to-operate at load:** dominated by (a) WeasyPrint CPU for PDF and (b) the LLM tokens the *agent* spends assembling `data` (the render itself is deterministic, near-free). No queueing/rate-limiting on generation — a mission fan-out that emits many documents renders them serially in-process.
- **Observability:** generation logs data keys and unresolved vars (`generation_service.py:145, 176`); there is **no metric** for generation success/failure, render latency, or unresolved-rate — you cannot currently answer "how many documents rendered clean this week." (Feeds §G / T3.)

---

## G. Quality metric — how to measure and track

Today: **effectively unmeasured.** The only quality signal in the whole plane is `agent_reports.grade` (1–5 human grade, `report_service.py:489-526`) — and grades weren't inspectable as populated in the real-data pass. Generated *documents* have **no** quality metric.

Proposed tracked numbers (feed T3 harness), cheapest first:
1. **Clean-render rate** = generations with `unresolved == [] and unknown == []` ÷ total. The data already exists in-process (`RenderedHtml.unresolved`, `resolved.unknown`) — just persist it on the deliverable `extra` and aggregate. **Today: unknown (not recorded).** This is the single highest-value metric and is nearly free.
2. **Durable-link liveness** = fraction of deliverable `download_url`s that resolve 200 (directly measures F030). **Today: ~0% after 1h for S3-backed docs** by construction.
3. **Template-lane coverage** = share of generations using a block template vs the legacy fallback (`blocks_from_legacy`). Tracks the migration off Jinja. **Today: heavily legacy** (5 of 7 seeds are Jinja; real data shows no block docs at all).
4. **Report grade distribution** = mean/median `agent_reports.grade` and % graded (`report_service.get_stats`, `:528-577`). Already computed — surface it. **Today: computable but unsurfaced; population unverified.**
5. **Deliverable freshness** = age of newest client-facing artifact per workspace — would have **caught the 2026-06-16 outage on day one** (`evidence/data/deliverables.md`). **Today: 18 days stale, unmonitored.**

An LLM-judge rubric (brand adherence, completeness vs requested sections, no unresolved markers) calibrated on a small human-graded set would cover the fuzzy "is this client-quality" question — but #1 and #5 are the ones to ship first because they're free and would have caught the two worst real failures.

---

## H. Cost note

*(Informational.)*
- **Render itself:** deterministic, no LLM tokens. PDF = WeasyPrint CPU (seconds for long docs — the real cost); DOCX/XLSX = cheap pure-Python; S3 PUT + presign = negligible.
- **Agent-side:** the tokens the LLM spends composing the `data` payload (sections/metrics/etc.) — the dominant variable cost, borne in `auto-core`/mission synthesis, not here. `generate_document` is one tool call; a template-schema-aware flow adds `platform_list_templates` + `platform_get_template_schema` round-trips (2 extra cheap tool calls).
- **Flywheel tax:** every generated doc + report triggers ingestion (embedding cost) via `ingest_agent_output` — small per-doc, but it's an always-on cost attached to generation.
- **Attachments:** inbound doc extraction is bounded by an 80k-char budget (`resolver.py:29`); images inline <500 KB else signed URL — bounded, no runaway.

No cost gate needed at pilot scale; the only cost worth watching is WeasyPrint CPU if documents get long/high-volume.

---

## I. UX / surface

Current surface: `/deliverables` with three tabs — **Outputs** (the `v_workspace_outputs` gallery), **Blogs**, **Templates** (the block **Template Studio**: `TemplateStudio.tsx` + `BlockEditor.tsx` + `BrandKitDialog.tsx` + `PreviewPane.tsx` + `VariablePicker.tsx`) — plus a separate `/documents` page (RAG document management, distinct concern) and a mission-scoped `mission-deliverables-panel.tsx`.

Concrete IA/UX changes, North-Star ranked:
1. **Surface deliverable freshness + the outage in Command Center.** A "no client-facing output in 18 days" state must be *loud*, not silent. This is the #1 UX failure the real data exposed — the platform produced nothing for 2.5 weeks and no surface said so. Add a Command Center tile driven by metric G-5.
2. **Show unresolved/unknown variables as a pre-finalise blocker in the Studio and in the agent result.** The data's already there (§B.3) — turn the visible red marker into an explicit "3 unresolved fields — fix before sending" gate (closes C-4). Clients should never receive a `[[company.address]]` document.
3. **Adopt Plate for the editor** (§E, C-7) — the single biggest perceived-quality lift for anyone authoring a template. Drag-drop, slash menu, real inline tables.
4. **Add the loop/repeat block to the editor** (C-1) so a user can build an invoice/line-item table visually — the most-requested business document.
5. **Fix the download UX (F030)** — the "Download" link in the gallery must never 404; point it at a stable app route that re-mints on demand.
6. **Deliverables ≠ Documents naming** — two top-level surfaces both about "files" confuse; per the canonical-terms rule, "Deliverable" is the client-output word — consider folding RAG `/documents` under a clearer "Knowledge Sources" label so "Deliverables" unambiguously means client output.

---

## J. Upgrade path — prioritised (impact × effort), North-Star-judged

| # | Change | Impact | Effort | Why (North Star) |
|---|---|---|---|---|
| J1 | **Fix F030** — persist the app-relative path as `download_url`; let the re-mint endpoint own presigning (`generation_service.py:642-645`, `api/document_generation.py:559-596`) | High | Small | The client-facing artifact currently **rots after 1h** — the most direct hit to "what clients see." Cheapest high-impact fix; half-built already. |
| J2 | **Fix F031** — add `template_id` to the registry `generate_document` ToolSpec (`tool_registry.py:1185-1218`) | Med | Trivial | Unblocks id-driven template generation on the **non-chat autonomy lane** (missions/board/scheduled) — the exact path the North Star cares about. |
| J3 | **Ship deliverable-freshness metric + Command Center tile** (G-1, G-5) | High | Small | Would have caught the 2026-06-16 silent outage on day one. Turns "quality is a vibe" into a tracked number for this plane. |
| J4 | **Add a `repeat`/`each` block primitive** (bind `data.*` array → child sub-tree, Carbone `[i]` semantics) + migrate the 5 Jinja seeds to blocks; retire the legacy `template_content` lane | High | Med | Closes the invoice/line-item gap (C-1) — the most common client business document — and unifies onto one schema, deleting the legacy renderer. |
| J5 | **Enforce the unresolved gate** — make callers refuse-to-finalise (or flag) when `unresolved`/`unknown` non-empty (`generation_service.py:176-180`) | Med | Small | Stops clients receiving `[[…]]`-marked documents; activates a primitive that already exists. |
| J6 | **Adopt Plate for the editor** (MIT; seam already Plate-shaped) | High | Med | Biggest perceived-quality lift for template authoring; stops hand-rolling `BlockEditor`. |
| J7 | **Persist clean-render + template-lane metrics** on deliverable `extra`; surface report-grade distribution | Med | Small | Feeds the T3 harness; makes document quality trackable per workspace. |
| J8 | **Brand the XLSX header** from the kit (`generation_service.py:397`) | Low | Trivial | Removes a visible cross-format branding inconsistency. |
| J9 | **De-risk WeasyPrint for long/high-volume docs** — offload render to a worker/Gotenberg or evaluate Typst | Low (now) / High (at scale) | Med | Only if generation volume or document length grows; not a pilot-phase priority. |

**One-line judgement:** the fusion (brand kit + variables + agent tool + registry + flywheel) is genuinely worth keeping and is the module's real value, but it's currently unproven in production, can't render the most common client document (line items), and hands clients a link that expires in an hour — fix F030/F031 and freshness first (days), then add the loop block and adopt Plate.

---

### Evidence index
- Internal: `orchestrator/modules/documents/{blocks/schema.py, blocks/html_renderer.py, blocks/docx_renderer.py, blocks/legacy_mapper.py, block_starters.py, seed_templates.py, generation_service.py, brand_kit.py, template_service.py, conversion_service.py, variables/resolver.py, variables/catalog.py}`; `orchestrator/services/{deliverable_service.py, report_service.py}`; `orchestrator/modules/attachments/resolver.py`; `orchestrator/api/{deliverables.py, document_generation.py}`; `orchestrator/modules/tools/discovery/actions_documents.py`; `orchestrator/modules/agents/services/agent_platform_tools.py:721-840`; `orchestrator/services/coordinator_service.py:905-945`; `frontend/components/documents/blocks/*`, `frontend/app/deliverables/page.tsx`.
- Real data: `reports/dossiers/evidence/data/deliverables.md`, `evidence/data/census.md`, `evidence/real-data-inventory.md`.
- Residual map (July close-out): `reports/dossiers/evidence/phase0-residual-map.md` — F030 (:475), F031 (:541), F089 FIXED (:521).
- Competitors: [carbone.io](https://carbone.io/) · [carbone.io/pricing.html](https://carbone.io/pricing.html) · [carbone loops](https://carbone.io/documentation/design/repetitions/with-arrays.html) · [blocknotejs.org](https://www.blocknotejs.org/) · [github.com/TypeCellOS/BlockNote](https://github.com/TypeCellOS/BlockNote) · [platejs.org](https://platejs.org/) · [github.com/udecode/plate](https://github.com/udecode/plate) · [nutrient.io/ai/infrastructure](https://www.nutrient.io/ai/infrastructure/) · [github.com/PSPDFKit-labs/nutrient-agent-skill](https://github.com/PSPDFKit-labs/nutrient-agent-skill) · [docxtpl.readthedocs.io](https://docxtpl.readthedocs.io/) · [weasyprint.com comparison](https://weasyprint.com/2026/05/02/weasyprint-vs-others-which-pdf-generator-should-you-choose/) · [speedata PDF benchmark](https://news.speedata.de/2026/02/10/typesetting-benchmark/) · [typst.app/blog](https://typst.app/blog/2025/automated-generation/).
