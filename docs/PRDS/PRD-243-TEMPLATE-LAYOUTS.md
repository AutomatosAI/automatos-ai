# PRD-243: Categories are layouts — and tables that fill from data

> **Status:** BUILT 2026-09-15 on `feat/prd-243-template-presets` (cut from `main` @ `e74f84953`, #739). Grounded @ Gerard's test of PRD-242 on the local stack the same day.

---

## The review (2026-09-15, screenshots of the Studio)

1. *"All categories — doesn't matter what I select, nothing changes. The idea of templates is the categories would be pre-programmed: a letter would have business details, address; invoices and so on. Basic set-ups the user can select and edit, not build everything from scratch."* — Category was a tag on a blank page. He had opened the legacy "Executive Summary" as a copy and got an empty editor with one heading, then built a letter out of Heading blocks.
2. *"Every header has lines under it."* — the HTML renderer drew an accent rule under every `h1` and a hairline under every `h2`, so his name and email (both headings) came out underlined.
3. Implicit: an invoice needs a line-items table whose rows an agent supplies. Blocks v1 only had a typed-in table, so an invoice starter could not exist (the dossier's J4 gap, deferred since PRD-190).

## Framing (CLAUDE.md §3)

**Extension.** Same schema, same renderers, same Studio. Three additions: a preset per category, a `data_table` block, and a picker in front of "New template".

## Decisions

- **D1 · A category is the layout you start from, and still the tag agents filter by.** `modules/documents/presets.py` holds one complete, brand-aware block layout per category (letter, invoice, report, proposal, contract, data, general). `New template` opens the picker; `Change layout` in the editor swaps a draft's blocks for another category's layout (identity kept); copying a legacy (non-block) template starts from its category's layout instead of an empty heading.
- **D2 · Starters are seeded FROM the presets and refreshed in place.** One starter per category, `created_by="system"`, copy-on-customise. `seed_starter_templates` now refreshes a platform-owned starter whose columns drifted from the preset and never touches a row a person created under the same name (`starter_outcome`, pure, tested). Local edition: on the next backend boot. SaaS: at provisioning only — existing SaaS workspaces keep their old starters until a "refresh starters" action exists (open question 1). `block_starters.py` deleted (replaced).
- **D3 · `data_table` block.** Rows come from a `data.*` list (`data.line_items`, `data.metrics`, `data.pricing`, `data.rows`); columns carry key/label/alignment; an empty or missing list is **unresolved** (blocked at finalisation, like an empty chip) unless the author sets `empty_text`. Rendered in HTML (WeasyPrint) and DOCX. The Studio form edits list fields as JSON rows with a per-column hint; the gallery, the chip bar and the Use-with-Auto prompts mark them `name[]` and spell out the row keys, so an agent knows to pass a list of objects.
- **D4 · No rules under headings.** Headings are colour + weight + spacing only; table header rows keep the primary fill; body cells use a hairline bottom border. `h1` 22pt, `h2` 15pt, `h3` 12.5pt.
- **D5 · Every preset must render clean with its own sample data** against a filled brand kit and user — a parametrised test, so a starter can never ship a blocked document. Optional contact details (`company.address/email/phone/website`, `user.email`) carry `fallback=""`; the fields a document is *about* have none.

## Stories

- **S1 · `data_table` block** — `blocks/schema.py` (`DataTableColumn`, `DataTableBlock`: path must be `data.*`, unique keys), `blocks/validation.py` (`collect_list_fields`, discriminator tag), `blocks/html_renderer.py` + `blocks/docx_renderer.py` (`data=` kwarg; rows by key or position; empty policy), `variables/catalog.py::walk_dynamic` (shared reader; the resolver's `_walk` moved there), `generation_service.py` + the preview route thread the raw `data` through, `template_summary.py` adds `list_fields`.
- **S2 · Presets + seed** — `modules/documents/presets.py` (7 layouts, `preset_for`, `preset_payload`), `seed_templates.py` (`starter_columns`, `starter_outcome`, refresh in place), `GET /api/documents/templates/presets` (declared before `/templates/{id}`; manifest +1 → 806).
- **S3 · Studio** — `PresetPicker.tsx`, `presetDraft.ts` (`draftFromPreset`, `applyPresetLayout`, `blankDraft`, `sampleDataOf`, `isBlankDraft`), `DataTableEditor.tsx`, `BlockEditor.tsx` (menu: "Table (typed in)" vs "Table from data (agent fills rows)"), `TemplateStudio.tsx` (picker on New / Change layout / legacy copy), `TemplateEditor.tsx`, `PreviewDataForm.tsx` (JSON rows for list fields), `GenerateDocumentDialog.tsx`, `TemplateCards.tsx`, `promptSnippets.ts` (list fields spelled out; playbook step seeds `[]`), `TemplateGuide.tsx`, `tooltips.json` (category = layout + filter).
- **Tests** — `tests/test_prd243_data_table_and_presets.py` (schema rules, path/list collection, HTML + DOCX rendering, empty policy, every preset renders clean, payload derivation, starter refresh outcomes, no heading rules); vitest `templateFields`, `promptSnippets`, `presetDraft`.

## How it reads now

New template → pick **Invoice** → a branded invoice with your details, a bill-to block, invoice number/date/due, a line-items table (`line_items[]` — description, quantity, unit price, total), totals and payment terms. Name it, save. **Use with Auto → Chat** now says: *"Fill these fields from your research: client_name, … line_items. line_items is a list of rows, each with description, quantity, unit_price, total."*

## Open questions (Gerard's call)

1. **Refreshing starters in existing SaaS workspaces** — the seeder only runs at provisioning there. A super-admin "refresh starters" action (or a boot-time sweep) would push preset updates to every workspace; not built.
2. **Legacy Jinja seeds** (Basic Report, Invoice, Executive Summary, Meeting Notes, Data Export) still seed alongside the block starters, and `generate()` still defaults PDF generation to "Basic Report" when no template is named. With a block invoice and report now existing, they could be retired (PRD-167's original "100% block seeds" metric) — but that changes the agent's default document look; your call.
3. **Repeat/each beyond tables** (J4's full form — repeated sections per item) is not built; `data_table` covers the invoice/report/proposal/data cases that were blocking.
