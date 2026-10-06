# Ralph Review Prompt — PRD-255 Brand Kit v2, Wave 1 (the v2 kit, and every renderer reading it)

You are a fresh-context **adversarial reviewer**. The build claims PRD-255 Wave 1 is complete. Find where:
- a renderer still paints the raw primary (or the accent) on an element that has a role, so a document is still "orange from top to bottom";
- a v1 kit does not read as a readable v2 kit, or a stored role loses to a derived one;
- a text role can be saved, or derived, below its contrast target;
- a currency, a date style, a logo variant or a font appears that the kit does not have;
- an existing workspace, template, social template or token name breaks;
- the agent path to the kit is wider than the owner's REST path;
- a story's evidence does not exist.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd255w1.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

The base is the `origin/main` merge-base. Never file a finding against a line this diff does not touch, unless this wave's change makes it reachable. Read:
- `scripts/ralph/prd-255w1.json` (binding, with its `decisions` and each story's `notes`: the loop's choices for the PRD's ambiguities are the contract, not findings);
- `docs/PRDS/PRD-255-BRAND-KIT-V2.md` (FR-1..FR-12, §7);
- the `ownerTest` list in the JSON (what the owner checks by hand; the loop must not claim it).

## Hunt list: every item is a confirmed-risk class

1. **Roles, not raw colours (FR-3, FR-6).**
   - `page_style.py`, the DOCX writer (and F356's `design_tokens.py` / `docx_style.py` / `docx_tables.py` when on the base), the legacy Jinja templates and `xlsx_render.py` read the roles from `effective_palette`, never `primary_color`, for any element with a role.
   - Under `accent_use: sparing` (the default for every kit, Decision Q1): headings and the table header fill are not the accent; the accent is the title rule, key numbers, links, one element per section.
   - Small text in an accent below 4.5:1 on paper prints in `heading`.
   - A heading or header still in the accent under sparing = HIGH.
2. **Derivation (US-003, FR-2).**
   - `derive_palette` reuses `_least` / `_most` / `contrast` (no second WCAG implementation); pure; no settings migration; a stored role always wins.
   - The four kits of the tests (Automatos, Harbourline, dark-only, light-only) derive every text role ≥ 4.5:1 on paper and on `surface_2`; `heading` is never the primary.
   - `paper_palette` and `stage_palette` output unchanged for a kit with no v2 roles (socials of existing workspaces render as before).
   - A readable-on-paper claim that a test doesn't prove = HIGH.
3. **Validation (US-001, US-002, FR-5).**
   - The PUT refuses a role below its target with a 422 naming the role and the ratio; contrast is measured on the EFFECTIVE paper.
   - Every new field has bounds and a default as named constants; `extra="forbid"` still holds on every model; `BrandKitPatch` and the tool schema stay in parity (the extended test proves nested parity).
   - Tone: plain strings stay valid on read and write, and every reader of `voice.tone` goes through one helper (nothing still iterates it as strings).
   - A field that saves unvalidated = HIGH.
4. **Uploads and the agent path (US-002, US-008).**
   - `/brand-kit/logo-dark` and `/brand-kit/logo-mono` (POST/GET/DELETE) are behind the same permission as `/brand-kit/logo-mark`, reuse its helpers, scope to the caller's workspace, and are in the committed route manifest with their methods.
   - `logo_dark_path` / `logo_mono_path` are server-managed: no PUT or tool call can point them at a file.
   - `platform_update_brand_kit` keeps `admin_only`; it validates through the same writer as the PUT.
   - A wider path to the kit, or a cross-workspace read = CRITICAL.
5. **Documents (US-004).**
   - Sizes from `type_scale`, gaps and margins from `spacing_unit_pt` / `page_margin_mm`, the letterhead logo at `logo_rules.letterhead_mm`; DOCX Heading 1–3 / Normal / Table / Caption from the same tokens, with a header and a footer carrying page numbers.
   - Currency: symbol and two decimals only when the kit has a currency; none invented (FR-7). `date.long` and every rendered date follow `date_style` (FR-8). `brand.sign_off` is a chip and the Branded Letter's default closing name.
   - The render tests read REAL PDFs back (the accent share ≤ 15% of the inked area on page 1 of EVERY starter, heading colours, sizes, an accent change touching only accent elements). A test that asserts on the CSS string alone where the AC says "measured on the rendered page" = HIGH.
6. **Spreadsheets and socials (US-005, US-006).**
   - The sheet's header on `surface_2` + `heading` (accent only when bold), zebra rows on `surface`, the body font and size from `type_scale.body`, the currency format from the kit, the logo from `logo_rules`.
   - `brand_tokens` emits every token name the base emitted; the v2 roles are used when set and today's derivation otherwise; the size ratios default to 1 (a template without them renders as before); the dark logo only when set (FR-9).
   - The media-render CI lane ran `social_template_previews.py` with both kits and `hyperframes check` passed (read the job log). A lane that skipped although the wave touched its paths = HIGH.
7. **The page (US-007).** Every section in the AC exists; "Reset to derived" removes the stored role; the 422 shows under its role; calls through `apiClient`; components ≤ 150 lines; compact CSS only in the compact region; vitest proves each section. The browser check is the owner's: a claim of it = CRITICAL.
8. **The rules block (US-008).** `rules_for_kit` lists each role with set/derived, the type scale summary, the currency, the date style and the logo-variant paths; the 30 s kit cache still bounds reads.
9. **Scope and conventions.**
   - No Alembic revision. No hosted re-seed job, no type-scale presets, no onboarding change (the decisions).
   - No `os.getenv` outside `config.py`; no hardcoded colour or size outside named constants; every commit DCO-signed; no `node_modules`; no new dependency (WeasyPrint `<70` intact); functions ≤ 50 code lines and nesting ≤ 4 on touched code; no file over 800 lines grown; replaced paths deleted (no `_legacy`, no V2 copies); pushes only to this branch.
10. **Claims.** Nothing in the diff, the commit messages or the DONE marks may claim a browser check, the contact sheet, or a run on the owner's stack. A claimed check = CRITICAL.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff **in the foreground**. Any CRITICAL or HIGH it reports is a finding.
- Run `bash scripts/ralph/acceptance-prd255w1.sh` yourself. A green build with a red gate is a finding.
- Check CI: `gh run list --branch feat/prd-255-w1-brand-kit-v2 --workflow test.yml --limit 5`, then the jobs of HEAD's run.
- Spot-check three `DONE` acceptance criteria at random against the code and the CI log. Evidence that does not exist = CRITICAL.
- **Nothing runs on the owner's machine:** no server, docker, browser, pytest or database. CI is the evidence. Never end your turn with anything running in the background.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 5-line summary noting:
  - (a) where each renderer reads its roles (file:function), traced for a heading, a table header and a KPI number under `sparing`;
  - (b) the derivation proof (the four kits' test names and the ratios they assert);
  - (c) the accent-share render test and the CI run it passed in;
  - (d) the agent path to the kit (who can call it, through which writer);
  - (e) the owner's next step: the `ownerTest` list in `scripts/ralph/prd-255w1.json`.

  Final line: `REVIEW_PASS`
- **Findings:**
  1. Append `P255-RVW-n` stories to `scripts/ralph/prd-255w1.json` (n continues from the highest existing `P255-RVW-` number), each with a title, `file:line` evidence, mechanical ACs (each ending with "CI green on HEAD"), `priority` after the last story, `passes: false`.
  2. Commit with `git commit -s -m 'chore(prd-255): Wave 1 review findings → fix stories'`, then push to `origin feat/prd-255-w1-brand-kit-v2`.

  Final line: `REVIEW_FINDINGS`
