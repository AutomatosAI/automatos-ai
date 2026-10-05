# PRD-255: Brand Kit v2, one brand system for every output

**Status:** Draft · **Owner:** Gerard · **Written:** 5 Oct 2026 (TESTER, from nights 10/10b and Gerard's brand-board discussion)
**Type:** Extension (the kit, the palette and the renderers exist; this makes them one system) plus a small net-new part (the brand board, the Brand designer agent)

## 1. Introduction

The product thesis (4 Oct): *Automatos becomes your brand.* Every document, spreadsheet and social post a business makes should look like it came from the same studio.

Nights 10 and 10b (4–5 Oct) showed where we are. The kit reaches the outputs, but it is too thin to make them look professional:

- **The kit is four colours and two fonts** (`modules/documents/brand_kit.py`: primary, secondary, accent, text, `font_family`, `heading_font`). The colours have no role, so a renderer paints `primary` on everything: titles, table headers, KPI tiles. On the Automatos kit (orange #c44a1a) every document is orange from top to bottom (Gerard, 5 Oct: "a lot of orange in there").
- **Socials already use roles; documents don't.** `core/brand_palette.py` derives contrast-checked role tokens (paper, ink, on-paper, primary-on-paper, …) for the social images and videos (`core/media_render_bundle.brand_tokens`). The document stylesheet (`modules/documents/blocks/page_style.py`, after #969) and the spreadsheet writer (`xlsx_render.py`) read the raw kit colours. So one kit gives two visual systems.
- **No type scale, spacing or logo rules**, so each template guesses its sizes (F350 and F356: one heading size everywhere, logo sizes 45/25/0 mm before #969).
- **The kit can't say what reaches the page:** no currency, no date style, no `brand.sign_off` chip (F347, F350), no logo for dark backgrounds.
- **Nobody helps the owner build a kit.** The Brand kit page is a form. Auto can read and update it (`platform_update_brand_kit`), but nobody proposes one, shows it, or checks the result by looking at it.

The reference Gerard shared (a brand guideline board: colour roles with codes, a type scale H1 48/56 to caption 12/16, a spacing scale, logo uses on light and dark, tone words with meanings, applications) is the standard to aim for. It works because of restraint: an off-white page, near-black text, one accent used sparingly, strong size contrast and generous space.

This PRD has three parts:

1. **Kit v2:** a role-based brand system that every renderer reads.
2. **The brand board:** the kit shown as a one-page guideline, as a document and a social card.
3. **The Brand designer agent:** Auto delegates, and the designer proposes, renders, looks and revises.

## 2. Goals

- One kit drives **every** output, from the same tokens: PDF and DOCX documents (block and legacy templates), XLSX, social images, carousels and social video.
- **The accent is used sparingly.** On every starter, at most about 15% of the inked area is the accent colour, measured on the rendered page (the "orange" problem).
- Every template's type sizes, spacing and logo size come from the kit, not from the template.
- **Every existing workspace gets the v2 kit,** derived from its current kit, with nothing to fill in and nothing broken (Gerard: "everyone gets the updated brand kit").
- The owner can see their whole brand on one page (the brand board) and change it through Auto in plain words ("less orange", "warmer", "more space").
- The night re-run (10c) grades both the brand kit and the templates, and the socials night (11) is the second brand-kit run.

## 3. User stories

### Part 1: Kit v2

#### US-001: Colour roles in the kit
**Description:** As an owner, I want my kit to say which colour is for text, page, surface and accent, so documents use my accent as a highlight, not everywhere.

**Acceptance criteria:**
- [ ] `BrandKit` gains a `palette` object with roles: `ink` (body text), `heading` (headings), `paper` (page background), `surface` and `surface_2` (cards, zebra rows, table header fill), `accent` (highlights), `accent_2` (optional second accent), `muted` (secondary text) and `rule` (hairlines). Each is a 6-digit hex.
- [ ] Each role has an **optional usage hint** for the accent: `accent_use: "sparing" | "bold"` (default `sparing`).
- [ ] `PUT /api/documents/brand-kit` validates the roles (hex, contrast: ink and heading on paper ≥ 4.5:1, accent text on paper ≥ 4.5:1 or used for large text only), with a 422 that names the role and the failing ratio.
- [ ] `GET` returns the stored roles, or the derived ones (US-003), and says which (`palette_source: "set" | "derived"`).
- [ ] Unit tests for validation and contrast.

#### US-002: Type scale, spacing and logo rules in the kit
**Description:** As an owner, I want my kit to hold my type sizes, spacing and logo rules, so every template looks like it came from one designer.

**Acceptance criteria:**
- [ ] `BrandKit` gains `type_scale`: `display`, `h1`, `h2`, `h3`, `body`, `small` and `caption`, each `{size_pt, line_pt, weight}`. The default is a professional document scale (e.g. h1 22/28 600, h2 15/20 600, h3 12/16 600, body 10/15 400, small 8.5/12, caption 7.5/10).
- [ ] `spacing_unit_pt` (default 4) and `page_margin_mm` (default 18).
- [ ] `logo_rules`: `letterhead_mm` (default 16), `clear_space` (in logo heights, default 0.5), `min_mm` (default 8).
- [ ] Logo variants: `logo_dark_path` (for dark backgrounds) and `logo_mono_path` (single colour), uploaded like the logo (`/brand-kit/logo-dark`, `/brand-kit/logo-mono`). A variant that isn't set is never invented; renderers fall back as in FR-9.
- [ ] Locale: `currency` (ISO 4217, e.g. GBP) and `date_style` (`"d MMMM yyyy"` default, `"MMMM d, yyyy"`).
- [ ] Voice: each tone word can carry a one-line meaning (`tone: [{word, meaning}]`; plain strings stay valid).
- [ ] Unit tests for each field's validation and defaults.

#### US-003: Every existing kit becomes a v2 kit, derived
**Description:** As an existing customer, I want my kit upgraded without filling anything in, so my documents look better the day this ships.

**Acceptance criteria:**
- [ ] A pure function `derive_palette(kit) -> palette`, in `core/brand_palette.py` next to `paper_palette` and reusing its contrast search, maps a v1 kit to roles:
  - `ink` = the kit's text colour, darkened to ≥ 10:1 on paper;
  - `heading` = near-black from the text colour's hue (not the primary);
  - `paper` = white or the kit's light colour;
  - `surface` and `surface_2` = the paper, tinted 4% and 8% toward the secondary;
  - `accent` = the primary, darkened only as far as AA needs;
  - `accent_2` = the secondary if it differs in hue;
  - `muted` and `rule` from the ink.
- [ ] Derivation runs at read time, so there's no migration of `workspace.settings` (the kit is JSON). A stored role always wins over a derived one.
- [ ] On the Automatos kit (#c44a1a / #1d3658 / #1a1a2e), the derived palette gives near-black headings, an off-white or white page, and orange only as `accent`.
- [ ] Tests: the Automatos kit, the Harbourline night kit (#1E3A5F / #C26A2E), a dark-only kit and a light-only kit all derive readable palettes (every text role ≥ 4.5:1).

#### US-004: Documents read the kit's tokens
**Description:** As an owner, I want every PDF and Word document to use my colour roles, type scale and spacing, so they look consistent and not painted in one colour.

**Acceptance criteria:**
- [ ] `blocks/page_style.py` builds its CSS from the v2 tokens:
  - headings in `heading`, the title in `heading` with an `accent` rule;
  - table headers on `surface_2` with `heading` text, or on `accent` only when `accent_use` is `bold`;
  - zebra rows on `surface`, hairlines in `rule`;
  - KPI tiles on `surface` with the number in `accent`;
  - sizes from `type_scale`, margins and gaps from `spacing_unit_pt` / `page_margin_mm`, the logo at `logo_rules.letterhead_mm`.
- [ ] The DOCX renderer (`blocks/docx_renderer.py`) maps the same tokens to Word styles (Heading 1–3, Normal, Table, Caption) and gets a header and footer with page numbers (F356).
- [ ] The legacy Jinja templates (Basic Report, Invoice, Executive Summary) get the same variables (`brand.palette.*`, `brand.type.*`) and the four old-style starters are restyled to match (F356 part 3).
- [ ] Amounts print with the kit's `currency` (symbol and two decimals), and `date.long` follows `date_style`.
- [ ] A `brand.sign_off` chip exists, and the Branded Letter uses it as its default closing name.
- [ ] Render tests (the F331/F350 pattern, real PDFs read back):
  - the accent covers ≤ 15% of the inked area on each starter's page 1;
  - headings are not the accent colour;
  - the type sizes match the kit;
  - changing `accent` changes only accent elements.

#### US-005: Spreadsheets read the kit's tokens
**Description:** As an owner, I want my spreadsheets to match my documents.

**Acceptance criteria:**
- [ ] `xlsx_render.py`:
  - the header row uses `surface_2` + `heading` (or `accent` when `bold`);
  - zebra rows use `surface`;
  - the body font and size come from `type_scale.body`;
  - number formats use the kit's `currency`;
  - the logo uses `logo_rules`.
- [ ] Test: the header fill, the font and the currency format on a generated sheet.

#### US-006: Socials read the same tokens
**Description:** As an owner, I want my social cards, carousels and videos to be clearly the same brand as my invoices.

**Acceptance criteria:**
- [ ] `core/media_render_bundle.brand_tokens` reads the v2 roles when set (`paper`, `ink`, `accent`, …) and falls back to today's derivation otherwise. It keeps every existing token name (`--brand-ink`, `--brand-on-paper`, …), so all 18 social templates keep working.
- [ ] The social templates' display and body sizes scale from `type_scale.display` / `body`, keeping their own pixel scale (a ratio, not points).
- [ ] The dark logo variant is used on dark stages when set (it's never generated).
- [ ] Tests:
  - every social template renders with the Automatos kit and the Harbourline kit;
  - the hyperframes contrast check passes;
  - the accent is not the background of a text-heavy card when `accent_use` is `sparing`.

#### US-007: The Brand kit page shows and edits v2
**Description:** As an owner, I want to see and change my roles, type scale, spacing and logo variants on the Brand kit page.

**Acceptance criteria:**
- [ ] The Brand kit tab has sections:
  - **Colours:** each role as a swatch with its hex and a contrast badge, and "derived" or "set" per role.
  - **Type:** the scale with a live sample line per level.
  - **Spacing and logo:** the unit, margins, letterhead logo size and clear space, with a preview.
  - **Logo variants:** upload dark and mono.
  - **Locale:** currency and date style.
  - **Voice:** tone words with meanings.
- [ ] "Reset to derived" per role.
- [ ] Saving shows a contrast error inline (from US-001's 422).
- [ ] vitest for the sections; typecheck passes.
- [ ] Verify in browser using dev-browser skill.

#### US-008: Agents and Auto see and change the whole kit
**Description:** As an owner, I want to tell Auto "make the orange an accent only" or "more space between sections" and have it change my kit.

**Acceptance criteria:**
- [ ] `platform_update_brand_kit` (`actions_brand_kit_update.py`) accepts every v2 field. Its schema equals `BrandKit`'s fields (the existing schema-parity test is extended).
- [ ] `platform_get_brand_kit` returns the v2 kit, including derived roles marked `derived`.
- [ ] The session agents' rules block (`services/brand_rules.py rules_for_kit`) carries the roles (not just the four colours), the type scale summary, the currency, the date style and the logo-variant file paths (F332 pattern).
- [ ] Tests: the action round-trips every field; the rules block lists the roles.

### Part 2: The brand board

#### US-009: A brand board template
**Description:** As an owner, I want one page that shows my whole brand, so I can check it, share it and hand it to anyone who makes things for me.

**Acceptance criteria:**
- [ ] A new block starter "Brand Board" (category `brand`, PDF and DOCX) laid out from the kit only, with no data fields to fill:
  - the logo large, plus the variants on light and dark;
  - the colour roles as swatches with hex codes;
  - the type scale with samples;
  - the spacing and logo clear space;
  - tone words with meanings;
  - three miniature applications rendered from the starters (invoice, letter, social card).
- [ ] It is seeded for every workspace, under the existing starter-refresh rule.
- [ ] A social version, "Brand board" (`social_image`, 4:5 and 9:16), built from the same data.
- [ ] Render test with the Automatos kit; the page fits on one A4 page.

#### US-010: The brand board on the Brand kit page
**Description:** As an owner, I want to see my brand board on the Brand kit page and download it.

**Acceptance criteria:**
- [ ] A "Brand board" preview (the rendered page 1) at the top of the Brand kit tab, refreshed after each save.
- [ ] Download PDF and download PNG buttons.
- [ ] Verify in browser using dev-browser skill.

### Part 3: The Brand designer agent

#### US-011: A Brand designer agent, seeded
**Description:** As an owner, I want a designer on my team who builds and improves my brand, so I don't need an agency.

**Acceptance criteria:**
- [ ] A "Brand designer" agent is seeded per workspace (seed pattern in `core/seeds/`). It runs on the subscription runtime (`runtime: cli, provider: claude`), because it must read images.
- [ ] Its instructions:
  - analyse the logo (shape, colours, sector, tone, sophistication) before proposing anything;
  - derive everything from the logo;
  - accent used sparingly;
  - check every output by looking at its rendered page;
  - change the kit only through `platform_update_brand_kit`, and only after the owner approves.
- [ ] Hosted and local editions: the agent is created with no host; a ticket waits for a host as other session agents do.

#### US-012: Agents can render and look
**Description:** As the Brand designer, I need to see the page I made, so I can judge it and revise.

**Acceptance criteria:**
- [ ] A read-only session tool `render_preview(template_id | deliverable_id, page=1)` writes a PNG of the page into the session's folder (reusing F349's session copy, #970, and the page-1 thumbnail renderer from F353 / issue #947) and returns its path, so the agent can open it.
- [ ] It is limited to the ticket's own workspace; tested.

#### US-013: Agents can create and edit templates
**Description:** As the Brand designer, I need to build the owner's templates (a quote, a price list, a carousel) in the studio's block format.

**Acceptance criteria:**
- [ ] Session/platform actions `create_template(name, category, format, blocks, sample_data)` and `update_template(id, blocks, …)`, through the same validation as `POST`/`PUT /api/documents/templates` (blocks schema, chips). The 3-file tool pattern (`modules/tools/README.md`), registered via `registry.register` (hierarchy gate).
- [ ] Created templates are tagged `made-by:<agent>` and listed in the studio like any template.
- [ ] They can never change or delete a starter (copy-to-customise only, as the studio does).
- [ ] Tests: create, edit, a refusal on a starter, and validation errors are passed back.

#### US-014: Auto runs the brand flow
**Description:** As an owner, I want to say "help me design my brand" to Auto and be walked through it.

**Acceptance criteria:**
- [ ] When the owner asks Auto to design or improve the brand, kit or templates, Auto files a ticket to the Brand designer. Auto does not do the design itself (thesis: Auto delegates).
- [ ] The designer's ticket flow:
  1. read the logo;
  2. propose a kit (as a card the owner approves, with the brand board rendered from the proposal and not yet saved);
  3. on approval, save the kit;
  4. produce a sample set (invoice, letter, proposal, three social cards) as Deliverables;
  5. report back with the board and the sample set.
- [ ] "Less orange", "warmer" or "more space" on the card sends it back for a revision.
- [ ] Test: Auto routes a brand ask to the designer (routing test); the designer's card carries the proposal and the board.

## 4. Functional requirements

- **FR-1:** `BrandKit` v2 adds `palette` (roles), `accent_use`, `type_scale`, `spacing_unit_pt`, `page_margin_mm`, `logo_rules`, `logo_dark_path`, `logo_mono_path`, `currency`, `date_style` and `voice.tone` with meanings. Every v1 field keeps working.
- **FR-2:** a missing v2 field is derived at read time from v1 (US-003). A stored field always wins. There's no settings migration.
- **FR-3:** `core/brand_palette.py` is the single source of colour roles for documents, spreadsheets and socials. No renderer reads `primary_color` directly for an element that has a role.
- **FR-4:** every renderer (block HTML/PDF, DOCX, legacy Jinja, XLSX, social image, carousel, video) reads type sizes, spacing and the logo size from the kit.
- **FR-5:** contrast: every text role is ≥ 4.5:1 on its surface (≥ 3:1 for large text), enforced on save (422) and in derivation.
- **FR-6:** `accent_use: sparing` (the default) limits the accent to highlights: title rule, key numbers, links, one element per section. The table header fill and headings are not the accent.
- **FR-7:** amounts use the kit's currency (symbol and two decimals); no renderer adds a currency the kit doesn't have.
- **FR-8:** `date.long` and every rendered date follow `date_style`.
- **FR-9:** logo variants are used when set and never generated: a dark background uses `logo_dark` if set, otherwise the logo on a light chip; mono uses `logo_mono` if set, otherwise the logo.
- **FR-10:** the brand board (document and social) is rendered only from the kit and refreshed on save.
- **FR-11:** the Brand designer changes the kit only after the owner approves a proposal card, and creates templates only through the validated template actions.
- **FR-12:** both editions (local and hosted). Everything is scoped to the caller's workspace.

## 5. Non-goals

- No AI-generated logo, and no change to the owner's logo (the board shows the logo as uploaded).
- No print or CMYK output (brand-assets-print-eps is separate).
- No new font hosting: fonts are the kit's uploaded files or system or web-safe families.
- No per-template colour overrides UI in this PRD (a template still can't override the kit's roles).
- Not an image generator for the board's lifestyle photos: the board uses the kit, the starters and the style references only.

## 6. Design considerations

- **The reference standard:** Gerard's brand-board image (5 Oct): an off-white page, near-black type, one accent used sparingly, a strong type scale, a 4/8 spacing grid, logo on light and dark, tone words with meanings.
- **Reuse:**
  - `core/brand_palette.py` (`paper_palette`, `stage_palette`, contrast search);
  - `core/media_render_bundle.brand_tokens`;
  - `blocks/page_style.py` and `letterhead()` (#969);
  - `brand_assets()` / `rules_for_kit` (#953, #961);
  - the brand-references style profile (`brand_style.py`, PRD-251B) for the board's visual-language strip;
  - the Template Studio editor and preview.
- **Starters:** after US-004 every starter is re-rendered and checked by eye (the TESTER contact sheet) before merge.

## 7. Technical considerations

- **The kit is JSON** on `workspace.settings['brand_kit']`, so no Alembic migration is needed. The read path derives v2 (FR-2).
- `BrandKit` is a Pydantic model with `extra=forbid` in places (`BrandVoice`). Add the fields explicitly, and keep `BrandKitPatch` and `platform_update_brand_kit`'s schema in parity (an existing test).
- **Starter refresh:** the local edition re-seeds starters at boot. Hosted workspaces get new starter rows only at creation, which is today's rule. **To meet "everyone gets the update"**, render-level changes (tokens, sizes) apply to every template at render time without a re-seed. Only new starters (Brand Board) and starter-layout changes need the seeder; a one-off re-seed job for existing hosted workspaces is an open question (Q2).
- WeasyPrint is pinned <70 (#966). Page numbers use `@page` margin boxes; DOCX uses python-docx fields.
- **Performance:** derivation is pure and cached with the kit (`brand_rules` already caches reads for 30 s).
- **Code shape:** `page_style.py`, `docx_renderer.py` and `xlsx_render.py` get the token mapping as small new functions; nothing grows past 800 lines.

## 8. Success metrics

- **Accent share** ≤ 15% of the inked area on every starter's page 1 (measured by a test on rendered pixels).
- **Night 10c**, a re-run of the templates night in c1 with the brand kit added to the theme:
  - ≥ 80% of documents graded 4+ (night 10b: the persona's own templates around 4, starters around 2.5–3);
  - zero "Courier/DejaVu fallback" fonts;
  - zero documents where headings are the accent colour;
  - the kit changes (role, scale, logo variant) show in the next render on every route.
- **Night 11 (socials)**: the second brand-kit run. Every social family renders with the same roles as the documents; Gerard rates the side-by-side "same brand?" yes.
- **Brand flow:** from "help me design my brand" to an approved kit plus a sample set in one ticket, with ≤ 2 revisions, in a test workspace.

## 9. Open questions

1. **Accent default for existing kits:** `sparing` for everyone (it changes how today's documents look, intentionally), or keep `bold` for kits saved before v2? (Recommended: sparing.)
2. **A hosted re-seed:** run a one-off job so existing hosted workspaces get the new starter layouts and the Brand Board, or only new workspaces?
3. **Does the designer also edit social templates' layouts**, or only the kit plus document templates in this PRD?
4. **A brand board as an onboarding step:** show it at the end of onboarding (PRD-222) for every new workspace?
5. **The type default:** one professional scale for all, or two presets ("editorial" and "compact") the owner picks from?

## 10. Testing plan (nights)

- **Night 10c (templates + brand kit), in c1, on main after this PRD's waves merge:**
  1. The night-10b theme, plus a brand-kit block: change each role, the type scale, the spacing and the logo variants through the Brand kit page, through Auto in plain words, and through the Brand designer.
  2. Regenerate the same five documents after each change.
  3. A contact sheet each morning.
  4. A brand-board check.
- **Night 11 (socials):** the same kit across every social family. The second brand-kit run.
- Both are graded with the night-10 rubric plus two new axes: **brand consistency** (same roles across outputs) and **restraint** (accent share).

## Related

Findings F332, F336, F347, F350, F352, F353, F356 (docs/testing/FINDINGS-LEDGER.md); PRs #953, #961, #969, #970, #971; PRD-251 (Socials), PRD-251B (the studio and style references), PRD-243 (block starters); issue #947 (thumbnails).
