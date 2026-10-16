# PRD-259: Skins in Studio (Style axis, Marketplace item, Auto companion)

Status: decisions taken 14 Oct 2026 (§10). Moves to `automatos-ai/docs/PRDS/` for the build. Companion: `automatos-play` PRD-004 (the Skin Kit that makes skins).

## 1. Introduction

Studio already has two appearance settings (PRD-244): **Style** (Classic | Studio, cookie `automatos-style`, applied by the server as `<html class="studio">`) and **Tone** (Light | Dark | System, next-themes `.dark`). Both are in one dropdown (`components/ui/theme-toggle.tsx`).

This PRD adds **skins** to the Style list. A skin is a data-only appearance package made with the Automatos Skin Kit (`automatos-play`). It builds on Classic or Studio and replaces their colour tokens, fonts and shape, and can bring Auto (or its own mascot) into the chat. Skins are shared through the Marketplace as a new item type: a workspace **installs** one, and from then on its members can pick it in the Style dropdown. A developer can also load a skin straight from a file, which is the loop the Skin Kit uses.

The goal (Gerard, 14 Oct): make the real Automatos UI personal (mood, age, colours, fun, even a Jarvis feel) without rebuilding pages. **Studio's pages, layouts and components stay the same; only their look changes.**

## 2. Goals

- Installed skins appear in the Style dropdown next to Classic and Studio, and switch instantly without breaking Tone.
- No flash of the wrong look on load.
- A skin can't run code: it is validated data and assets, turned into CSS by Studio itself.
- Every skin is readable: contrast is enforced at upload, not hoped for.
- A new **Skins** item type in the Marketplace: submit, review, install and uninstall.
- An optional Auto companion in Studio's chat, driven by the same stream events Studio already parses.
- The worst fixed-colour files are moved onto tokens, so skins reach the corners.

## 3. The skin format, `automatos-skin/v1`

One file, `<name>.autoskin`: a zip holding `skin.json` plus an optional `assets/` folder. **Studio owns the schema** (`frontend/lib/skins/schema.ts`, also served at `GET /api/skins/schema`). The Skin Kit syncs its copy from there.

`skin.json`:
- `schema`, `slug` (kebab-case, unique per author), `name` (≤40 characters), `version` (semver), `author`, `description` (≤280), `licence`;
- `base`: `studio` or `classic`;
- `tones`: `["light", "dark"]` or either one alone;
- `tokens.light` and `tokens.dark`: values for **allowed token names only** (list below), as HSL triplets (`"16 100% 50%"`) or `#rrggbb` (normalised to triplets);
- `fonts`: `sans`, `serif` and `mono`, each the name of a font Studio ships or a font file in `assets/fonts/` (woff2 only);
- `shape`: `radius` (0–24 px), `glow` (0–40);
- `mascot`: `{mode: "off" | "auto" | "custom", shell?, light?}`. `custom` needs the 12 state clips in `assets/mascot/` (`<state>.webm`, `.mp4` and `.png`);
- `preview`: `assets/preview.png`.

**Allowed tokens** (taken from `app/globals.css`):
- the shadcn core set: `background, foreground, card(-foreground), popover(-foreground), primary(-foreground), secondary(-foreground), muted(-foreground), accent(-foreground), destructive(-foreground), border, input, ring`;
- semantic: `success, warning, info, agent` and their foregrounds;
- `chart-1..5` and `stage-1..9`;
- Studio aliases: `olive, warn-ink, on-strong, border-2, accent-2, navy`.

The pre-wrapped aliases (`paper, bg-1, bg-2, fg, fg-2, fg-3`) are derived by Studio from the core tokens and are never set by a skin.

**Validation** runs at import and upload, server-side, with the same rules in the frontend and the kit:
- schema and allowed names only;
- value formats;
- **AA contrast:** every text pair (`foreground` on `background`, `card-foreground` on `card`, `primary-foreground` on `primary`, `muted-foreground` on `background` and so on) at least 4.5:1, and `ring` on `background` at least 3:1;
- fonts are woff2 by magic bytes, at most 400 KB each;
- mascot media sniffed (webm, mp4, png), at most 8 MB in total;
- preview PNG at most 1 MB;
- package at most 15 MB;
- no other files, no path traversal, no symlinks.

Sizes are settings (`SKIN_MAX_PACKAGE_MB` and the rest) in `config.py` and `config-surface.json`.

**Never accepted:** CSS, JavaScript, SVG, HTML, URLs, or any field not in the schema.

## 4. How Studio applies a skin

- **Style list:** Classic, Studio, then the workspace's installed skins under a "Skins" label, plus "Add a skin…" (load a file) and "Browse skins" (opens the Marketplace tab).
- **Choosing a skin:**
  - the cookie `automatos-style` becomes `skin:<id>`;
  - the server sets the base class (`studio`, or none for Classic) plus `data-skin="<id>"` on `<html>`;
  - the existing `isStudio` branches keep working from the base, so layouts are untouched.
- **The skin's CSS:**
  - Studio generates one stylesheet from the validated numbers only: `html[data-skin="<id>"]{…light tokens…} html.dark[data-skin="<id>"]{…dark tokens…}`, scoped so it outranks the `.studio` and `.studio.dark` blocks;
  - plus `@font-face` rules for the skin's font files and `--radius`/`--skin-glow`;
  - it is cached in localStorage and injected by a small inline pre-paint script (the pattern proven in Play's `lib/looks/prepaint.ts`), so there's no flash;
  - Studio's CSP already allows inline style and script (`next.config.js`).
- **Tones:** a dark-only skin greys out Light while it is active and forces Dark. Switching back restores the user's tone.
- **Assets** are served by `GET /api/skins/{id}/assets/{path}`, only for files in the validated manifest, with their checked type, `X-Content-Type-Options: nosniff` and long caching by version.
- **Falling back:** if the active skin is uninstalled, invalid or missing, the base style is used and a quiet notice says so.

## 5. The Marketplace item

Modelled on the existing patterns:
- **Data like Packages** (`marketplace_packages`);
- **a stored bundle like Plugins** (`core/services/marketplace_s3.py`, a `skins/{slug}/{version}/` prefix through `core.storage`, never a new client);
- **installed by reference like Widgets** (`widget_installations`).

- **Tables (one Alembic revision on the current head, plus moving the two head pins):**
  - `marketplace_skins`: slug, name, version, author user, `skin` JSONB, asset prefix, preview path, `owner_type` (`marketplace` | `workspace`), `workspace_id` (set for private imports), `status` (pending | approved | rejected), `rejection_reason`, featured, install count, timestamps;
  - `skin_installations`: workspace, skin, version, installed by, installed and uninstalled timestamps.
- **Routes** (new `api/marketplace_skins.py`; every route goes in `route-manifest.json`):
  - `POST /api/skins/import`: load a file into **this workspace only** (`owner_type=workspace`, no review, installed at once). This is the developer loop;
  - `POST /api/marketplace/skins`: submit a file for the Marketplace. The status is pending unless the author is trusted, using the existing 5-approved rule counted on skins;
  - `GET /api/marketplace/skins`, `GET /api/marketplace/skins/{id}`: browse (approved only, unless admin), with search and featured;
  - `POST /api/marketplace/skins/{id}/approve`, `POST …/reject` (a reason is required; skins get a real reject, unlike agents today), `POST …/toggle-featured`: admin only;
  - `POST /api/marketplace/skins/{id}/install`, `DELETE …/install`: per workspace;
  - `GET /api/skins/installed`: this workspace's skins for the dropdown;
  - `GET /api/skins/schema`;
  - `GET /api/skins/{id}/assets/{path}`.
- **Marketplace UI:**
  - a **top-level Skins tab** next to Agents, Playbooks, LLMs and the rest (Gerard, 14 Oct): preview images in a grid, a detail view with light and dark previews, Install, Uninstall and "Use now";
  - Submit from a file;
  - an admin review queue with Approve and Reject.
- **Tenant isolation:** private skins and installations are always filtered by `workspace_id`.
- **Both editions:** in local, the operator is Marketplace admin.

## 6. Auto, the chat companion

- A skin with `mascot.mode` set to `auto` or `custom` shows a companion in Studio's chat: the Studio chat ledger's side column, and the classic chat panel. Each user can hide it.
- It is driven by the events Studio's chat parser already reads (`frontend/lib/chat/hooks.ts`), through the mascot reducer and player ported from Play (`lib/mascot/*`, already tested):
  - the 12 states, switching at the clip boundary;
  - the WebM/MP4 choice per browser;
  - the reduced-motion PNG;
  - a ring drawn by the page.
- **The built-in Auto clips ship with Studio** (rights settled 14 Oct, §10): the 12 states as WebM, MP4 and PNG under `frontend/public/auto/`, about 4.5 MB in total.

## 7. Making skins reach the corners

- **`--radius` is fixed globally (Gerard, 14 Oct).** It is used by Tailwind (`rounded-lg/md/sm`) but never defined in `globals.css`, so those utilities fall back to square corners today. Define it in `:root` (Classic, `0.5rem`, the shadcn default) and `.studio` (a value matching Studio's hard-coded 8–10 px radii), and let skins override it. Corners change across the app: review screenshots in the PR.
- **Fixed colours:** 535 fixed Tailwind palette classes and hex values in 91 files bypass tokens. Sweep the ten worst files onto tokens: `components/analytics/analytics-admin.tsx`, `analytics-composio.tsx`, `analytics-recommendations.tsx`, `components/documents/document-details-modal.tsx`, `delete-confirmation-modal.tsx`, `components/agents/org-chart-node.tsx`, `components/chatbot/markdown-components.tsx`, `sheet-artifact.tsx`, `text-artifact.tsx`, `components/marketplace/featured-showcase-card.tsx`. Status colours map to `success`, `warning`, `destructive` and `info`.
- **`dark:` is fixed globally (Gerard, 14 Oct).** All `dark:` classes are dead today: Tailwind 3.3.3 ignores `darkMode: ['variant', …]` (the `variant` form arrived in 3.4.1). Set `darkMode: ['class', '.dark']`, which 3.3.3 supports and which gives the same `:is(.dark &)` selector. The 135 `dark:` uses in 24 files then render for the first time, so audit each one (they were written but never seen) and fix or remove the wrong ones in the same story.

## 8. Stories (one wave each, in order)

- S-001 schema and validator (frontend `lib/skins/` and backend `services/skins/`, same rules, contrast pairs), `GET /api/skins/schema`
- S-002 tables, migration and storage (`skins/` prefix), config settings
- S-003 import route plus assets route (private workspace skins)
- S-004 the Style dropdown: Skins group, Add a skin…, generated CSS, pre-paint, the tone lock for dark-only skins, fallback
- S-005 Marketplace routes: submit, browse, approve and reject, install and uninstall, featured
- S-006 the Marketplace Skins tab, submit form and admin review queue
- S-007 the Auto companion in chat (port Play's reducer and player), with a per-user hide toggle
- S-008 sweep the ten worst fixed-colour files onto tokens
- S-009 define `--radius` globally (Classic and Studio), with before/after screenshots in the PR
- S-010 fix `darkMode` to `['class', '.dark']` and audit the 135 `dark:` uses in 24 files
- S-011 docs: skins in `docs/`, and the Studio half of the Skin Kit's guide

## 9. Non-goals (Gerard's to revisit)

- Layout or shell variants (a Jarvis HUD layout). Later, as named shells Studio itself ships and a skin picks by name, never as code.
- Skins changing page structure or components.
- Mobile Studio (mobile stays Classic until PRD-246).
- Paid skins.

## 10. Decisions (Gerard, 14 Oct 2026) and open questions

1. **Mascot rights: settled.**
   - The hero image was generated on Higgsfield's free plan (registration credits); the 12 clips were made with paid API credit.
   - Higgsfield's own help page says you own your outputs (Terms of Use §4.4) and that "the Terms of Use do not tie commercial use rights to a specific plan." Some third-party reviews claim otherwise; Higgsfield's page is the primary source.
   - So Auto ships in Studio.
2. **Tab:** Skins is its own top-level Marketplace tab, like Agents and LLMs.
3. **`--radius`:** fixed globally (S-009).
4. **`dark:`:** fixed globally (S-010).
5. **Merging:** the `automatos-play` PRs merge later. The repo keeps its name for now.
6. **Open:** install is per workspace and the active choice is per browser. Should an admin be able to set a workspace default skin?
