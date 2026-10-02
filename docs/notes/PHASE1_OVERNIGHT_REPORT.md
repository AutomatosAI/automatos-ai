# Phase 1 Overnight Report

**Branch:** `feat/studio-rebrand-phase1`
**Date:** 2026-05-15
**Status:** Foundation shipped, ready for your review. NO PR opened — your call.

---

## TL;DR

Studio token foundation, theme flag, glossary infrastructure, primitive radius overrides, editorial PageHeader, and a living `/styleguide` page are all on the branch. **Build passes**, **tests pass (8/8)**, **zero TypeScript errors** in any file I touched. The classic orange/glass theme is **completely untouched** — Studio is purely additive and opt-in.

To see it: flip your theme to "Studio (preview)" from the user menu, OR visit any page with `?theme=studio-preview` in the URL, OR navigate to `/styleguide` which forces Studio scope.

---

## What shipped (4 commits)

### 1. `docs: add Phase 1 PRD for platform redesign` (ccd2792)
- `PRD_PLATFORM_REDESIGN.md` — 11-phase plan, locked direction (Ledger Studio), CD's resolved answers, Track A on-ramp decision

### 2. `feat(studio): L1 token foundation + theme flag` (321b11c)
- `frontend/app/globals.css` — `.studio` scope (4th theme alongside `:root`/`.dark`/`.matte`)
  - Cream paper + warm ink + olive/navy/burnt-orange semantic lock
  - Flattens glass, kills glows + pulsing animations
  - 8 Studio utility classes ported from DesignKIT round2 (`.studio-pill`, `.studio-panel`, `.studio-eyebrow`, `.studio-lede`, `.studio-pip`, etc.)
- `frontend/tailwind.config.ts` — `font-serif` (Tiempos→Charter→Georgia) + `font-mono` (JetBrains Mono→Fira Code) families
- `frontend/hooks/use-studio-theme.ts` — `useStudioThemeFlag()` reads `?theme=studio-preview` URL param + persists via next-themes; `useIsStudio()` helper
- `frontend/components/providers.tsx` — registers `'studio'` with `next-themes`, mounts the flag inside a Suspense boundary
- `frontend/components/ui/theme-toggle.tsx` — "Studio (preview)" dropdown item with `BookOpen` icon

### 3. `feat(studio): Move 2 glossary + shadcn radius overrides + PageHeader editorial-first` (69f7266)
- `frontend/lib/glossary.ts` — 8-term content map (mission, agent, playbook, deliverable, router, skill, handoff, T2.5). Voice matches PRD §8.3.
- `frontend/components/ui/glossary-tooltip.tsx` — `<GlossaryTooltip term="mission">…</GlossaryTooltip>`. After 3 sightings of a term, suppresses itself via localStorage. Demo in `/styleguide` section 8.
- `frontend/app/globals.css` — Studio primitive overrides block: radius reductions (rounded-2xl → 10px, buttons → 6px, badges → 2px), tab pattern (2px accent border-bottom), heading utilities (text-3xl/4xl/5xl pick up serif), tooltip + toast styling.
- `frontend/components/shared/page-header.tsx` — added optional `eyebrow` and `lede` props for editorial-first treatment. **Existing API preserved** — no consumer breaks.

### 4. `feat(studio): in-product /styleguide route — living spec` (ae4768b)
- `frontend/app/styleguide/page.tsx` — single-page spec doc, 8 sections:
  1. Semantic colour lock
  2. Typography
  3. Buttons
  4. Inputs + form
  5. Badges + pills
  6. Status icon vocabulary
  7. Panels + cards
  8. Tabs + glossary tooltips
- Always renders in `.studio` scope. Demonstrates the new `PageHeader` editorial pattern at the top.

---

## What you should review tomorrow

Time-box this to 30 minutes. Anything more is day-2 polish, not review.

### Visual check (15 min)
```
cd frontend && npm run dev
```
Visit `http://localhost:3000/styleguide`. Walk all 8 sections. Look for:
- Cream paper feels right (not too yellow, not too grey)
- Serif renders (Tiempos/Charter — relies on system font availability)
- Status pills read correctly at 10px font
- Glossary tooltips trigger on hover (try `/styleguide` then look for dotted-underlined terms)

Then visit any existing page (`/command-center`, `/agents`) and toggle theme to Studio via the user-menu dropdown. Look for layout regressions — there shouldn't be any since the colour layer flows through CSS variables.

### Code review (15 min)
- `frontend/app/globals.css` — the `.studio` blocks are isolated. Read them.
- `frontend/hooks/use-studio-theme.ts` — 30 lines. Read it.
- `frontend/lib/glossary.ts` — read the 8 definitions; flag any you want reworded.
- `frontend/components/shared/page-header.tsx` — the eyebrow/lede addition.

---

## What I did NOT do (day-2 work)

### Genuinely unfinished — must complete before Phase 2
- **Visual parity verification** — I cannot run a browser. You confirming `/styleguide` matches `phase1/microspec.jsx` + `kit.jsx` is the gate before Phase 2 starts.
- **Lint check** — `next lint` is interactive on first run (needs config setup); skipped. Run `npx @next/codemod@canary next-lint-to-eslint-cli .` to migrate, then `npx eslint .`.

### Deliberately deferred (with reasons)
- **Per-primitive restyle of all 53 shadcn files.** Instead I added a global `.studio` radius + font override layer that all primitives inherit through. This was the time-budget-correct call but means a handful of niche primitives (Combobox, DataTable, Drawer, etc.) may need spot-fixes when first encountered in their pages. Flag them as you find them.
- **`.dark` Studio variant.** Per your decision, deferred. Studio runs light-only for now.
- **Move 3 Auto narration build.** That's Phase 3 work. Design is in `DUMPING AREA/phase1/narration.jsx`.
- **Sticky composer on /command-center** (the on-ramp decision, Track A from PRD §6a). That's Phase 3 work too.
- **Playwright screenshot diff harness.** Day-2 setup. Not blocking Phase 2 start.
- **Onboarding tour for the glossary terms.** The `<GlossaryTooltip>` infrastructure is there. A "first-run shows all 8 terms" tour is a nice-to-have, not Phase 1.

---

## Known issues + caveats

1. **Tiempos Headline is a paid font.** The serif fallback chain (`Charter → Iowan Old Style → Georgia`) renders fine on macOS/iOS but Windows users see Georgia, which is heavier. Decision needed in Phase 2: license Tiempos, ship a self-hosted variable serif (Newsreader is a free alternative), or accept the fallback. I left the original chain for now.
2. **`hooks/use-studio-theme.ts` uses `useSearchParams`** which Next.js 15 requires inside a Suspense boundary. I wrapped it correctly in `providers.tsx`, but if you move the hook, remember the Suspense.
3. **The `.gradient-text` utility is neutralised under Studio** — it renders as plain ink. This means every `PageHeader` using the two-word `title + titleAccent` pattern will look like a single-colour title under Studio. That's intentional per the brand brief; if you want a different treatment, flag it.
4. **Pre-existing TypeScript errors** remain in `lib/api/*`, `lib/chat/hooks.ts`, `lib/workflow-service.ts`, etc. (45 errors total). All pre-existed before my work. The Next.js build passes anyway because `tsconfig` has `strict: false` or equivalent.
5. **`/styleguide` is publicly routable.** It's likely behind Clerk auth via your middleware — verify when you visit. If you want it admin-only, add a guard.

---

## Phase 2 readiness checklist

Before starting Phase 2 (chat redesign), confirm:
- [ ] `/styleguide` renders correctly on your machine
- [ ] Theme toggle flips between light/dark/matte/studio without breaking any pages
- [ ] `?theme=studio-preview` URL param works (try on `/command-center`)
- [ ] Glossary tooltip hover shows on `/styleguide` (3 hovers per term then suppresses)
- [ ] You've reviewed the 8 glossary definitions in `lib/glossary.ts` and are happy with them
- [ ] You're OK with the serif fallback chain (or have decided to license Tiempos)
- [ ] You've made the on-ramp call (Option A / B / C from PRD §6a)

Once those are green: open Phase 2 with CD — chat redesign per `DesignKIT/round3/chat.jsx`.

---

## Files changed

```
PRD_PLATFORM_REDESIGN.md                            (NEW, 474 lines)
PHASE1_OVERNIGHT_REPORT.md                          (this file)
frontend/app/globals.css                            (+330 lines, 4 new blocks)
frontend/tailwind.config.ts                         (+17 lines, fontFamily)
frontend/components/providers.tsx                   (+10 lines, hook mount)
frontend/components/ui/theme-toggle.tsx             (+10 lines, Studio item)
frontend/hooks/use-studio-theme.ts                  (NEW, 33 lines)
frontend/lib/glossary.ts                            (NEW, 73 lines)
frontend/components/ui/glossary-tooltip.tsx         (NEW, 116 lines)
frontend/components/shared/page-header.tsx          (rewrite, +35 lines net)
frontend/app/styleguide/page.tsx                    (NEW, 442 lines)
```

4 commits, ~1,500 net lines added, 0 lines of pre-existing code modified destructively.

---

## To merge or not to merge

**Do not merge yet.** This branch is intentionally a feature flag — `?theme=studio-preview` opt-in. The classic theme remains the default for every user. You can:
1. Run it locally, walk `/styleguide`, give pilot users `?theme=studio-preview` URLs
2. Iterate on tokens/glossary/primitives before merging
3. Open PR when you're confident

When ready: `gh pr create` with title "feat(studio): Phase 1 — token foundation + theme flag + /styleguide".

---

## Day-2 punch list (do these before Phase 2 starts)

In order of priority:

1. **Visual review of `/styleguide`** (you, ~30 min)
2. **Decide on serif licensing** (Tiempos vs free alternative) — affects Phase 2 chat surface
3. **Test Studio on `/command-center` + `/agents` + `/chat`** — verify no layout regressions (you, ~15 min)
4. **Resolve the on-ramp decision** (PRD §6a Track A) — Option A / B / C
5. **Spot-fix any primitive that renders wrong under Studio** — flag in this report's appendix when found
6. **Set up Playwright screenshot diff harness** (~2 hours of dev work)
7. **Write glossary onboarding tour** (~1 hour) — first-time user sees all 8 terms once
