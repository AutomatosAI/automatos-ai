# Studio Shell Rollout Report

**Branch:** `feat/studio-shell-chrome` (forked off `feat/studio-rebrand-phase1`)
**Commits:** 2 this session
**Status:** SIDE-B + HEAD-A + TICK-A live behind Studio theme flag. NO PR yet.

---

## TL;DR

Rolled out CD's round-4 shell (sidebar + header + ticker) as the chrome for every page rendered in Studio mode. Classic theme is **completely untouched** — the switch is conditional on `.studio` class + desktop viewport. Pilot users see no change unless they opt into Studio preview.

To see it: flip the theme toggle to **Studio** (BookOpen icon in user menu) and navigate any page. Sidebar is now 232px cream-paper-on-cream with grouped nav, workspace pill, and mini-stats foot. Header is editorial — cmdK search + utilities cluster (Docs/Help/Alerts/Theme/Profile). Ticker is the 30px paper strip with 7 mono KPI cells.

---

## What landed (2 commits)

### `37eb0d9` — fonts + chrome CSS + menu config

**`app/layout.tsx`** — loads three Google Fonts via `next/font/google`:
- **Geist** (sans body)
- **Geist Mono** (mono detail layer)
- **Newsreader** (warm display serif — replaces unlicensed Tiempos Headline)

Variables `--font-geist-sans` / `--font-geist-mono` / `--font-newsreader` sit on `<html>`. The `.studio` block in globals.css reads from them with graceful fallback chains (Tiempos → Charter → Georgia for serif; JetBrains Mono → SF Mono for mono; system stack for sans).

**`app/globals.css`** — new chrome CSS under `.studio` scope (~310 lines added). Five class groups:

| Group | Purpose |
|---|---|
| `.sh-side` | 232px labelled sidebar — brand row, workspace card, group headers, nav items (active state with paper lift), footer, mini-stats grid |
| `.sh-brand` | Wordmark + 22px circle glyph with CSS-drawn sailboat |
| `.sh-headbar` + `.sh-utils` | Editorial header bar — cmdK input, icon buttons (Docs/Help/Alerts with 9+ badge), separator, profile pill |
| `.sh-ticker` | 30px paper strip with mono cells, LIVE olive dot, semantic tones (ok/err/info), right-aligned clock |
| `.sh-shell` | Full-page layout container (flex row sidebar+main) |

**`lib/studio-menu.ts`** — single source of truth for sidebar nav. 13 primary items in 3 groups (OPERATIONS / WORKFORCE / WORKSPACE) + 2 footer items (Docs external + Settings). Lucide icons, descriptions, hrefs. Plus `resolveActiveMenuId(pathname)` with smart route matching (`/chat/[id]` → chat, `/missions/[id]` → assign, `/activity` → cmd, `/admin/*` → admin, `/playbooks` → assign).

### `fd83acf` — StudioSidebar + Header + Ticker + MainLayout switch

**`components/layout/studio-sidebar.tsx`** — SIDE-B labelled rail
- Reads menu from `studio-menu.ts`
- Workspace switcher pill (props: name, meta, mark, click handler — currently displays "Automatos AI · pilot · 11 op", click handler unwired)
- 3-group nav with `usePathname()` → `resolveActiveMenuId()` for active state
- Footer renders Docs as `<a target="_blank">` + Settings as Next `<Link>`
- Optional `alerts` prop for per-item badges (e.g. `{ assign: '!' }` shows burnt-orange alert)
- Optional `showStats` prop for the mini-stats foot (defaults true; shows tick · $/dec · cache)

**`components/layout/studio-ticker.tsx`** — TICK-A 30px paper strip
- 7 default cells: UPTIME, CACHE, $/DEC, P50, ERR/HR, T2.5, QUEUE
- Tones: `ok` (olive), `err` (burnt orange), `info` (navy), `null` (default)
- `cells` prop for real metrics wiring later
- LIVE olive dot + right-aligned formatted clock (date + WET time + tick 5s)

**`components/layout/studio-header.tsx`** — HEAD-A editorial
- cmdK search trigger (visual stub — wire to GlobalSearch later via `onSearchClick`)
- Docs link (external), Help button (no-op), Alerts bell with badge (9+ if >9)
- **Theme toggle preserved** — existing `<ThemeToggle>` from `components/ui/theme-toggle.tsx`. Switching back to classic still works.
- Profile pill reads from Clerk's `useUser()` — shows initial in serif disc + first name + chevron
- Accessible: ARIA labels on every button, keyboard navigable

**`components/layout/main-layout.tsx`** — conditional render
- Uses `useIsStudio()` from `hooks/use-studio-theme`
- When `.studio` AND not mobile: renders `<div className="sh-shell">` with `<StudioSidebar /> + <main>{StudioTicker + StudioHeader + children}</main>`
- Else: renders the existing classic layout unchanged (Sidebar + Header)
- Mobile keeps the Sheet drawer pattern on both themes (Studio mobile sidebar is a future micro-pass)
- AutoWidget remains visible on both

---

## Verification

- **Build**: ✅ Next.js 15 build clean, all 45 routes compile, 0 errors
- **Tests**: ✅ 8/8 vitest passing
- **TS errors in my new files**: 0
- **Theme switch round-trip**: code-verified — flipping Studio → Classic via ThemeToggle removes the `.studio` class, MainLayout re-renders with classic chrome. No remount of `<Providers>` or auth state lost.

---

## Behavioural preservation

What the new shell **preserves** vs. the classic layout:

| Behaviour | Status |
|---|---|
| Theme toggle (light/dark/matte/studio) | ✅ Preserved — embedded in StudioHeader |
| Workspace state via WorkspaceProvider | ✅ Untouched (sidebar visual only; data still flows) |
| Route active state | ✅ via `resolveActiveMenuId(pathname)` |
| Sign-out flow | ⚠️ Stubbed — onProfileClick prop is non-interactive currently. Wire Clerk's `UserButton` or custom dropdown in Phase 2. |
| Mobile sheet drawer | ✅ Falls back to classic on mobile, both themes |
| AutoWidget visibility | ✅ Same logic, both layouts |
| Onboarding tour (Shepherd) | ✅ Preserved (`useAutoTour` still runs) |

---

## What's NOT yet wired (day-2 work, none urgent)

1. **Sign-out / profile dropdown** — `onProfileClick` is a no-op. Replace with Clerk's `<UserButton />` or a custom dropdown that includes sign out, profile, billing, etc. The existing classic header used `ProfileMenu`; we can drop that into StudioHeader.
2. **Workspace switcher click** — `onWorkspaceClick` is a no-op. Wire to the existing `WorkspaceSelector` dropdown (it's already in the platform).
3. **cmdK search** — visual stub. Hook to `GlobalSearch` already provided by `<Providers>`.
4. **Mobile Studio sidebar** — currently falls back to classic Sheet on mobile. The CD shell has a hover-expand (SIDE-C) variant that could double as mobile drawer.
5. **Real ticker metrics** — defaults are pilot snapshot. Wire a metrics hook when the API is online.
6. **Page sub-tabs** — `STUDIO_PAGE_TABS` is exported but not rendered yet. Phase 2 chat / Phase 3 cmd-centre will use it (HEAD-B / HEAD-C patterns).

---

## Visual diff vs. last night's state

When you flip Studio preview tomorrow:

**Before this branch:**
- Sidebar: 64–256px collapsible glass card with custom workspace pill, theme toggle in body, user dropdown at bottom
- Header: `glass-card` strip with workspace selector + theme toggle + profile, full-width pages

**After this branch:**
- Sidebar: 232px cream-paper sidebar with labelled 3-group nav, workspace pill at top, footer Docs/Settings + mini-stats
- Header: 30px paper ticker strip + editorial header (cmdK + utilities) — cleaner, mono-driven
- Brand wordmark in serif at sidebar top, with CSS-drawn sail glyph
- Active route lifts as white paper on cream, accent burnt-orange icon
- Page content area: `max-w-[1720px]`, generous padding, no `gradient-bg` (was the bluish wash)

---

## Risk + rollback

If anything breaks:
- **Per-component**: revert one of the two commits independently
- **Whole shell**: revert `fd83acf` — MainLayout falls back to classic chrome, the new components sit unused but don't render
- **Theme**: existing theme toggle still works — flipping to classic restores the old layout instantly

The new components are gated by `useIsStudio()` so even if the imports fail at module level the classic path is unaffected.

---

## Next sprint candidates

**Same overnight pattern** — say the word and I'll run:

1. **Wire profile dropdown + workspace switcher + cmdK** (~1 day) — close the behavioural gaps so the Studio shell has full functional parity with classic.
2. **Phase 2 chat redesign** (`/chat` 3-column layout per round3/chat.jsx) (~1 week)
3. **Phase 3 command-centre** (ticker is reusable; we already have the bones — add the KPI sparkline cards + Auto narration strip) (~1.5 weeks)
4. **Wire `STUDIO_PAGE_TABS` rendering** for chat + cmd-centre + agents (~1 day)

---

## Branch state

```
fd83acf  feat(studio shell): StudioSidebar + Header + Ticker + MainLayout switch
37eb0d9  feat(studio shell): fonts + chrome CSS + menu config
```

Forked off `feat/studio-rebrand-phase1` (HEAD: 769a14f at fork time). Both branches now share Sprint 0 + 0.5 work.

`feat/studio-shell-chrome` is at origin. No PR opened. Ready for visual review.

Sleep well.
