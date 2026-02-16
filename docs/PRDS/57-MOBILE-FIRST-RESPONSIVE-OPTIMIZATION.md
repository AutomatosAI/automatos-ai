# PRD-57: Mobile-First Responsive Optimization

**Version:** 1.0
**Status:** Planning Phase
**Date:** February 16, 2026
**Author:** Claude Code + Mobile-First Design Skill
**Prerequisites:** None (touches existing components only)
**Blocks:** None

---

## Executive Summary

Automatos AI is a desktop-first application that currently renders on mobile but is **not optimized for mobile use**. The sidebar is always visible (eating 64px), stats bars consume excessive vertical space, card grids don't auto-switch to compact list views, and touch targets are too small. This PRD converts Automatos to a mobile-first responsive app that looks and feels native on phones and tablets while preserving the full desktop experience.

### The Problems (From Real Device Testing)

| Issue | Impact | Screenshot Evidence |
|-------|--------|-------------------|
| Sidebar (`w-16`) always visible on mobile | Loses 64px (18%) of a 360px screen | Chat, Workflows, Marketplace pages |
| Hamburger `<Menu>` button exists but sidebar has no off-canvas mode | Button is misleading — clicks toggle a fixed sidebar, not a drawer | Header component |
| Stats bars (4 metric cards) stack vertically on mobile | Consume 400-500px of vertical space before content starts | Workflow Management, Marketplace |
| Category filter pills wrap into 4+ rows | Pushes content far below fold | Marketplace Agents tab (now fixed — horizontal scroll added) |
| Cards default to grid view on mobile | Grid cards are cramped, list/line view exists but doesn't auto-activate | Agent Roster, Marketplace tabs |
| Chat page has fixed `w-[320px]` docked panel | Overflows on mobile screens | Chat page |
| Jira floating widget overlaps content | Covers bottom-right interactive elements | All pages |
| `p-6` padding on main content | 24px padding is excessive on 360px screens | All pages |
| `backdrop-blur-xl` on sidebar + header | Performance drag on mobile GPUs | Global layout |
| No `useMobile` / `useMediaQuery` hook | Can't programmatically adapt behavior to screen size | Codebase-wide |

### The Solution

A **phased mobile optimization** that:
1. **Hides sidebar on mobile**, replaces with off-canvas sheet triggered by hamburger
2. **Hides stats bars on mobile** — not essential, saves 400px+ vertical space
3. **Auto-switches to list/line view** on mobile for all card-based pages
4. **Creates a `useIsMobile()` hook** for programmatic responsive behavior
5. **Optimizes spacing, touch targets, and performance** for mobile devices
6. **Optional bottom navigation bar** for thumb-friendly mobile navigation

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Breakpoint Strategy](#2-breakpoint-strategy)
3. [Phase 1: Foundation — useMobile Hook & Sidebar](#3-phase-1-foundation)
4. [Phase 2: Layout & Spacing Optimization](#4-phase-2-layout--spacing)
5. [Phase 3: Page-by-Page Mobile Optimization](#5-phase-3-page-by-page)
6. [Phase 4: Touch & Performance](#6-phase-4-touch--performance)
7. [Phase 5: Progressive Enhancement](#7-phase-5-progressive-enhancement)
8. [Component Change Matrix](#8-component-change-matrix)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Testing Strategy](#10-testing-strategy)
11. [Risk Assessment](#11-risk-assessment)

---

## 1. Design Principles

Applying the **mobile-first design skill**:

| Principle | Application |
|-----------|-------------|
| **Start at 320px** | All new CSS begins mobile, enhances upward |
| **Touch targets: 48x48px minimum** | Buttons, nav items, card actions |
| **Relative units** | `rem`, `%`, `vw` — not fixed `px` for layout |
| **Content over chrome** | Hide sidebar, stats, reduce padding on mobile |
| **Progressive enhancement** | Mobile = core experience. Tablet/desktop = enhanced |
| **Performance budget** | Reduce `backdrop-blur` on mobile, lazy-load animations |
| **Thumb zone optimization** | Primary actions in bottom 60% of screen |

### Visual Identity on Mobile

The glass morphism dark theme with orange accents is the brand. On mobile:
- Keep the dark background + glass cards
- **Reduce** `backdrop-blur` from `xl` (24px) to `sm` (4px) on mobile for perf
- Keep orange glow accents but reduce `box-shadow` spread
- Maintain card borders and subtle gradients — they render fine on mobile

---

## 2. Breakpoint Strategy

Using Tailwind's default breakpoints (no custom breakpoints needed):

| Token | Width | Device Class | Layout |
|-------|-------|-------------|--------|
| (base) | 0-639px | Phones | Single column, no sidebar, list view, reduced padding |
| `sm:` | 640px+ | Large phones landscape | Same as base, slightly wider cards |
| `md:` | 768px+ | Tablets portrait | 2-column grids, stats bar visible, sidebar still hidden |
| `lg:` | 1024px+ | Tablets landscape / small laptops | Sidebar visible (collapsed), 3-column grids |
| `xl:` | 1280px+ | Desktops | Full layout as today |

**Key breakpoint:** `lg:` (1024px) is the sidebar visibility threshold. Below `lg:` = mobile/tablet experience.

---

## 3. Phase 1: Foundation

### 3.1 Create `useIsMobile()` Hook

**File:** `frontend/hooks/use-mobile.ts`

```typescript
import { useState, useEffect } from 'react'

const MOBILE_BREAKPOINT = 768  // md breakpoint
const TABLET_BREAKPOINT = 1024 // lg breakpoint

export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`)
    const handler = (e: MediaQueryListEvent | MediaQueryList) => setIsMobile(e.matches)
    handler(mql)
    mql.addEventListener('change', handler)
    return () => mql.removeEventListener('change', handler)
  }, [])

  return isMobile
}

export function useIsTabletOrBelow(): boolean {
  const [isTablet, setIsTablet] = useState(false)

  useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${TABLET_BREAKPOINT - 1}px)`)
    const handler = (e: MediaQueryListEvent | MediaQueryList) => setIsTablet(e.matches)
    handler(mql)
    mql.addEventListener('change', handler)
    return () => mql.removeEventListener('change', handler)
  }, [])

  return isTablet
}
```

### 3.2 Sidebar: Hidden on Mobile, Off-Canvas Sheet

**Current behavior:** Sidebar is `fixed left-0` with `w-16` always. Hamburger toggles `w-16` → `w-64`.

**New behavior:**
- Below `lg:` → Sidebar is completely hidden (`hidden lg:block`)
- Hamburger button triggers a `<Sheet>` (Radix `sheet.tsx` already exists) that slides in from left
- Sheet contains the full expanded sidebar navigation
- Sheet closes on nav link click or overlay tap
- Above `lg:` → Current collapsed sidebar behavior preserved

**Changes to `main-layout.tsx`:**

```tsx
// Pseudocode — actual implementation
export function MainLayout({ children }: MainLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const isDesktop = !useIsTabletOrBelow()

  return (
    <div className="min-h-screen gradient-bg">
      {/* Desktop: existing sidebar */}
      {isDesktop && (
        <Sidebar collapsed={sidebarCollapsed} onToggle={setSidebarCollapsed} />
      )}

      {/* Mobile: off-canvas sheet */}
      {!isDesktop && (
        <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
          <SheetContent side="left" className="w-72 p-0 glass-card">
            <MobileSidebar onNavigate={() => setMobileMenuOpen(false)} />
          </SheetContent>
        </Sheet>
      )}

      {/* Main content — no left margin on mobile */}
      <div className={cn(
        'transition-all duration-300',
        isDesktop ? 'ml-16' : 'ml-0'
      )}>
        <Header onMenuClick={() => {
          isDesktop ? setSidebarCollapsed(!sidebarCollapsed) : setMobileMenuOpen(true)
        }} />
        <main className="p-3 md:p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
```

### 3.3 Header: Mobile-Optimized

**Current:** `px-6 py-4` with full brand lockup (logo + "AUTOMATOS A.I." text)

**Mobile changes:**
- Reduce padding: `px-3 py-2 md:px-6 md:py-4`
- Hamburger button: always visible on mobile (remove `lg:hidden`)
- Brand: hide "AUTOMATOS A.I." text on small screens, show only logo mark
- Keep notifications + profile compact

```tsx
{/* Brand text — hidden on small screens */}
<span className="hidden sm:inline ml-3 text-lg font-semibold tracking-wide">
  AUTOMATOS A.I.
</span>
```

---

## 4. Phase 2: Layout & Spacing

### 4.1 Stats Bar: Hidden on Mobile

**Component:** `frontend/components/shared/stats-bar.tsx`

The 4 metric cards at the top of pages (Workflow Management, Marketplace, etc.) are informational but not essential for mobile task completion.

**Change:**
```tsx
// stats-bar.tsx — add hidden on mobile
<div className={cn(
  'hidden md:grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6',
  className
)}>
```

This hides the entire stats bar below `md:` (768px). On tablets, show 2 columns. On desktop, show 4 columns.

**Pages affected:**
- Workflow Management (`workflow-management.tsx`)
- Community Marketplace (`marketplace-homepage.tsx`)
- Agent Management (`agent-management.tsx`)
- Tools Dashboard (`tools-dashboard.tsx`)
- Analytics pages (`analytics-*.tsx`)

### 4.2 Main Content Padding

**Current:** `p-6` (24px all sides) + `max-w-7xl`

**New:** `p-3 md:p-6` — 12px on mobile, 24px on desktop

### 4.3 Page Header Spacing

**Component:** `frontend/components/shared/page-header.tsx` (if exists) or inline headers

**Changes:**
- Title font size: `text-2xl md:text-3xl lg:text-4xl`
- Description: `text-sm md:text-base`
- Button groups: stack vertically on mobile, horizontal on desktop
- Gap between header and content: `space-y-4 md:space-y-6`

---

## 5. Phase 3: Page-by-Page Optimization

### 5.1 Chat Page (`/chat`)

**Priority: HIGH** — Most used page on mobile

| Current | Mobile Optimized |
|---------|-----------------|
| `h-[calc(100vh-8rem)]` fixed height | `h-[calc(100dvh-3.5rem)]` using `dvh` for mobile address bar |
| Docked sidebar `w-[320px]` | Hidden on mobile, accessible via button/sheet |
| `p-6` message padding | `p-3` message padding |
| Message input at bottom | Sticky bottom, full-width, larger touch targets |

**Key changes:**
- Use `100dvh` (dynamic viewport height) to handle mobile browser chrome
- Chat history sidebar → slide-over sheet on mobile
- Message input: min-height 48px, attachment/model buttons stay compact
- Quick action buttons (Create Agent, Knowledge Base, etc.) → horizontal scroll row

### 5.2 Agent Management (`/agents`)

| Current | Mobile Optimized |
|---------|-----------------|
| Grid view default: `grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4` | **Auto list view on mobile** |
| Stats bar visible | Hidden on mobile |
| ViewToggle in header | Respect mobile default |

**Key change — auto list view on mobile:**

Modify `useViewMode` hook to accept a mobile override:

```typescript
export function useViewMode(pageKey: string, defaultMode: ViewMode = 'grid'): [ViewMode, (m: ViewMode) => void] {
  const isMobile = useIsMobile()

  // On mobile, always default to list. User can still toggle.
  const effectiveDefault = isMobile ? 'list' : defaultMode

  // ... rest of hook, using effectiveDefault
}
```

**Agents already have a line/list view** (`agent-roster.tsx` uses `ViewToggle` + `useViewMode`). The line view shows compact rows: icon + name + status badge + model + tasks + kebab menu. This is perfect for mobile.

### 5.3 Workflow Management (`/workflows`)

| Current | Mobile Optimized |
|---------|-----------------|
| Stats cards (4) consume full viewport | Hidden on mobile |
| Recipe cards in grid | List view on mobile |
| Workflow execution: `grid-cols-1 lg:grid-cols-4` | Already responsive |
| ReactFlow workflow builder | Simplified view / read-only on mobile with pinch-zoom |

**Special handling for ReactFlow:**
- ReactFlow is fundamentally a desktop tool. On mobile:
  - Show workflow as a step list (not canvas)
  - Or show read-only canvas with pinch-to-zoom
  - "Open in Desktop" prompt for editing

### 5.4 Community Marketplace (`/marketplace`)

| Current | Mobile Optimized |
|---------|-----------------|
| Stats bar (Total Items, Categories, Featured, Installs) | Hidden on mobile |
| Tab bar (Applications, Agents, Recipes, LLMs, Capabilities, Skills) | Horizontal scroll (already there) |
| Category pills wrapping 4+ rows | Horizontal scroll (user already fixed) |
| Integration cards in 3-column grid | **Auto list view** |
| Search bar full width | Full width (already works) |

**Marketplace cards in list mode:**
- Logo (32px) + App name + category badge + Tools/Triggers count + install button
- Single row per integration — fits ~8-10 visible on screen vs. 1-2 in grid

### 5.5 Tools & Integrations (`/tools`)

- Stats bar: hidden on mobile
- Tool cards: auto list view
- Category filters: already horizontal scroll

### 5.6 Knowledge Bases (`/documents`)

- Document library: auto list view (already has `ViewToggle`)
- Upload area: full-width, larger drop zone for touch
- Tabs: horizontal scroll if needed

### 5.7 Dashboard (`/dashboard`)

| Current | Mobile Optimized |
|---------|-----------------|
| `grid-cols-1 lg:grid-cols-3` main layout | Already single-col on mobile |
| Metric cards at top | `grid-cols-2` compact on mobile (these ARE the dashboard content) |
| Charts (recharts) | Full-width, touch-friendly tooltips |
| Sidebar cards panel | Stacked below main content (already happens) |

**Dashboard is unique** — stats/metrics ARE the content here, so they stay visible but more compact:
- Metric cards: `grid-cols-2` on mobile (2x2 instead of 1x4)
- Reduce card padding from `p-4` to `p-3`

### 5.8 Analytics (`/analytics`)

- Charts: responsive containers, `aspect-ratio` instead of fixed heights
- Tables: horizontal scroll wrapper
- Stats bars: hidden on mobile

### 5.9 Settings (`/settings`)

- Tab navigation: horizontal scroll
- Form fields: full-width on mobile
- Action buttons: full-width stacked

---

## 6. Phase 4: Touch & Performance

### 6.1 Touch Target Compliance

Per mobile-first-design skill: **minimum 48x48px touch targets**.

| Component | Current | Fix |
|-----------|---------|-----|
| Sidebar nav items | `py-2 px-3` (~36px height) | `min-h-[48px]` on mobile sheet |
| ViewToggle buttons | `h-7 w-7` (28px) | `h-9 w-9 md:h-7 md:w-7` |
| Kebab menu triggers | `w-8 h-8` (32px) | `min-w-[44px] min-h-[44px]` |
| Category filter pills | Variable | `min-h-[40px] px-4` |
| Header action buttons | `size="icon"` (40px) | OK (40px is close enough with padding) |

### 6.2 Performance: Reduce Blur & Animations on Mobile

**`backdrop-blur` is expensive on mobile.** Safari on iOS particularly struggles.

**Changes to `globals.css`:**
```css
/* Reduce glass effects on mobile */
@media (max-width: 767px) {
  .glass-card {
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
  }

  .glass-panel {
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
  }
}

/* Reduce motion for mobile performance */
@media (max-width: 767px) and (prefers-reduced-motion: no-preference) {
  /* Disable framer-motion page transitions on mobile */
  .motion-reduce-mobile * {
    animation-duration: 0.1s !important;
    transition-duration: 0.1s !important;
  }
}
```

**Also consider:**
- Lazy-load below-fold components on mobile
- Reduce `box-shadow` spread on `.card-glow`
- Use `will-change: transform` sparingly

### 6.3 Floating Widget Repositioning

**Jira/Pilot Helper floating widget** currently overlaps bottom-right content.

**On mobile:**
- Move to a fixed bottom bar, or
- Reduce size to 40px circle, or
- Hide behind a "+" FAB that shows both Chat Widget + Jira when tapped
- Ensure it doesn't overlap the message input on chat page

### 6.4 Scroll Behavior

- Category pill bars: `overflow-x-auto` with `-webkit-overflow-scrolling: touch` and `scrollbar-hide` (user already fixed)
- Tables: wrap in `overflow-x-auto` with scroll snap
- Card sections: vertical scroll (natural)

---

## 7. Phase 5: Progressive Enhancement (Optional / Future)

### 7.1 Bottom Navigation Bar (Mobile)

Replace the sidebar with a **fixed bottom tab bar** on mobile for the 5 most-used pages:

```
[ Chat ] [ Agents ] [ Workflows ] [ Marketplace ] [ More... ]
```

- "More..." opens a bottom sheet with remaining nav items
- Active tab has orange indicator
- 56px height, translucent glass background
- Main content gets `pb-16` to account for bottom bar

**This is optional** — the hamburger + sheet approach works fine. Bottom nav is a UX upgrade for power users.

### 7.2 PWA (Progressive Web App)

- Add `manifest.json` with app name, icons, theme color
- Add `<meta name="viewport">` optimization
- Add `<meta name="apple-mobile-web-app-capable">`
- Service worker for offline shell caching
- "Add to Home Screen" prompt

### 7.3 Gesture Support

- Swipe right from left edge → open sidebar sheet
- Pull down on lists → refresh data
- Swipe left on cards → reveal quick actions (delete, edit)

---

## 8. Component Change Matrix

| Component | File | Changes | Priority |
|-----------|------|---------|----------|
| `useIsMobile` hook | `hooks/use-mobile.ts` | **NEW** — media query hook | P0 |
| `MainLayout` | `components/layout/main-layout.tsx` | Hide sidebar on mobile, Sheet nav, reduce padding | P0 |
| `Sidebar` | `components/layout/sidebar.tsx` | Add `hidden lg:block`, create `MobileSidebar` variant | P0 |
| `Header` | `components/layout/header.tsx` | Reduce padding, hide brand text on mobile | P0 |
| `StatsBar` | `components/shared/stats-bar.tsx` | Add `hidden md:grid` | P0 |
| `useViewMode` | `hooks/use-view-mode.ts` | Auto-default to `list` on mobile | P1 |
| `ViewToggle` | `components/shared/view-toggle.tsx` | Larger touch targets on mobile | P1 |
| `globals.css` | `app/globals.css` | Reduce blur on mobile, perf media queries | P1 |
| Chat page | `app/chat/page.tsx` + chatbot components | `100dvh`, sticky input, sheet for history | P1 |
| Agent Roster | `components/agents/agent-roster.tsx` | Auto list view | P1 |
| Marketplace tabs | `components/marketplace/*.tsx` | Auto list view, compact cards | P1 |
| Workflow Management | `components/workflows/workflow-management.tsx` | Hide stats, list view | P2 |
| Tools Dashboard | `components/tools/tools-dashboard.tsx` | Hide stats, list view | P2 |
| Document Library | `components/documents/document-library.tsx` | Auto list view | P2 |
| Dashboard | `components/dashboard/dashboard.tsx` | Compact metric cards | P2 |
| Analytics | `components/analytics/*.tsx` | Responsive charts, scroll tables | P2 |
| ChatWidget | `components/chatbot/chat-widget.tsx` | Reposition/resize on mobile | P3 |
| Bottom Nav | **NEW** `components/layout/mobile-nav.tsx` | Optional bottom tab bar | P3 |

---

## 9. Implementation Roadmap

### Phase 1 — Foundation (Priority: Critical)
**Estimated scope: ~8 components**

1. Create `useIsMobile()` and `useIsTabletOrBelow()` hooks
2. Modify `MainLayout` — conditionally render sidebar vs. sheet
3. Create `MobileSidebar` component (reuses nav items, renders in Sheet)
4. Update `Header` — responsive padding, brand text visibility
5. Update `StatsBar` — `hidden md:grid`
6. Update main content padding — `p-3 md:p-6`
7. Test on iPhone SE (320px), iPhone 14 (390px), iPad mini (768px)

### Phase 2 — Card Views & Content (Priority: High)

1. Modify `useViewMode` to auto-default `list` on mobile
2. Update `ViewToggle` touch targets
3. Verify list/line view renders well on all card pages:
   - Agent Roster
   - Marketplace Applications, Agents, Recipes, LLMs, Skills, Plugins
   - Document Library
   - Workflow Recipes
   - Tools Dashboard
4. Chat page — `dvh` height fix, sticky input, history sheet

### Phase 3 — Polish & Performance (Priority: Medium)

1. `globals.css` — mobile blur reduction
2. Floating widget repositioning
3. Dashboard metric card compact layout
4. Analytics chart responsiveness
5. Table horizontal scroll wrappers
6. Touch target audit across all interactive elements

### Phase 4 — Progressive Enhancement (Priority: Low / Future)

1. Bottom navigation bar (optional)
2. PWA manifest + service worker
3. Gesture support (swipe to open nav)
4. "Open in Desktop" prompts for complex views (ReactFlow)

---

## 10. Testing Strategy

### Device Matrix

| Device | Screen | OS | Priority |
|--------|--------|----|----------|
| iPhone SE | 375x667 | iOS 17+ | P0 |
| iPhone 14/15 | 390x844 | iOS 17+ | P0 |
| iPhone 14 Pro Max | 430x932 | iOS 17+ | P1 |
| Samsung Galaxy S24 | 360x780 | Android 14 | P1 |
| iPad mini | 768x1024 | iPadOS 17 | P1 |
| iPad Air/Pro | 820x1180 | iPadOS 17 | P2 |

### Test Scenarios

1. **Navigation:** Hamburger opens sheet → tap nav link → sheet closes → correct page loads
2. **Sidebar hidden:** No 64px gap on mobile, full-width content
3. **Stats hidden:** No stats bar below 768px
4. **Auto list view:** Cards render as compact list rows on mobile
5. **Chat:** Full viewport height, input stays above keyboard, messages scroll
6. **Touch targets:** All tappable elements pass 48px minimum
7. **Performance:** No jank scrolling, blur effects reduced, animations smooth
8. **Orientation:** Landscape phone still works (no fixed heights breaking)
9. **Keyboard:** Virtual keyboard doesn't hide input fields
10. **Zoom:** Pinch-to-zoom works on all pages, content doesn't break

### Browser Testing

- Safari (iOS) — Primary. Test `dvh`, `-webkit-backdrop-filter`, safe-area-inset
- Chrome (Android) — Secondary
- Firefox mobile — Tertiary

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Desktop regression from mobile-first changes | Medium | High | Use Tailwind responsive prefixes only — base styles don't change existing desktop behavior since current styles already use `md:`, `lg:` prefixes |
| `useIsMobile` SSR mismatch (hydration error) | High | Medium | Default to `false` (desktop) on server, update on mount. Wrap in `useEffect`. |
| Sheet sidebar missing nav items vs. desktop sidebar | Low | Medium | Single `navigationItems` array shared between both components |
| Performance: too many media query listeners | Low | Low | One shared hook instance via context if needed |
| ReactFlow unusable on mobile | High | Medium | Show step-list fallback, don't try to make canvas work on 360px |

---

## Appendix A: Current File References

| File | Path | Relevance |
|------|------|-----------|
| Main Layout | `frontend/components/layout/main-layout.tsx` | Core layout shell |
| Sidebar | `frontend/components/layout/sidebar.tsx` | Fixed sidebar, needs mobile hide |
| Header | `frontend/components/layout/header.tsx` | Hamburger + brand |
| Stats Bar | `frontend/components/shared/stats-bar.tsx` | Shared stats component |
| View Toggle | `frontend/components/shared/view-toggle.tsx` | Grid/list switch |
| useViewMode | `frontend/hooks/use-view-mode.ts` | View mode persistence |
| Global CSS | `frontend/app/globals.css` | Glass morphism, animations |
| Tailwind Config | `frontend/tailwind.config.ts` | Default breakpoints |
| Sheet UI | `frontend/components/ui/sheet.tsx` | Radix sheet (for mobile nav) |
| Drawer UI | `frontend/components/ui/drawer.tsx` | Vaul drawer (alternative) |

## Appendix B: Design References

The app uses a **dark glass morphism** design language:
- Background: deep dark (`hsl(var(--background))`)
- Cards: semi-transparent glass with blur + subtle border gradient
- Accent: orange (`hsl(var(--primary))`) — brand color
- Semantic colors: green (success), blue (info), purple (agent), yellow (warning)
- Typography: system font stack, white/off-white text

On mobile, this aesthetic should be **preserved but lightened** — less blur, fewer shadows, same color palette. The app should feel like a premium mobile companion, not a shrunk desktop app.
