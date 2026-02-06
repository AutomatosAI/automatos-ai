# PRD: Automatos AI Platform Design System

## Introduction

The Automatos AI platform has grown organically with 269 components across 26 page routes. While the foundation is solid (CSS variables, glass morphism, Radix UI + shadcn/ui), agents and developers frequently deviate from brand guidelines — inventing colors, using square edges where rounds are expected, creating inconsistent buttons, and breaking the visual language.

This PRD defines a **Design System Enforcement** effort: lock down every reusable component to match the Automatos brand identity, fix known inconsistencies (modals, dropdowns, search bars, light theme contrast), and produce an in-repo living reference document (`DESIGN_SYSTEM.md`) that all future work must follow.

**The core principle: change one CSS variable or component, and every instance across the platform changes with it.**

## Goals

- Enforce a single source of truth for all UI patterns via `globals.css` + component library
- Fix all inconsistent border-radius (modals, selects, dropdowns must match brand)
- Add glass effect + orange glow to all modals/dialogs
- Make all dropdowns and select triggers pill-shaped
- Standardize search bar focus states with orange highlight
- Fix light theme contrast issues (surface separation, border visibility, readability)
- Standardize page headers with the two-color pattern (white + orange)
- Define clear button hierarchy (solid primary CTA vs orange-outline secondary)
- Standardize item states (default, hover, selected) across all list/grid items
- Create an in-repo `DESIGN_SYSTEM.md` that agents/developers must reference
- Eliminate all hardcoded color values in favor of CSS variables
- Ensure WCAG AA contrast compliance in both themes

## User Stories

### US-01: Shared PageHeader component
**Description:** A reusable `PageHeader` component in `components/shared/page-header.tsx` that renders the two-color h1 pattern + subtitle + actions slot.
**Status:** COMPLETE
**Acceptance Criteria:**
- [x] Props: `title`, `titleAccent`, `subtitle?`, `actions?: ReactNode`
- [x] Uses `text-3xl font-bold`, `gradient-text` for accent word
- [x] Includes framer-motion fade-in
- [x] Used by Agents, Tools, Workflows, Marketplace pages

---

### US-02: Shared StatsBar component
**Description:** A reusable `StatsBar` component that renders a responsive grid of stat cards with glass-card + card-glow styling.
**Status:** COMPLETE
**Acceptance Criteria:**
- [x] Props: `stats: StatItem[]`, `loading?`, `glow?`
- [x] StatItem has `label`, `value`, `change?`, `icon: LucideIcon`, `iconColor?`
- [x] Icon colors use semantic tokens (--success, --info, --agent, --primary)
- [x] Used by Agents, Tools, Workflows, Marketplace pages

---

### US-03: Shared SearchInput component
**Description:** A reusable pill-shaped `SearchInput` that wraps Input with Search icon and optional loading spinner.
**Status:** COMPLETE
**Acceptance Criteria:**
- [x] Props: `value`, `onChange(string)`, `placeholder?`, `loading?`
- [x] `rounded-full bg-secondary/50 border-secondary` with orange focus glow
- [x] Used by Agents, Tools, Workflows, Marketplace pages

---

### US-04: Shared StatusBadge component
**Description:** A semantic `StatusBadge` with 8 variants (success, active, warning, error, info, neutral, purple, primary) using CSS variables.
**Status:** COMPLETE
**Acceptance Criteria:**
- [x] Props: `status`, `children`, `dot?`, `size?: 'sm' | 'default'`
- [x] Colors use semantic CSS variables (--success, --info, --warning, --destructive, --agent)
- [x] `rounded-full` with two size options
- [x] Available for use across all pages

---

### US-05: Shared ItemCard component
**Description:** A slot-based card component for grid/list items with glass-card styling.
**Status:** COMPLETE
**Acceptance Criteria:**
- [x] Slot props: `icon`, `title`, `subtitle`, `titleBadges`, `description`, `meta`, `actions`, `children`
- [x] `glass-card card-glow`, `Separator` before actions
- [x] Optional framer-motion animation
- [x] Available for use in card grids

---

### US-06: Shared FilterTabs component
**Description:** A reusable tabs wrapper with consistent styling and trailing slot for search/sort controls.
**Status:** COMPLETE
**Acceptance Criteria:**
- [x] Props: `tabs: FilterTab[]`, `value`, `onValueChange`, `trailing?`, `children`
- [x] `bg-secondary/40 backdrop-blur`, icons + responsive text
- [x] Re-exports TabsContent for convenience
- [x] Used by Agents and Workflows pages

---

### US-07: Migrate Agents page to shared components
**Description:** Replace inline header, stats, search, tabs in `agent-management.tsx` with shared components.
**Status:** COMPLETE
**Acceptance Criteria:**
- [x] Uses `PageHeader`, `StatsBar`, `SearchInput`, `FilterTabs`
- [x] Hardcoded icon colors replaced with semantic tokens
- [x] `bg-brand-primary` replaced with default Button variant
- [x] No new TS errors introduced

---

### US-08: Migrate Tools page to shared components
**Description:** Replace inline header, stats, search in `tools-dashboard.tsx` with shared components.
**Status:** COMPLETE
**Acceptance Criteria:**
- [x] Uses `PageHeader`, `StatsBar`, `SearchInput`
- [x] Hardcoded icon colors replaced with semantic tokens
- [x] No new TS errors introduced

---

### US-09: Migrate Workflows page to shared components
**Description:** Replace inline header, stats, search, tabs in `workflow-management.tsx` with shared components.
**Status:** COMPLETE
**Acceptance Criteria:**
- [x] Uses `PageHeader`, `StatsBar`, `SearchInput`, `FilterTabs`
- [x] Hardcoded icon colors replaced with semantic tokens
- [x] Hardcoded error colors replaced with --destructive variable
- [x] No new TS errors introduced

---

### US-10: Migrate Marketplace page to shared components
**Description:** Replace inline header, stats, search in `marketplace-homepage.tsx` with shared components.
**Status:** COMPLETE
**Acceptance Criteria:**
- [x] Uses `PageHeader`, `StatsBar`, `SearchInput`
- [x] Stats cards use semantic icon colors
- [x] TabsTrigger overrides removed (base component styling is correct)
- [x] No new TS errors introduced

---

### US-11: Migrate all remaining modals/pages
**Description:** Migrate remaining page-level components (marketplace tabs, recipe cards, etc.) to use shared components where applicable.
**Status:** PENDING
**Acceptance Criteria:**
- [ ] Marketplace plugins tab uses StatusBadge for verified/enabled states
- [ ] Recipe cards in recipes-tab.tsx use ItemCard structure
- [ ] Remaining hardcoded colors in page-level components replaced with semantic tokens
- [ ] Native `<button>` elements replaced with Button component

---

### US-12: Light theme contrast validation
**Description:** Validate all pages in light theme for WCAG AA compliance.
**Status:** PENDING
**Acceptance Criteria:**
- [ ] All text meets 4.5:1 contrast ratio (normal text) or 3:1 (large text)
- [ ] Glass card borders are visible in light theme
- [ ] Orange primary has sufficient contrast against white card backgrounds
- [ ] Toggle light/dark theme on all 4 main pages and verify no visual regressions

---

## Functional Requirements

- **FR-1:** All UI components MUST use CSS variables for colors — no hardcoded hex/rgba values outside of variable definitions
- **FR-2:** All modals/dialogs MUST have: `rounded-2xl` border-radius, glass-card background, orange glow border effect
- **FR-3:** All select/dropdown triggers MUST be pill-shaped (`rounded-full`)
- **FR-4:** All search bars MUST highlight with orange border + glow on focus
- **FR-5:** All page headers MUST use the two-color pattern (foreground + primary orange)
- **FR-6:** Only ONE primary (solid orange) CTA button per view; secondary actions use outline variant
- **FR-7:** All interactive list items MUST implement 3 states: default (muted), hover (orange border), selected (solid orange)
- **FR-8:** All sub-navigation tabs MUST use pill-shaped tab bar with glass background
- **FR-9:** All icons MUST come from Lucide React — no other icon libraries
- **FR-10:** Light theme MUST meet WCAG AA contrast ratios for all text and interactive elements
- **FR-11:** Border-radius MUST be controlled via CSS variables / Tailwind config — no arbitrary values in components
- **FR-12:** All animation durations MUST use the standard transition (220ms ease) unless explicitly overridden for a documented reason
- **FR-13:** A `DESIGN_SYSTEM.md` file MUST exist in the repo and be kept current with all brand rules
- **FR-14:** Modal headers MUST follow the standard pattern: icon + two-color title on left, actions + close on right

## Non-Goals (Out of Scope)

- No redesign of the overall layout/navigation architecture
- No new pages or features — this is purely about visual consistency
- No migration to a different CSS framework or component library
- No Figma/design tool deliverables — the codebase IS the design system
- No changes to business logic or API layer
- No new dependencies (keep existing Tailwind + Radix + shadcn stack)
- No Storybook or component playground (may be a future effort)
- No responsive/mobile redesign (separate effort)

## Design Considerations

### Brand Color Palette (Dark Theme — Primary)

| Token | HSL | Hex (approx) | Usage |
|-------|-----|---------------|-------|
| `--primary` | `16 100% 60%` | `#FF6B35` | Brand orange — CTAs, accents, focus rings |
| `--background` | `0 0% 6%` | `#0F0F0F` | App background |
| `--card` | `0 0% 8%` | `#141414` | Card/surface background |
| `--foreground` | `0 0% 98%` | `#FAFAFA` | Primary text |
| `--muted-foreground` | `0 0% 65%` | `#A6A6A6` | Secondary text |
| `--border` | `0 0% 15%` | `#262626` | Borders and dividers |
| `--destructive` | `0 84% 60%` | `#EF4444` | Danger/delete actions |
| `--success` | `160 84% 39%` | `#10B981` | Success states |
| `--warning` | `43 96% 56%` | `#F59E0B` | Warning states |
| `--info` | `217 91% 60%` | `#3B82F6` | Info states |
| `--agent` | `271 91% 65%` | `#A855F7` | Agent-related UI |

### Brand Color Palette (Light Theme)

| Token | HSL | Hex (approx) | Usage |
|-------|-----|---------------|-------|
| `--primary` | `16 100% 50%` | `#FF4D00` | Slightly darker orange for contrast on white |
| `--background` | `0 0% 97%` | `#F7F7F7` | App background (slightly off-white) |
| `--card` | `0 0% 100%` | `#FFFFFF` | Card surfaces |
| `--foreground` | `0 0% 10%` | `#1A1A1A` | Primary text |
| `--muted-foreground` | `0 0% 35%` | `#595959` | Secondary text (darker than current for contrast) |
| `--border` | `0 0% 75%` | `#BFBFBF` | Borders (stronger than current) |

### Border Radius Scale

| Token | Value | Tailwind | Usage |
|-------|-------|----------|-------|
| `--radius-sm` | `0.5rem` | `rounded-lg` | Small elements (checkboxes, small badges) |
| `--radius-md` | `0.75rem` | `rounded-xl` | Medium elements (dropdown items, small buttons) |
| `--radius-lg` | `1rem` | `rounded-2xl` | Standard (buttons, inputs, cards) |
| `--radius-xl` | `1.5rem` | `rounded-3xl` | Large containers (modals, panels) |
| `--radius-full` | `9999px` | `rounded-full` | Pills (badges, search bars, select triggers, tabs) |

### Glass Effect Specification

```css
/* Standard glass card */
background: hsla(var(--card) / var(--glass-card-alpha));
backdrop-filter: blur(18px);
border: 1px solid hsla(var(--primary) / var(--glass-border-alpha));
border-radius: var(--radius-lg);
box-shadow: 0 18px 45px hsla(0 0% 0% / var(--glass-shadow-alpha));

/* Hover: orange glow intensifies */
border-color: hsla(var(--primary) / var(--glass-border-alpha-hover));
box-shadow: 0 0 46px hsla(var(--primary) / var(--glass-glow-alpha));

/* Modal glass: same as card but with stronger glow */
box-shadow: 0 0 60px hsla(var(--primary) / 0.15),
            0 25px 50px hsla(0 0% 0% / 0.5);
```

### Button Hierarchy (Visual Reference)

```
[  Enable  ]  ← Primary: solid orange bg, white text (1 per view)
[ Details  ]  ← Outline: orange border, transparent bg, foreground text
  Refresh    ← Ghost: no border, text only, hover highlights
  Cancel     ← Ghost: no border, muted text
[  Delete  ]  ← Destructive: solid red bg, white text
```

### Item State Pattern

```
DEFAULT:    [ developer tools  36 ]  ← muted text, subtle border
HOVER:      [ developer tools  36 ]  ← orange border glow, brighter text
SELECTED:   [ developer tools  36 ]  ← solid orange bg, white text, dark counter
```

## Technical Considerations

- **Existing stack:** Next.js 14 + React 18 + Tailwind 3.3 + Radix UI + shadcn/ui + CVA
- **All changes must be backwards-compatible** — modifying base components (`button.tsx`, `dialog.tsx`, etc.) will automatically propagate to all usages
- **CSS variable changes in `globals.css`** are the highest-leverage changes — they cascade everywhere
- **Component-level changes** in `components/ui/` cascade to all 269 component files that import them
- **Clerk auth theme** in `providers.tsx` has hardcoded dark theme colors that should be migrated to use CSS variables where the Clerk API allows
- **Framer Motion** transitions should align with the 220ms standard timing
- **No new dependencies required** — everything can be done with the existing stack

### Implementation Order

1. **US-01** — Shared PageHeader component -- COMPLETE
2. **US-02** — Shared StatsBar component -- COMPLETE
3. **US-03** — Shared SearchInput component -- COMPLETE
4. **US-04** — Shared StatusBadge component -- COMPLETE
5. **US-05** — Shared ItemCard component -- COMPLETE
6. **US-06** — Shared FilterTabs component -- COMPLETE
7. **US-07** — Migrate Agents page -- COMPLETE
8. **US-08** — Migrate Tools page -- COMPLETE
9. **US-09** — Migrate Workflows page -- COMPLETE
10. **US-10** — Migrate Marketplace page -- COMPLETE
11. **US-11** — Migrate remaining modals/pages -- PENDING
12. **US-12** — Light theme contrast validation -- PENDING

## Success Metrics

- Zero hardcoded color values outside of CSS variable definitions
- All modals have glass effect + orange glow (visual audit of every dialog)
- All dropdowns are pill-shaped (visual audit of every select)
- Light theme passes WCAG AA contrast check (automated tool verification)
- Page headers all follow two-color pattern (visual audit of all 26 routes)
- `DESIGN_SYSTEM.md` exists and is comprehensive enough for an agent to build a new page without deviating from brand

## Open Questions

1. Should the `DESIGN_SYSTEM.md` be enforced by a linter or pre-commit hook that checks for hardcoded colors?
2. The reference images show `rounded-full` (pill) tabs — but the current tabs use `rounded-2xl`. Confirm: should ALL tabs become full-pill, or keep the slightly rounded rectangle?
3. Should we add a `--radius-base` variable and derive all others from it, or keep independent radius variables?
