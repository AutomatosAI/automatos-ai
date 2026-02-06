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

### US-001: Create DESIGN_SYSTEM.md reference document
**Description:** As a developer or AI agent, I need a single in-repo reference document that defines every brand rule so I never have to guess or improvise.

**Acceptance Criteria:**
- [ ] `DESIGN_SYSTEM.md` exists at repo root (or `frontend/` root)
- [ ] Documents all brand colors with hex, HSL, and CSS variable names
- [ ] Documents border-radius rules for every component type
- [ ] Documents button hierarchy (primary, secondary, outline, ghost, destructive)
- [ ] Documents glass effect specifications (blur, border, shadow, glow)
- [ ] Documents modal/dialog styling requirements
- [ ] Documents search bar and input focus states
- [ ] Documents dropdown/select pill-shape requirement
- [ ] Documents page header two-color pattern
- [ ] Documents item state patterns (default, hover, selected)
- [ ] Documents icon library (Lucide React only)
- [ ] Documents spacing scale and typography
- [ ] Documents dark and light theme differences
- [ ] Documents animation/transition standards
- [ ] Includes visual examples or ASCII diagrams where helpful

---

### US-002: Fix dialog/modal to use glass effect with orange glow
**Description:** As a user, I want all popup modals to have the curved edges, orange glow, and glass effect that match the Automatos brand, not the stock shadcn default.

**Acceptance Criteria:**
- [ ] `dialog.tsx` DialogContent uses `glass-card` or equivalent glass styling
- [ ] DialogContent uses `rounded-2xl` border-radius (not `sm:rounded-lg`)
- [ ] DialogContent has orange glow border matching `card-glow` effect
- [ ] Glass backdrop-blur of 18px applied
- [ ] Orange glow visible on the dialog border (subtle, using `--primary` variable)
- [ ] Dark theme: semi-transparent dark background with orange border glow
- [ ] Light theme: frosted glass appearance with visible surface separation
- [ ] Close button (X) uses `rounded-full` for consistency
- [ ] All existing dialog usages across the app inherit new styling automatically
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### US-003: Make all select/dropdown triggers pill-shaped
**Description:** As a user, I want all dropdowns to be pill-shaped to match the brand language shown in the design reference images.

**Acceptance Criteria:**
- [ ] `select.tsx` SelectTrigger uses `rounded-full` (pill shape)
- [ ] SelectContent dropdown panel uses `rounded-2xl` with glass effect
- [ ] SelectItem hover state uses `rounded-xl`
- [ ] All existing select usages across the app inherit new styling automatically
- [ ] Any custom dropdown components also use pill-shaped triggers
- [ ] Focus state shows orange ring (`ring-ring` which maps to `--ring: primary`)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### US-004: Standardize search bar with orange focus highlight
**Description:** As a user, I want all search bars to highlight with an orange border/glow when focused, matching the Automatos brand.

**Acceptance Criteria:**
- [ ] `input.tsx` focus state shows orange border (not just ring)
- [ ] Search inputs use `rounded-full` (pill shape) — distinct from regular text inputs which use `rounded-2xl`
- [ ] On focus: orange border + subtle orange glow shadow (matching reference image)
- [ ] Create a `SearchInput` compound component that wraps Input with search icon + pill shape + orange focus glow
- [ ] All search bar instances across the platform use this component
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### US-005: Standardize page headers with two-color pattern
**Description:** As a user, I want all page headers to follow the pattern: first word in white/foreground, second word in orange (primary), matching "Agent **Management**", "Community **Marketplace**" etc.

**Acceptance Criteria:**
- [ ] Create a `PageHeader` component that accepts `title` (splits into white + orange parts)
- [ ] Component also accepts `description` (muted foreground subtitle)
- [ ] Component accepts optional right-side action buttons slot
- [ ] Orange word uses `gradient-text` class (gradient from #ff6b35 to #ff4500)
- [ ] All existing page headers refactored to use this component
- [ ] Consistent spacing: title is `text-3xl font-bold`, description is `text-sm text-muted-foreground`
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### US-006: Fix button hierarchy — reduce "too much orange"
**Description:** As a developer, I need clear rules for when to use solid orange (primary CTA only) vs orange-outline (secondary actions) so the UI doesn't become a wall of orange.

**Acceptance Criteria:**
- [ ] **Primary** (`default` variant): Solid orange fill — used for ONE main CTA per view (e.g., "Create Agent", "Enable for Workspace")
- [ ] **Secondary action** (`outline` variant): Orange border, transparent/glass background — used for secondary actions (e.g., "Details", "Save Changes", "Refresh")
- [ ] **Tertiary** (`ghost` variant): No border, text only with hover highlight — used for less important actions
- [ ] **Destructive** stays red for delete/dangerous actions
- [ ] Document the hierarchy clearly in `DESIGN_SYSTEM.md`
- [ ] Audit all pages to fix button variants: marketplace cards should have "Details" as outline, "Enable" as primary
- [ ] Save buttons in modal headers should use `outline` variant (per reference image showing ghost-style "Save Changes")
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### US-007: Standardize item states (default, hover, selected)
**Description:** As a user, I want list items, category pills, and grid items to have consistent visual states matching the reference images.

**Acceptance Criteria:**
- [ ] **Default state**: Muted text, subtle border, dark background
- [ ] **Hover state**: Orange border glow, text becomes brighter, slight scale or lift
- [ ] **Selected/Active state**: Solid orange background, white text, counter badge becomes dark
- [ ] Create a reusable `SelectableItem` or apply via CSS classes (`.item-default`, `.item-hover`, `.item-selected`)
- [ ] Category pills in marketplace follow this 3-state pattern
- [ ] Tab-like navigation items follow this pattern
- [ ] Sidebar items follow this pattern
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### US-008: Fix light theme contrast
**Description:** As a user on light theme, I want clear surface separation, readable text, and visible borders so the UI doesn't look washed out.

**Acceptance Criteria:**
- [ ] `--border` in light theme increased from `0 0% 82%` to stronger value (e.g., `0 0% 75%`)
- [ ] `--card` given slight off-white tint or shadow to separate from background
- [ ] Glass card borders are clearly visible in light theme (increase `--glass-border-alpha` for light)
- [ ] Card hover glow is visible in light theme (not invisible against white)
- [ ] All text meets WCAG AA contrast ratio (4.5:1 for normal text, 3:1 for large text)
- [ ] `--muted-foreground` in light theme dark enough for readability (currently `0 0% 40%` — check if sufficient)
- [ ] Orange primary color has sufficient contrast against white card backgrounds
- [ ] Sidebar, header, and navigation clearly distinguished from main content area
- [ ] Test all pages in light theme — no invisible text, borders, or buttons
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### US-009: Replace all hardcoded color values with CSS variables
**Description:** As a developer, I need all colors to flow through CSS variables so theme changes propagate everywhere automatically.

**Acceptance Criteria:**
- [ ] `globals.css`: Replace all `rgba(255, 107, 53, ...)` with `hsla(var(--primary) / ...)`
- [ ] `globals.css`: Replace all `rgba(16, 185, 129, ...)` with semantic variable (create `--success` if needed)
- [ ] `globals.css`: Replace all `rgba(239, 68, 68, ...)` with `hsla(var(--destructive) / ...)`
- [ ] `globals.css`: Replace all `rgba(59, 130, 246, ...)` with semantic variable (create `--info` if needed)
- [ ] `globals.css`: Replace all `rgba(245, 158, 11, ...)` with semantic variable (create `--warning` if needed)
- [ ] `globals.css`: Replace all `rgba(168, 85, 247, ...)` with semantic variable (create `--agent` or `--purple` if needed)
- [ ] `gradient-accent` and `gradient-text` classes use CSS variables instead of hardcoded hex
- [ ] Add missing semantic color variables to `:root` and `.dark`: `--success`, `--warning`, `--info`, `--agent`
- [ ] Sidebar icon colors use CSS variables instead of named Tailwind colors
- [ ] Clerk auth theme uses CSS variables where possible
- [ ] No raw hex or rgba color values remain in `globals.css` (except in the variable definitions themselves)
- [ ] Typecheck passes

---

### US-010: Standardize sub-tabs navigation
**Description:** As a user, I want all sub-navigation tabs (like "General | Persona | Resources | Skills | Plugins | Model | Tools") to use the same pill-shaped tab bar pattern.

**Acceptance Criteria:**
- [ ] `tabs.tsx` TabsList uses pill-shaped container with `rounded-full` border and glass background
- [ ] TabsTrigger active state: solid background with `rounded-full` pill shape
- [ ] TabsTrigger inactive: transparent with hover state
- [ ] Matches the reference image showing the rounded tab bar with "General" selected
- [ ] All tab instances across the app inherit this styling
- [ ] Consistent padding and font size across all tab bars
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### US-011: Standardize modal header pattern
**Description:** As a user, I want all modal headers to follow the same pattern: icon + title + subtitle on the left, action button(s) + close button on the right.

**Acceptance Criteria:**
- [ ] Create a `DialogHeader` pattern/component that includes: left side (icon + title + subtitle), right side (action buttons + close)
- [ ] Title uses the two-color pattern (first word white, second word orange) where applicable
- [ ] "Save Changes" button in modal headers uses `outline` variant with icon
- [ ] Orange accent line/border at top of modal header (per reference image)
- [ ] Close button (X) is circular with ghost styling
- [ ] All existing modals refactored to use this consistent header
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### US-012: Add CSS variable for border-radius scale
**Description:** As a developer, I need a single `--radius` variable that controls the border-radius scale across all components, so changing one value updates everything.

**Acceptance Criteria:**
- [ ] Add `--radius: 1.5rem` to `:root` in `globals.css` (matches current `rounded-2xl` = 1rem... adjust as needed)
- [ ] Define radius scale: `--radius-sm`, `--radius-md`, `--radius-lg`, `--radius-xl`, `--radius-full`
- [ ] All components reference these variables through Tailwind config
- [ ] `tailwind.config.ts` borderRadius section uses these variables
- [ ] Changing `--radius` in one place updates buttons, cards, inputs, modals, tabs, badges
- [ ] Document the radius scale in `DESIGN_SYSTEM.md`
- [ ] Typecheck passes

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

### Implementation Order (Recommended)

1. **US-009** — Replace hardcoded colors (foundation work, no visual changes)
2. **US-012** — Add radius CSS variables (foundation work)
3. **US-008** — Fix light theme contrast (high-impact, variable-only changes)
4. **US-002** — Fix dialogs/modals (high-visibility fix)
5. **US-003** — Fix dropdowns to pill shape (high-visibility fix)
6. **US-004** — Fix search bars (high-visibility fix)
7. **US-010** — Fix sub-tabs (medium visibility)
8. **US-005** — Standardize page headers (medium visibility)
9. **US-006** — Fix button hierarchy (requires page-by-page audit)
10. **US-007** — Standardize item states (requires component-by-component work)
11. **US-011** — Standardize modal headers (requires modal-by-modal refactor)
12. **US-001** — Write DESIGN_SYSTEM.md (captures everything done above)

## Success Metrics

- Zero hardcoded color values outside of CSS variable definitions
- All modals have glass effect + orange glow (visual audit of every dialog)
- All dropdowns are pill-shaped (visual audit of every select)
- Light theme passes WCAG AA contrast check (automated tool verification)
- Page headers all follow two-color pattern (visual audit of all 26 routes)
- `DESIGN_SYSTEM.md` exists and is comprehensive enough for an agent to build a new page without deviating from brand

## Open Questions

1. Should the `DESIGN_SYSTEM.md` be enforced by a linter or pre-commit hook that checks for hardcoded colors?
2. Should we create a `PageHeader` React component or just document the pattern and expect developers to apply the Tailwind classes?
3. The reference images show `rounded-full` (pill) tabs — but the current tabs use `rounded-2xl`. Confirm: should ALL tabs become full-pill, or keep the slightly rounded rectangle?
4. For the two-color header pattern, should it always be "last word = orange" or should we support custom splits (e.g., "Community **Marketplace**" vs "Agent **Configuration** Communication")?
5. Should we add a `--radius-base` variable and derive all others from it, or keep independent radius variables?
