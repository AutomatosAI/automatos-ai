# Automatos AI Platform — Design System

> **This is the authoritative brand reference for the Automatos AI platform.**
> Every developer and AI agent MUST follow these guidelines. Do not invent colors, border-radius values, button styles, or component patterns. If it's not documented here, ask before creating it.

---

## Table of Contents

1. [Brand Identity](#1-brand-identity)
2. [Color System](#2-color-system)
3. [Typography](#3-typography)
4. [Spacing & Layout](#4-spacing--layout)
5. [Border Radius](#5-border-radius)
6. [Glass Morphism & Elevation](#6-glass-morphism--elevation)
7. [Buttons](#7-buttons)
8. [Inputs & Search Bars](#8-inputs--search-bars)
9. [Select / Dropdowns](#9-select--dropdowns)
10. [Tabs & Sub-Navigation](#10-tabs--sub-navigation)
11. [Cards](#11-cards)
12. [Modals / Dialogs](#12-modals--dialogs)
13. [Page Headers](#13-page-headers)
14. [Badges & Labels](#14-badges--labels)
15. [Item States](#15-item-states)
16. [Icons](#16-icons)
17. [Animations & Transitions](#17-animations--transitions)
18. [Dark Theme](#18-dark-theme)
19. [Light Theme](#19-light-theme)
20. [Do's and Don'ts](#20-dos-and-donts)

---

## 1. Brand Identity

**Automatos AI** is a premium AI orchestration platform. The visual language communicates:
- **Sophistication** — dark surfaces, glass effects, subtle glows
- **Energy** — orange accents that draw attention without overwhelming
- **Clarity** — clean typography, generous spacing, obvious hierarchy
- **Consistency** — every screen feels like the same product

**Logo:** `/public/brand/automatos-mark-hi.png`
**Font:** System font stack (no custom web fonts)
**Icon Library:** Lucide React (exclusively — no other icon libraries)

---

## 2. Color System

All colors are defined as CSS custom properties in `globals.css` using HSL format.
**NEVER use hardcoded hex, rgb, or rgba values in components.** Always reference the CSS variable.

### Core Brand Colors

| Token | Dark Theme | Light Theme | Usage |
|-------|-----------|-------------|-------|
| `--primary` | `16 100% 60%` | `16 100% 50%` | Brand orange. CTAs, focus rings, accents |
| `--background` | `0 0% 6%` | `0 0% 97%` | Page background |
| `--foreground` | `0 0% 98%` | `0 0% 10%` | Primary text |
| `--card` | `0 0% 8%` | `0 0% 100%` | Card / surface background |
| `--card-foreground` | `0 0% 98%` | `0 0% 10%` | Text on cards |
| `--secondary` | `0 0% 12%` | `0 0% 96%` | Secondary surfaces (tab bars, subtle fills) |
| `--muted` | `0 0% 12%` | `0 0% 95%` | Muted backgrounds |
| `--muted-foreground` | `0 0% 65%` | `0 0% 35%` | Secondary / helper text |
| `--border` | `0 0% 15%` | `0 0% 75%` | Borders and dividers |
| `--input` | `0 0% 15%` | `0 0% 75%` | Input borders |
| `--ring` | `16 100% 60%` | `16 100% 50%` | Focus ring (orange) |

### Semantic Colors

| Token | Value | Usage |
|-------|-------|-------|
| `--destructive` | `0 84% 60%` | Delete, error, danger |
| `--success` | `160 84% 39%` | Success states, completed |
| `--warning` | `43 96% 56%` | Warnings, caution |
| `--info` | `217 91% 60%` | Informational |
| `--agent` | `271 91% 65%` | Agent-related UI elements |

### Using Colors in Components

```tsx
// CORRECT — uses CSS variable via Tailwind
<div className="bg-primary text-primary-foreground" />
<div className="border-border hover:border-primary/30" />
<div className="text-muted-foreground" />

// WRONG — hardcoded values
<div style={{ color: '#FF6B35' }} />
<div className="text-orange-400" />
<div style={{ background: 'rgba(255, 107, 53, 0.3)' }} />
```

### Using Color with Opacity

```css
/* CORRECT */
background: hsla(var(--primary) / 0.2);
border-color: hsla(var(--border) / 0.5);

/* WRONG */
background: rgba(255, 107, 53, 0.2);
```

### Gradient Accent

The brand gradient is used for text highlights and accent fills:
```css
/* Defined in globals.css as .gradient-text and .gradient-accent */
background: linear-gradient(135deg, hsl(var(--primary)) 0%, hsl(var(--primary) / 0.8) 100%);
```

---

## 3. Typography

### Font Stack
```css
font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
```

### Monospace (logs, code)
```css
font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Monaco, monospace;
```

### Scale

| Element | Tailwind Class | Size |
|---------|---------------|------|
| Page title | `text-3xl font-bold` | 1.875rem / 30px |
| Section heading | `text-2xl font-semibold` | 1.5rem / 24px |
| Card title | `text-lg font-semibold` | 1.125rem / 18px |
| Body text | `text-sm` | 0.875rem / 14px |
| Helper / muted | `text-xs text-muted-foreground` | 0.75rem / 12px |
| Badge text | `text-xs font-semibold` | 0.75rem / 12px |
| Button text | `text-sm font-medium` | 0.875rem / 14px |

---

## 4. Spacing & Layout

Use the Tailwind spacing scale. Standard spacing values:

| Context | Value | Tailwind |
|---------|-------|----------|
| Inline spacing (between icon & text) | 0.5rem | `gap-2` |
| Between form fields | 1rem | `gap-4` or `space-y-4` |
| Card padding | 1.5rem | `p-6` |
| Section spacing | 2rem | `gap-8` or `space-y-8` |
| Page margin | 1.5rem | `p-6` |

### Layout Grid
- Sidebar: fixed width (collapsed ~64px, expanded ~240px)
- Main content: fluid, max-width container
- Cards in grids: `grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6`

---

## 5. Border Radius

**Everything in Automatos is rounded. No sharp corners.**

| Element | Radius | Tailwind | CSS Variable |
|---------|--------|----------|-------------|
| Badges, pills, search bars, select triggers, tabs | Full pill | `rounded-full` | `--radius-full: 9999px` |
| Buttons (default) | Extra large | `rounded-2xl` | `--radius-lg: 1rem` |
| Cards, modals, panels | Extra large | `rounded-2xl` | `--radius-lg: 1rem` |
| Inputs, textareas | Extra large | `rounded-2xl` | `--radius-lg: 1rem` |
| Dropdown menu content | Large | `rounded-xl` | `--radius-md: 0.75rem` |
| Dropdown menu items | Medium | `rounded-lg` | `--radius-sm: 0.5rem` |
| Checkboxes, small controls | Small | `rounded-md` | `calc(var(--radius-sm) - 2px)` |
| Avatars | Full circle | `rounded-full` | — |

**Rule: When in doubt, use `rounded-2xl`. Never use sharp corners (`rounded-none` or `rounded-sm`).**

---

## 6. Glass Morphism & Elevation

Glass is the signature Automatos visual effect. Use it on **cards, modals, panels, and floating elements**.

### Glass Card (Standard)
```css
.glass-card {
  background: hsla(var(--card) / var(--glass-card-alpha));
  backdrop-filter: blur(18px);
  border: 1px solid hsla(var(--glass-border) / var(--glass-border-alpha));
  border-radius: 1.5rem;
  box-shadow: 0 18px 45px hsla(0 0% 0% / var(--glass-shadow-alpha)),
              inset 0 1px 0 hsla(0 0% 100% / var(--card-highlight-alpha));
}
```

### Glass Card Hover
```css
.glass-card:hover {
  border-color: hsla(var(--glass-border-hover) / var(--glass-border-alpha-hover));
  box-shadow: 0 0 46px hsla(var(--primary) / var(--glass-glow-alpha)),
              0 18px 45px hsla(0 0% 0% / var(--glass-shadow-alpha-hover));
}
```

### Orange Glow (for modals and important elements)
```css
.card-glow {
  box-shadow: 0 18px 45px hsla(0 0% 0% / var(--card-shadow-alpha)),
              0 0 0 1px hsla(var(--primary) / var(--card-accent-alpha)),
              0 0 24px hsla(var(--primary) / var(--card-accent-alpha));
}
```

### Glass Tuning Variables

| Variable | Dark | Light | Purpose |
|----------|------|-------|---------|
| `--glass-card-alpha` | `0.55` | `0.86` | Card background transparency |
| `--glass-panel-alpha` | `0.92` | `0.95` | Panel background transparency |
| `--glass-border-alpha` | `0.12` | `0.18` | Border opacity |
| `--glass-border-alpha-hover` | `0.22` | `0.20` | Border opacity on hover |
| `--glass-glow-alpha` | `0.16` | `0.10` | Orange glow intensity |
| `--glass-shadow-alpha` | `0.45` | `0.10` | Drop shadow intensity |

---

## 7. Buttons

Buttons use the `Button` component from `components/ui/button.tsx` with CVA variants.

### Hierarchy (Most Important → Least Important)

| Variant | Appearance | When to Use |
|---------|-----------|-------------|
| `default` (Primary) | Solid orange bg, white text | **ONE main CTA per view.** "Create Agent", "Enable", "Submit" |
| `outline` (Secondary) | Orange border, glass bg, foreground text | Secondary actions. "Details", "Save Changes", "Refresh", "Export" |
| `ghost` (Tertiary) | No border, hover highlight | Toolbar actions, less important. "Cancel", icon buttons |
| `destructive` | Solid red bg, white text | Dangerous actions only. "Delete", "Remove" |
| `link` | Underlined text | Inline navigation links |
| `secondary` | Muted bg, foreground text | Neutral actions that don't need orange |

### Size Scale

| Size | Class | Height | Radius | Usage |
|------|-------|--------|--------|-------|
| `sm` | `size="sm"` | 36px (h-9) | `rounded-xl` | Compact areas, table rows |
| `default` | (default) | 40px (h-10) | `rounded-2xl` | Standard buttons |
| `lg` | `size="lg"` | 44px (h-11) | `rounded-xl` | Hero CTAs, important actions |
| `icon` | `size="icon"` | 40x40px | `rounded-2xl` | Icon-only buttons |

### Rules
- **Maximum ONE solid orange (primary) button per view/section**
- "Save Changes" in modal headers = `outline` variant
- "Details" on cards = `outline` variant
- "Enable" / "Create" = `default` (primary) variant
- Always include `transition-colors` (built into the base class)
- Disabled state reduces opacity to 50%

---

## 8. Inputs & Search Bars

### Standard Input
```tsx
// Uses rounded-2xl, glass background, orange focus ring
<Input placeholder="Enter value..." />
```
- Border: `border-border/50`
- Background: `bg-background/50` with `backdrop-blur`
- Focus: orange ring (`ring-ring`) + orange border hint (`border-primary/30`)
- Radius: `rounded-2xl`

### Search Bar (Pill Shape)
Search bars are **pill-shaped** and have a stronger orange focus state:
```tsx
// SearchInput component — pill shape with orange glow on focus
<SearchInput placeholder="Search recipes..." />
```
- Shape: `rounded-full` (full pill)
- Focus: orange border + orange glow shadow
- Icon: `Search` from Lucide React, positioned inside on the left
- All search bars across the platform MUST use this same pattern

---

## 9. Select / Dropdowns

### Trigger
- Shape: **pill-shaped** (`rounded-full`)
- Border: `border-border/50`
- Background: glass effect (`bg-background/50 backdrop-blur`)
- Focus: orange ring
- Chevron icon on right side

### Dropdown Content
- Shape: `rounded-xl` with glass background
- Border: subtle glass border
- Items: `rounded-lg` with hover highlight
- Animation: zoom-in/fade-in on open

### Rules
- ALL dropdowns/selects in the platform must be pill-shaped triggers
- Never use square or slightly-rounded select triggers
- The dropdown content panel can be slightly less rounded than the trigger

---

## 10. Tabs & Sub-Navigation

### Tab Bar Container
- Shape: `rounded-full` (full pill) — the entire tab bar is a pill
- Background: `bg-secondary/40` with `backdrop-blur` glass effect
- Border: `border-border/40`
- Padding: `p-1` internal

### Tab Trigger (Individual Tab)
- Shape: `rounded-full` (pill within the pill)
- **Active:** solid background (`bg-background`), foreground text, subtle shadow, visible border
- **Inactive:** transparent, muted text
- **Hover:** slight background highlight (`bg-background/60`)
- Font: `text-sm font-medium`
- Each tab includes an icon (from Lucide) + label text

### Rules
- All tab-bar navigation must use this pill-in-pill pattern
- The active tab should be clearly distinguishable (elevated, bordered)
- Tabs must be used for in-page navigation, not for filtering (use category pills for filtering)

---

## 11. Cards

All cards use the `Card` component which applies the `glass-card` class.

### Structure
```tsx
<Card>
  <CardHeader>
    <CardTitle>Title</CardTitle>
    <CardDescription>Subtitle text</CardDescription>
  </CardHeader>
  <CardContent>
    {/* Main content */}
  </CardContent>
  <CardFooter>
    {/* Action buttons */}
  </CardFooter>
</Card>
```

### Styling
- Background: glass effect (semi-transparent + blur)
- Border: subtle, glows orange on hover
- Radius: `rounded-2xl`
- Shadow: elevation shadow + inset highlight
- Hover: orange glow intensifies, border becomes orange-tinted
- Padding: `p-6` for header, content, footer

### Rules
- **Never create a card without using the `Card` component**
- Never override the glass-card styling with inline styles
- Card grids: `gap-6` between cards
- Cards in listings should all be the same height (use flex/grid alignment)

---

## 12. Modals / Dialogs

Modals are the **highest-elevation** element. They MUST have the strongest glass + glow effect.

### Required Styling
- Radius: `rounded-2xl` (not `sm:rounded-lg`)
- Background: glass card effect (`glass-card` class or equivalent)
- Border: orange-tinted glow (using `card-glow` effect)
- Shadow: strong drop shadow + orange glow
- Overlay: `bg-black/80` backdrop
- Top accent: subtle orange gradient line at top of modal header

### Header Pattern
```
┌─── orange accent line ──────────────────────────────┐
│  🔧 Agent Configuration                  [Save] [X] │
│     Communication                                    │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Modal content here...                               │
│                                                      │
└──────────────────────────────────────────────────────┘
```

- Left: icon + title (two-color: white + orange) + subtitle
- Right: action button(s) (`outline` variant) + close button (`ghost`, circular)

### Rules
- **NEVER use the stock shadcn dialog styling** — always apply glass + glow
- All modals must have `rounded-2xl`
- Close button must be circular (`rounded-full`)
- Modal max-width: `max-w-lg` (small), `max-w-2xl` (medium), `max-w-4xl` (large)

---

## 13. Page Headers

Every page in the platform has a header following the **two-color pattern**.

### Pattern
```
First Word(s)  OrangeWord
Subtitle description text in muted foreground

                                    [Status] [Action] [+ Create]
```

### Rules
- Title: `text-3xl font-bold`
- First word(s): `text-foreground` (white in dark, black in light)
- Last word: `text-transparent bg-clip-text gradient-text` (orange gradient)
- Subtitle: `text-sm text-muted-foreground`
- Right side: status badges + action buttons (outline variant) + primary CTA

### Examples
- **Agent** <span style="color:orange">**Management**</span>
- **Community** <span style="color:orange">**Marketplace**</span>
- **Workflow** <span style="color:orange">**Builder**</span>
- **Agent** <span style="color:orange">**Configuration**</span>

---

## 14. Badges & Labels

Badges are **always pill-shaped** (`rounded-full`).

### Variants
| Variant | Appearance | Usage |
|---------|-----------|-------|
| `default` | Orange bg, white text | Active/enabled states |
| `secondary` | Muted bg, foreground text | Neutral info, counts |
| `destructive` | Red bg, white text | Errors, failed states |
| `outline` | Border only, foreground text | Tags, categories |

### Status Badges
| Status | Color | Example |
|--------|-------|---------|
| Verified / Safe | Green (`--success`) | ✓ Verified Safe |
| Enabled | Orange (`--primary`) | Enabled |
| Pending | Yellow (`--warning`) | Pending |
| Error / Failed | Red (`--destructive`) | Failed |
| Connected | Green | ● Connected |
| Disabled | Muted | Disabled |

### Rules
- Always `rounded-full` — never square badges
- Small: `text-xs px-2.5 py-0.5`
- Always use the `Badge` component, never create custom badge styling

---

## 15. Item States

Interactive list items, category pills, and selectable grid items follow a **3-state pattern**.

### Default State
- Background: transparent or `bg-secondary/20`
- Text: `text-muted-foreground`
- Border: `border-border/30` (subtle)

### Hover State
- Background: slight lift
- Text: `text-foreground` (brighter)
- Border: `border-primary/40` (orange glow)
- Optional: subtle orange box-shadow

### Selected / Active State
- Background: `bg-primary` (solid orange)
- Text: `text-primary-foreground` (white)
- Border: `border-primary`
- Counter badges inside become `bg-primary-foreground/20` (dark on orange)

### Rules
- Transitions between states: 220ms ease
- All three states must be visually distinct
- Selected state must be obviously different from hover (solid fill vs glow)

---

## 16. Icons

### Library
**Lucide React** (`lucide-react`) — this is the ONLY icon library.

```tsx
import { Search, Settings, Plus, X, ChevronDown } from 'lucide-react';
```

### Sizes
| Context | Size | Tailwind |
|---------|------|----------|
| Inline with text | 16px | `h-4 w-4` |
| Button icon | 16px | `h-4 w-4` |
| Card header icon | 20px | `h-5 w-5` |
| Feature icon | 24px | `h-6 w-6` |
| Large decorative | 32-48px | `h-8 w-8` to `h-12 w-12` |

### Icon Colors
- Default: `text-muted-foreground` (inherits)
- Active/accent: `text-primary` (orange)
- On colored backgrounds: `text-primary-foreground` (white)
- Semantic: use `text-green-500`, `text-red-500` etc. for status only via semantic tokens

### Rules
- **NEVER import icons from other libraries** (no Font Awesome, Heroicons, etc.)
- Use `strokeWidth={2}` (Lucide default) — don't vary stroke width
- Pair icons with text labels when possible for accessibility

---

## 17. Animations & Transitions

### Standard Transition
```css
transition: all 220ms ease;
/* or specific properties */
transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
```

### Tailwind Shorthand
```tsx
className="transition-colors"    // color changes
className="transition-all"       // everything
```

### Named Animations (defined in globals.css / tailwind.config.ts)

| Animation | Duration | Usage |
|-----------|----------|-------|
| `fade-in` | 500ms | Elements appearing on page load |
| `slide-in` | 300ms | Elements sliding in from left |
| `pulse-glow` | 2s infinite | Pulsing orange glow on active items |
| `stage-pulse` | 2s infinite | Active workflow stages |
| `avatar-pulse` | 2s infinite | Active agent avatars |
| `shimmer` | 2s infinite | Progress bar loading state |

### Rules
- Standard transitions: 220ms ease
- Respect `prefers-reduced-motion` — all animations are disabled for users who prefer reduced motion
- Never add new animation keyframes without documenting them here
- Hover transitions should feel snappy (220ms), not sluggish

---

## 18. Dark Theme

Dark theme is the **primary/flagship** experience.

### Key Properties
- Background: very dark (`0 0% 6%` ≈ `#0F0F0F`)
- Cards: slightly lighter (`0 0% 8%` ≈ `#141414`)
- Text: near-white (`0 0% 98%`)
- Borders: subtle (`0 0% 15%`)
- Glass effects: more transparent card backgrounds, stronger shadows
- Orange glow: more prominent against dark surfaces

### Dark Theme Glass Tuning
- Cards are more transparent (alpha `0.55`)
- Shadows are deeper (alpha `0.45`)
- Orange glow is more visible (alpha `0.16`)
- This creates the signature "floating glass" look

---

## 19. Light Theme

Light theme must be clean, high-contrast, and clearly readable.

### Key Properties
- Background: off-white (`0 0% 97%` — NOT pure white)
- Cards: white (`0 0% 100%`)
- Text: near-black (`0 0% 10%`)
- Borders: **stronger** than dark theme (`0 0% 75%`) for clear surface separation
- Muted text: darkened (`0 0% 35%`) for WCAG AA compliance

### Light Theme Glass Tuning
- Cards are more opaque (alpha `0.86`)
- Shadows are softer (alpha `0.10`)
- Orange glow is more subtle (alpha `0.10`)
- Borders compensate — clearer line between surfaces

### Contrast Requirements (WCAG AA)
- Normal text (< 18px): minimum 4.5:1 contrast ratio
- Large text (>= 18px bold or >= 24px): minimum 3:1 contrast ratio
- Interactive elements: minimum 3:1 against adjacent colors
- **Test every component in light theme** — if it disappears or becomes unreadable, the contrast is wrong

---

## 20. Do's and Don'ts

### DO
- Use CSS variables for ALL colors
- Use the `Button`, `Card`, `Input`, `Select`, `Badge`, `Dialog` components from `components/ui/`
- Use `rounded-2xl` as the default radius
- Use `rounded-full` for pills, badges, search bars, select triggers, tabs
- Use `glass-card` for card backgrounds
- Use Lucide React for all icons
- Follow the button hierarchy (one primary CTA per view)
- Test both dark AND light themes
- Follow the two-color page header pattern

### DON'T
- Hardcode hex/rgb/rgba colors in components
- Use `rounded-sm`, `rounded-md`, or `rounded-none` (unless for very small controls)
- Create custom button styling outside the Button component
- Use icons from libraries other than Lucide React
- Build modals without glass effect + orange glow
- Use square select/dropdown triggers
- Put more than one solid orange button per view section
- Skip the focus state on interactive elements
- Use `!important` to override design system styles
- Create one-off component styles that don't match these guidelines

---

## File Reference

| File | Purpose |
|------|---------|
| `app/globals.css` | Master stylesheet — CSS variables, glass effects, animations |
| `tailwind.config.ts` | Tailwind theme extension — colors, radius, animations |
| `components/ui/button.tsx` | Button component with CVA variants |
| `components/ui/card.tsx` | Card component with glass-card styling |
| `components/ui/dialog.tsx` | Modal/dialog component |
| `components/ui/input.tsx` | Text input component |
| `components/ui/select.tsx` | Select/dropdown component |
| `components/ui/tabs.tsx` | Tabs navigation component |
| `components/ui/badge.tsx` | Badge/label component |
| `components/theme-provider.tsx` | Dark/light theme provider |
| `components/providers.tsx` | Root providers including Clerk theme |

---

*Last updated: 2026-02-06*
*Maintained by: Automatos AI Platform Team*
