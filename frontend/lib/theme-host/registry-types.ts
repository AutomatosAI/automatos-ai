/**
 * PRD-261 A-02a — the types of the Style axis and the theme registry.
 *
 * The axis is Classic, Studio or a theme: `theme:<ref>`, where the ref names a theme compiled
 * into the app (`builtin:<id>`) or one installed in the workspace (`install:<uuid>`). Each theme
 * is one registry entry (lib/theme-host/registry.ts); nothing else in the core names a theme.
 */

/** A theme's name on the axis. An install id is a lowercase uuid. */
export type ThemeRef = `builtin:${string}` | `install:${string}`

export type UiStyle = 'classic' | 'studio' | `theme:${ThemeRef}`

/** The design system a theme is drawn on, where it has no design of its own. */
export type ThemeBase = 'studio' | 'classic'

/** The places a theme may fill, each versioned with its props. */
export type SlotId = 'frame@1' | 'lookEditor@1' | 'styleSheet@1' | 'notFound@1' | 'asleep@1'
  | 'page-header.mascot@1' | 'page-header.art@1' | 'empty-state.mascot@1'

/** Who sets the tone under the theme: its look, the user, or always one tone. */
export type ThemeTonePolicy = 'from-look' | 'user' | 'light' | 'dark'

/** A page of the theme's own, listed in the search dialog while the theme is the style. */
export interface ThemeNavItem { readonly id: string; readonly label: string; readonly path: string }

export interface ThemeEntry {
  readonly ref: ThemeRef; readonly name: string; readonly base: ThemeBase; readonly htmlClass: string
  readonly minWidth: number; readonly tone: ThemeTonePolicy; readonly kind: 'builtin' | 'installed'
  readonly reviewed: boolean; readonly ownsPages: readonly string[]; readonly routes: readonly string[]
  readonly nav: readonly ThemeNavItem[]; readonly slots: readonly SlotId[]
  /**
   * PRD-261 W-A6b (gap G2): an installed theme's id (theme.json `id`), which the host's look store keeps
   * its looks under (installedLookTheme). Builtin entries have none: builtin Play's looks keep their PRD-260 keys.
   */
  readonly themeId?: string
}

/**
 * What a theme's places render: 'theme' (the theme's version), 'fallback' (the core's: no theme,
 * or the window is narrower than the theme's minWidth) or 'pending' (a theme, the width not known
 * yet: on the server and while hydrating).
 */
export type ThemeLayout = 'pending' | 'theme' | 'fallback'
