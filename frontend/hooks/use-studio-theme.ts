'use client';

import { useUiStyleOptional } from '@/contexts/ui-style-context';
import { useIsBelowWidth } from '@/hooks/use-below-width';
import type { ThemeEntry, ThemeLayout } from '@/lib/theme-host/registry-types';

/**
 * Whether the Studio layout renders. Use in components that need to branch
 * on the design system (the route forks, the chrome); styling flows via CSS.
 *
 * PRD-244 D1: Studio is the Style axis (Classic | Studio | a theme), not a tone.
 * The style is known on the server (cookie → app/layout.tsx → UiStyleProvider),
 * so this is stable across SSR and hydration and never flips after mount.
 * Outside the provider (isolated mounts, tests) it is Classic.
 *
 * PRD-260 D2: true under a theme drawn on Studio too, so every page without a
 * design of the theme's keeps rendering Studio's version inside its frame.
 */
export function useIsStudio(): boolean {
  return useUiStyleOptional()?.isStudio ?? false;
}

/**
 * PRD-261 A-02a: the active theme's registry entry — its frame and the pages it
 * owns render its version. Null for Classic and Studio, and outside the provider.
 */
export function useActiveTheme(): ThemeEntry | null {
  return useUiStyleOptional()?.activeTheme ?? null;
}

/** The server has no window: not known. */
const WIDTH_ON_SERVER = null;

/**
 * PRD-260 P260-FX-25 (M69), PRD-261 A-02a — the one width source for a theme:
 * MainLayout, ThemeSlot's pages and the 404, the theme-only pages and Conversations
 * all read it. The width comes from useIsBelowWidth (hooks/use-below-width.ts:
 * useSyncExternalStore over matchMedia, the reader useIsTabletOrBelow shares),
 * against the active theme's minWidth, so a mount in the browser knows it on its
 * first render (a width that starts false and is corrected by an effect mounted
 * the theme for one render on phones and tablets: a deep link consumed, the cold
 * start raced, requests fired, 12 posters fetched). On the server and while
 * hydrating it is 'pending', and those places render nothing of the theme until
 * the width is known.
 */
export function useThemeLayout(): ThemeLayout {
  const minWidth = useActiveTheme()?.minWidth ?? null;
  const below = useIsBelowWidth(minWidth, WIDTH_ON_SERVER);
  if (minWidth === null) return 'fallback';
  if (below === null) return 'pending';
  return below ? 'fallback' : 'theme';
}
