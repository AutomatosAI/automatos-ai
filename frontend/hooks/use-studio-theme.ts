'use client';

import { useCallback, useSyncExternalStore } from 'react';

import { useUiStyleOptional } from '@/contexts/ui-style-context';
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

const hasMatchMedia = (): boolean => typeof window.matchMedia === 'function';

/** Narrower than the theme's minWidth. At 1024 px this is the band useIsTabletOrBelow (hooks/use-mobile.ts) reads. */
const belowWidthQuery = (minWidth: number): string => `(max-width: ${minWidth - 1}px)`;

function subscribeToWidth(minWidth: number | null, onChange: () => void): () => void {
  if (minWidth === null || !hasMatchMedia()) return () => undefined;
  const query = window.matchMedia(belowWidthQuery(minWidth));
  query.addEventListener('change', onChange);
  return () => query.removeEventListener('change', onChange);
}

/** Below the theme's width now; a browser that can't say is a wide one, as hooks/use-mobile.ts reads it. */
function belowWidthNow(minWidth: number | null): boolean | null {
  if (minWidth === null) return false;
  return hasMatchMedia() && window.matchMedia(belowWidthQuery(minWidth)).matches;
}

/** The server has no window: not known. */
const widthOnServer = (): boolean | null => null;

/**
 * PRD-260 P260-FX-25 (M69), PRD-261 A-02a — the one width source for a theme:
 * MainLayout, ThemeSlot's pages and the 404, the theme-only pages and Conversations
 * all read it. The width comes from useSyncExternalStore over matchMedia,
 * against the active theme's minWidth, so a mount in the browser knows it on its
 * first render (useIsTabletOrBelow starts false, which mounted the theme for one
 * render on phones and tablets: a deep link consumed, the cold start raced,
 * requests fired, 12 posters fetched). On the server and while hydrating it is
 * 'pending', and those places render nothing of the theme until the width is known.
 */
export function useThemeLayout(): ThemeLayout {
  const minWidth = useActiveTheme()?.minWidth ?? null;
  const subscribe = useCallback((onChange: () => void) => subscribeToWidth(minWidth, onChange), [minWidth]);
  const snapshot = useCallback(() => belowWidthNow(minWidth), [minWidth]);
  const below = useSyncExternalStore(subscribe, snapshot, widthOnServer);
  if (minWidth === null) return 'fallback';
  if (below === null) return 'pending';
  return below ? 'fallback' : 'theme';
}
