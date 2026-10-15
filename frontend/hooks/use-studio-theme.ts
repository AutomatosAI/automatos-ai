'use client';

import { useSyncExternalStore } from 'react';

import { useUiStyleOptional } from '@/contexts/ui-style-context';

/**
 * Whether the Studio layout renders. Use in components that need to branch
 * on the design system (the route forks, the chrome); styling flows via CSS.
 *
 * PRD-244 D1: Studio is the Style axis (Classic | Studio | Play), not a tone.
 * The style is known on the server (cookie → app/layout.tsx → UiStyleProvider),
 * so this is stable across SSR and hydration and never flips after mount.
 * Outside the provider (isolated mounts, tests) it is Classic.
 *
 * PRD-260 D2: true under Play too, so every page without a Play design keeps
 * rendering Studio's version inside Play's frame.
 */
export function useIsStudio(): boolean {
  return useUiStyleOptional()?.isStudio ?? false;
}

/**
 * PRD-260 D2: whether Play is the style — Play's frame and the Play-designed
 * pages render Play's version. False outside the provider, like useIsStudio.
 */
export function useIsPlay(): boolean {
  return useUiStyleOptional()?.isPlay ?? false;
}

/**
 * PRD-260 D7: Play's frame and pages render from this width up; below it, the
 * fallback, as for Studio. The band useIsTabletOrBelow (hooks/use-mobile.ts)
 * reads, so Play's switch and the fallback chrome agree.
 */
export const PLAY_MIN_WIDTH_PX = 1024;
const BELOW_PLAY_WIDTH = `(max-width: ${PLAY_MIN_WIDTH_PX - 1}px)`;

/**
 * What Play's places render: 'play' (Play's version), 'fallback' (the core's:
 * not Play, or Play below PLAY_MIN_WIDTH_PX) or 'pending' (Play, the width not
 * known yet: on the server and while hydrating).
 */
export type PlayLayout = 'pending' | 'play' | 'fallback';

const hasMatchMedia = (): boolean => typeof window.matchMedia === 'function';

function subscribeToWidth(onChange: () => void): () => void {
  if (!hasMatchMedia()) return () => undefined;
  const query = window.matchMedia(BELOW_PLAY_WIDTH);
  query.addEventListener('change', onChange);
  return () => query.removeEventListener('change', onChange);
}

/** Below Play's width now; a browser that can't say is a wide one, as hooks/use-mobile.ts reads it. */
const belowPlayWidthNow = (): boolean | null => hasMatchMedia() && window.matchMedia(BELOW_PLAY_WIDTH).matches;

/** The server has no window: not known. */
const widthOnServer = (): boolean | null => null;

/**
 * PRD-260 P260-FX-25 (M69) — the one width source for Play: MainLayout, the
 * route switch, the 404, the Play-only pages and Conversations all read it.
 * The width comes from useSyncExternalStore over matchMedia, so a mount in the
 * browser knows it on its first render (useIsTabletOrBelow starts false, which
 * mounted Play for one render on phones and tablets: a deep link consumed, the
 * cold start raced, requests fired, 12 posters fetched). On the server and while
 * hydrating it is 'pending', and those places render nothing of Play until the
 * width is known.
 */
export function usePlayLayout(): PlayLayout {
  const isPlay = useIsPlay();
  const below = useSyncExternalStore(subscribeToWidth, belowPlayWidthNow, widthOnServer);
  if (!isPlay) return 'fallback';
  if (below === null) return 'pending';
  return below ? 'fallback' : 'play';
}
