'use client';

import { useCallback, useSyncExternalStore } from 'react';

/**
 * PRD-260 P260-FX-25 (M69) — "is the window narrower than this width", known on a browser
 * mount's first render: useSyncExternalStore over matchMedia, never a state that starts at a
 * guess and is corrected by an effect. The one reader behind useThemeLayout
 * (hooks/use-studio-theme.ts, a theme's minWidth) and useIsTabletOrBelow (hooks/use-mobile.ts,
 * 1024 px), so the two ask the same query and agree on each side of a width.
 *
 * The server has no window: there, and while hydrating, the answer is the caller's `onServer`.
 */

const hasMatchMedia = (): boolean => typeof window.matchMedia === 'function';

/** Narrower than `widthPx`. */
const belowWidthQuery = (widthPx: number): string => `(max-width: ${widthPx - 1}px)`;

function subscribeToWidth(widthPx: number | null, onChange: () => void): () => void {
  if (widthPx === null || !hasMatchMedia()) return () => undefined;
  const query = window.matchMedia(belowWidthQuery(widthPx));
  query.addEventListener('change', onChange);
  return () => query.removeEventListener('change', onChange);
}

/** Below the width now; no width is never below it, and a browser that can't say is a wide one. */
function belowWidthNow(widthPx: number | null): boolean {
  if (widthPx === null) return false;
  return hasMatchMedia() && window.matchMedia(belowWidthQuery(widthPx)).matches;
}

/**
 * Whether the window is narrower than `widthPx` (null: no width, never narrower), following the
 * window as it crosses it. `onServer` is what the server renders and hydration reads.
 */
export function useIsBelowWidth<S extends boolean | null>(widthPx: number | null, onServer: S): boolean | S {
  const subscribe = useCallback((onChange: () => void) => subscribeToWidth(widthPx, onChange), [widthPx]);
  const snapshot = useCallback(() => belowWidthNow(widthPx), [widthPx]);
  const serverSnapshot = useCallback(() => onServer, [onServer]);
  return useSyncExternalStore<boolean | S>(subscribe, snapshot, serverSnapshot);
}
