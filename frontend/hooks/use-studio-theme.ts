'use client';

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
