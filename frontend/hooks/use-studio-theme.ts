'use client';

import { useUiStyleOptional } from '@/contexts/ui-style-context';

/**
 * Whether the Studio style is active. Use in components that need to branch
 * on the design system (the route forks, the chrome); styling flows via CSS.
 *
 * PRD-244 D1: Studio is the Style axis (Classic | Studio), not a tone. The
 * style is known on the server (cookie → app/layout.tsx → UiStyleProvider),
 * so this is stable across SSR and hydration and never flips after mount.
 * PRD-259 §4: a skin built on Studio is Studio here (its base drives the
 * layout; a skin never changes one). Outside the provider (isolated mounts,
 * tests) it is Classic.
 */
export function useIsStudio(): boolean {
  return useUiStyleOptional()?.isStudio === true;
}
