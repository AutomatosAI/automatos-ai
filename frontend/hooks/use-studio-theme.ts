'use client';

import { useEffect, useState } from 'react';
import { useTheme } from 'next-themes';
import { useSearchParams } from 'next/navigation';

const STUDIO_FLAG_VALUE = 'studio-preview';

/**
 * Detects `?theme=studio-preview` in the URL and persists the Studio theme via
 * next-themes. Once flipped on, persists across pages until the user picks a
 * different theme via the menu toggle. PRD §3 / Phase 1 feature flag.
 *
 * Mounted once near the top of the tree (e.g. in Providers). No DOM render.
 */
export function useStudioThemeFlag(): void {
  const params = useSearchParams();
  const { setTheme, theme } = useTheme();

  useEffect(() => {
    const requested = params?.get('theme');
    if (requested === STUDIO_FLAG_VALUE && theme !== 'studio') {
      setTheme('studio');
    }
  }, [params, setTheme, theme]);
}

/**
 * Returns whether the Studio theme is currently active. Use in components that
 * need to branch on theme (rare — most styling should flow via CSS variables).
 */
export function useIsStudio(): boolean {
  const { theme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Avoid hydration mismatch: server render returns false, client decides post-mount.
  if (!mounted) return false;
  return theme === 'studio' || resolvedTheme === 'studio';
}
