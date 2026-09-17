'use client';

import { useEffect, useState } from 'react';
import { useTheme } from 'next-themes';

/**
 * Returns whether the Studio theme is currently active. Use in components that
 * need to branch on theme (rare — most styling should flow via CSS variables).
 *
 * PRD-244: Studio is chosen in the theme picker only (the URL flag is gone);
 * the remaining branches on this hook are deleted wave by wave (D2, D3), which
 * is what removes the classic-first paint, since next-themes only knows the
 * theme in the browser.
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
