import { useSyncExternalStore } from 'react'

/**
 * PRD-260 — a browser preference the host's look runtime and the theme SDK follow
 * (lib/theme-host/looks/provider.tsx, lib/theme-sdk/context.ts). It reads as "reduced motion"
 * on the server, so the first paint never starts a loop the user asked not to see.
 * PRD-261 A-19f1: the clip format hook (useVideoFormat) left with builtin Play's mascot.
 */

const REDUCED_MOTION = '(prefers-reduced-motion: reduce)'

const hasMatchMedia = (): boolean => typeof window.matchMedia === 'function'

function subscribeReducedMotion(onChange: () => void): () => void {
  if (!hasMatchMedia()) return () => undefined
  const query = window.matchMedia(REDUCED_MOTION)
  query.addEventListener('change', onChange)
  return () => query.removeEventListener('change', onChange)
}

const reducedMotionNow = (): boolean => hasMatchMedia() && window.matchMedia(REDUCED_MOTION).matches

const reducedMotionOnServer = (): boolean => true

/** True when the user asks for reduced motion (and on the server). */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(subscribeReducedMotion, reducedMotionNow, reducedMotionOnServer)
}
