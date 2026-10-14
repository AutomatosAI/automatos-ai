import { useSyncExternalStore } from 'react'

/**
 * PRD-260 — browser preferences Play's theme and Auto follow. They read as
 * "reduced motion" on the server, so the first paint never starts a loop
 * the user asked not to see.
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
