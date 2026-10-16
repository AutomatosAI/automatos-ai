import { useSyncExternalStore } from 'react'

import { videoFormat, type VideoFormat } from '@/lib/mascot/media'

/**
 * PRD-260 — browser preferences Play's theme and Auto follow. They read as
 * "reduced motion" and "no video" on the server, so the first paint never
 * starts a loop the user asked not to see: Auto's still shows first and his
 * clip starts after hydration.
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

const noSubscription = (): (() => void) => () => undefined

const formatNow = (): VideoFormat => videoFormat(navigator.userAgent, navigator.maxTouchPoints ?? 0)

const formatOnServer = (): VideoFormat | null => null

/** The clip format for this browser (lib/mascot/media.ts); null on the server. */
export function useVideoFormat(): VideoFormat | null {
  return useSyncExternalStore(noSubscription, formatNow, formatOnServer)
}
