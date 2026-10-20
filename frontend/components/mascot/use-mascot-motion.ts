'use client'

import { useCallback, useSyncExternalStore } from 'react'

import { usePrefersReducedMotion } from '@/hooks/use-media-preferences'
import { isMascotPaused, setMascotPaused, subscribeMascotPause } from '@/lib/mascot/pause'

/**
 * PRD-261 A-13b (F10, WCAG 2.2.2) — whether Auto may move: not under the OS
 * reduced-motion setting, and not while the user has paused him (the Pause
 * Auto control, lib/mascot/pause.ts). On the server he is a still and nothing
 * is paused, so the first paint never starts a loop and never shows a control
 * the browser may not need.
 */

export interface MascotMotion {
  /** The OS asks for reduced motion (and on the server): stills only, and no Pause control. */
  readonly reducedMotion: boolean
  /** The user paused Auto in this browser. */
  readonly paused: boolean
  /** True when Auto stands still for either reason: his still shows and his ring stops. */
  readonly still: boolean
  /** Pause Auto, or play him again, in every tab of this browser. */
  readonly togglePaused: () => void
}

const notPausedOnServer = (): boolean => false

const pausedNow = (): boolean => isMascotPaused()

export function useMascotMotion(): MascotMotion {
  const reducedMotion = usePrefersReducedMotion()
  const paused = useSyncExternalStore(subscribeMascotPause, pausedNow, notPausedOnServer)
  const togglePaused = useCallback(() => setMascotPaused(!isMascotPaused()), [])
  return { reducedMotion, paused, still: reducedMotion || paused, togglePaused }
}
