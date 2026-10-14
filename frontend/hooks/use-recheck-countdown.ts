'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * PRD-260 §4 — the asleep screen's automatic recheck (Ember mock 07), ported
 * from Automatos Play's useBackendStatus, whose timer asked again every 10
 * seconds while the backend was asleep. Here the screen counts the seconds
 * down, calls `check` when they run out, and starts over; Check again calls it
 * at once and starts the count over too. The timer lives only while the
 * screen that uses it is mounted.
 */

export const SECOND_MS = 1000

export interface RecheckCountdown {
  /** Whole seconds until the next automatic check; 0 while one is being asked. */
  readonly remaining: number
  /** Check again now, and start the count over. */
  readonly checkNow: () => void
}

export function useRecheckCountdown(periodSeconds: number, check: () => void): RecheckCountdown {
  const checkRef = useRef(check)
  const [round, setRound] = useState(0)
  const [remaining, setRemaining] = useState(periodSeconds)

  useEffect(() => {
    checkRef.current = check
  }, [check])

  // Each round counts down from the full period, one second at a time.
  useEffect(() => {
    setRemaining(periodSeconds)
    const timer = window.setInterval(() => setRemaining((seconds) => seconds - 1), SECOND_MS)
    return () => window.clearInterval(timer)
  }, [round, periodSeconds])

  // Exactly at zero (never again for a stray tick below it): ask, then start a new round.
  useEffect(() => {
    if (remaining !== 0) return
    checkRef.current()
    setRound((value) => value + 1)
  }, [remaining])

  const checkNow = useCallback(() => {
    checkRef.current()
    setRound((value) => value + 1)
  }, [])

  return { remaining: Math.max(remaining, 0), checkNow }
}
