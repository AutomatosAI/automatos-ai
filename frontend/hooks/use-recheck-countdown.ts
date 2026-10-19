'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * PRD-260 §4 — the asleep screen's automatic recheck (Ember mock 07), ported
 * from Automatos Play's useBackendStatus, whose timer asked again every 10
 * seconds while the backend was asleep. Here the screen counts the seconds
 * down, calls `check` when they run out, and starts over; Check again calls it
 * at once and starts the count over too. The timer lives only while the
 * screen that uses it is mounted.
 *
 * While a check is in flight (`checking`: the ask and its retries, longer than
 * the period against a frozen backend, see BACKEND_HEALTH_TIMEOUT_MS in
 * use-backend-health.ts) the count holds at the full period and
 * the screen says Checking… instead (P260-FX-21, M94); the next round counts
 * from when the check ends, so the automatic recheck never cuts into one.
 */

export const SECOND_MS = 1000

export interface RecheckCountdown {
  /** Whole seconds until the next automatic check; the full period while a check is in flight. */
  readonly remaining: number
  /** Check again now, and start the count over. */
  readonly checkNow: () => void
}

export function useRecheckCountdown(periodSeconds: number, check: () => void, checking = false): RecheckCountdown {
  const checkRef = useRef(check)
  const [round, setRound] = useState(0)
  const [remaining, setRemaining] = useState(periodSeconds)

  useEffect(() => {
    checkRef.current = check
  }, [check])

  // Each round counts down from the full period, one second at a time; a check in flight holds it there.
  useEffect(() => {
    setRemaining(periodSeconds)
    if (checking) return undefined
    const timer = window.setInterval(() => setRemaining((seconds) => seconds - 1), SECOND_MS)
    return () => window.clearInterval(timer)
  }, [round, periodSeconds, checking])

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
