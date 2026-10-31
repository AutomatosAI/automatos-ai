'use client'

import { useEffect, useState } from 'react'

/**
 * Whole seconds left until `deadlineIso`, ticking once a second; 0 once it has passed, null
 * without a deadline. Moved out of MissionApprovalWidget (its auto-proceed countdown) so the
 * pairing code in Settings → Phones & devices (PRD-266 US-005c) counts down the same way.
 */
export function useCountdown(deadlineIso?: string): number | null {
  const [secsLeft, setSecsLeft] = useState<number | null>(null)
  useEffect(() => {
    if (!deadlineIso) {
      setSecsLeft(null)
      return
    }
    const tick = () => {
      const ms = new Date(deadlineIso).getTime() - Date.now()
      setSecsLeft(Math.max(0, Math.round(ms / 1000)))
    }
    tick()
    const t = setInterval(tick, 1000)
    return () => clearInterval(t)
  }, [deadlineIso])
  return secsLeft
}
