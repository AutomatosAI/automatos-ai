'use client'

import type { ReactNode } from 'react'

import { useBackendHealth } from '@/hooks/use-backend-health'
import { useBackendMascot } from '@/hooks/use-backend-mascot'
import { isLocal } from '@/lib/auth-edition'

import { AsleepScreen, asleepDetail } from './asleep-screen'

/**
 * PRD-260 §4 — while the backend doesn't answer, Play's frame is "Automatos is
 * asleep" (Ember mock 07: the whole window, as Automatos Play's LiveShell
 * shows it); the frame and its page mount again, fresh, once it answers.
 * Ported from LiveShell and useBackendStatus: Play's proxy check becomes the
 * core's GET /health through apiClient (useBackendHealth, the connection
 * pill's own query, so Check again refreshes the pill too). Auto takes
 * backend-down and backend-up through Play's reducer (useBackendMascot), and
 * his nap is the gate: the screen opens once he has gone to sleep (never idle
 * first) and stays until /health answers. A recheck in flight is not an
 * answer: react-query reports a query that has never succeeded as loading
 * again while it asks, which must not flash the page back in between. Down is
 * /health failing on every retry, never one failed request (useBackendHealth,
 * P260-RVW-1): the swap unmounts the page and whatever was typed in it. A gate
 * that mounts while the backend is down (browser Back) shows the screen on its
 * first render, so the page never mounts in between (P260-FX-21, M89); while a
 * check is in flight the screen says Checking… (M94).
 */

/** How often the asleep screen asks again on its own (Automatos Play's RECHECK_MS). */
export const ASLEEP_RECHECK_MS = 10_000
const MS_PER_SECOND = 1000

export function AsleepGate({ children }: { readonly children: ReactNode }) {
  const backend = useBackendHealth()
  const auto = useBackendMascot(backend.connection)
  if (!auto.asleep || backend.connection === 'connected') return <>{children}</>
  return (
    <AsleepScreen
      mascot={auto.mascot}
      onClipEnded={auto.onClipEnded}
      detail={asleepDetail(backend.host)}
      onCheck={backend.check}
      checking={backend.checking}
      recheckSeconds={Math.round(ASLEEP_RECHECK_MS / MS_PER_SECOND)}
      local={isLocal}
    />
  )
}
