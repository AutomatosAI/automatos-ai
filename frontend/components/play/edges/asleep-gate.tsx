'use client'

import { type ReactNode, type RefObject, useEffect, useRef, useState } from 'react'

import { useBackendHealth } from '@/hooks/use-backend-health'
import { useBackendMascot } from '@/hooks/use-backend-mascot'
import { isLocal } from '@/lib/auth-edition'
import { refocus } from '@/lib/play/focus'

import { ASLEEP_TITLE, AsleepScreen, asleepDetail } from './asleep-screen'

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
 *
 * The swap drops keyboard focus with the frame, so it is put back (P260-FX-22,
 * M21): the asleep screen focuses its heading as it opens, and after a nap the
 * frame's <main> (the wakeFocus ref) takes focus when it mounts again. One
 * live region outside the swap, present from the start so it is heard, says
 * each change: asleep, then back.
 */

/** How often the asleep screen asks again on its own (Automatos Play's RECHECK_MS). */
export const ASLEEP_RECHECK_MS = 10_000
const MS_PER_SECOND = 1000
/** What the gate's live region says once the page is back after a nap. */
export const AWAKE_LINE = 'Automatos is back'

export interface AsleepGateProps {
  readonly children: ReactNode
  /** What takes focus when the page mounts again after a nap: the frame's <main>. */
  readonly wakeFocus?: RefObject<HTMLElement>
}

/** The live region's words: nothing until the first nap, then asleep, then back (reset while rendering). */
function useNapAnnouncement(asleep: boolean): string {
  const [napped, setNapped] = useState(asleep)
  if (asleep && !napped) setNapped(true)
  if (asleep) return ASLEEP_TITLE
  return napped ? AWAKE_LINE : ''
}

/** After a nap the frame mounts again: `target` takes the focus the swap dropped. */
function useWakeFocus(asleep: boolean, target: RefObject<HTMLElement> | undefined) {
  const wasAsleep = useRef(asleep)
  useEffect(() => {
    if (asleep) {
      wasAsleep.current = true
      return
    }
    if (!wasAsleep.current) return
    wasAsleep.current = false
    refocus(target?.current)
  }, [asleep, target])
}

export function AsleepGate({ children, wakeFocus }: AsleepGateProps) {
  const backend = useBackendHealth()
  const auto = useBackendMascot(backend.connection)
  const asleep = auto.asleep && backend.connection !== 'connected'
  const announcement = useNapAnnouncement(asleep)
  useWakeFocus(asleep, wakeFocus)
  return (
    <>
      {asleep ? (
        <AsleepScreen
          mascot={auto.mascot}
          onClipEnded={auto.onClipEnded}
          detail={asleepDetail(backend.host)}
          onCheck={backend.check}
          checking={backend.checking}
          recheckSeconds={Math.round(ASLEEP_RECHECK_MS / MS_PER_SECOND)}
          local={isLocal}
        />
      ) : (
        children
      )}
      <p className="sr-only" role="status">
        {announcement}
      </p>
    </>
  )
}
