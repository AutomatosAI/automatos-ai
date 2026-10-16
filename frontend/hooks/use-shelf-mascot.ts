'use client'

import { useCallback, useEffect, useReducer } from 'react'

import { useBackendHealth } from '@/hooks/use-backend-health'
import { useCelebrateNew } from '@/hooks/use-celebrate-new'
import { captionFor } from '@/lib/mascot/media'
import { INITIAL_MASCOT, mascotReducer } from '@/lib/mascot/reducer'
import type { MascotName } from '@/lib/mascot/types'
import { kindOf, shelfCaption, type ShelfView } from '@/lib/play/shelf'

/**
 * PRD-260 §4, D6 — Auto in the shelf's header (Ember mock 04), as Automatos
 * Play's ShelfScreen drives him: proud of the count while he idles, a
 * celebration once for a Deliverable this browser hasn't seen
 * (useCelebrateNew), oops when the shelf doesn't load until it does (Try
 * again), and asleep while the backend doesn't answer (useBackendHealth, the
 * frame's GET /health).
 */

export interface ShelfMascot {
  readonly mascot: MascotName
  /** The shelf's line while he idles or celebrates; his own caption otherwise (asleep, oops). */
  readonly caption?: string
  readonly onClipEnded: (state: MascotName) => void
}

function idleCaption(state: MascotName, total: number | null): string | undefined {
  return state === 'idle' || state === 'celebrate' ? shelfCaption(total) : undefined
}

export function useShelfMascot(view: ShelfView): ShelfMascot {
  const [state, dispatch] = useReducer(mascotReducer, INITIAL_MASCOT)
  const { connection } = useBackendHealth()
  const celebrate = useCallback(() => dispatch({ type: 'deliverable-added' }), [])
  const onClipEnded = useCallback((mascot: MascotName) => dispatch({ type: 'clip-ended', state: mascot }), [])
  // A shelf that loads ends a failed load's oops. Declared before useCelebrateNew,
  // so this effect runs first and a new Deliverable still celebrates.
  useEffect(() => {
    if (view.status === 'ready') dispatch({ type: 'request-ok' })
  }, [view.status])
  const latest = useCelebrateNew(view.status === 'ready' ? view.items : null, celebrate)

  useEffect(() => {
    if (connection === 'down') dispatch({ type: 'backend-down' })
    if (connection === 'connected') dispatch({ type: 'backend-up' })
  }, [connection])
  // A shelf that failed with the backend down is the nap, not a snag.
  useEffect(() => {
    if (view.status === 'failed' && connection !== 'down') dispatch({ type: 'request-failed' })
  }, [view.status, connection])

  const total = view.status === 'ready' ? view.total : null
  const caption =
    state.mascot === 'celebrate' && latest ? captionFor('celebrate', kindOf(latest)) : idleCaption(state.mascot, total)
  return { mascot: state.mascot, caption, onClipEnded }
}
