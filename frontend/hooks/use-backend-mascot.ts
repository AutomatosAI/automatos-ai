'use client'

import { useCallback, useEffect, useReducer } from 'react'

import type { BackendConnection } from '@/hooks/use-backend-health'
import { INITIAL_MASCOT, mascotReducer } from '@/lib/mascot/reducer'
import type { MascotName } from '@/lib/mascot/types'

/**
 * PRD-260 §4, D6 — Auto on the asleep screen, driven by Play's reducer from the
 * backend's connection (useBackendHealth, GET /health): backend-down when the
 * connection is down, so he sleeps; backend-up when it answers again, so he
 * wakes to idle. Play's frame keeps this mounted across both, so it sees the
 * whole nap (components/play/edges/asleep-gate.tsx).
 */

export interface BackendMascot {
  /** The state to show: AutoMascot's `state`. */
  readonly mascot: MascotName
  /** The reducer has taken backend-down: Auto is asleep until backend-up. */
  readonly asleep: boolean
  /** AutoMascot's onClipEnded. */
  readonly onClipEnded: (state: MascotName) => void
}

export function useBackendMascot(connection: BackendConnection): BackendMascot {
  const [state, dispatch] = useReducer(mascotReducer, INITIAL_MASCOT)

  useEffect(() => {
    if (connection === 'down') dispatch({ type: 'backend-down' })
    if (connection === 'connected') dispatch({ type: 'backend-up' })
  }, [connection])

  const onClipEnded = useCallback((ended: MascotName) => dispatch({ type: 'clip-ended', state: ended }), [])

  return { mascot: state.mascot, asleep: state.backendDown, onClipEnded }
}
