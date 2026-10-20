'use client'

import { useCallback, useEffect, useReducer } from 'react'

import type { HostHealth } from '@automatos/theme-sdk'

import { INITIAL_MASCOT, mascotReducer } from '../lib/mascot/reducer'
import type { MascotName, MascotState } from '../lib/mascot/types'

/**
 * PRD-260 §4, D6 — Auto on the asleep screen, driven by Play's reducer from the
 * backend's connection (the SDK's useHealth, GET /health): backend-down when the
 * connection is down, so he sleeps; backend-up when it answers again, so he
 * wakes to idle. Play's frame keeps this mounted across both, so it sees the
 * whole nap (components/play/edges/asleep-gate.tsx). It starts from the
 * connection it mounts with (P260-FX-21, M89): mounted while the backend is
 * down, Auto is asleep on the first render, so the gate never mounts its page
 * (and that page's requests) before the asleep screen.
 */

/** The backend's connection as the SDK's useHealth reports it ('checking' | 'connected' | 'down'). */
type BackendConnection = HostHealth['connection']

export interface BackendMascot {
  /** The state to show: AutoMascot's `state`. */
  readonly mascot: MascotName
  /** The reducer has taken backend-down: Auto is asleep until backend-up. */
  readonly asleep: boolean
  /** AutoMascot's onClipEnded. */
  readonly onClipEnded: (state: MascotName) => void
}

/** Auto as the connection finds him: asleep when it is down, else Play's starting state. */
export function initialBackendMascot(connection: BackendConnection): MascotState {
  return connection === 'down' ? mascotReducer(INITIAL_MASCOT, { type: 'backend-down' }) : INITIAL_MASCOT
}

export function useBackendMascot(connection: BackendConnection): BackendMascot {
  const [state, dispatch] = useReducer(mascotReducer, connection, initialBackendMascot)

  useEffect(() => {
    if (connection === 'down') dispatch({ type: 'backend-down' })
    if (connection === 'connected') dispatch({ type: 'backend-up' })
  }, [connection])

  const onClipEnded = useCallback((ended: MascotName) => dispatch({ type: 'clip-ended', state: ended }), [])

  return { mascot: state.mascot, asleep: state.backendDown, onClipEnded }
}
