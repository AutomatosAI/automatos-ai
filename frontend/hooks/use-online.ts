'use client'

import { useSyncExternalStore } from 'react'

/**
 * PRD-260 P260-FX-21 (M83) — is this browser online? navigator.onLine, kept
 * current by the window's online and offline events. The asleep screen reads it
 * so an offline user is told so, never that nothing is broken on their side.
 * Online on the server, where the screen never renders.
 */

const CONNECTIVITY_EVENTS = ['online', 'offline'] as const

function subscribeConnectivity(onChange: () => void): () => void {
  CONNECTIVITY_EVENTS.forEach((name) => window.addEventListener(name, onChange))
  return () => CONNECTIVITY_EVENTS.forEach((name) => window.removeEventListener(name, onChange))
}

const onlineNow = (): boolean => navigator.onLine

const onlineOnServer = (): boolean => true

/** True while the browser says it is online. */
export function useOnline(): boolean {
  return useSyncExternalStore(subscribeConnectivity, onlineNow, onlineOnServer)
}
