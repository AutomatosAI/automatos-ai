/**
 * PRD-266 US-005b — who hears about a 401 from the API.
 *
 * apiClient reports the parsed body of every 401 it receives here (`reportUnauthorized`, one call
 * in api-client.ts's error path), and a listener decides what the refusal means for it. The local
 * edition's browser pairing gate is the one listener today: a paired browser whose token was
 * revoked, or an install that started asking for paired devices, sends it back to "Pair this
 * browser" (components/settings/phones-devices/pair-browser-gate.tsx). With no listener (SaaS)
 * a report does nothing, so the request fails exactly as before.
 */

/** Hears the parsed body of a 401 (undefined when the answer had none). */
export type UnauthorizedListener = (body: unknown) => void

const listeners = new Set<UnauthorizedListener>()

/** Hear every 401 apiClient receives from now on; the returned function stops it. */
export function onUnauthorized(listener: UnauthorizedListener): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/** Tell every listener about a 401. A listener that throws is logged and never breaks the request. */
export function reportUnauthorized(body: unknown): void {
  Array.from(listeners).forEach((listener) => {
    try {
      listener(body)
    } catch (error) {
      console.error('[api] a 401 listener failed', error)
    }
  })
}
