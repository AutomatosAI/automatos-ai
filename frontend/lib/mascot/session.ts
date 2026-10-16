/**
 * PRD-260 D6 — hello plays on the first open of Play in a browser session,
 * ported from Automatos Play's lib/mascot/session.ts. The session remembers it
 * in sessionStorage; where that is blocked (a private window, a sandbox), it
 * plays once per page load instead.
 */

export const HELLO_KEY = 'automatos-play.hello-played'

const PLAYED = '1'

let playedWithoutStorage = false

export type SessionSource = () => Storage

const browserSession: SessionSource = () => window.sessionStorage

/** True once per browser session: the caller dispatches app-open. Later calls say false. */
export function claimFirstOpen(storage: SessionSource = browserSession): boolean {
  try {
    const store = storage()
    if (store.getItem(HELLO_KEY) !== null) return false
    store.setItem(HELLO_KEY, PLAYED)
    return true
  } catch (error) {
    console.warn("[mascot] sessionStorage isn't available, so hello plays once per page load:", error)
    if (playedWithoutStorage) return false
    playedWithoutStorage = true
    return true
  }
}
