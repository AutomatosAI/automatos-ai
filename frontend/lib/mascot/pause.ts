/**
 * PRD-261 A-13b (PRD-260 review F10, M24; WCAG 2.2.2 Pause, Stop, Hide) —
 * whether the user has paused Auto in this browser. Auto's clips loop, and his
 * ring pulses, for longer than five seconds; without this the only way to stop
 * them was the OS reduced-motion setting. The flag is a per-browser preference
 * kept in localStorage, so it survives a state change, a remount and a reload,
 * and every Auto on the page (and in other tabs, through the `storage` event)
 * follows one switch. localStorage may be missing or blocked (a private
 * window, a sandbox): the flag then lives for this page load only, and the
 * first failure is logged once.
 */

export const PAUSE_KEY = 'automatos-play.mascot-paused'

const PAUSED = '1'

export type PauseStorage = () => Storage

const browserStorage: PauseStorage = () => window.localStorage

/** The flag when localStorage can't hold it, and whether that has been said once. */
let pausedWithoutStorage = false
let warnedWithoutStorage = false

const listeners = new Set<() => void>()

function warnOnce(error: unknown): void {
  if (warnedWithoutStorage) return
  warnedWithoutStorage = true
  console.warn("[mascot] localStorage isn't available, so pausing Auto lasts for this page only:", error)
}

/** True while the user has paused Auto in this browser. */
export function isMascotPaused(storage: PauseStorage = browserStorage): boolean {
  try {
    return storage().getItem(PAUSE_KEY) === PAUSED
  } catch (error) {
    warnOnce(error)
    return pausedWithoutStorage
  }
}

/** Pause (true) or play (false) every Auto in this browser, and tell the ones on screen. */
export function setMascotPaused(paused: boolean, storage: PauseStorage = browserStorage): void {
  pausedWithoutStorage = paused
  try {
    if (paused) storage().setItem(PAUSE_KEY, PAUSED)
    else storage().removeItem(PAUSE_KEY)
  } catch (error) {
    warnOnce(error)
  }
  listeners.forEach((listener) => listener())
}

/**
 * Calls `onChange` when the flag changes here or in another tab (a `storage`
 * event for PAUSE_KEY, or a cleared store: key null). For useSyncExternalStore.
 */
export function subscribeMascotPause(onChange: () => void): () => void {
  listeners.add(onChange)
  const onStorage = (event: StorageEvent): void => {
    if (event.key === PAUSE_KEY || event.key === null) onChange()
  }
  window.addEventListener('storage', onStorage)
  return () => {
    listeners.delete(onChange)
    window.removeEventListener('storage', onStorage)
  }
}
