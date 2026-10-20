/**
 * PRD-261 G3 (P260-FX-22, M21) — 'Switch to Play' hands focus to the screen's heading, also when
 * the switch lands on an installed theme. A switch to an installed theme is never made in place:
 * the page reloads into it (PA-13, contexts/ui-style-context.tsx setStyle), which drops what
 * ThemeSlotPage remembered about the note that offered the switch. So the note leaves word in this
 * tab's sessionStorage just before the reload (one key, holding the page's path), and the next
 * load's ThemeSlotPage takes it once: read and cleared together, so it never outlives that load.
 *
 * Storage errors (blocked site data, a full store) mean "no word left", never a throw: the switch
 * still happens, only the focus stays where the browser puts it.
 */

/** This tab's sessionStorage key: the path whose screen heading takes focus on the next load. */
export const SWITCH_FOCUS_STORAGE_KEY = 'automatos-theme-switch-focus'

const LOG_PREFIX = '[theme-host]'

export type SwitchFocusStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>

/** This tab's sessionStorage, or null where reading the property itself throws (blocked site data). */
function tabStorage(): SwitchFocusStorage | null {
  try {
    return window.sessionStorage
  } catch (error) {
    console.warn(LOG_PREFIX, 'sessionStorage is not available', error)
    return null
  }
}

function currentPath(): string {
  return window.location.pathname
}

/** Just before a switch reloads the page: the next load of `path` hands focus to its screen's heading. */
export function rememberHeadingFocusAfterReload(
  path: string = currentPath(),
  storage: SwitchFocusStorage | null = tabStorage(),
): void {
  if (storage === null) return
  try {
    storage.setItem(SWITCH_FOCUS_STORAGE_KEY, path)
  } catch (error) {
    console.warn(LOG_PREFIX, 'the focus after the switch could not be kept for the reload', error)
  }
}

/** Once per load: whether the switch before the reload asked `path`'s heading to take focus. Clears the word. */
export function takeHeadingFocusAfterReload(
  path: string = currentPath(),
  storage: SwitchFocusStorage | null = tabStorage(),
): boolean {
  if (storage === null) return false
  try {
    const wanted = storage.getItem(SWITCH_FOCUS_STORAGE_KEY)
    if (wanted === null) return false
    storage.removeItem(SWITCH_FOCUS_STORAGE_KEY)
    return wanted === path
  } catch (error) {
    console.warn(LOG_PREFIX, 'the focus after the switch could not be read', error)
    return false
  }
}
