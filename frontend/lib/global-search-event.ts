/**
 * The global search's open event: a header's Search button dispatches it and
 * useGlobalSearch (hooks/use-global-search.ts) opens the dialog on it. One
 * name for the Studio header and Play's frame (PRD-260 T-004).
 */
export const GLOBAL_SEARCH_OPEN_EVENT = 'automatos:global-search-open'

/** Opens the global search dialog (the ⌘K dialog), wherever the button sits. */
export function openGlobalSearch(): void {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new Event(GLOBAL_SEARCH_OPEN_EVENT))
}
