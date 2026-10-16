/**
 * PRD-260 P260-FX-22 (M20, M21) — where keyboard focus goes when the control
 * that had it leaves the page. A browser drops focus to <body> when the focused
 * element is removed (the asleep gate's swap, Switch to Play, the last Load
 * more): these put it on the place that took over instead, and only when focus
 * really was dropped, never taking it from where the user or the page (an
 * autoFocus field) already put it.
 */

/** Focus was dropped: nothing has it, or only <body>. */
export function focusWasDropped(doc: Document = document): boolean {
  const active = doc.activeElement
  return active === null || active === doc.body
}

/** Focus `target` (a heading or <main> with tabIndex={-1}, a card's link) when focus was dropped. */
export function refocus(target: HTMLElement | null | undefined): void {
  if (target && focusWasDropped(target.ownerDocument)) target.focus()
}

/**
 * Focus what `selector` finds once it is on the page: at once, or when code
 * still loading renders it (a screen behind a dynamic import). Returns the stop
 * for an effect's cleanup, so a page left before then never takes focus.
 */
export function refocusWhenShown(selector: string, doc: Document = document): () => void {
  const find = () => doc.querySelector<HTMLElement>(selector)
  const now = find()
  if (now) {
    refocus(now)
    return () => undefined
  }
  const observer = new MutationObserver(() => {
    const shown = find()
    if (!shown) return
    observer.disconnect()
    refocus(shown)
  })
  observer.observe(doc.body, { childList: true, subtree: true })
  return () => observer.disconnect()
}
