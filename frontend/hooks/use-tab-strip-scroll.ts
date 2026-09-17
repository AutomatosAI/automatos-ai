import { useEffect, useRef } from 'react'
import { useIsTabletOrBelow } from './use-mobile'

/**
 * Keeps the active tab of a Studio `.cc-tabs` strip visible.
 *
 * The strips already scroll sideways (`overflow-x: auto` on `.cc-tabs`),
 * which is all a desktop needs because every tab fits. On a compact viewport
 * the strip is wider than the screen, so a tab that is active on mount —
 * driven by `?tab=` in the URL, or simply the seventh of seven on a phone —
 * starts off-screen and reads as missing.
 *
 * Attach the ref to the strip and pass the active key:
 *
 *   const tabs = useTabStripScroll(activeTab)
 *   <nav className="cc-tabs" ref={tabs}>…</nav>
 *
 * ONE hook for every strip — the Command Centre, the Assignments hub, the
 * three bespoke Studio pages and the shared `FilterTabs` (PRD-246 M5: the
 * behaviour is shared, not copied per surface).
 */
export function useTabStripScroll<T extends HTMLElement = HTMLElement>(activeKey: string) {
  const ref = useRef<T>(null)
  const isCompact = useIsTabletOrBelow()

  useEffect(() => {
    // On a desktop the whole strip is on screen; leave the scroll alone.
    if (!isCompact) return
    // `.active` is how every Studio strip marks its current tab.
    const active = ref.current?.querySelector<HTMLElement>('.cc-tab.active')
    // jsdom implements no layout, so `scrollIntoView` is absent under test.
    active?.scrollIntoView?.({ block: 'nearest', inline: 'center' })
  }, [activeKey, isCompact])

  return ref
}
