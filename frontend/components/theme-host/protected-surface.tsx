'use client'

import type { ReactNode } from 'react'

/**
 * PRD-261 A-12c (PA-15) — a protected surface: a card the owner must always see before deciding
 * (a tool approval, a Mission plan approval), whatever a theme's stylesheet says.
 *
 * Theme CSS stays unlayered (docs/theme-host/css-layer-model.md), so the cards are protected by
 * placement, not by a cascade layer. Two elements wrap the card: the host (`data-protected-surface`)
 * and its inner wrapper (`data-protected-surface-inner`, option (2) of PA-15: no shadow root). Each
 * carries every declaration of PROTECTED_SURFACE_GUARD inline with `!important`. An important
 * declaration attached to the element beats every important or normal stylesheet declaration
 * (CSS Cascade 4: author important inline > author important sheet > author normal), and the
 * server's scope check refuses `!important` in theme CSS anyway, so no theme rule can hide,
 * fade, move off-screen or switch off clicks on these two elements. `visibility` and
 * `pointer-events` inherit, so the host also gives the card back what a theme takes from an
 * ancestor; `display`, `opacity` and position on an ancestor outside the host are not reachable
 * from here, which is why, for code inside the page, this is a review rule and not a guarantee
 * (PRD §4.4; e2e/browser/fixtures/hello-a12c/README.md).
 *
 * React writes inline styles only from a `style` prop and cannot mark one `!important`, so the
 * guard is set through CSSStyleDeclaration.setProperty in a ref callback (it runs at commit,
 * before the browser paints the card). Neither element takes a `style` prop, so React never
 * rewrites the attribute afterwards.
 */

export const PROTECTED_SURFACE_ATTR = 'data-protected-surface'
export const PROTECTED_SURFACE_INNER_ATTR = 'data-protected-surface-inner'

/** The cards a theme may never hide. */
export type ProtectedSurfaceName = 'tool-approval' | 'mission-approval'

const IMPORTANT = 'important'
/** Both wrappers fill their parent, so the card's own `h-full` keeps working inside them. */
const FILL_CLASS = 'h-full w-full'

/**
 * The inline guard, every value `!important`. The first five are PA-15's; `static` position
 * also voids any top/left/right/bottom a theme sets (an off-screen move), and transform, filter
 * and clip-path close the other ways to move, fade or cut away an element of the theme's choosing.
 */
export const PROTECTED_SURFACE_GUARD: Readonly<Record<string, string>> = Object.freeze({
  display: 'block',
  visibility: 'visible',
  opacity: '1',
  'pointer-events': 'auto',
  position: 'static',
  transform: 'none',
  filter: 'none',
  'clip-path': 'none',
  overflow: 'visible',
})

/** A ref callback: puts the guard on the element as it mounts (null on unmount: nothing to do). */
export function guardProtectedElement(element: HTMLElement | null): void {
  if (element === null) return
  for (const [property, value] of Object.entries(PROTECTED_SURFACE_GUARD)) {
    element.style.setProperty(property, value, IMPORTANT)
  }
}

export interface ProtectedSurfaceProps {
  readonly surface: ProtectedSurfaceName
  readonly children: ReactNode
}

/** Wraps a protected card in its guarded host and inner wrapper. */
export function ProtectedSurface({ surface, children }: ProtectedSurfaceProps) {
  return (
    <div data-protected-surface={surface} ref={guardProtectedElement} className={FILL_CLASS}>
      <div data-protected-surface-inner="" ref={guardProtectedElement} className={FILL_CLASS}>
        {children}
      </div>
    </div>
  )
}
