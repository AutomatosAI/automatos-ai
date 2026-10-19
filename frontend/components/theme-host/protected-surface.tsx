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
 * fade, move, shrink or switch off clicks on these two elements. The scope check also refuses
 * every theme selector that names either attribute (css_scope.py), so no rule reaches the card
 * or what it holds by name. `visibility` and `pointer-events` inherit, so the host also gives the
 * card back what a theme takes from an ancestor. Review rules, not guarantees (PRD §4.4;
 * docs/theme-host/css-layer-model.md): `display`, `opacity` or position on an ancestor, a generic
 * selector such as `button` matching the card's elements, an overlay, and code in the page.
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
 * also voids any top/left/right/bottom a theme sets (an off-screen move). The rest close every
 * other property that hides, moves, shrinks, cuts away or skips painting an element on its own:
 * transform and its individual properties (translate, scale, rotate) and offset-path; filter,
 * clip-path, mask and -webkit-mask; content-visibility, contain and zoom; margin (a negative
 * margin moves a static box); z-index (a flex item stacks even when static); and width and
 * height, the same 100% as FILL_CLASS, so a theme cannot collapse the box the card's `h-full`
 * reads. Both wrappers are the only child of a core element (WidgetWrapper's `h-full` div, Play's
 * card cell), so no core rule sets these on them and the guard changes no layout.
 */
export const PROTECTED_SURFACE_GUARD: Readonly<Record<string, string>> = Object.freeze({
  display: 'block',
  visibility: 'visible',
  opacity: '1',
  'pointer-events': 'auto',
  position: 'static',
  transform: 'none',
  translate: 'none',
  scale: 'none',
  rotate: 'none',
  'offset-path': 'none',
  filter: 'none',
  'clip-path': 'none',
  mask: 'none',
  '-webkit-mask': 'none',
  'content-visibility': 'visible',
  contain: 'none',
  zoom: '1',
  margin: '0px',
  'z-index': 'auto',
  width: '100%',
  height: '100%',
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
