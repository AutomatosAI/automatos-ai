/**
 * PRD-264 EUAI-11a — can a person see the AI label right now?
 *
 * CSS alone can't guarantee the label is visible against an arbitrary theme's stylesheet
 * (PRD §9), so the host checks it at run time: the label must be displayed (no `display: none`
 * on it or on any element around it), visible (`visibility` on the label itself, which
 * inherits), opaque enough (the product of its own and its ancestors' `opacity`), not shrunk
 * to nothing, and wholly inside the viewport. The first rule it breaks is the answer; null
 * means it can be seen. Being drawn over by another element is not checked here (a review rule,
 * docs/theme-host/css-layer-model.md).
 */

export type AiLabelHiddenReason = 'detached' | 'display' | 'visibility' | 'opacity' | 'size' | 'viewport'

/** Below this combined opacity the words are too faint to count as perceivable. */
export const MIN_LABEL_OPACITY = 0.5
/** A box narrower or shorter than this holds no readable text. */
export const MIN_LABEL_SIZE_PX = 4
/** Sub-pixel rounding at the viewport's edge is not "outside" it. */
export const VIEWPORT_EDGE_TOLERANCE_PX = 1

const HIDDEN_VISIBILITY = new Set(['hidden', 'collapse'])

/** The element and every element around it, up to <html>. */
function selfAndAncestors(element: Element): Element[] {
  const chain: Element[] = []
  for (let node: Element | null = element; node !== null; node = node.parentElement) chain.push(node)
  return chain
}

/** A computed opacity as a number; an empty or unparsable value is fully opaque. */
function opacityOf(style: CSSStyleDeclaration): number {
  const value = Number.parseFloat(style.opacity)
  return Number.isFinite(value) ? value : 1
}

function hiddenByStyle(element: Element, view: Window): AiLabelHiddenReason | null {
  const styles = selfAndAncestors(element).map((node) => view.getComputedStyle(node))
  if (styles.some((style) => style.display === 'none')) return 'display'
  if (HIDDEN_VISIBILITY.has(styles[0].visibility)) return 'visibility'
  const opacity = styles.reduce((product, style) => product * opacityOf(style), 1)
  return opacity < MIN_LABEL_OPACITY ? 'opacity' : null
}

function hiddenByBox(element: Element, view: Window): AiLabelHiddenReason | null {
  const box = element.getBoundingClientRect()
  if (box.width < MIN_LABEL_SIZE_PX || box.height < MIN_LABEL_SIZE_PX) return 'size'
  const edge = VIEWPORT_EDGE_TOLERANCE_PX
  const inside =
    box.left >= -edge && box.top >= -edge && box.right <= view.innerWidth + edge && box.bottom <= view.innerHeight + edge
  return inside ? null : 'viewport'
}

/** Why the label can't be seen, or null when it can. */
export function aiLabelHiddenReason(element: Element | null, view: Window): AiLabelHiddenReason | null {
  if (element === null || !element.isConnected) return 'detached'
  return hiddenByStyle(element, view) ?? hiddenByBox(element, view)
}
