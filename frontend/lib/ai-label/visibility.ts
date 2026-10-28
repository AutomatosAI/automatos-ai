/**
 * PRD-264 EUAI-11a — can a person see the AI label right now?
 *
 * CSS alone can't guarantee the label is visible against an arbitrary theme's stylesheet
 * (PRD §9), so the host checks it at run time. A label that fails is drawn again in the host's own
 * layer (components/ai-label/protected-caption.tsx), which costs nothing but the copy, so the check
 * is strict: whatever it can't confirm reads as hidden. The words sit in the label's text element
 * (AI_LABEL_TEXT_ATTR); the label can be seen only when, in this order:
 *
 * - it is in the page, and neither its words nor any element around them is `display: none`;
 *   the words are `visibility: visible`; the product of their and their ancestors' opacity is at
 *   least MIN_LABEL_OPACITY;
 * - nothing around the words cuts or repaints them: no `clip-path`, mask or `filter`, and no
 *   `content-visibility: hidden`, on the words, the label or an ancestor ('paint');
 * - the font is at least MIN_FONT_SIZE_PX ('font');
 * - the colour the words are filled with (`-webkit-text-fill-color`, else `color`) is not
 *   transparent and has at least MIN_TEXT_CONTRAST against the colour behind them: the first
 *   opaque `background-color` up the ancestors with the translucent ones in between painted over
 *   it, or the page's own white (and black too, where the root allows a dark scheme) ('color');
 * - the label's box is at least MIN_LABEL_SIZE_PX each way ('size') and inside the viewport
 *   ('viewport');
 * - the words themselves, measured as a Range, are at least a real line of text (MIN_FONT_SIZE_PX
 *   tall, MIN_GLYPH_WIDTH_EM of the font per character wide), wholly inside the viewport, and
 *   inside the label's own box when the label clips its overflow: so `text-indent`, a shrunk or
 *   clipped box and a scale transform all fail ('text');
 * - nothing is drawn over them: at the start, middle and end of each line of the words, the
 *   topmost element is the text element itself ('covered'). The label is `pointer-events: none` (it never takes a
 *   click), which hit-testing skips, so the label and its words take `pointer-events: auto` for the
 *   probe only, synchronously, and get their own value back. A pseudo-element of the label drawn
 *   over its words hits as the label, not the text element, so it fails too.
 *
 * Where the browser can't measure the words or hit-test (no Range rects, no elementsFromPoint),
 * the label reads as hidden. Not read: a background image behind the words, a font whose glyphs
 * are blank, and code the theme runs in the page; those stay review rules
 * (docs/theme-host/css-layer-model.md).
 */
import { canvasGrounds, contrastRatio, isOpaque, over, parseCssColor, type Rgba } from './contrast'

export type AiLabelHiddenReason =
  | 'detached'
  | 'display'
  | 'visibility'
  | 'opacity'
  | 'paint'
  | 'font'
  | 'color'
  | 'size'
  | 'viewport'
  | 'text'
  | 'covered'

/** On the label's element that holds its words. */
export const AI_LABEL_TEXT_ATTR = 'data-ai-label-text'

/** Below this combined opacity the words are too faint to count as perceivable. */
export const MIN_LABEL_OPACITY = 0.5
/** A box narrower or shorter than this holds no readable text. */
export const MIN_LABEL_SIZE_PX = 4
/** Sub-pixel rounding at an edge is not "outside" it. */
export const VIEWPORT_EDGE_TOLERANCE_PX = 1
/** Smaller type than this is not a caption anyone reads. */
export const MIN_FONT_SIZE_PX = 9
/** The least contrast the words keep against what is behind them (WCAG's floor for large text). */
export const MIN_TEXT_CONTRAST = 3
/** The narrowest a character is drawn, as a share of the font size, so a squeezed line reads as hidden. */
export const MIN_GLYPH_WIDTH_EM = 0.3

const HIDDEN_VISIBILITY = new Set(['hidden', 'collapse'])
const NO_PAINT_CHANGE = new Set(['', 'none'])
const PAINT_PROPERTIES = ['clip-path', 'mask-image', '-webkit-mask-image', 'filter'] as const
const PROBE_PROPERTY = 'pointer-events'

interface Box {
  readonly left: number
  readonly top: number
  readonly right: number
  readonly bottom: number
  readonly width: number
  readonly height: number
}

/** The element and every element around it, up to <html> (or the top of its shadow tree). */
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

/** The first value of `property` the chain gives, nearest first (a DOM without a renderer resolves no inheritance). */
function nearestValue(styles: readonly CSSStyleDeclaration[], property: string): string {
  return styles.map((style) => style.getPropertyValue(property).trim()).find((value) => value !== '') ?? ''
}

function hiddenByStyle(styles: readonly CSSStyleDeclaration[]): AiLabelHiddenReason | null {
  if (styles.some((style) => style.display === 'none')) return 'display'
  if (HIDDEN_VISIBILITY.has(styles[0].visibility)) return 'visibility'
  const opacity = styles.reduce((product, style) => product * opacityOf(style), 1)
  if (opacity < MIN_LABEL_OPACITY) return 'opacity'
  const repainted = (style: CSSStyleDeclaration) =>
    PAINT_PROPERTIES.some((property) => !NO_PAINT_CHANGE.has(style.getPropertyValue(property).trim())) ||
    style.getPropertyValue('content-visibility').trim() === 'hidden'
  return styles.some(repainted) ? 'paint' : null
}

function hiddenByFont(styles: readonly CSSStyleDeclaration[]): AiLabelHiddenReason | null {
  const size = Number.parseFloat(nearestValue(styles, 'font-size'))
  return Number.isFinite(size) && size < MIN_FONT_SIZE_PX ? 'font' : null
}

/** What is painted behind the words: one colour, or two where the page's own ground may be white or black. */
function groundsBehind(styles: readonly CSSStyleDeclaration[], view: Window): readonly Rgba[] | null {
  const layers: Rgba[] = []
  for (const style of styles) {
    const color = parseCssColor(style.backgroundColor || 'transparent')
    if (color === null) return null
    if (isOpaque(color)) return [layers.reduceRight((ground, layer) => over(layer, ground), color)]
    layers.push(color)
  }
  const mayBeDark = view.getComputedStyle(view.document.documentElement).colorScheme.includes('dark')
  return canvasGrounds(mayBeDark).map((page) => layers.reduceRight((ground, layer) => over(layer, ground), page))
}

function hiddenByColour(styles: readonly CSSStyleDeclaration[], view: Window): AiLabelHiddenReason | null {
  const fill = parseCssColor(nearestValue(styles, '-webkit-text-fill-color') || styles[0].color)
  const grounds = groundsBehind(styles, view)
  if (fill === null || grounds === null || fill.a === 0) return 'color'
  const faint = grounds.some((ground) => contrastRatio(over(fill, ground), ground) < MIN_TEXT_CONTRAST)
  return faint ? 'color' : null
}

function insideViewport(box: Box, view: Window): boolean {
  const edge = VIEWPORT_EDGE_TOLERANCE_PX
  return box.left >= -edge && box.top >= -edge && box.right <= view.innerWidth + edge && box.bottom <= view.innerHeight + edge
}

function inside(inner: Box, outer: Box): boolean {
  const edge = VIEWPORT_EDGE_TOLERANCE_PX
  return inner.left >= outer.left - edge && inner.top >= outer.top - edge &&
    inner.right <= outer.right + edge && inner.bottom <= outer.bottom + edge
}

function hiddenByBox(label: Element, view: Window): AiLabelHiddenReason | null {
  const box = label.getBoundingClientRect()
  if (box.width < MIN_LABEL_SIZE_PX || box.height < MIN_LABEL_SIZE_PX) return 'size'
  return insideViewport(box, view) ? null : 'viewport'
}

/** The words as the browser lays them out: their whole box and each line's. */
interface Measured {
  readonly box: Box
  readonly lines: readonly Box[]
}

/** The words' own boxes; null where the browser can't measure text. */
function measureWords(words: Element): Measured | null {
  const range = words.ownerDocument.createRange()
  if (typeof range.getBoundingClientRect !== 'function' || typeof range.getClientRects !== 'function') return null
  range.selectNodeContents(words)
  const lines = Array.from(range.getClientRects()).filter((line) => line.width > 0 && line.height > 0)
  return { box: range.getBoundingClientRect(), lines }
}

/** Whether the label cuts off what overflows its box (any overflow but visible). */
function clipsOverflow(style: CSSStyleDeclaration): boolean {
  const clipping = (value: string) => value !== '' && value !== 'visible'
  return clipping(style.overflowX || style.overflow) || clipping(style.overflowY || style.overflow)
}

function hiddenText(label: Element, words: Element, measured: Measured | null, fontSize: number, view: Window): AiLabelHiddenReason | null {
  if (measured === null || measured.lines.length === 0) return 'text'
  const { box } = measured
  const characters = Array.from((words.textContent ?? '').trim()).length
  const minWidth = characters * fontSize * MIN_GLYPH_WIDTH_EM
  if (box.height < MIN_FONT_SIZE_PX || box.width < minWidth || !insideViewport(box, view)) return 'text'
  return clipsOverflow(view.getComputedStyle(label)) && !inside(box, label.getBoundingClientRect()) ? 'text' : null
}

/** The elements under (x, y), topmost first, with `probed` taking hits for the moment of the call. */
function elementsAt(root: Document | ShadowRoot, probed: readonly HTMLElement[], points: ReadonlyArray<readonly [number, number]>) {
  const saved = probed.map((element) => [element.style.getPropertyValue(PROBE_PROPERTY), element.style.getPropertyPriority(PROBE_PROPERTY)])
  probed.forEach((element) => element.style.setProperty(PROBE_PROPERTY, 'auto', 'important'))
  try {
    return points.map(([x, y]) => root.elementsFromPoint(x, y))
  } finally {
    probed.forEach((element, index) => {
      const [value, priority] = saved[index]
      if (value === '') element.style.removeProperty(PROBE_PROPERTY)
      else element.style.setProperty(PROBE_PROPERTY, value, priority)
    })
  }
}

/** The start, middle and end of a line of the words, at its middle height. */
function samplePoints(line: Box): ReadonlyArray<readonly [number, number]> {
  const inset = Math.min(2, line.width / 4)
  const y = line.top + line.height / 2
  return [line.left + inset, line.left + line.width / 2, line.right - inset].map((x) => [x, y] as const)
}

/** Whether anything is drawn over the words at the start, middle or end of any of their lines. */
function hiddenByCover(label: Element, words: Element, measured: Measured | null): AiLabelHiddenReason | null {
  const root = words.getRootNode() as Document | ShadowRoot
  if (measured === null || typeof root.elementsFromPoint !== 'function') return 'covered'
  const probed = [label, words].filter((element): element is HTMLElement => 'style' in element)
  const points = measured.lines.flatMap(samplePoints)
  return elementsAt(root, probed, points).every((hits) => hits[0] === words) ? null : 'covered'
}

/** The element that holds the label's words: its text element, or the label itself. */
function wordsOf(label: Element): Element {
  return label.querySelector(`[${AI_LABEL_TEXT_ATTR}]`) ?? label
}

/** Why the label can't be seen, or null when it can. */
export function aiLabelHiddenReason(element: Element | null, view: Window): AiLabelHiddenReason | null {
  if (element === null || !element.isConnected) return 'detached'
  const words = wordsOf(element)
  const styles = selfAndAncestors(words).map((node) => view.getComputedStyle(node))
  const early = hiddenByStyle(styles) ?? hiddenByFont(styles) ?? hiddenByColour(styles, view) ?? hiddenByBox(element, view)
  if (early !== null) return early
  const measured = measureWords(words)
  const fontSize = Number.parseFloat(nearestValue(styles, 'font-size')) || MIN_FONT_SIZE_PX
  return hiddenText(element, words, measured, fontSize, view) ?? hiddenByCover(element, words, measured)
}
