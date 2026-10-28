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
 * - nothing around the words cuts or repaints them: no `clip-path`, mask or `filter`, no
 *   `content-visibility: hidden`, and no `-webkit-text-stroke` wider than MAX_TEXT_STROKE_PX (a stroke
 *   in the background's colour blots the glyphs while the fill passes), on the words, the label or an
 *   ancestor ('paint');
 * - the font is at least MIN_FONT_SIZE_PX ('font');
 * - the colour the words are filled with (`-webkit-text-fill-color`, else `color`) is not
 *   transparent and has at least MIN_TEXT_CONTRAST against the colour behind them: the first
 *   opaque `background-color` up the ancestors with the translucent ones in between painted over
 *   it, or the page's own white (and black too, where the root allows a dark scheme) ('color');
 * - no ::first-line or ::first-letter rule reaches the words or an element around them ('paint'),
 *   and, where the browser reports those pseudo-elements' styles, theirs keep the words opaque,
 *   visible and at contrast too ('color'): line-pseudos.ts;
 * - neither the words' element nor the label has a ::before or ::after (any `content` but none or
 *   normal): one of the words' own pseudo-elements hit-tests as the words, so the hit-test below
 *   can't see it drawn over them, and one that adds text changes what the label says ('paint').
 *   Only these two: an ancestor's pseudo-element hit-tests as that ancestor, so the hit-test sees
 *   it, and themes paint the page's own (scan lines on <body>, say). Where the browser doesn't
 *   report pseudo-element styles (CSS.supports), the read is skipped, as in line-pseudos.ts;
 * - the label's box is at least MIN_LABEL_SIZE_PX each way ('size') and inside the viewport
 *   ('viewport');
 * - the words themselves, measured as a Range, are at least a real line of text (MIN_FONT_SIZE_PX
 *   tall, MIN_GLYPH_WIDTH_EM of the font per character wide), wholly inside the viewport, and
 *   inside the label's own box when the label clips its overflow: so `text-indent`, a shrunk or
 *   clipped box and a scale transform all fail ('text');
 * - nothing is drawn over them: at SAMPLE_POINTS_PER_LINE points evenly across each line of the words
 *   and JITTERED_POINTS_PER_LINE more at random, new each check (so a striped overlay can't be laid
 *   between known points), the topmost element is the text element itself ('covered'). Hit-testing skips anything with
 *   `pointer-events: none`: the label itself (it never takes a click), and an overlay a theme could
 *   make click-through to hide behind. So for the probe only, in the same task (nothing paints
 *   meanwhile), a transient constructed stylesheet makes everything hittable, `*, *::before,
 *   *::after` with `pointer-events: auto !important` (theme CSS can't be `!important`: the server's
 *   scope check refuses it), and is taken out in a `finally`; being constructed, it adds no element a
 *   `:has()` rule could notice. Only an inline `!important` beats it: the host's own protected copy,
 *   which must not count as a cover. A pseudo-element of the label drawn over its
 *   words hits as the label, not the text element, so it fails too.
 *
 * Where the browser can't measure the words or hit-test (no Range rects, no elementsFromPoint),
 * the label reads as hidden. Not read: a background image behind the words, a font whose glyphs
 * are blank, and code the theme runs in the page; those stay review rules
 * (docs/theme-host/css-layer-model.md).
 */
import { canvasGrounds, contrastRatio, isOpaque, over, parseCssColor, type Rgba } from './contrast'
import { lineRulesReach, lineStylesHide, pseudoStylesReadable } from './line-pseudos'

export type AiLabelHiddenReason =
  /** The check itself threw (a theme's code may have patched what it calls): the label counts as hidden. */
  | 'unchecked'
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
/** A text stroke wider than this can blot the glyphs out (in the background's colour). */
export const MAX_TEXT_STROKE_PX = 0.5
/** Hit-test points evenly across each line of the words, ends included. */
export const SAMPLE_POINTS_PER_LINE = 9
/** And this many more per line, at random places, new each check. */
export const JITTERED_POINTS_PER_LINE = 2

const HIDDEN_VISIBILITY = new Set(['hidden', 'collapse'])
const NO_PAINT_CHANGE = new Set(['', 'none'])
const PAINT_PROPERTIES = ['clip-path', 'mask-image', '-webkit-mask-image', 'filter'] as const
const OWN_PSEUDOS = ['::before', '::after'] as const
const NO_CONTENT = new Set(['', 'none', 'normal'])
/** Keeps a hit-test point this far inside a line's ends, and its random height inside the line's middle half. */
const LINE_END_INSET_PX = 2
const LINE_MIDDLE = { from: 0.25, span: 0.5 }
/** The transient sheet that makes everything hittable for the moment of the probe. */
export const AI_LABEL_PROBE_ATTR = 'data-ai-label-probe'
const PROBE_RULE = '*, *::before, *::after { pointer-events: auto !important; }'

/** A box in viewport px, as a browser measures it. */
export interface Box {
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

/** The text stroke's width in px: the longhand, else the shorthand's leading length; 0 when there is none. */
function textStrokePx(style: CSSStyleDeclaration): number {
  const longhand = Number.parseFloat(style.getPropertyValue('-webkit-text-stroke-width'))
  const shorthand = Number.parseFloat(style.getPropertyValue('-webkit-text-stroke'))
  return Math.max(Number.isFinite(longhand) ? longhand : 0, Number.isFinite(shorthand) ? shorthand : 0)
}

function repainted(style: CSSStyleDeclaration): boolean {
  return (
    PAINT_PROPERTIES.some((property) => !NO_PAINT_CHANGE.has(style.getPropertyValue(property).trim())) ||
    style.getPropertyValue('content-visibility').trim() === 'hidden' ||
    textStrokePx(style) > MAX_TEXT_STROKE_PX
  )
}

function hiddenByStyle(styles: readonly CSSStyleDeclaration[]): AiLabelHiddenReason | null {
  if (styles.some((style) => style.display === 'none')) return 'display'
  if (HIDDEN_VISIBILITY.has(styles[0].visibility)) return 'visibility'
  const opacity = styles.reduce((product, style) => product * opacityOf(style), 1)
  if (opacity < MIN_LABEL_OPACITY) return 'opacity'
  return styles.some(repainted) ? 'paint' : null
}

/** Whether the words' element or the label has a ::before or ::after (see the module's note on why only these). */
function hiddenByOwnPseudos(label: Element, words: Element, view: Window): AiLabelHiddenReason | null {
  if (!pseudoStylesReadable(view, '::before')) return null
  const elements = words === label ? [label] : [words, label]
  const drawn = (element: Element) =>
    OWN_PSEUDOS.some((pseudo) => !NO_CONTENT.has(view.getComputedStyle(element, pseudo).content.trim()))
  return elements.some(drawn) ? 'paint' : null
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

function hiddenByColour(styles: readonly CSSStyleDeclaration[], grounds: readonly Rgba[] | null): AiLabelHiddenReason | null {
  const fill = parseCssColor(nearestValue(styles, '-webkit-text-fill-color') || styles[0].color)
  if (fill === null || grounds === null || fill.a === 0) return 'color'
  const faint = grounds.some((ground) => contrastRatio(over(fill, ground), ground) < MIN_TEXT_CONTRAST)
  return faint ? 'color' : null
}

/** What ::first-line and ::first-letter do to the words (line-pseudos.ts). */
function hiddenByLinePseudos(chain: readonly Element[], grounds: readonly Rgba[], view: Window): AiLabelHiddenReason | null {
  if (lineRulesReach(chain, view.document)) return 'paint'
  const limits = { minOpacity: MIN_LABEL_OPACITY, minContrast: MIN_TEXT_CONTRAST }
  return lineStylesHide(chain, grounds, view, limits) ? 'color' : null
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

/** Whether the document takes constructed stylesheets (adoptedStyleSheets), which put no element in the page. */
function adoptsSheets(doc: Document): boolean {
  const view = doc.defaultView
  return Array.isArray(doc.adoptedStyleSheets) && typeof view?.CSSStyleSheet === 'function' && 'replaceSync' in view.CSSStyleSheet.prototype
}

/**
 * Runs `probe` with everything in the document hittable, then takes the sheet out in the `finally`, all in
 * this task, so no frame paints with it. A constructed sheet (adoptedStyleSheets) adds no element, so no
 * theme rule can tell it is there (a `<style>` element could be seen by `:has()`); a browser without them
 * gets a `<style>` at the end of <html>, outside <head> and <body>, which the label's and the copy's
 * observers watch.
 */
function withEverythingHittable<T>(doc: Document, probe: () => T): T {
  if (adoptsSheets(doc)) {
    const sheet = new (doc.defaultView as Window & typeof globalThis).CSSStyleSheet()
    sheet.replaceSync(PROBE_RULE)
    doc.adoptedStyleSheets = [...doc.adoptedStyleSheets, sheet]
    try {
      return probe()
    } finally {
      doc.adoptedStyleSheets = doc.adoptedStyleSheets.filter((adopted) => adopted !== sheet)
    }
  }
  const element = doc.createElement('style')
  element.setAttribute(AI_LABEL_PROBE_ATTR, '')
  element.textContent = PROBE_RULE
  doc.documentElement.appendChild(element)
  try {
    return probe()
  } finally {
    element.remove()
  }
}

/** The elements under each point, topmost first, with everything hittable for the moment of the call. */
function elementsAt(root: Document | ShadowRoot, points: ReadonlyArray<readonly [number, number]>): Element[][] {
  const doc = root.ownerDocument ?? (root as Document)
  return withEverythingHittable(doc, () => points.map(([x, y]) => root.elementsFromPoint(x, y)))
}

/**
 * SAMPLE_POINTS_PER_LINE points evenly across a line of the words at its middle height, ends included,
 * and JITTERED_POINTS_PER_LINE more at random across it and inside its middle half: bounded, and new
 * each check, so no overlay laid out between known points goes unseen for long.
 */
export function samplePoints(line: Box, random: () => number = Math.random): ReadonlyArray<readonly [number, number]> {
  const inset = Math.min(LINE_END_INSET_PX, line.width / 4)
  const span = line.width - 2 * inset
  const middle = line.top + line.height / 2
  const steps = SAMPLE_POINTS_PER_LINE - 1
  const even = Array.from({ length: SAMPLE_POINTS_PER_LINE }, (_, step) => [line.left + inset + (span * step) / steps, middle] as const)
  const jittered = Array.from(
    { length: JITTERED_POINTS_PER_LINE },
    () => [line.left + inset + span * random(), line.top + line.height * (LINE_MIDDLE.from + LINE_MIDDLE.span * random())] as const,
  )
  return [...even, ...jittered]
}

/** Whether anything is drawn over the words at the start, middle or end of any of their lines. */
function hiddenByCover(words: Element, measured: Measured | null): AiLabelHiddenReason | null {
  const root = words.getRootNode() as Document | ShadowRoot
  if (measured === null || typeof root.elementsFromPoint !== 'function') return 'covered'
  const points = measured.lines.flatMap((line) => samplePoints(line))
  return elementsAt(root, points).every((hits) => hits[0] === words) ? null : 'covered'
}

/** The element that holds the label's words: its text element, or the label itself. */
function wordsOf(label: Element): Element {
  return label.querySelector(`[${AI_LABEL_TEXT_ATTR}]`) ?? label
}

/** Why the label can't be seen, or null when it can. */
export function aiLabelHiddenReason(element: Element | null, view: Window): AiLabelHiddenReason | null {
  if (element === null || !element.isConnected) return 'detached'
  const words = wordsOf(element)
  const chain = selfAndAncestors(words)
  const styles = chain.map((node) => view.getComputedStyle(node))
  const grounds = groundsBehind(styles, view)
  const early =
    hiddenByStyle(styles) ??
    hiddenByFont(styles) ??
    hiddenByColour(styles, grounds) ??
    hiddenByLinePseudos(chain, grounds ?? [], view) ??
    hiddenByOwnPseudos(element, words, view) ??
    hiddenByBox(element, view)
  if (early !== null) return early
  const measured = measureWords(words)
  const fontSize = Number.parseFloat(nearestValue(styles, 'font-size')) || MIN_FONT_SIZE_PX
  return hiddenText(element, words, measured, fontSize, view) ?? hiddenByCover(words, measured)
}
