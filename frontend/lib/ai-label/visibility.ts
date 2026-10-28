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
 *   `content-visibility: hidden` or a `mix-blend-mode` other than normal, on the words, the label or an ancestor
 *   ('paint');
 * - the font is at least MIN_FONT_SIZE_PX ('font');
 * - the colour the words are filled with (`-webkit-text-fill-color`, else `color`) is not
 *   transparent and has at least MIN_TEXT_CONTRAST against every ground behind them at every point
 *   the check hit-tests, read from the paint stack there (ground.ts: whatever element paints under the
 *   words, the label's ancestor or not, its background colour and gradients, faded by its opacity, down
 *   to the first opaque colour or the page; a background image that isn't a gradient fails) ('color');
 * - no ::first-line or ::first-letter rule reaches the words or an element around them ('paint'),
 *   and, where the browser reports those pseudo-elements' styles, theirs keep the words opaque,
 *   visible and at contrast too ('color'): line-pseudos.ts;
 * - the words' text stroke doesn't blot them, and neither the words' element nor the label has a
 *   ::before or ::after, read from the rules and, where the browser reports them, the computed
 *   styles ('paint'): own-paint.ts;
 * - the label's box is at least MIN_LABEL_SIZE_PX each way ('size') and inside the viewport
 *   ('viewport');
 * - the words themselves, measured as a Range, are at least a real line of text (MIN_FONT_SIZE_PX
 *   tall, MIN_GLYPH_WIDTH_EM of the font per character wide), wholly inside the viewport, and
 *   inside the label's own box when the label clips its overflow: so `text-indent`, a shrunk or
 *   clipped box and a scale transform all fail ('text');
 * - nothing is drawn over them: at SAMPLE_POINTS_PER_LINE points evenly across each line of the words
 *   and JITTERED_POINTS_PER_LINE more at random, new each check (so a striped overlay can't be laid
 *   between known points), the topmost element is the text element itself ('covered'). The points are
 *   hit-tested once per check with everything hittable (hit-test.ts), and the same stacks give the
 *   ground above. A pseudo-element of the label drawn over its words hits as the label, not the text
 *   element, so it fails too.
 *
 * Where the browser can't measure the words or hit-test (no Range rects, no elementsFromPoint),
 * the label reads as hidden, and the ground is read from the words' ancestors instead. Not read:
 * what a replaced element under the words draws, an element's ::before or ::after under them, a
 * backdrop-filter or mix-blend-mode, a font whose glyphs are blank, and code the theme runs in the
 * page; those stay review rules (docs/theme-host/css-layer-model.md).
 */
import { contrastRatio, over, parseCssColor, type Rgba } from './contrast'
import { groundsBehind, groundsUnderWords, opacityOf } from './ground'
import { type Box, type Measured, type PointStack, measureWords, stacksUnderWords } from './hit-test'
import { lineRulesReach, lineStylesHide } from './line-pseudos'
import { ownPseudosDrawn, strokeHides } from './own-paint'
import { type StyleRuleRead, readableStyleRules } from './pseudo-rules'

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
export { AI_LABEL_PROBE_ATTR, type Box, JITTERED_POINTS_PER_LINE, SAMPLE_POINTS_PER_LINE, samplePoints } from './hit-test'

const HIDDEN_VISIBILITY = new Set(['hidden', 'collapse'])
const NO_PAINT_CHANGE = new Set(['', 'none'])
/** A blend mode can make the words match whatever is under them (difference, multiply, …): only `normal` reads. */
const NO_BLEND = new Set(['', 'normal'])
const PAINT_PROPERTIES = ['clip-path', 'mask-image', '-webkit-mask-image', 'filter'] as const

/** The element and every element around it, up to <html> (or the top of its shadow tree). */
function selfAndAncestors(element: Element): Element[] {
  const chain: Element[] = []
  for (let node: Element | null = element; node !== null; node = node.parentElement) chain.push(node)
  return chain
}

/** The first value of `property` the chain gives, nearest first (a DOM without a renderer resolves no inheritance). */
function nearestValue(styles: readonly CSSStyleDeclaration[], property: string): string {
  return styles.map((style) => style.getPropertyValue(property).trim()).find((value) => value !== '') ?? ''
}

function repainted(style: CSSStyleDeclaration): boolean {
  return (
    PAINT_PROPERTIES.some((property) => !NO_PAINT_CHANGE.has(style.getPropertyValue(property).trim())) ||
    style.getPropertyValue('content-visibility').trim() === 'hidden' ||
    !NO_BLEND.has(style.getPropertyValue('mix-blend-mode').trim())
  )
}

function hiddenByStyle(styles: readonly CSSStyleDeclaration[]): AiLabelHiddenReason | null {
  if (styles.some((style) => style.display === 'none')) return 'display'
  if (HIDDEN_VISIBILITY.has(styles[0].visibility)) return 'visibility'
  const opacity = styles.reduce((product, style) => product * opacityOf(style), 1)
  if (opacity < MIN_LABEL_OPACITY) return 'opacity'
  return styles.some(repainted) ? 'paint' : null
}

/** The words' text stroke, as their element computes it (own-paint.ts). */
function hiddenByStroke(styles: readonly CSSStyleDeclaration[], grounds: readonly Rgba[], fontSize: number): AiLabelHiddenReason | null {
  const stroke = {
    width: nearestValue(styles, '-webkit-text-stroke-width'),
    shorthand: nearestValue(styles, '-webkit-text-stroke'),
    color: nearestValue(styles, '-webkit-text-stroke-color'),
    fill: nearestValue(styles, '-webkit-text-fill-color') || styles[0].color,
  }
  return strokeHides(stroke, grounds, fontSize, MIN_TEXT_CONTRAST) ? 'paint' : null
}

function hiddenByFont(styles: readonly CSSStyleDeclaration[]): AiLabelHiddenReason | null {
  const size = Number.parseFloat(nearestValue(styles, 'font-size'))
  return Number.isFinite(size) && size < MIN_FONT_SIZE_PX ? 'font' : null
}

function hiddenByColour(styles: readonly CSSStyleDeclaration[], grounds: readonly Rgba[] | null): AiLabelHiddenReason | null {
  const fill = parseCssColor(nearestValue(styles, '-webkit-text-fill-color') || styles[0].color)
  if (fill === null || grounds === null || fill.a === 0) return 'color'
  const faint = grounds.some((ground) => contrastRatio(over(fill, ground), ground) < MIN_TEXT_CONTRAST)
  return faint ? 'color' : null
}

/** What ::first-line and ::first-letter do to the words (line-pseudos.ts). */
function hiddenByLinePseudos(
  chain: readonly Element[],
  grounds: readonly Rgba[],
  view: Window,
  rules: readonly StyleRuleRead[],
): AiLabelHiddenReason | null {
  if (lineRulesReach(chain, rules)) return 'paint'
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

/** Whether anything is drawn over the words at any point the check hit-tested ('covered' where it couldn't hit-test). */
function hiddenByCover(words: Element, stacks: readonly PointStack[] | null): AiLabelHiddenReason | null {
  if (stacks === null) return 'covered'
  return stacks.every(({ hits }) => hits[0] === words) ? null : 'covered'
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
  const styled = hiddenByStyle(styles) ?? hiddenByFont(styles)
  if (styled !== null) return styled
  // the words' boxes and what is under them, read once: the ground below and the cover check use the same stacks
  const measured = measureWords(words)
  const stacks = stacksUnderWords(words, measured)
  const underWords = groundsUnderWords(words, stacks, view)
  const grounds = underWords === undefined ? groundsBehind(styles, view) : underWords
  const fontSize = Number.parseFloat(nearestValue(styles, 'font-size')) || MIN_FONT_SIZE_PX
  const rules = readableStyleRules(view.document)
  const painted =
    hiddenByColour(styles, grounds) ??
    hiddenByStroke(styles, grounds ?? [], fontSize) ??
    hiddenByLinePseudos(chain, grounds ?? [], view, rules) ??
    (ownPseudosDrawn(element, words, view, rules) ? 'paint' : null) ??
    hiddenByBox(element, view)
  if (painted !== null) return painted
  return hiddenText(element, words, measured, fontSize, view) ?? hiddenByCover(words, stacks)
}
