/**
 * PRD-264 EUAI-11a — what is painted behind the AI label's words, for the label's colour check (visibility.ts).
 *
 * The ground is read from the browser's paint stack, not the label's ancestors: at each point the check
 * hit-tests (hit-test.ts: everything hittable for the moment), the elements under the words, from the words'
 * own element down, are what paints behind them, whoever's they are. So a theme can't paint the corner under
 * a fixed label in the words' own colour on an element that isn't the label's ancestor. At each point:
 *
 * - each element, from the words' own element down, paints its background colour, then its background image
 *   layers over it (gradient-parse.ts, gradient-paint.ts: the colour a gradient paints at that very point,
 *   where its tile lies under it, background-tile.ts), each scaled by the element's effective opacity (its own
 *   times every ancestor's), so an element at opacity 0 paints nothing and can't fake a ground;
 * - the walk stops at the first opaque background colour; below the last element is the page's root
 *   background (body's, where <html> paints none) and then the page itself, white, or white and black where
 *   the root allows a dark scheme;
 * - the translucent layers are blended down onto that ground, and every colour they could make is a ground.
 *
 * The words must keep their contrast against every ground at every point: one point over a patch of their
 * own colour is enough to fail. What can't be read fails closed (the ground is unknown, so the label counts
 * as hidden): a background colour the browser reports in a form nothing reads, a background image that isn't
 * a gradient (`url()`, `image-set()` and the rest), or more possible grounds at one point than
 * MAX_GROUNDS_PER_POINT. Not read: what a replaced element draws (an <img>, <canvas>, <video>, <iframe> or
 * <svg>, whose background is read but not its content), an element's ::before and ::after (they hit-test as
 * the element, so where they paint is unknown), a backdrop-filter's effect on what is behind, and a
 * mix-blend-mode. Adding a replaced element takes markup, which is code; the rest stay review rules
 * (docs/theme-host/css-layer-model.md).
 *
 * Where no hit-test reaches the words (a browser with no elementsFromPoint, a label that isn't laid out), the
 * ground is read the old way, groundsBehind: the label's ancestors, nearest first, to the first opaque one.
 */
import { colourReaches, type Point, type Rect, tileAt } from './background-tile'
import { canvasGrounds, isOpaque, over, parseCssColor, type Rgba } from './contrast'
import { splitTopLevel } from './css-tokens'
import { type Gradient, parseGradient } from './gradient-parse'
import { gradientColours, gradientColoursAt } from './gradient-paint'
import type { PointStack } from './hit-test'

/** More possible grounds than this at one point and the ground counts as unknown. */
export const MAX_GROUNDS_PER_POINT = 4096

const CLEAR: Rgba = Object.freeze({ r: 0, g: 0, b: 0, a: 0 })

/** One element's paint, read once per check: null fields are what can't be read. */
interface Painter {
  readonly style: CSSStyleDeclaration
  readonly rect: Rect
  readonly opacity: number
  readonly colour: Rgba | null
  /** Its image layers, top first; a null layer is `none`. Null: a layer isn't a gradient, or a gradient can't be read. */
  readonly layers: ReadonlyArray<Gradient | null> | null
}

/** What one element paints at a point, faded by its opacity: its background colour (null: none there), then its image layers, bottom first, each a set of colours it may be there. */
interface Paint {
  readonly colour: Rgba | null
  readonly layers: ReadonlyArray<readonly Rgba[]>
}

/** A computed opacity as a number; an empty or unparsable value is fully opaque. */
export function opacityOf(style: CSSStyleDeclaration): number {
  const value = Number.parseFloat(style.opacity)
  return Number.isFinite(value) ? value : 1
}

/** The element above `element`: its parent, or across a shadow root, its host. */
function parentOf(element: Element): Element | null {
  if (element.parentElement !== null) return element.parentElement
  const root = element.getRootNode() as Partial<ShadowRoot>
  return root.host ?? null
}

/** Its image layers as gradients, top first; null where one is no gradient or can't be read. */
function layersOf(style: CSSStyleDeclaration): Array<Gradient | null> | null {
  const value = style.getPropertyValue('background-image').trim()
  if (value === '' || value === 'none') return []
  const parsed = splitTopLevel(value, ',').map((layer) => (layer === 'none' ? 'none' : parseGradient(layer, style.color)))
  if (parsed.some((layer) => layer === null || layer === 'unreadable')) return null
  return parsed.map((layer) => (layer === 'none' ? null : (layer as Gradient)))
}

function faded(colour: Rgba, opacity: number): Rgba {
  return opacity >= 1 ? colour : { ...colour, a: colour.a * opacity }
}

/** Reads each element's paint once per check. */
class PaintReader {
  private readonly painters = new Map<Element, Painter>()
  private readonly opacities = new Map<Element, number>()

  constructor(private readonly view: Window) {}

  /** The product of the element's opacity and every ancestor's. */
  opacity(element: Element): number {
    const known = this.opacities.get(element)
    if (known !== undefined) return known
    const parent = parentOf(element)
    const value = opacityOf(this.view.getComputedStyle(element)) * (parent === null ? 1 : this.opacity(parent))
    this.opacities.set(element, value)
    return value
  }

  painter(element: Element): Painter {
    const known = this.painters.get(element)
    if (known !== undefined) return known
    const style = this.view.getComputedStyle(element)
    const painter: Painter = {
      style,
      rect: this.paintsCanvas(element) ? viewportOf(this.view) : element.getBoundingClientRect(),
      opacity: this.opacity(element),
      colour: parseCssColor(style.backgroundColor || 'transparent'),
      layers: layersOf(style),
    }
    this.painters.set(element, painter)
    return painter
  }

  /** Whether <html> paints a background of its own; else body's is the page's root background. */
  rootPaints(): boolean {
    const { colour, layers } = this.painter(this.view.document.documentElement)
    return colour === null || layers === null || colour.a > 0 || layers.length > 0
  }

  /** Whether the element's background is the page's root background, which covers the whole page wherever its box ends. */
  private paintsCanvas(element: Element): boolean {
    const doc = this.view.document
    return element === doc.documentElement || (element === doc.body && !this.rootPaints())
  }

  /** What `element` paints at `point`, faded by its opacity; null where it can't be read. */
  paintAt(element: Element, point: Point): Paint | null {
    const { style, rect, opacity, colour, layers } = this.painter(element)
    if (colour === null || layers === null) return null
    const reaches = colour.a > 0 && colourReaches(style, rect, point)
    const painted: Rgba[][] = []
    for (let index = layers.length - 1; index >= 0; index -= 1) {
      const layer = layers[index]
      const tile = layer === null ? null : tileAt(style, index, rect, this.view, point)
      if (layer === null || tile === null || tile.covers === 'no') continue
      const colours = tile.covers === 'yes' ? gradientColoursAt(layer, tile, tile.x, tile.y) : [...gradientColours(layer), CLEAR]
      painted.push(colours.map((paint) => faded(paint, opacity)))
    }
    return { colour: reaches ? faded(colour, opacity) : null, layers: painted }
  }
}

function viewportOf(view: Window): Rect {
  return { left: 0, top: 0, width: view.innerWidth, height: view.innerHeight }
}

/** The page under every element: white, or white and black where the root allows a dark scheme. */
function pageGrounds(view: Window, doc: Document): readonly Rgba[] {
  return canvasGrounds(view.getComputedStyle(doc.documentElement).colorScheme.includes('dark'))
}

/** Every opaque colour `paints` (bottom element last) make over `start`, deduplicated; null past MAX_GROUNDS_PER_POINT. */
function blendDown(start: readonly Rgba[], paints: readonly Paint[]): Rgba[] | null {
  let grounds = new Map(start.map((ground): [string, Rgba] => [keyOf(ground), ground]))
  const bottomFirst = [...paints].reverse().flatMap(({ colour, layers }) => [...(colour === null ? [] : [[colour]]), ...layers])
  for (const layer of bottomFirst) {
    const next = new Map<string, Rgba>()
    grounds.forEach((ground) => layer.forEach((paint) => {
      const blended = over(paint, ground)
      next.set(keyOf(blended), blended)
    }))
    if (next.size > MAX_GROUNDS_PER_POINT) return null
    grounds = next
  }
  return [...grounds.values()]
}

function keyOf(colour: Rgba): string {
  return `${Math.round(colour.r)},${Math.round(colour.g)},${Math.round(colour.b)}`
}

/**
 * The elements behind the words at one point, from the words' own element down: those the hit-test found,
 * with the page's root background last (<html>, and body where its background is the page's), wherever their
 * boxes end. Null where the hit-test didn't reach the words.
 */
function behindWords(hits: readonly Element[], words: Element, reader: PaintReader): Element[] | null {
  const from = hits.indexOf(words)
  if (from < 0) return null
  const doc = words.ownerDocument
  const [root, body] = [doc.documentElement, doc.body]
  const between = hits.slice(from).filter((element) => element !== root && element !== body)
  const bodyPaintsHere = body !== null && (hits.includes(body) || !reader.rootPaints())
  return [...between, ...(bodyPaintsHere ? [body] : []), root]
}

/** Every ground behind the words at one point; null where one can't be read. */
function groundsAt(elements: readonly Element[], point: Point, reader: PaintReader, view: Window): Rgba[] | null {
  const paints: Paint[] = []
  for (const element of elements) {
    const paint = reader.paintAt(element, point)
    if (paint === null) return null
    paints.push(paint)
    if (paint.colour !== null && isOpaque(paint.colour)) return blendDown([paint.colour], paints)
  }
  return blendDown(pageGrounds(view, view.document), paints)
}

/**
 * Every ground behind the words across the points the check hit-tested: null when one can't be read (the label
 * counts as hidden), undefined when no hit-test reached the words (the caller reads the ancestors instead).
 */
export function groundsUnderWords(words: Element, stacks: readonly PointStack[] | null, view: Window): readonly Rgba[] | null | undefined {
  const reader = new PaintReader(view)
  const grounds = new Map<string, Rgba>()
  let reached = false
  for (const { x, y, hits } of stacks ?? []) {
    const elements = behindWords(hits, words, reader)
    if (elements === null) continue
    reached = true
    const here = groundsAt(elements, { x, y }, reader, view)
    if (here === null) return null
    here.forEach((ground) => grounds.set(keyOf(ground), ground))
  }
  return reached ? [...grounds.values()] : undefined
}

/** What is painted behind the words by their ancestors alone, nearest first: one colour, or two where the page may be white or black. */
export function groundsBehind(styles: readonly CSSStyleDeclaration[], view: Window): readonly Rgba[] | null {
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
