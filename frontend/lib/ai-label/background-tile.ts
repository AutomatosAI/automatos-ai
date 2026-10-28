/**
 * PRD-264 EUAI-11a — where an element's background paints, for the AI label's ground check (ground.ts):
 * whether its background colour reaches a point (its clip box), and, for each image layer, whether a tile
 * of it lies under the point and where in that tile the point falls (its clip, origin, size, position,
 * repeat and attachment, as CSS lays them out), so a gradient is read at the point it paints there. A
 * corner bracket drawn as a small no-repeat gradient doesn't count where the label's words are, and a grid
 * of 1 px lines counts only on a line.
 *
 * A value this module doesn't read leaves the layer 'maybe' under the point: the check then takes every
 * colour the layer could paint there, and nothing at all too. A layer clipped to text paints only inside
 * its element's own glyphs, so it is no ground for the label's words.
 */
import { type CssLength, cssLength, listItem, resolveLength, splitTopLevel } from './css-tokens'

export interface Rect {
  readonly left: number
  readonly top: number
  readonly width: number
  readonly height: number
}

export interface Point {
  readonly x: number
  readonly y: number
}

/** Under the point: no tile of the layer; a tile, its size and where in it the point is; or a layout this module can't read. */
export type TileAt =
  | { readonly covers: 'no' }
  | { readonly covers: 'maybe' }
  | { readonly covers: 'yes'; readonly width: number; readonly height: number; readonly x: number; readonly y: number }

type Repeat = 'repeat' | 'space' | 'round' | 'no-repeat'

const NO_TILE: TileAt = Object.freeze({ covers: 'no' })
const MAYBE_TILE: TileAt = Object.freeze({ covers: 'maybe' })
const SIDES = ['top', 'right', 'bottom', 'left'] as const
const REPEATS: ReadonlySet<string> = new Set(['repeat', 'space', 'round', 'no-repeat'])
const AXIS_SHARES = {
  x: new Map([['left', 0], ['center', 50], ['right', 100]]),
  y: new Map([['top', 0], ['center', 50], ['bottom', 100]]),
} as const
const WHOLE_AREA = new Set(['auto', 'auto auto', 'cover', 'contain'])
const START = '0%'

function widths(style: CSSStyleDeclaration, property: (side: string) => string): readonly number[] {
  return SIDES.map((side) => Number.parseFloat(style.getPropertyValue(property(side))) || 0)
}

function inset(rect: Rect, [top, right, bottom, left]: readonly number[]): Rect {
  return { left: rect.left + left, top: rect.top + top, width: rect.width - left - right, height: rect.height - top - bottom }
}

/** The element's border, padding or content box; null for a box this module doesn't read ('text' among them). */
function boxOf(style: CSSStyleDeclaration, rect: Rect, box: string): Rect | null {
  if (box === 'border-box') return rect
  const padding = inset(rect, widths(style, (side) => `border-${side}-width`))
  if (box === 'padding-box') return padding
  return box === 'content-box' ? inset(padding, widths(style, (side) => `padding-${side}`)) : null
}

function contains(rect: Rect, point: Point): boolean {
  return point.x >= rect.left && point.x < rect.left + rect.width && point.y >= rect.top && point.y < rect.top + rect.height
}

function clipOf(style: CSSStyleDeclaration, layer: number): string {
  const clip = style.getPropertyValue('background-clip') || style.getPropertyValue('-webkit-background-clip')
  return listItem(clip, layer, 'border-box')
}

/** Whether the element's background colour reaches `point`: inside the clip box of its bottom layer, never clipped to text. */
export function colourReaches(style: CSSStyleDeclaration, rect: Rect, point: Point): boolean {
  const layers = Math.max(1, splitTopLevel(style.getPropertyValue('background-image') || 'none', ',').length)
  const clipped = boxOf(style, rect, clipOf(style, layers - 1))
  return clipped !== null && contains(clipped, point)
}

/** A tile's size in the positioning area; null for a value this module doesn't read. */
function tileSize(value: string, area: Rect): readonly [number, number] | null {
  if (WHOLE_AREA.has(value)) return [area.width, area.height]
  const [across, down = 'auto'] = splitTopLevel(value, ' ')
  const extentOf = (token: string, extent: number) => {
    const length = token === 'auto' ? null : cssLength(token)
    return token === 'auto' ? extent : length === null ? null : resolveLength(length, extent)
  }
  const [width, height] = [extentOf(across, area.width), extentOf(down, area.height)]
  return width === null || height === null ? null : [width, height]
}

function axisPosition(value: string, shares: ReadonlyMap<string, number>): CssLength | null {
  const [first, second] = splitTopLevel(value, ' ')
  const share = shares.get(first ?? '')
  if (second === undefined) return share === undefined ? cssLength(first ?? START) : { px: 0, pct: share }
  const offset = share === undefined ? null : cssLength(second)
  if (offset === null || share === undefined) return null
  return share === 100 ? { px: -offset.px, pct: 100 - offset.pct } : { px: offset.px, pct: share + offset.pct }
}

/** The layer's position on each axis, from the longhands, or the shorthand where those are unset. */
function positionOf(style: CSSStyleDeclaration, layer: number): readonly [CssLength | null, CssLength | null] {
  const across = listItem(style.getPropertyValue('background-position-x'), layer, '')
  const down = listItem(style.getPropertyValue('background-position-y'), layer, '')
  if (across !== '' && down !== '') return [axisPosition(across, AXIS_SHARES.x), axisPosition(down, AXIS_SHARES.y)]
  const both = splitTopLevel(listItem(style.getPropertyValue('background-position'), layer, `${START} ${START}`), ' ')
  return both.length === 2 ? [axisPosition(both[0], AXIS_SHARES.x), axisPosition(both[1], AXIS_SHARES.y)] : [null, null]
}

function repeatOf(value: string): readonly [Repeat, Repeat] | null {
  if (value === 'repeat-x') return ['repeat', 'no-repeat']
  if (value === 'repeat-y') return ['no-repeat', 'repeat']
  const [across, down = across] = splitTopLevel(value, ' ')
  return REPEATS.has(across) && REPEATS.has(down) ? [across as Repeat, down as Repeat] : null
}

/** A 'round' tile stretched to fit a whole number of times. */
function rounded(size: number, extent: number, repeat: Repeat): number {
  return repeat === 'round' && size > 0 ? extent / Math.max(1, Math.round(extent / size)) : size
}

/** Where `point` falls in the tile along one axis; null where no tile lies under it. */
function along(point: number, start: number, size: number, repeat: Repeat): number | null {
  const offset = point - start
  if (repeat !== 'no-repeat') return ((offset % size) + size) % size
  return offset >= 0 && offset < size ? offset : null
}

function viewportOf(view: Window): Rect {
  return { left: 0, top: 0, width: view.innerWidth, height: view.innerHeight }
}

/** The positioning area: the viewport for a fixed layer, else the origin box. */
function areaOf(style: CSSStyleDeclaration, rect: Rect, layer: number, view: Window): Rect | null {
  if (listItem(style.getPropertyValue('background-attachment'), layer, 'scroll') === 'fixed') return viewportOf(view)
  return boxOf(style, rect, listItem(style.getPropertyValue('background-origin'), layer, 'padding-box'))
}

/** Where the `layer`th background layer of an element with border box `rect` has a tile under `point`, if anywhere. */
export function tileAt(style: CSSStyleDeclaration, layer: number, rect: Rect, view: Window, point: Point): TileAt {
  const clip = clipOf(style, layer)
  if (clip === 'text') return NO_TILE
  const clipped = boxOf(style, rect, clip)
  if (clipped !== null && !contains(clipped, point)) return NO_TILE
  const area = areaOf(style, rect, layer, view)
  const size = area === null ? null : tileSize(listItem(style.getPropertyValue('background-size'), layer, 'auto'), area)
  const [across, down] = positionOf(style, layer)
  const repeat = repeatOf(listItem(style.getPropertyValue('background-repeat'), layer, 'repeat'))
  if (clipped === null || area === null || size === null || across === null || down === null || repeat === null) return MAYBE_TILE
  const [width, height] = [rounded(size[0], area.width, repeat[0]), rounded(size[1], area.height, repeat[1])]
  if (!(width > 0 && height > 0)) return NO_TILE
  const x = along(point.x, area.left + resolveLength(across, area.width - width), width, repeat[0])
  const y = along(point.y, area.top + resolveLength(down, area.height - height), height, repeat[1])
  if (x === null || y === null) return NO_TILE
  return repeat.includes('space') ? MAYBE_TILE : { covers: 'yes', width, height, x, y }
}
