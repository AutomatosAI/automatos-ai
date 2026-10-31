/**
 * PRD-264 EUAI-11a — which colour a gradient (gradient-parse.ts) paints at a point of its box, for the AI
 * label's ground check (ground.ts).
 *
 * Where the geometry reads, the answer is the one colour the browser paints there: the point's place along
 * the gradient line (linear), its distance out along the ending shape (radial) or its angle round the centre
 * (conic), the stops placed as CSS places them (missing positions spaced evenly, none before the one ahead
 * of it), a hint's curve, a repeating gradient's period, and the blend between the two stops either side in
 * premultiplied sRGB, as a legacy-colour gradient blends. Where it doesn't (an unread geometry, a degenerate
 * box or shape, stops that don't place), the answer is every colour the gradient could paint: its stops and
 * the blends a quarter, half and three quarters of the way between neighbours, since a ramp from dark to light
 * passes through every grey between them, the words' own among them.
 */
import type { Rgba } from './contrast'
import { resolveLength } from './css-tokens'
import type { Gradient, GradientEntry, GradientGeometry, LinearDirection } from './gradient-parse'

/** Where the worst case also reads the blend between two neighbouring stops, as shares of the way. */
export const BETWEEN_STOPS = [0.25, 0.5, 0.75] as const

export interface PaintBox {
  readonly width: number
  readonly height: number
}

/** Where a point is along a gradient: `t`, its share of the gradient line, and the line's length in px (0: px don't place). */
interface Along {
  readonly t: number
  readonly length: number
}

/** Stops placed along the line as shares of it, and the hint between each stop and the next (null: half-way). */
interface Placed {
  readonly colours: readonly Rgba[]
  readonly at: readonly number[]
  readonly hints: ReadonlyArray<number | null>
}

const CLEAR: Rgba = Object.freeze({ r: 0, g: 0, b: 0, a: 0 })
const DEGREES_PER_TURN = 360
const HALF = 0.5
const SAME_PLACE = 1e-9

/** `from` blended `share` of the way to `to`, premultiplied, as CSS blends a gradient's stops. */
export function blend(from: Rgba, to: Rgba, share: number): Rgba {
  const a = from.a + (to.a - from.a) * share
  if (a <= 0) return CLEAR
  const mix = (first: number, second: number) => (first * from.a * (1 - share) + second * to.a * share) / a
  return { r: mix(from.r, to.r), g: mix(from.g, to.g), b: mix(from.b, to.b), a }
}

function stopColours(entries: readonly GradientEntry[]): Rgba[] {
  return entries.flatMap((entry) => ('colour' in entry ? [entry.colour] : []))
}

/** Every colour `gradient` could paint anywhere: its stops, and blends between each stop and the next. */
export function gradientColours(gradient: Gradient): Rgba[] {
  const colours = stopColours(gradient.entries)
  const blends = colours.slice(1).flatMap((next, index) => BETWEEN_STOPS.map((share) => blend(colours[index], next, share)))
  return [...colours, ...blends]
}

function linearAlong(direction: LinearDirection, box: PaintBox, x: number, y: number): Along | null {
  const radians = 'angle' in direction ? (direction.angle * Math.PI) / (DEGREES_PER_TURN / 2) : 0
  const raw = 'angle' in direction
    ? [Math.sin(radians), -Math.cos(radians)]
    : [direction.corner.x * box.height, direction.corner.y * box.width]
  const norm = Math.hypot(raw[0], raw[1])
  if (norm === 0) return null
  const [dx, dy] = [raw[0] / norm, raw[1] / norm]
  const length = Math.abs(box.width * dx) + Math.abs(box.height * dy)
  if (length <= 0) return null
  return { t: ((x - box.width / 2) * dx + (y - box.height / 2) * dy) / length + HALF, length }
}

type Radial = Extract<GradientGeometry, { kind: 'radial' }>

/** The ending shape's radii, as CSS sizes a radial gradient in a box with its centre at (cx, cy). */
function radiiOf(geometry: Radial, box: PaintBox, cx: number, cy: number): readonly [number, number] {
  const { size, circle } = geometry
  if ('radii' in size) return [resolveLength(size.radii[0], box.width), resolveLength(size.radii[1], box.height)]
  const across = [Math.abs(cx), Math.abs(box.width - cx)]
  const down = [Math.abs(cy), Math.abs(box.height - cy)]
  const pick = size.extent.startsWith('closest') ? Math.min : Math.max
  const [sideX, sideY] = [pick(...across), pick(...down)]
  if (size.extent.endsWith('side')) return circle ? [pick(sideX, sideY), pick(sideX, sideY)] : [sideX, sideY]
  if (circle) return [Math.hypot(sideX, sideY), Math.hypot(sideX, sideY)]
  return [sideX * Math.SQRT2, sideY * Math.SQRT2]
}

function radialAlong(geometry: Radial, box: PaintBox, x: number, y: number): Along | null {
  const cx = resolveLength(geometry.at.x, box.width)
  const cy = resolveLength(geometry.at.y, box.height)
  const [rx, ry] = radiiOf(geometry, box, cx, cy)
  if (!(rx > 0 && ry > 0)) return null
  return { t: Math.hypot((x - cx) / rx, (y - cy) / ry), length: rx }
}

function conicAlong(geometry: Extract<GradientGeometry, { kind: 'conic' }>, box: PaintBox, x: number, y: number): Along {
  const cx = resolveLength(geometry.at.x, box.width)
  const cy = resolveLength(geometry.at.y, box.height)
  const degrees = (Math.atan2(x - cx, cy - y) * (DEGREES_PER_TURN / 2)) / Math.PI - geometry.from
  return { t: (((degrees % DEGREES_PER_TURN) + DEGREES_PER_TURN) % DEGREES_PER_TURN) / DEGREES_PER_TURN, length: 0 }
}

function alongAt(geometry: GradientGeometry, box: PaintBox, x: number, y: number): Along | null {
  if (geometry.kind === 'linear') return linearAlong(geometry.direction, box, x, y)
  return geometry.kind === 'radial' ? radialAlong(geometry, box, x, y) : conicAlong(geometry, box, x, y)
}

/** A position as a share of the line; NaN where px can't place (a conic gradient's line has no length). */
function shareOf(position: { readonly px: number; readonly pct: number }, length: number): number {
  if (position.px !== 0 && length <= 0) return Number.NaN
  return position.pct / 100 + (length > 0 ? position.px / length : 0)
}

/** Positions CSS leaves out: the first at 0, the last at 1, none before the one ahead, the rest spaced evenly. */
function fixUp(raw: ReadonlyArray<number | null>): number[] {
  const known = raw.map((value, index) => value ?? (index === 0 ? 0 : index === raw.length - 1 ? 1 : null))
  let highest = Number.NEGATIVE_INFINITY
  const clamped = known.map((value) => {
    if (value === null) return null
    highest = Math.max(highest, value)
    return highest
  })
  const placedAt = clamped.flatMap((value, index) => (value === null ? [] : [index]))
  return clamped.map((value, index) => {
    if (value !== null) return value
    const after = placedAt.find((placed) => placed > index) as number
    const before = placedAt[placedAt.indexOf(after) - 1]
    const [from, to] = [clamped[before] as number, clamped[after] as number]
    return from + ((to - from) * (index - before)) / (after - before)
  })
}

/** The stops placed along a line `length` px long; null where they don't place (a hint at an end, two in a row, a px in a conic). */
function place(entries: readonly GradientEntry[], length: number): Placed | null {
  const colours: Rgba[] = []
  const raw: Array<number | null> = []
  const hints: Array<number | null> = []
  for (const entry of entries) {
    if ('hint' in entry) {
      if (colours.length === 0 || hints[colours.length - 1] !== null) return null
      hints[colours.length - 1] = shareOf(entry.hint, length)
      continue
    }
    colours.push(entry.colour)
    raw.push(entry.at === null ? null : shareOf(entry.at, length))
    hints.push(null)
  }
  if (colours.length === 0 || hints[colours.length - 1] !== null || [...raw, ...hints].some((value) => Number.isNaN(value))) return null
  return { colours, at: fixUp(raw), hints: hints.slice(0, -1) }
}

/** The blend's share at `t` between stops at `from` and `to`, bent by a hint (CSS's curve through the hint at half-way). */
function shareBetween(t: number, from: number, to: number, hint: number | null): number {
  const share = (t - from) / (to - from)
  if (hint === null) return share
  const bend = (hint - from) / (to - from)
  if (bend <= 0) return 1
  if (bend >= 1) return 0
  return share ** (Math.log(HALF) / Math.log(bend))
}

/** The colours at `t` along placed stops: one, or both where two stops sit at the very place. */
function colourAt(placed: Placed, t: number): Rgba[] {
  const { colours, at, hints } = placed
  const last = colours.length - 1
  if (t <= at[0]) return [colours[0]]
  if (t >= at[last]) return [colours[last]]
  const index = at.findIndex((position, step) => step < last && t >= position && t <= at[step + 1])
  const [from, to] = [at[index], at[index + 1]]
  if (to - from < SAME_PLACE) return [colours[index], colours[index + 1]]
  return [blend(colours[index], colours[index + 1], shareBetween(t, from, to, hints[index]))]
}

/** `t` folded into a repeating gradient's period; null where the period is empty. */
function folded(placed: Placed, t: number): number | null {
  const first = placed.at[0]
  const period = placed.at[placed.at.length - 1] - first
  if (period < SAME_PLACE) return null
  return first + ((((t - first) % period) + period) % period)
}

/** The one colour (two at a hard stop) `gradient` paints at (x, y) of a box; null where its geometry can't say. */
function exactColours(gradient: Gradient, box: PaintBox, x: number, y: number): Rgba[] | null {
  if (gradient.geometry === null || !(box.width > 0 && box.height > 0)) return null
  const along = alongAt(gradient.geometry, box, x, y)
  const placed = along === null ? null : place(gradient.entries, along.length)
  if (along === null || placed === null) return null
  const t = gradient.repeating ? folded(placed, along.t) : along.t
  return t === null ? null : colourAt(placed, t)
}

/** The colours `gradient` paints at (x, y) of a `box`: the one there where its geometry reads, else every one it could. */
export function gradientColoursAt(gradient: Gradient, box: PaintBox, x: number, y: number): Rgba[] {
  return exactColours(gradient, box, x, y) ?? gradientColours(gradient)
}
