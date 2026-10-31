/**
 * PRD-264 EUAI-11a — reading a computed CSS gradient for the AI label's ground check (ground.ts): its
 * colour stops and hints, and the geometry gradient-paint.ts needs to tell which colour it paints at a
 * point (a linear gradient's angle or corner, a radial gradient's shape, size and centre, a conic
 * gradient's start angle and centre).
 *
 * What it can't read, it says so: a layer that is no gradient (`url()`, `image-set()` and the rest) reads
 * as null, so the ground counts as unknown; a stop whose colour can't be read makes the gradient
 * 'unreadable', which fails closed too. A gradient whose geometry or colour space this module doesn't
 * read (a prefixed `-webkit-` gradient, `in oklch`) keeps its stops with no geometry, and the check then
 * takes every colour it could paint.
 */
import { parseCssColor, type Rgba } from './contrast'
import { type CssLength, cssAngle, cssCall, cssLength, splitTopLevel } from './css-tokens'

export interface CssPosition {
  readonly x: CssLength
  readonly y: CssLength
}

/** A CSS angle (0deg points up, clockwise), or `to` a corner: the signs of its sides (right and bottom are 1). */
export type LinearDirection = { readonly angle: number } | { readonly corner: { readonly x: 1 | -1; readonly y: 1 | -1 } }

export type RadialExtent = 'closest-side' | 'farthest-side' | 'closest-corner' | 'farthest-corner'

export type GradientGeometry =
  | { readonly kind: 'linear'; readonly direction: LinearDirection }
  | {
      readonly kind: 'radial'
      readonly circle: boolean
      readonly size: { readonly extent: RadialExtent } | { readonly radii: readonly [CssLength, CssLength] }
      readonly at: CssPosition
    }
  | { readonly kind: 'conic'; readonly from: number; readonly at: CssPosition }

/** A colour stop; `at` null where the gradient leaves it to be spaced evenly. A conic stop's angle is a share of the turn. */
export interface GradientStop {
  readonly colour: Rgba
  readonly at: CssLength | null
}

/** Where the blend between two stops is half-way. */
export interface GradientHint {
  readonly hint: CssLength
}

export type GradientEntry = GradientStop | GradientHint

export interface Gradient {
  readonly repeating: boolean
  /** null: a geometry or colour space this module doesn't read; only the gradient's colours are known. */
  readonly geometry: GradientGeometry | null
  readonly entries: readonly GradientEntry[]
}

type Kind = GradientGeometry['kind']

const GRADIENT_FUNCTION = /^(-webkit-|-moz-)?(repeating-)?(linear|radial|conic)-gradient$/
const COLOUR_FUNCTIONS = new Set(['rgb', 'rgba', 'hsl', 'hsla', 'hwb', 'lab', 'lch', 'oklab', 'oklch', 'color', 'color-mix', 'light-dark', 'device-cmyk'])
/** Words a gradient's first argument uses that are no colour. */
const PRELUDE_WORDS = new Set([
  'to', 'at', 'from', 'in', 'left', 'right', 'top', 'bottom', 'center', 'circle', 'ellipse', 'closest-side', 'closest-corner',
  'farthest-side', 'farthest-corner', 'shorter', 'longer', 'increasing', 'decreasing', 'hue', 'srgb', 'srgb-linear', 'display-p3',
  'a98-rgb', 'prophoto-rgb', 'rec2020', 'lab', 'oklab', 'xyz', 'xyz-d50', 'xyz-d65', 'hsl', 'hwb', 'lch', 'oklch',
])
const EXTENTS: ReadonlySet<string> = new Set(['closest-side', 'farthest-side', 'closest-corner', 'farthest-corner'])
const SIDE_ANGLES: ReadonlyMap<string, number> = new Map([['top', 0], ['right', 90], ['bottom', 180], ['left', 270]])
const X_SHARES: ReadonlyMap<string, number> = new Map([['left', 0], ['center', 50], ['right', 100]])
const Y_SHARES: ReadonlyMap<string, number> = new Map([['top', 0], ['center', 50], ['bottom', 100]])
const CENTRE: CssPosition = { x: { px: 0, pct: 50 }, y: { px: 0, pct: 50 } }
const DEFAULT_DIRECTION: LinearDirection = { angle: 180 }
const DEGREES_PER_TURN = 360
/** Interpolation in the space a legacy colour gradient uses anyway. */
const READ_SPACE = 'srgb'

function isPosition(token: string, kind: Kind): boolean {
  return cssLength(token) !== null || (kind === 'conic' && cssAngle(token) !== null)
}

function isColourToken(token: string, kind: Kind): boolean {
  if (isPosition(token, kind) || cssAngle(token) !== null) return false
  const call = cssCall(token)
  return call === null ? !PRELUDE_WORDS.has(token.toLowerCase()) : COLOUR_FUNCTIONS.has(call.name)
}

/** A stop's position: a length, or for a conic gradient an angle as a share of the turn. */
function positionOf(token: string, kind: Kind): CssLength | null {
  const angle = kind === 'conic' ? cssAngle(token) : null
  if (angle !== null && cssLength(token) === null) return { px: 0, pct: (angle / DEGREES_PER_TURN) * 100 }
  return cssLength(token)
}

function colourOf(token: string, currentColour: string): Rgba | null {
  return parseCssColor(token.toLowerCase() === 'currentcolor' ? currentColour : token)
}

/** One argument after the first: a stop (a colour and up to two positions) or a hint; null when it reads as neither. */
function entriesOf(argument: string, kind: Kind, currentColour: string): GradientEntry[] | null {
  const tokens = splitTopLevel(argument, ' ')
  const colours = tokens.filter((token) => !isPosition(token, kind))
  const positions = tokens.filter((token) => isPosition(token, kind)).map((token) => positionOf(token, kind))
  if (positions.some((position) => position === null) || positions.length > 2) return null
  const at = positions as CssLength[]
  if (colours.length === 0) return at.length === 1 ? [{ hint: at[0] }] : null
  const colour = colours.length === 1 ? colourOf(colours[0], currentColour) : null
  if (colour === null) return null
  return at.length === 0 ? [{ colour, at: null }] : at.map((position) => ({ colour, at: position }))
}

/** Tokens of an axis position (`50%`, `right 10px`, `center`); null for anything else. */
function axisOf(tokens: readonly string[], shares: ReadonlyMap<string, number>): CssLength | null {
  const [first, second] = tokens
  const share = shares.get(first?.toLowerCase() ?? '')
  if (tokens.length === 1) return share === undefined ? cssLength(first) : { px: 0, pct: share }
  const offset = tokens.length === 2 && share !== undefined ? cssLength(second) : null
  if (offset === null) return null
  return share === 100 ? { px: -offset.px, pct: 100 - offset.pct } : { px: offset.px, pct: share + offset.pct }
}

/** The tokens after `at`: one or two values, or edges with offsets; null for anything else. */
export function cssPosition(tokens: readonly string[]): CssPosition | null {
  const words = tokens.map((token) => token.toLowerCase())
  const vertical = Y_SHARES.has(words[0] ?? '') && words[0] !== 'center'
  if (words.length === 1 && vertical) return { x: CENTRE.x, y: axisOf(words, Y_SHARES) as CssLength }
  if (words.length === 1) {
    const x = axisOf(words, X_SHARES)
    return x === null ? null : { x, y: CENTRE.y }
  }
  if (words.length < 2 || words.length > 4) return null
  const half = words.length === 2 ? 1 : 2
  const [first, second] = [tokens.slice(0, half), tokens.slice(half)]
  const swapped = vertical
  const x = axisOf(swapped ? second : first, X_SHARES)
  const y = axisOf(swapped ? first : second, Y_SHARES)
  return x === null || y === null ? null : { x, y }
}

function linearGeometry(tokens: readonly string[]): GradientGeometry | null {
  if (tokens.length === 0) return { kind: 'linear', direction: DEFAULT_DIRECTION }
  const angle = tokens.length === 1 ? cssAngle(tokens[0]) : null
  if (angle !== null) return { kind: 'linear', direction: { angle } }
  const sides = tokens[0].toLowerCase() === 'to' ? tokens.slice(1).map((side) => side.toLowerCase()) : []
  const toSide = sides.length === 1 ? SIDE_ANGLES.get(sides[0]) : undefined
  if (toSide !== undefined) return { kind: 'linear', direction: { angle: toSide } }
  const across = sides.find((side) => side === 'left' || side === 'right')
  const down = sides.find((side) => side === 'top' || side === 'bottom')
  if (sides.length !== 2 || across === undefined || down === undefined) return null
  return { kind: 'linear', direction: { corner: { x: across === 'right' ? 1 : -1, y: down === 'bottom' ? 1 : -1 } } }
}

function radialSize(tokens: readonly string[]): { readonly circle: boolean; readonly size: Extract<GradientGeometry, { kind: 'radial' }>['size'] } | null {
  const words = tokens.map((token) => token.toLowerCase())
  const lengths = tokens.filter((token) => cssLength(token) !== null).map((token) => cssLength(token) as CssLength)
  const extent = words.find((word) => EXTENTS.has(word)) as RadialExtent | undefined
  const rest = words.filter((word) => word !== 'circle' && word !== 'ellipse' && !EXTENTS.has(word) && cssLength(word) === null)
  if (rest.length > 0 || (extent !== undefined && lengths.length > 0) || lengths.length > 2) return null
  const circle = words.includes('circle') || (lengths.length === 1 && !words.includes('ellipse'))
  if (lengths.length === 0) return { circle, size: { extent: extent ?? 'farthest-corner' } }
  if (lengths.length === 1 && (!circle || lengths[0].pct !== 0)) return null
  return { circle, size: { radii: [lengths[0], lengths[lengths.length - 1]] } }
}

function radialGeometry(tokens: readonly string[]): GradientGeometry | null {
  const split = tokens.findIndex((token) => token.toLowerCase() === 'at')
  const shape = radialSize(split < 0 ? tokens : tokens.slice(0, split))
  const at = split < 0 ? CENTRE : cssPosition(tokens.slice(split + 1))
  return shape === null || at === null ? null : { kind: 'radial', ...shape, at }
}

function conicGeometry(tokens: readonly string[]): GradientGeometry | null {
  const words = tokens.map((token) => token.toLowerCase())
  const fromAt = words.indexOf('from')
  const atAt = words.indexOf('at')
  const from = fromAt < 0 ? 0 : cssAngle(tokens[fromAt + 1] ?? '')
  const at = atAt < 0 ? CENTRE : cssPosition(tokens.slice(atAt + 1, fromAt > atAt ? fromAt : undefined))
  const known = (fromAt < 0 ? 0 : 2) + (atAt < 0 ? 0 : tokens.length - atAt - (fromAt > atAt ? tokens.length - fromAt : 0))
  if (from === null || at === null || known !== tokens.length) return null
  return { kind: 'conic', from, at }
}

/** The geometry a gradient's first argument sets (none set: the default), after its colour space; null where it isn't read. */
function geometryOf(kind: Kind, prelude: readonly string[]): GradientGeometry | null {
  const spaceAt = prelude.findIndex((token) => token.toLowerCase() === 'in')
  if (spaceAt >= 0 && (prelude[spaceAt + 1] ?? '').toLowerCase() !== READ_SPACE) return null
  const tokens = spaceAt < 0 ? prelude : [...prelude.slice(0, spaceAt), ...prelude.slice(spaceAt + 2)]
  if (kind === 'linear') return linearGeometry(tokens)
  return kind === 'radial' ? radialGeometry(tokens) : conicGeometry(tokens)
}

/**
 * A computed background layer as a gradient: null when it is no gradient (`url()`, `image-set()`, …),
 * 'unreadable' when a stop's colour or the gradient's shape can't be read. `currentColour` is the
 * element's own computed colour, for a `currentcolor` stop.
 */
export function parseGradient(layer: string, currentColour: string): Gradient | 'unreadable' | null {
  const call = cssCall(layer)
  const match = call?.name.match(GRADIENT_FUNCTION)
  if (!call || !match) return null
  const kind = match[3] as Kind
  const args = splitTopLevel(call.args, ',')
  const first = splitTopLevel(args[0] ?? '', ' ')
  const hasPrelude = !first.some((token) => isColourToken(token, kind))
  const entries = (hasPrelude ? args.slice(1) : args).map((argument) => entriesOf(argument, kind, currentColour))
  if (entries.length === 0 || entries.some((entry) => entry === null)) return 'unreadable'
  const geometry = match[1] === undefined ? geometryOf(kind, hasPrelude ? first : []) : null
  return { repeating: match[2] !== undefined, geometry, entries: entries.flat() as GradientEntry[] }
}
