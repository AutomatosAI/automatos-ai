/**
 * PRD-264 EUAI-11a — the colour half of "can a person see the AI label?": the colour its words are
 * painted in against the colour behind them, as the browser computed both (visibility.ts).
 *
 * A computed colour reaches here as `rgb()`/`rgba()` (every legacy sRGB colour), `color(srgb …)`, or
 * another space a theme wrote (`oklch()`, `lab()`, `color(display-p3 …)`). The first two are read as
 * text; any other is painted on a 1 x 1 canvas and read back as sRGB, so a colour this module can't
 * read in words still gets an honest answer. A value nothing can read is null, and the check then
 * treats the label as hidden. `Canvas` and `CanvasText` (what a DOM with no renderer, such as the
 * test runner's, reports unresolved) read as their light-scheme colours.
 */

export interface Rgba {
  /** Red, green and blue, 0 to 255. */
  readonly r: number
  readonly g: number
  readonly b: number
  /** Alpha, 0 to 1. */
  readonly a: number
}

const WHITE: Rgba = Object.freeze({ r: 255, g: 255, b: 255, a: 1 })
const BLACK: Rgba = Object.freeze({ r: 0, g: 0, b: 0, a: 1 })
const CLEAR: Rgba = Object.freeze({ r: 0, g: 0, b: 0, a: 0 })
const KEYWORDS: Readonly<Record<string, Rgba>> = Object.freeze({ transparent: CLEAR, canvas: WHITE, canvastext: BLACK })

const CHANNEL_MAX = 255
const OPAQUE = 0.999
const NUMBER = '(-?[\\d.]+(?:e[-+]?\\d+)?)'
const ALPHA = '(?:\\s*[,/]\\s*([\\d.]+%?))?'
const RGB_FUNCTION = new RegExp(`^rgba?\\(\\s*${NUMBER}[\\s,]+${NUMBER}[\\s,]+${NUMBER}${ALPHA}\\s*\\)$`)
const SRGB_FUNCTION = new RegExp(`^color\\(\\s*srgb\\s+${NUMBER}\\s+${NUMBER}\\s+${NUMBER}${ALPHA}\\s*\\)$`)

/** The canvas that reads a colour in any other space back as sRGB; null where there is none. */
let paintContext: OffscreenCanvasRenderingContext2D | null | undefined

function clampChannel(value: number): number {
  return Math.min(CHANNEL_MAX, Math.max(0, value))
}

function alphaOf(text: string | undefined): number {
  if (text === undefined) return 1
  const value = Number.parseFloat(text)
  if (!Number.isFinite(value)) return 1
  return Math.min(1, Math.max(0, text.endsWith('%') ? value / 100 : value))
}

function fromMatch(match: RegExpMatchArray | null, scale: number): Rgba | null {
  if (match === null) return null
  const [r, g, b] = [match[1], match[2], match[3]].map((channel) => clampChannel(Number.parseFloat(channel) * scale))
  return [r, g, b].every(Number.isFinite) ? { r, g, b, a: alphaOf(match[4]) } : null
}

function canvasContext(): OffscreenCanvasRenderingContext2D | null {
  if (paintContext !== undefined) return paintContext
  paintContext = typeof OffscreenCanvas === 'function'
    ? new OffscreenCanvas(1, 1).getContext('2d', { willReadFrequently: true })
    : null
  return paintContext
}

/** Paints `text` on the canvas and reads the pixel back; a colour the canvas refuses reads as clear. */
function fromCanvas(text: string): Rgba | null {
  const context = canvasContext()
  if (context === null) return null
  context.clearRect(0, 0, 1, 1)
  context.fillStyle = 'rgba(0, 0, 0, 0)'
  context.fillStyle = text
  context.fillRect(0, 0, 1, 1)
  const [r, g, b, a] = context.getImageData(0, 0, 1, 1).data
  return { r, g, b, a: a / CHANNEL_MAX }
}

/** A computed CSS colour as sRGB with alpha, or null when it can't be read. */
export function parseCssColor(value: string): Rgba | null {
  const text = value.trim().toLowerCase()
  if (text === '') return null
  return KEYWORDS[text] ?? fromMatch(text.match(RGB_FUNCTION), 1) ?? fromMatch(text.match(SRGB_FUNCTION), CHANNEL_MAX) ?? fromCanvas(text)
}

export function isOpaque(color: Rgba): boolean {
  return color.a >= OPAQUE
}

/** `top` painted over the opaque `bottom`. */
export function over(top: Rgba, bottom: Rgba): Rgba {
  const mix = (front: number, back: number) => front * top.a + back * (1 - top.a)
  return { r: mix(top.r, bottom.r), g: mix(top.g, bottom.g), b: mix(top.b, bottom.b), a: 1 }
}

function linear(channel: number): number {
  const value = channel / CHANNEL_MAX
  return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
}

function luminance(color: Rgba): number {
  return 0.2126 * linear(color.r) + 0.7152 * linear(color.g) + 0.0722 * linear(color.b)
}

/** The WCAG contrast ratio of two opaque colours, 1 to 21. */
export function contrastRatio(first: Rgba, second: Rgba): number {
  const [light, dark] = [luminance(first), luminance(second)].sort((x, y) => y - x)
  return (light + 0.05) / (dark + 0.05)
}

/** The page's own ground where no element paints one: white, and black too when the page may be dark. */
export function canvasGrounds(mayBeDark: boolean): readonly Rgba[] {
  return mayBeDark ? [WHITE, BLACK] : [WHITE]
}
