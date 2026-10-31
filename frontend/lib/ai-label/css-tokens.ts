/**
 * PRD-264 EUAI-11a — reading the pieces of a computed CSS value the AI label's ground check needs
 * (gradient.ts, background-tile.ts): lists split outside parentheses and quotes, a function call's name
 * and arguments, a length as a share of a reference plus px (`12px`, `40%`, `calc(100% - 7px)`), and an
 * angle in degrees. Anything else reads as null, and the callers then take the worst case or fail closed.
 */

/** A length as `pct` percent of whatever it is measured against, plus `px`. */
export interface CssLength {
  readonly px: number
  readonly pct: number
}

export interface CssCall {
  readonly name: string
  readonly args: string
}

const NUMBER = '[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:e[-+]?\\d+)?'
const PX_LENGTH = new RegExp(`^(${NUMBER})px$`, 'i')
const PERCENTAGE = new RegExp(`^(${NUMBER})%$`)
const PLAIN_ZERO = /^[-+]?0*\.?0+$/
const ANGLE = new RegExp(`^(${NUMBER})(deg|grad|rad|turn)$`, 'i')
const DEGREES_PER_UNIT: Readonly<Record<string, number>> = { deg: 1, grad: 0.9, rad: 180 / Math.PI, turn: 360 }
const CALL = /^([-a-z0-9]+)\(([\s\S]*)\)$/i
const CALC_OPERATOR = /\s+([-+])\s+/
const WHITESPACE = /\s/

/** Whether `ch` ends a piece of a list split at `separator`. */
function separates(ch: string, separator: ',' | ' '): boolean {
  return separator === ',' ? ch === ',' : WHITESPACE.test(ch)
}

/** `text` split at `separator` (a comma, or any whitespace) outside parentheses and quotes; pieces trimmed, empty ones dropped. */
export function splitTopLevel(text: string, separator: ',' | ' '): string[] {
  const pieces: string[] = []
  let depth = 0
  let quote: string | null = null
  let start = 0
  for (let index = 0; index < text.length; index += 1) {
    const ch = text[index]
    if (quote !== null) {
      if (ch === '\\') index += 1
      else if (ch === quote) quote = null
    } else if (ch === '"' || ch === "'") {
      quote = ch
    } else if (ch === '(' || ch === ')') {
      depth = Math.max(0, depth + (ch === '(' ? 1 : -1))
    } else if (depth === 0 && separates(ch, separator)) {
      pieces.push(text.slice(start, index))
      start = index + 1
    }
  }
  pieces.push(text.slice(start))
  return pieces.map((piece) => piece.trim()).filter((piece) => piece !== '')
}

/** A function call's lower-cased name and its arguments as written; null for anything else. */
export function cssCall(token: string): CssCall | null {
  const match = token.trim().match(CALL)
  return match ? { name: match[1].toLowerCase(), args: match[2] } : null
}

/** `calc()` of px and percentages joined by + and -; null for anything else (a product, a nested unit). */
function calcLength(args: string): CssLength | null {
  const parts = args.trim().split(CALC_OPERATOR)
  let total: CssLength = { px: 0, pct: 0 }
  for (let index = 0; index < parts.length; index += 2) {
    const sign = index > 0 && parts[index - 1] === '-' ? -1 : 1
    const term = cssLength(parts[index].replace(/^\(([\s\S]*)\)$/, '$1'))
    if (term === null) return null
    total = { px: total.px + sign * term.px, pct: total.pct + sign * term.pct }
  }
  return total
}

/** A length in px, a percentage, a bare 0, or `calc()` of those; null for anything else. */
export function cssLength(token: string): CssLength | null {
  const text = token.trim()
  if (PLAIN_ZERO.test(text)) return { px: 0, pct: 0 }
  const px = text.match(PX_LENGTH)
  if (px) return { px: Number(px[1]), pct: 0 }
  const pct = text.match(PERCENTAGE)
  if (pct) return { px: 0, pct: Number(pct[1]) }
  const call = cssCall(text)
  return call?.name === 'calc' ? calcLength(call.args) : null
}

/** `length` measured against `reference` px. */
export function resolveLength(length: CssLength, reference: number): number {
  return (length.pct / 100) * reference + length.px
}

/** An angle in degrees (deg, grad, rad or turn, or a bare 0); null for anything else. */
export function cssAngle(token: string): number | null {
  const text = token.trim()
  if (PLAIN_ZERO.test(text)) return 0
  const match = text.match(ANGLE)
  return match ? Number(match[1]) * DEGREES_PER_UNIT[match[2].toLowerCase()] : null
}

/** The `index`th item of a computed list (a layer's value: the list repeats when shorter), or `fallback` when the list is empty. */
export function listItem(list: string, index: number, fallback: string): string {
  const items = splitTopLevel(list, ',')
  return items.length === 0 ? fallback : items[index % items.length]
}
