/**
 * PRD-264 EUAI-11a — what the AI label's words and the label itself paint besides the words, for the label's
 * check (visibility.ts). Both mark the label hidden as 'paint'.
 *
 * - Their own ::before and ::after. One of the words' own pseudo-elements hit-tests as the words, so the
 *   check's hit-test can't see it drawn over them, and one that adds text changes what the label says. Read
 *   two ways, so neither depends on the other: the rules (pseudo-rules.ts), any rule whose selector names
 *   ::before or ::after (or the single colon) on the words' element or the label and sets `content` other
 *   than none or normal (a pseudo-element with no content is never drawn); and, where the browser reports
 *   pseudo-element styles (CSS.supports), their computed `content`. Only these two elements: an ancestor's
 *   pseudo-element hit-tests as that ancestor, so the hit-test sees it, and themes paint the page's own
 *   (scan lines on <body>, say), which must not read as hiding the label.
 * - The text stroke, as the words' element computes it (it inherits, and the words are what it paints):
 *   wider than MAX_TEXT_STROKE_PX, and either thick enough to fill the glyphs' counters (over
 *   THICK_TEXT_STROKE_EM of the font) or in a colour within MIN_TEXT_CONTRAST of what is behind the words (a
 *   stroke in the background's colour blots the glyphs while the fill passes). A transparent stroke paints
 *   nothing. `text-shadow` is not read: it paints beneath the glyphs.
 */
import { contrastRatio, over, parseCssColor, type Rgba } from './contrast'
import { pseudoStylesReadable } from './line-pseudos'
import { type StyleRuleRead, pseudoRulesReach } from './pseudo-rules'

/** A stroke this thin or thinner can't hide a glyph, whatever its colour. */
export const MAX_TEXT_STROKE_PX = 0.5
/** A stroke wider than this share of the font fills the glyphs' counters, whatever its colour. */
export const THICK_TEXT_STROKE_EM = 0.15

const OWN_PSEUDO_NAMES: ReadonlySet<string> = new Set(['before', 'after'])
const MENTIONS_OWN_PSEUDO = /before|after/i
const OWN_PSEUDOS = ['::before', '::after'] as const
const NO_CONTENT = new Set(['', 'none', 'normal'])
const LENGTH = /^(-?(?:\d+\.?\d*|\.\d+))(px|em)?$/i
/** The line-width keywords' lengths (CSS Backgrounds 3), for a DOM that leaves a keyword unresolved. */
const KEYWORD_PX: Readonly<Record<string, number>> = { thin: 1, medium: 3, thick: 5 }

function drawsContent(value: string): boolean {
  return !NO_CONTENT.has(value.trim().toLowerCase())
}

/** Whether the words' element or the label has a ::before or ::after drawn. */
export function ownPseudosDrawn(label: Element, words: Element, view: Window, rules: readonly StyleRuleRead[]): boolean {
  const targets = words === label ? [label] : [words, label]
  const setsContent = (rule: CSSStyleRule) => drawsContent(rule.style.getPropertyValue('content'))
  if (pseudoRulesReach(rules, OWN_PSEUDO_NAMES, MENTIONS_OWN_PSEUDO, targets, setsContent)) return true
  if (!pseudoStylesReadable(view, '::before')) return false
  return targets.some((element) => OWN_PSEUDOS.some((pseudo) => drawsContent(view.getComputedStyle(element, pseudo).content)))
}

/** A length in px (px, em of `fontSize`, a bare number, or a line-width keyword); null for anything else. */
function lengthPx(value: string, fontSize: number): number | null {
  const text = value.trim().toLowerCase()
  if (text in KEYWORD_PX) return KEYWORD_PX[text]
  const match = text.match(LENGTH)
  if (match === null) return null
  const amount = Number.parseFloat(match[1])
  return match[2] === 'em' ? amount * fontSize : amount
}

/** The words' text stroke as their element computes it (a DOM without inheritance hands the nearest declared values). */
export interface StrokeRead {
  readonly width: string
  /** The `-webkit-text-stroke` shorthand, for a width the longhand doesn't give. */
  readonly shorthand: string
  readonly color: string
  /** The fill, which the stroke takes when it names no colour (currentcolor). */
  readonly fill: string
}

/** Whether the words' text stroke hides them: wide enough, and thick or in a colour too close to what is behind. */
export function strokeHides(stroke: StrokeRead, grounds: readonly Rgba[], fontSize: number, minContrast: number): boolean {
  const leading = stroke.shorthand.trim().split(/\s+/)[0] ?? ''
  const width = lengthPx(stroke.width, fontSize) ?? lengthPx(leading, fontSize) ?? 0
  if (width <= MAX_TEXT_STROKE_PX) return false
  if (width > THICK_TEXT_STROKE_EM * fontSize) return true
  const color = parseCssColor(stroke.color || stroke.fill)
  if (color === null) return true
  if (color.a === 0) return false
  return grounds.some((ground) => contrastRatio(over(color, ground), ground) < minContrast)
}
