/**
 * PRD-264 EUAI-11a — the ::first-line and ::first-letter half of the AI label's check (visibility.ts).
 *
 * getComputedStyle(element) doesn't show what those pseudo-elements paint, so a theme rule such as
 * `p[role=note]::first-line { color: transparent }` could blank the words while the element reads as
 * fine. Two reads, either of which marks the label hidden:
 *
 * - The rules ('paint'), read as pseudo-rules.ts reads them: any rule whose selector names
 *   ::first-line or ::first-letter (or the legacy single colon) and reaches the words or any element
 *   around them restyles the label, whatever it sets.
 * - The computed styles ('color'), where the browser reports pseudo-element styles (it says so through
 *   CSS.supports): each element's ::first-line and ::first-letter must keep its opacity, stay visible,
 *   and fill the words with a colour at least MIN_TEXT_CONTRAST against its own background painted
 *   over what is behind the words. A DOM with no CSS namespace (the test runner) skips this read.
 */
import { contrastRatio, over, parseCssColor, type Rgba } from './contrast'
import { type StyleRuleRead, pseudoRulesReach } from './pseudo-rules'

const LINE_PSEUDOS = ['::first-line', '::first-letter'] as const
const NAMES_LINE_PSEUDO = /::?first-(?:line|letter)\b/i
const HIDDEN_VISIBILITY = new Set(['hidden', 'collapse'])

type WithCss = Window & { readonly CSS?: { readonly supports?: (condition: string) => boolean } }

/** Whether a rule styles the ::first-line or ::first-letter of the words or an element around them, whatever it sets. */
export function lineRulesReach(chain: readonly Element[], rules: readonly StyleRuleRead[]): boolean {
  return pseudoRulesReach(rules, NAMES_LINE_PSEUDO, chain)
}

function lineStyleHides(style: CSSStyleDeclaration, grounds: readonly Rgba[], minOpacity: number, minContrast: number): boolean {
  const opacity = Number.parseFloat(style.opacity)
  if ((Number.isFinite(opacity) && opacity < minOpacity) || HIDDEN_VISIBILITY.has(style.visibility)) return true
  const fill = parseCssColor(style.getPropertyValue('-webkit-text-fill-color').trim() || style.color)
  const background = parseCssColor(style.backgroundColor || 'transparent')
  if (fill === null || background === null || fill.a === 0) return true
  return grounds.some((ground) => {
    const behind = over(background, ground)
    return contrastRatio(over(fill, behind), behind) < minContrast
  })
}

/**
 * Whether the browser reports `pseudo`'s styles through getComputedStyle: every browser that paints it
 * says so through CSS.supports. A DOM with no CSS namespace (the test runner) can't, and skips the read.
 */
export function pseudoStylesReadable(view: Window, pseudo: string): boolean {
  const css = (view as WithCss).CSS
  return typeof css?.supports === 'function' && css.supports(`selector(${pseudo})`)
}

/** Whether the computed ::first-line or ::first-letter of the words or an element around them hides them. */
export function lineStylesHide(
  chain: readonly Element[],
  grounds: readonly Rgba[],
  view: Window,
  limits: { readonly minOpacity: number; readonly minContrast: number },
): boolean {
  if (!pseudoStylesReadable(view, '::first-line')) return false
  return chain.some((element) =>
    LINE_PSEUDOS.some((pseudo) => lineStyleHides(view.getComputedStyle(element, pseudo), grounds, limits.minOpacity, limits.minContrast)),
  )
}
