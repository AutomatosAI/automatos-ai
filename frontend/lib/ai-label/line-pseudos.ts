/**
 * PRD-264 EUAI-11a — the ::first-line and ::first-letter half of the AI label's check (visibility.ts).
 *
 * getComputedStyle(element) doesn't show what those pseudo-elements paint, so a theme rule such as
 * `p[role=note]::first-line { color: transparent }` could blank the words while the element reads as
 * fine. Two reads, either of which marks the label hidden:
 *
 * - The rules ('paint'). Every rule the page's stylesheets hold (inside @media, @supports, @layer,
 *   @container and nested rules too) whose selector names ::first-line or ::first-letter (or the
 *   legacy single colon), and whose selector without them matches the words or any element around
 *   them, restyles the label, whatever it sets. A selector the browser can't match counts as
 *   matching. A sheet from another origin can't be read; a theme's never is one (the host serves it
 *   from the app's own origin, docs/theme-host/security-model.md).
 * - The computed styles ('color'), where the browser reports pseudo-element styles (it says so through
 *   CSS.supports): each element's ::first-line and ::first-letter must keep its opacity, stay visible,
 *   and fill the words with a colour at least MIN_TEXT_CONTRAST against its own background painted
 *   over what is behind the words. A DOM with no CSS namespace (the test runner) skips this read.
 */
import { contrastRatio, over, parseCssColor, type Rgba } from './contrast'

const LINE_PSEUDOS = ['::first-line', '::first-letter'] as const
const NAMES_LINE_PSEUDO = /::?first-(?:line|letter)\b/i
const LINE_PSEUDO_EVERYWHERE = /::?first-(?:line|letter)\b/gi
const ANY_ELEMENT = '*'
const HIDDEN_VISIBILITY = new Set(['hidden', 'collapse'])

type WithCss = Window & { readonly CSS?: { readonly supports?: (condition: string) => boolean } }

/** A stylesheet's rules; none for a sheet the page may not read (another origin's: the browser throws). */
function rulesOf(sheet: CSSStyleSheet): CSSRule[] {
  try {
    return Array.from(sheet.cssRules)
  } catch {
    return []
  }
}

/** The rules, with every rule a group or a style rule holds (media, supports, layer, container, nesting). */
function flatten(rules: readonly CSSRule[]): CSSRule[] {
  return rules.flatMap((rule) => {
    const inner = (rule as Partial<CSSGroupingRule>).cssRules
    return inner ? [rule, ...flatten(Array.from(inner))] : [rule]
  })
}

/** Whether `selector` matches an element of `chain`; a selector the browser can't match counts as matching. */
function matchesChain(selector: string, chain: readonly Element[]): boolean {
  return chain.some((element) => {
    try {
      return element.matches(selector)
    } catch {
      return true
    }
  })
}

/** Whether a stylesheet rule styles the ::first-line or ::first-letter of the words or an element around them. */
export function lineRulesReach(chain: readonly Element[], doc: Document): boolean {
  const rules = flatten(Array.from(doc.styleSheets).flatMap(rulesOf))
  return rules.some((rule) => {
    const selector = (rule as Partial<CSSStyleRule>).selectorText
    if (typeof selector !== 'string' || !NAMES_LINE_PSEUDO.test(selector)) return false
    const base = selector.replace(LINE_PSEUDO_EVERYWHERE, '').trim() || ANY_ELEMENT
    return matchesChain(base, chain)
  })
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

/** Whether the browser reports ::first-line styles through getComputedStyle (it says so through CSS.supports). */
function lineStylesReadable(view: Window): boolean {
  const css = (view as WithCss).CSS
  return typeof css?.supports === 'function' && css.supports('selector(::first-line)')
}

/** Whether the computed ::first-line or ::first-letter of the words or an element around them hides them. */
export function lineStylesHide(
  chain: readonly Element[],
  grounds: readonly Rgba[],
  view: Window,
  limits: { readonly minOpacity: number; readonly minContrast: number },
): boolean {
  if (!lineStylesReadable(view)) return false
  return chain.some((element) =>
    LINE_PSEUDOS.some((pseudo) => lineStyleHides(view.getComputedStyle(element, pseudo), grounds, limits.minOpacity, limits.minContrast)),
  )
}
