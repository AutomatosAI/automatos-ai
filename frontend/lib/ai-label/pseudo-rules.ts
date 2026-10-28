/**
 * PRD-264 EUAI-11a — reading the page's stylesheet rules for pseudo-elements that reach the AI label, for the
 * label's check (visibility.ts, line-pseudos.ts). getComputedStyle(element) shows nothing a pseudo-element
 * paints, and not every browser (nor the test runner) reports a pseudo-element's own computed style, so the
 * rules themselves are read too: this works wherever the page's stylesheets can be read, jsdom included.
 *
 * Every rule of every readable sheet counts, inside @media, @supports, @layer, @container and nested rules
 * too (a media query that doesn't match today may tomorrow). A nested rule's selector is read as it applies:
 * its `&` stands for its parent's selector, and one with no `&` sits inside it. A selector list is read one
 * selector at a time.
 * Selectors are read as CSS reads them, outside escapes and strings: Tailwind's `.hover\:after\:scale-100`
 * is one class, and its `\:after` names no pseudo-element. A selector that names the pseudo-element (with two
 * colons, or one for the four legacy ones) reaches an element when the selector without it matches that
 * element; one the browser can't match counts as matching. One that still names another pseudo-element without
 * it (`input::-webkit-search-cancel-button::before`) styles that pseudo-element's own, never an element's, so
 * it reaches none. A sheet from
 * another origin can't be read, and a theme's never is one: the host serves it from the app's own origin
 * (docs/theme-host/security-model.md).
 */

const ANY_ELEMENT = '*'
const NESTING = '&'
const OPENERS = new Set(['(', '['])
const CLOSERS = new Set([')', ']'])
const QUOTES = new Set(['"', "'"])

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

function isStyleRule(rule: CSSRule | null | undefined): rule is CSSStyleRule {
  return typeof (rule as Partial<CSSStyleRule> | null | undefined)?.selectorText === 'string'
}

/** The style rule a nested rule sits in, through any group rules between; null for a rule at the top. */
function nestingParent(rule: CSSRule): CSSStyleRule | null {
  for (let parent = rule.parentRule; parent !== null && parent !== undefined; parent = parent.parentRule) {
    if (isStyleRule(parent)) return parent
  }
  return null
}

/** A rule's selector as it applies: a nested rule's `&` is its parent's, and one without `&` sits inside it. */
function resolvedSelector(rule: CSSStyleRule): string {
  const parent = nestingParent(rule)
  if (parent === null) return rule.selectorText
  const outer = `:is(${resolvedSelector(parent)})`
  return selectorsOf(rule.selectorText)
    .map((selector) => (selector.includes(NESTING) ? selector.split(NESTING).join(outer) : `${outer} ${selector}`))
    .join(', ')
}

/** A style rule and its selector as it applies. */
export interface StyleRuleRead {
  readonly rule: CSSStyleRule
  readonly selector: string
}

/** Every style rule the page's stylesheets hold that it can read, with its selector as it applies, read once per check. */
export function readableStyleRules(doc: Document): StyleRuleRead[] {
  return flatten(Array.from(doc.styleSheets).flatMap(rulesOf))
    .filter(isStyleRule)
    .map((rule) => ({ rule, selector: resolvedSelector(rule) }))
}

/** A piece of a selector: one character outside escapes and strings (`plain`), or a whole escape or string. */
interface Piece {
  readonly text: string
  readonly plain: boolean
}

/** Where the escape starting at `start` (a backslash) ends: up to six hex digits and one space, or one character. */
function escapeEnd(text: string, start: number): number {
  const hex = /^[0-9a-f]{1,6}\s?/i.exec(text.slice(start + 1, start + 8))
  return Math.min(text.length, start + 1 + (hex ? hex[0].length : 1))
}

/** Where the string starting at `start` (a quote) ends, its escapes skipped. */
function stringEnd(text: string, start: number): number {
  let index = start + 1
  while (index < text.length && text[index] !== text[start]) index += text[index] === '\\' ? 2 : 1
  return Math.min(text.length, index + 1)
}

/**
 * A selector cut into pieces, so that what an escape or a string holds is never read as syntax: Tailwind's
 * `.hover\:after\:scale-100` is one class, and its `\:after` is no pseudo-element.
 */
function piecesOf(selector: string): Piece[] {
  const pieces: Piece[] = []
  for (let index = 0; index < selector.length; ) {
    const char = selector[index]
    const end = char === '\\' ? escapeEnd(selector, index) : QUOTES.has(char) ? stringEnd(selector, index) : index + 1
    pieces.push({ text: selector.slice(index, end), plain: end === index + 1 && char !== '\\' && !QUOTES.has(char) })
    index = end
  }
  return pieces
}

/** A selector list's selectors, split at the commas outside brackets, escapes and strings. */
export function selectorsOf(list: string): string[] {
  const parts: string[] = []
  let depth = 0
  let current = ''
  for (const piece of piecesOf(list)) {
    if (piece.plain && OPENERS.has(piece.text)) depth += 1
    if (piece.plain && CLOSERS.has(piece.text)) depth -= 1
    if (piece.plain && piece.text === ',' && depth === 0) {
      parts.push(current)
      current = ''
    } else current += piece.text
  }
  return [...parts, current].map((part) => part.trim()).filter((part) => part !== '')
}

/** What taking the pseudo-elements `names` off a selector leaves. */
export interface Stripped {
  /** The selector without them: the elements they belong to. */
  readonly base: string
  /** It named one of `names`. */
  readonly found: boolean
  /** It names another pseudo-element too, so it styles that one's, never an element's. */
  readonly other: boolean
}

const PSEUDO_NAME = /^[a-z0-9_-]$/i
/** The pseudo-elements CSS also takes with one colon. */
const SINGLE_COLON_PSEUDO_ELEMENTS = new Set(['before', 'after', 'first-line', 'first-letter'])
const TRAILING_COMBINATOR = /[>+~]\s*$/

/** The name of the pseudo-class or -element whose colons start at `pieces[start]`, and where it ends. */
function pseudoAt(pieces: readonly Piece[], start: number): { readonly name: string; readonly double: boolean; readonly end: number } {
  const double = pieces[start + 1]?.plain === true && pieces[start + 1].text === ':'
  let end = start + (double ? 2 : 1)
  let name = ''
  while (end < pieces.length && pieces[end].plain && PSEUDO_NAME.test(pieces[end].text)) {
    name += pieces[end].text
    end += 1
  }
  return { name: name.toLowerCase(), double, end }
}

/** `selector` with the pseudo-elements `names` taken off, read outside escapes, strings and brackets. */
export function stripPseudo(selector: string, names: ReadonlySet<string>): Stripped {
  const pieces = piecesOf(selector)
  let base = ''
  let depth = 0
  let found = false
  let other = false
  for (let index = 0; index < pieces.length; ) {
    const piece = pieces[index]
    if (piece.plain && OPENERS.has(piece.text)) depth += 1
    if (piece.plain && CLOSERS.has(piece.text)) depth -= 1
    const pseudo = piece.plain && piece.text === ':' && depth === 0 ? pseudoAt(pieces, index) : null
    const element = pseudo !== null && (pseudo.double || SINGLE_COLON_PSEUDO_ELEMENTS.has(pseudo.name))
    if (pseudo !== null && element && names.has(pseudo.name)) found = true
    else if (pseudo !== null && element) other = true
    if (pseudo !== null && element && names.has(pseudo.name)) index = pseudo.end
    else {
      base += piece.text
      index += 1
    }
  }
  const trimmed = base.trim()
  const whole = trimmed === '' ? ANY_ELEMENT : TRAILING_COMBINATOR.test(trimmed) ? `${trimmed} ${ANY_ELEMENT}` : trimmed
  return { base: whole, found, other }
}

/**
 * Whether `selector` matches one of `targets`. A rule in the page's sheets parsed, so its selector is one the
 * browser takes; should what is left once a pseudo-element is taken off still be one it can't match, that
 * counts as matching.
 */
function matchesAny(selector: string, targets: readonly Element[]): boolean {
  return targets.some((element) => {
    try {
      return element.matches(selector)
    } catch {
      return true
    }
  })
}

/**
 * Whether a rule among `rules` that `applies` styles one of the pseudo-elements `names` on one of `targets`.
 * `mentions` is a cheap test of a selector list's text, to skip the rules that can't name them.
 */
export function pseudoRulesReach(
  rules: readonly StyleRuleRead[],
  names: ReadonlySet<string>,
  mentions: RegExp,
  targets: readonly Element[],
  applies: (rule: CSSStyleRule) => boolean = () => true,
): boolean {
  return rules.some(
    ({ rule, selector: list }) =>
      mentions.test(list) &&
      applies(rule) &&
      selectorsOf(list).some((selector) => {
        const stripped = stripPseudo(selector, names)
        return stripped.found && !stripped.other && matchesAny(stripped.base, targets)
      }),
  )
}
