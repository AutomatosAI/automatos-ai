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
 * A selector that names the pseudo-element (with two colons or the legacy one) reaches an element when the
 * selector without it matches that element; one the browser can't match counts as matching. One that still
 * names another pseudo-element without it (`input::-webkit-search-cancel-button::before`) styles that
 * pseudo-element's own, never an element's, so it reaches none. A sheet from
 * another origin can't be read, and a theme's never is one: the host serves it from the app's own origin
 * (docs/theme-host/security-model.md).
 */

const ANY_ELEMENT = '*'
const PSEUDO_ELEMENT_MARK = '::'
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

/** A selector list's selectors, split at the commas outside brackets and strings. */
export function selectorsOf(list: string): string[] {
  const parts: string[] = []
  let depth = 0
  let quote = ''
  let start = 0
  for (let index = 0; index < list.length; index += 1) {
    const char = list[index]
    if (quote !== '') {
      if (char === quote && list[index - 1] !== '\\') quote = ''
    } else if (QUOTES.has(char)) quote = char
    else if (OPENERS.has(char)) depth += 1
    else if (CLOSERS.has(char)) depth -= 1
    else if (char === ',' && depth === 0) {
      parts.push(list.slice(start, index))
      start = index + 1
    }
  }
  return [...parts, list.slice(start)].map((part) => part.trim()).filter((part) => part !== '')
}

/** Whether `selector` matches one of `targets`; a selector the browser can't match counts as matching. */
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
 * Whether a rule among `rules` that `applies` styles the pseudo-element `names` matches (a global-free
 * pattern such as /::?before\b/i) on one of `targets`.
 */
export function pseudoRulesReach(
  rules: readonly StyleRuleRead[],
  names: RegExp,
  targets: readonly Element[],
  applies: (rule: CSSStyleRule) => boolean = () => true,
): boolean {
  const everywhere = new RegExp(names.source, 'gi')
  return rules.some(
    ({ rule, selector: list }) =>
      names.test(list) &&
      applies(rule) &&
      selectorsOf(list).some((selector) => {
        if (!names.test(selector)) return false
        const base = selector.replace(everywhere, '').trim() || ANY_ELEMENT
        return !base.includes(PSEUDO_ELEMENT_MARK) && matchesAny(base, targets)
      }),
  )
}
