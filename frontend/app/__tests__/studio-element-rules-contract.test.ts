// @vitest-environment node
/**
 * PRD-261 A-19e2 — the host side of P260-FX-01 (M01): the element rules app/globals.css puts on
 * every page, which a theme's own CSS has to win over. Every theme the host runs today is drawn on
 * Studio, so <html> carries `studio` with the theme's class, and Tailwind 3.3.3 emits globals.css's
 * @layer blocks unlayered: between two rules on one element, specificity decides. A theme wins with a
 * rule that starts with its <html> class (docs/frontend-architecture/play-theme.md, "Play's rules beat
 * Studio's element rules"): `html.<class> .<own class>` on its own element, `html.<class> .<surface>
 * <element>` inside the SDK's MarkdownView. That holds only while the host keeps these rules where
 * they are, so this test pins them: the list of rules that reach those elements, each one's
 * specificity, and each one under the theme's rule.
 *
 * The plan's shorthand for the ceiling, (0,1,1), is the plain element rules' (`.studio h1`,
 * `.md-view a`); two rules it names sit higher and always have (prd260-play-guards.test.ts reads them
 * so): `.studio code:not(pre code)` at (0,1,3) and the Studio markdown rules (`.studio .md-view a` and
 * its headings) at (0,2,1). Each is pinned at its own value, still below the theme's rule.
 *
 * Changing this list, or raising a rule in it, changes the theme contract: the kit's guides and its
 * copy of these rules (B-05g, test/fixtures/core/studio-element-rules.css) change with it.
 *
 * Replaces, in the core test ledger (plan §2.2.8), the core half of section (d) of
 * lib/__tests__/prd260-play-guards.test.ts ("finds the core rules it compares against in globals.css"
 * and its reading of their specificity); its theme half moves to the kit (B-05g, test/css-guards.test.ts).
 */
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

import { compareSpecificity, cssRules, elementOf, lastCompound, specificity, type Specificity } from '@/lib/__tests__/css-cascade'

const GLOBALS_CSS = path.resolve(__dirname, '..', 'globals.css')

/** The plan's figure: one class and one element, the plain element rules' specificity. */
const PLAIN_ELEMENT_RULE: Specificity = [0, 1, 1]

/** Studio's element rules: they reach a theme's own elements, since <html> carries `studio`. */
const STUDIO_ELEMENT_RULES: Readonly<Record<string, Specificity>> = {
  '.studio h1': PLAIN_ELEMENT_RULE,
  '.studio h2': PLAIN_ELEMENT_RULE,
  '.studio kbd': PLAIN_ELEMENT_RULE,
  '.studio code': PLAIN_ELEMENT_RULE,
  '.studio code:not(pre code)': [0, 1, 3],
}

/** The core markdown's element rules: they reach the elements inside the SDK's MarkdownView (.md-view). */
const MARKDOWN_ELEMENT_RULES: Readonly<Record<string, Specificity>> = {
  '.md-view h1': PLAIN_ELEMENT_RULE,
  '.md-view h2': PLAIN_ELEMENT_RULE,
  '.md-view h3': PLAIN_ELEMENT_RULE,
  '.md-view code': PLAIN_ELEMENT_RULE,
  '.md-view strong': PLAIN_ELEMENT_RULE,
  '.md-view a': PLAIN_ELEMENT_RULE,
  '.md-view pre code': [0, 1, 2],
  '.md-view a:hover': [0, 2, 1],
  '.md-view-compact h1': PLAIN_ELEMENT_RULE,
  '.md-view-compact h2': PLAIN_ELEMENT_RULE,
  '.md-view-compact h3': PLAIN_ELEMENT_RULE,
  '.studio .md-view h1': [0, 2, 1],
  '.studio .md-view h2': [0, 2, 1],
  '.studio .md-view h3': [0, 2, 1],
  '.studio .md-view a': [0, 2, 1],
}

/** The weakest rule a theme writes on its own element: its <html> class, then a class of its own. */
const THEME_OWN_RULE = 'html.theme .title'
/** ... and on an element inside a markdown surface: its <html> class, the surface's class, the element. */
const THEME_SURFACE_RULE = 'html.theme .surface h1'

/** The elements those rules restyle. */
const CONTRACT_ELEMENTS = new Set(['h1', 'h2', 'h3', 'kbd', 'code', 'a', 'strong'])
/** A class or an attribute on the element itself: the rule reaches only an element carrying the core's class. */
const OWN_CLASS_OR_ATTRIBUTE = /[.[]/
const PSEUDO_ARGUMENT = /\([^)]*\)/g

const named = (rules: Readonly<Record<string, Specificity>>) => Object.keys(rules)
const ALL_NAMED = [...named(STUDIO_ELEMENT_RULES), ...named(MARKDOWN_ELEMENT_RULES)]

function globalsSelectors(): string[] {
  return cssRules(fs.readFileSync(GLOBALS_CSS, 'utf8')).flatMap((rule) => [...rule.selectors])
}

/** Whether the selector lands on every one of the contract's elements it names, not only on one with a core class. */
function reachesContractElement(selector: string): boolean {
  const compound = lastCompound(selector)
  const element = elementOf(compound)
  if (!CONTRACT_ELEMENTS.has(element)) return false
  const own = compound.slice(element.length).replace(PSEUDO_ARGUMENT, '')
  return !OWN_CLASS_OR_ATTRIBUTE.test(own)
}

/** The named rules whose specificity is above their pinned value, or not below the theme's rule. */
function outOfContract(rules: Readonly<Record<string, Specificity>>, themeRule: string): string[] {
  const theme = specificity(themeRule)
  return Object.entries(rules).flatMap(([selector, pinned]) => {
    const own = specificity(selector)
    if (compareSpecificity(own, pinned) > 0) return [`${selector} is ${own.join(',')}, above ${pinned.join(',')}`]
    if (compareSpecificity(own, theme) >= 0) return [`${selector} ties with or beats ${themeRule}`]
    return []
  })
}

describe('A-19e2 — the element rules a theme has to win over (FX-01, the host side)', () => {
  it('reads specificity the way the contract is written', () => {
    expect(specificity('.studio h1')).toEqual(PLAIN_ELEMENT_RULE)
    expect(specificity('.studio code:not(pre code)')).toEqual([0, 1, 3])
    expect(specificity('.studio .md-view a')).toEqual([0, 2, 1])
    expect(specificity(THEME_OWN_RULE)).toEqual([0, 2, 1])
    expect(specificity(THEME_SURFACE_RULE)).toEqual([0, 2, 2])
    expect(reachesContractElement('.studio code:not(pre code)')).toBe(true)
    expect(reachesContractElement('.md-view a:hover')).toBe(true)
    expect(reachesContractElement('.studio a.rounded-xl')).toBe(false)
    expect(reachesContractElement('.studio h3.text-2xl')).toBe(false)
    expect(reachesContractElement('.studio .sh-main')).toBe(false)
  })

  it('finds every rule the contract names in app/globals.css', () => {
    const selectors = new Set(globalsSelectors())
    expect(ALL_NAMED.filter((selector) => !selectors.has(selector))).toEqual([])
  })

  it('names every globals.css rule that reaches those elements: a new one is a contract change', () => {
    const reaching = Array.from(new Set(globalsSelectors().filter(reachesContractElement)))
    expect(reaching.filter((selector) => !ALL_NAMED.includes(selector))).toEqual([])
  })

  it("keeps Studio's element rules at their pinned specificity, below a theme's rule on its own element", () => {
    expect(named(STUDIO_ELEMENT_RULES).filter((selector) => specificity(selector)[1] > 1)).toEqual([])
    expect(outOfContract(STUDIO_ELEMENT_RULES, THEME_OWN_RULE)).toEqual([])
  })

  it("keeps the markdown rules at their pinned specificity, below a theme's rule on a markdown surface", () => {
    expect(outOfContract(MARKDOWN_ELEMENT_RULES, THEME_SURFACE_RULE)).toEqual([])
  })

  it('keeps the plain element rules at one class and one element, the (0,1,1) the plan names', () => {
    const plain = ALL_NAMED.filter((selector) => /^\.[\w-]+ [a-z0-9]+$/.test(selector))
    expect(plain.length).toBeGreaterThan(0)
    expect(plain.filter((selector) => compareSpecificity(specificity(selector), PLAIN_ELEMENT_RULE) !== 0)).toEqual([])
  })
})
