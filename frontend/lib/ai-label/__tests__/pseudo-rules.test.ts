/**
 * PRD-264 EUAI-11a — how the AI label's check reads a selector for a pseudo-element (lib/ai-label/pseudo-rules.ts):
 * outside escapes and strings, as CSS reads it. A Tailwind class such as `hover:after:scale-100`
 * (`.hover\:after\:scale-100`) holds an escaped `\:after` that names no pseudo-element: reading it as one turned
 * Studio's own `data-[panel-group-direction=vertical]:after:left-0` utilities (shadcn's resizable handle) into a
 * selector the browser can't match, which the check counts as matching, so Studio's default /chat drew the
 * protected copy over a label that was plainly seen (CI run 40760817606).
 */
import { describe, expect, it } from 'vitest'

import { selectorsOf, stripPseudo } from '../pseudo-rules'

const OWN: ReadonlySet<string> = new Set(['before', 'after'])
const LINES: ReadonlySet<string> = new Set(['first-line', 'first-letter'])

describe('stripPseudo (PRD-264 EUAI-11a)', () => {
  it.each([
    [String.raw`.data-\[panel-group-direction\=vertical\]\:after\:left-0[data-panel-group-direction="vertical"]::after`,
      String.raw`.data-\[panel-group-direction\=vertical\]\:after\:left-0[data-panel-group-direction="vertical"]`],
    [String.raw`.hover\:after\:scale-100:hover::after`, String.raw`.hover\:after\:scale-100:hover`],
    [String.raw`.before\:absolute::before`, String.raw`.before\:absolute`],
    ['p[role="note"]:after', 'p[role="note"]'],
    ['[data-x="a::before"] span::before', '[data-x="a::before"] span'],
    ['span::before:hover', 'span:hover'],
  ])('takes the pseudo-element off %s, and leaves escapes and strings as they are', (selector, base) => {
    expect(stripPseudo(selector, OWN)).toEqual({ base, found: true, other: false })
  })

  it("finds no pseudo-element in a class's escaped colon", () => {
    expect(stripPseudo(String.raw`.hover\:after\:scale-100`, OWN)).toEqual({ base: String.raw`.hover\:after\:scale-100`, found: false, other: false })
    expect(stripPseudo(String.raw`.first-line\:uppercase`, LINES)).toMatchObject({ found: false })
  })

  it('reads the selector a lone pseudo-element or a trailing combinator leaves as any element', () => {
    expect(stripPseudo('::before', OWN).base).toBe('*')
    expect(stripPseudo('p > ::after', OWN).base).toBe('p > *')
  })

  it("marks a selector that is another pseudo-element's own: it reaches no element", () => {
    expect(stripPseudo('input::-webkit-search-cancel-button::before', OWN)).toMatchObject({ found: true, other: true })
  })

  it('reads the first-line pseudo-elements the same way', () => {
    expect(stripPseudo('p::first-line', LINES)).toEqual({ base: 'p', found: true, other: false })
    expect(stripPseudo('p:first-letter', LINES)).toEqual({ base: 'p', found: true, other: false })
  })
})

describe('selectorsOf (PRD-264 EUAI-11a)', () => {
  it('splits a list at the commas outside brackets, escapes and strings', () => {
    expect(selectorsOf(String.raw`.a\,b::before, :is(.c, .d)::after, [x="1,2"]`)).toEqual([
      String.raw`.a\,b::before`,
      ':is(.c, .d)::after',
      '[x="1,2"]',
    ])
  })
})
