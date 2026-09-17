/**
 * PRD-246 — the mobile scope gate.
 *
 * Studio on a phone is the design system in its compact form, not the desktop
 * shrunk (M3) and not the hybrid of Studio chrome around classic bodies that
 * PRD-244's default flip left behind (M1). This gate holds the conventions
 * every compact rule is built on:
 *
 *   · ONE region in `globals.css` owns every compact rule (M5). A `@media`
 *     block scattered elsewhere in the file is the thing this catches.
 *   · TWO bands, both already in `hooks/use-mobile.ts` (M2) — 1023px
 *     (`useIsTabletOrBelow`) and 767px (`useIsMobile`). No third breakpoint.
 *   · ONE safe-area helper, `.safe-bottom` (M4 / no second primitive).
 *   · Studio rules stay scoped, so nothing reaches Classic (M6).
 *
 * The per-surface coverage assertions are grouped at the end and grow as each
 * surface gets its compact form.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

const FRONTEND = path.resolve(__dirname, '..', '..')
const read = (rel: string) => readFileSync(path.join(FRONTEND, rel), 'utf8')
const css = read('app/globals.css')

const MARKER = '── Studio compact'
const END = 'end of the compact region'
/** The region, from the first character of its banner to its closing comment. */
const OPENS = css.lastIndexOf('/*', css.indexOf(MARKER))
const CLOSES = css.lastIndexOf('/*', css.indexOf(END))
const region = css.slice(OPENS, CLOSES)
/** The region with its comments stripped — declarations only. */
const declarations = region.replace(/\/\*[\s\S]*?\*\//g, '')
/** The declaration block of one `@media` band inside the region. */
const band = (maxWidth: number) => {
  const at = region.indexOf(`@media (max-width: ${maxWidth}px)`)
  expect(at, `the ${maxWidth}px band exists in the compact region`).toBeGreaterThan(-1)
  return region.slice(at, region.indexOf('\n  }', at))
}

describe('PRD-246 · the compact region', () => {
  it('is one region, inside @layer components, after the chrome guard', () => {
    expect(css.split(MARKER)).toHaveLength(2) // exactly one banner
    expect(region.length).toBeGreaterThan(0)
    expect(region, 'the region names itself as the home for compact rules').toContain(
      'THIS IS THE ONE HOME FOR COMPACT RULES',
    )
    const layer = css.indexOf('@layer components {')
    expect(css.indexOf('/* ── Page chrome never absorbs')).toBeGreaterThan(layer)
    expect(css.indexOf(MARKER), 'after the chrome guard').toBeGreaterThan(
      css.indexOf('/* ── Page chrome never absorbs'),
    )
    expect(css.indexOf(MARKER), 'inside @layer components').toBeLessThan(css.indexOf('\nhtml {'))
  })

  it('holds every compact media query — none is scattered through the file', () => {
    // The 8 `@media` blocks that predate PRD-246: the Studio desktop steps
    // (1279 chat rail, 1199/700 stats, 1199 summary, 700 roster, 1199
    // activity) and Classic's two (reduced motion, mobile paint cost).
    const outside = (css.slice(0, OPENS) + css.slice(CLOSES)).match(/@media/g) ?? []
    expect(outside, 'a new @media block belongs in the compact region').toHaveLength(8)
  })

  it('uses only the two breakpoints that already exist', () => {
    const hooks = read('hooks/use-mobile.ts')
    expect(hooks).toContain('MOBILE_BREAKPOINT = 768')
    expect(hooks).toContain('TABLET_BREAKPOINT = 1024')
    const widths = new Set((region.match(/max-width: (\d+)px/g) ?? []).map((m) => m))
    expect([...widths].sort()).toEqual(['max-width: 1023px', 'max-width: 767px'])
  })
})

describe('PRD-246 · safe area and touch targets', () => {
  it('has exactly one safe-area helper, and the Studio shell uses it', () => {
    expect(css.match(/safe-area-inset-bottom/g) ?? []).toHaveLength(2) // @supports + the declaration
    expect(css).toContain('.safe-bottom {')
    const layout = read('components/layout/main-layout.tsx')
    expect(layout, "the shell's bottom edge").toContain('className="sh-shell safe-bottom"')
    expect(layout, 'the Studio mobile rail sheet').toMatch(/'w-\[260px\][^']*safe-bottom'/)
    expect(region, 'no second safe-area definition in the compact region').not.toContain('safe-area-inset')
  })

  it('raises every tapped Studio control to 44px on a phone', () => {
    const phone = band(767)
    for (const sel of ['.sh-item', '.cc-tab', '.cc-btn', '.cc-filter-pill', '.cc-seg button']) {
      expect(phone, `${sel} is a thumb target`).toContain(sel)
    }
    expect(phone).toContain('min-height: 44px')
    // The rail lives in a sheet from 1024 down, so it is touch-sized there too.
    expect(band(1023)).toContain('.sh-item')
  })

  it('reads documents at the compact density on a phone, desktop rules intact', () => {
    const phone = band(767)
    expect(phone).toContain('.md-view { font-size: 12.5px; line-height: 1.55; }')
    expect(phone, 'block spacing too, not just type').toContain('.md-view > * + * { margin-top: 0.7em; }')
    // The opt-in class and the desktop metrics are untouched.
    expect(css).toContain('.md-view-compact { font-size: 12.5px; line-height: 1.55; }')
    expect(css.slice(0, OPENS)).toContain('font-size: 14px; line-height: 1.65;')
  })
})

describe('PRD-246 · nothing leaks into Classic', () => {
  it('scopes every Studio family rule in the region to a Studio root', () => {
    const offenders = declarations
      .split('\n')
      .filter((l) => /^\s{4}\.(cc|sh|entry|mis|pb|mkt|status|pg)-/.test(l))
    expect(offenders, 'hang the rule off :is(.studio, <own root>)').toEqual([])
  })

  it('never lets a named chrome row shrink', () => {
    expect(declarations).not.toMatch(/flex-shrink:\s*[1-9]/)
  })
})
