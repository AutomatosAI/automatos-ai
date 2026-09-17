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
    // The `@media` blocks that predate PRD-246 and are still desktop steps:
    // 1279 (chat rail), 1199 (stats 6→3), 1199 (summary), 700 (roster), 1199
    // (activity cards), and Classic's two (reduced motion, mobile paint
    // cost). US-002 moved the 700px stats step into the region at 768.
    const outside = (css.slice(0, OPENS) + css.slice(CLOSES)).match(/@media/g) ?? []
    expect(outside, 'a new @media block belongs in the compact region').toHaveLength(7)
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
    // A root styles itself — `.cc-page`, `.cc-cal-root`, `.sh-chat`,
    // `.sh-shell` are the Studio roots and exist on no Classic page.
    const ROOT = /^\s+\.(cc-page|cc-cal-root|sh-chat|sh-shell)\b/
    const offenders = declarations
      .split('\n')
      .filter((l) => /^\s+\.(cc|sh|entry|mis|pb|mkt|status|pg)-/.test(l) && !ROOT.test(l))
    expect(offenders, 'hang the rule off :is(.studio, <own root>)').toEqual([])
  })

  it('never lets a named chrome row shrink', () => {
    expect(declarations).not.toMatch(/flex-shrink:\s*[1-9]/)
  })
})

describe('PRD-246 · every surface that renders on a phone has a compact form', () => {
  const phone = () => band(767)

  it('Command Centre: the gutter, the tab strip, the stats and the board (US-002)', () => {
    const p = phone()
    // The strip's bleed matches the page's gutter — a wider bleed is
    // sideways page scroll, because .cc-page computes overflow-x: auto.
    expect(p).toContain('.cc-page { padding: 16px 16px 0; gap: 14px; }')
    expect(p).toContain(':is(.studio, .cc-page) .cc-tabs { margin: 0 -16px; padding: 0 16px; }')
    expect(p, 'six figures 2-up').toContain('grid-template-columns: repeat(2, minmax(0, 1fr))')
    expect(p, 'one row per status, snapped').toContain('scroll-snap-type: x mandatory')
    expect(p).toContain('scroll-snap-align: start')
    expect(p, 'the status head sticks inside its column').toMatch(
      /\.cc-kb-head \{ position: sticky/,
    )
    // The desktop board shape is intact: one row, scrolling sideways.
    expect(css.slice(0, OPENS)).toContain('grid-auto-flow: column')
  })

  it('Chat: one column, the sides behind the one Sheet (US-003)', () => {
    const c = band(1023)
    expect(c).toContain(':is(.studio, .sh-chat) .sh-chat-grid { grid-template-columns: 1fr; }')
    expect(c, 'the breadcrumb collapses to the title and the two controls').toContain('.sh-chat-crumb')
    expect(c).toContain('.sh-chat-bar .sh-chat-active')
    const shell = read('components/chatbot/studio-chat-shell.tsx')
    expect(shell).toContain("from '@/components/ui/sheet'")
    expect(shell, 'the sheets are portals — they need the inset themselves').toMatch(
      /SheetContent side="left"[\s\S]{0,80}safe-bottom/,
    )
  })

  it('Assignments hub: 1-up entries and cards, stacked meta, a wrapping head (US-004)', () => {
    const p = phone()
    expect(p).toContain(':is(.studio, .cc-page) :is(.entry-grid, .mis-cards, .pb-grid)')
    expect(p, 'the small marketplace tiles read 2-up').toContain('.mkt-strip { grid-template-columns: repeat(2, 1fr); }')
    expect(p, "a card's meta stacks beneath the title").toContain('.mis-card .row-meta')
    expect(p).toContain('.pb-card .stats { flex-wrap: wrap')
    expect(p).toContain('.status-head { flex-wrap: wrap; }')
    expect(p, 'the 7-column grouped row becomes two lines').toContain('.mis-row {')
    // The two card grids are classes, not inline styles a media query cannot
    // reach, and the desktop column counts stay outside the region.
    for (const [file, cls] of [
      ['components/assignments/studio/missions-body.tsx', 'mis-cards'],
      ['components/assignments/studio/playbooks-body.tsx', 'pb-grid'],
    ] as const) {
      expect(read(file), file).toContain(`className="${cls}"`)
      expect(css.slice(0, OPENS), cls).toMatch(new RegExp(`\\.${cls} \\{[^}]*grid-template-columns: repeat\\(`))
    }
    // `.status-head` stays a chrome row that refuses to shrink.
    expect(css.slice(0, OPENS)).toMatch(/flex-shrink: 0/)
  })

  it('the tab-strip behaviour is one hook, not a copy per surface', () => {
    const hook = read('hooks/use-tab-strip-scroll.ts')
    expect(hook).toContain('useIsTabletOrBelow')
    expect(hook).toContain('.cc-tab.active')
    for (const f of [
      'components/command-center/command-center-shell.tsx',
      'components/assignments/studio/assignments-hub.tsx',
    ]) expect(read(f), f).toContain('useTabStripScroll')
  })
})
