import { describe, it, expect } from 'vitest'
import { existsSync, readFileSync } from 'fs'
import path from 'path'

// PRD-154 S10 — honest pilot surfaces (BINDING D10). The studio chrome shipped
// two fabrications pilots would read as live truth:
//   * StudioTicker: 7 hardcoded ops KPIs (UPTIME 99.84%, CACHE 68%, $/DEC
//     $0.0027, P50 95ms, ERR/HR 6, T2.5 988 hits, QUEUE 14) behind a "LIVE" dot,
//     with no metrics source. PRD authorised removal — the component is deleted
//     and unmounted rather than left lying about live data.
//   * STUDIO_PAGE_TABS: fabricated per-tab counts (All 18, Outputs 41, Skills
//     24…) rendered as badges before any real fetch. Counts are dropped; pages
//     that have real numbers wire their own.
//
// Grep gate: the fabricated literals must be absent from the source.

const ROOT = path.resolve(__dirname, '..', '..')

describe('S10 studio chrome — no fabricated metrics', () => {
  it('deletes the fabricated StudioTicker component', () => {
    expect(existsSync(path.join(ROOT, 'components/layout/studio-ticker.tsx'))).toBe(false)
  })

  it('unmounts StudioTicker from the studio shell', () => {
    const layout = readFileSync(path.join(ROOT, 'components/layout/main-layout.tsx'), 'utf8')
    expect(layout).not.toContain('StudioTicker')
  })

  it('drops fabricated per-tab counts from STUDIO_PAGE_TABS', () => {
    const src = readFileSync(path.join(ROOT, 'lib/studio-menu.ts'), 'utf8')
    const start = src.indexOf('STUDIO_PAGE_TABS')
    expect(start).toBeGreaterThan(-1)
    // The declaration runs to the first `};` after it.
    const block = src.slice(start, src.indexOf('};', start))
    // Tab labels carry no digits, so any digit in this block is a fake count.
    expect(block).not.toMatch(/\d/)
    // None of the specific fabricated counts survive.
    for (const fake of ['18', '41', '24', '12', '9']) {
      expect(block).not.toContain(fake)
    }
  })

  it('renders tabs without a count badge', () => {
    const src = readFileSync(path.join(ROOT, 'components/layout/studio-page-tabs.tsx'), 'utf8')
    expect(src).not.toContain('sh-tab-ct')
  })
})
