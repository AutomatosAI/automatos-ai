import { describe, it, expect } from 'vitest'
import { existsSync, readFileSync, readdirSync, statSync } from 'fs'
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

  it('the header sub-nav (and its fabricated per-tab counts) is gone — Studio pages compose their own tabs', () => {
    const src = readFileSync(path.join(ROOT, 'lib/studio-menu.ts'), 'utf8')
    expect(src).not.toContain('STUDIO_PAGE_TABS')
    expect(existsSync(path.join(ROOT, 'components/layout/studio-page-tabs.tsx'))).toBe(false)
    expect(readFileSync(path.join(ROOT, 'components/layout/main-layout.tsx'), 'utf8')).not.toContain('StudioPageTabs')
    expect(readFileSync(path.join(ROOT, 'app/globals.css'), 'utf8')).not.toContain('.sh-tabs')
  })

  it('no Studio component renders the retired sub-nav count badge', () => {
    // The strip that carried fabricated counts is deleted (W5b); nothing may
    // resurrect its badge class under another name.
    const walk = (dir: string): string[] =>
      readdirSync(dir).flatMap((f) => {
        const full = path.join(dir, f)
        if (statSync(full).isDirectory()) return f === '__tests__' ? [] : walk(full)
        return f.endsWith('.tsx') ? [full] : []
      })
    const offenders = walk(path.join(ROOT, 'components')).filter((f) => readFileSync(f, 'utf8').includes('sh-tab-ct'))
    expect(offenders).toEqual([])
  })
})
