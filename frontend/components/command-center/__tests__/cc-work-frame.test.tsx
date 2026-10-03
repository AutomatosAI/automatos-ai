/**
 * PRD-252 R8 — a work surface can take the whole screen. The owner: "The Command
 * Centre doesn't scroll… we need to be able to scroll up so we can hide the
 * stats… full-screen board, full-screen calendar."
 *
 * The tab strip and the tab body sit in `.cc-work`, at least the page's visible
 * height, so the head and the stats scroll away and the strip sticks; Board and
 * Calendar fill it exactly. Below 1024 the frame is as it was. F219: the clock in
 * the eyebrow is written after mount, never on the server.
 */
import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { renderToString } from 'react-dom/server'
import { readFileSync } from 'fs'
import path from 'path'

const activeTab = vi.hoisted(() => ({ current: 'board' }))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), refresh: vi.fn() }),
  usePathname: () => '/command-center',
  useSearchParams: () => new URLSearchParams(`tab=${activeTab.current}`),
}))
vi.mock('@/hooks/use-activity-api', () => ({
  useActivityStats: () => ({ data: { working_now: 0, needs_attention: 0 } }),
  useActivityFeed: () => ({ data: { total: 0, items: [] } }),
  useActivitySchedule: () => ({ data: { scheduled: [] } }),
}))
vi.mock('@/hooks/use-board-event-stream', () => ({ useBoardEventStream: () => undefined }))
vi.mock('@/hooks/use-needs-you', () => ({ useNeedsYou: () => ({ data: { total: 0 } }) }))
vi.mock('@/hooks/use-watches-api', () => ({ useWatches: () => ({ data: { total: 0 } }) }))
vi.mock('@/hooks/use-approval-grants', () => ({ useQuestions: () => ({ data: { grants: [] } }) }))
vi.mock('../stats-strip', () => ({ StatsStrip: () => <div /> }))
vi.mock('../is-it-working-strip', () => ({ IsItWorkingStrip: () => <div /> }))
vi.mock('../summary-tab', () => ({ SummaryTab: () => <div>summary</div> }))
vi.mock('../board-tab', () => ({ BoardTab: () => <div>board</div> }))
vi.mock('../calendar-tab', () => ({ CalendarTab: () => <div>calendar</div> }))
vi.mock('../activity-tab', () => ({ ActivityTab: () => <div>activity</div> }))
vi.mock('../watchlist-tab', () => ({ WatchlistTab: () => <div>watchlist</div> }))
vi.mock('../governance-tab', () => ({ GovernanceTab: () => <div>governance</div> }))
vi.mock('../questions-tab', () => ({ QuestionsTab: () => <div>questions</div> }))
vi.mock('@/components/onboarding/trial-balance-pill', () => ({ TrialBalancePill: () => <div /> }))
vi.mock('@/components/onboarding/setup-checklist-card', () => ({ SetupChecklistCard: () => <div /> }))

import { CommandCenterShell } from '../command-center-shell'

const css = readFileSync(path.resolve(__dirname, '..', '..', '..', 'app', 'globals.css'), 'utf8')
const compact = css.slice(css.indexOf('@media (max-width: 1023px)'))

function frame(tab: string) {
  activeTab.current = tab
  const { container } = render(<CommandCenterShell />)
  return container.querySelector('.cc-page > .cc-work') as HTMLElement
}

describe('PRD-252 R8 — the Command Centre frame', () => {
  it('Board and Calendar fill the screen under the strip; other tabs grow under it', () => {
    for (const tab of ['board', 'calendar']) expect(frame(tab).className).toBe('cc-work fill')
    for (const tab of ['summary', 'activity', 'questions']) expect(frame(tab).className).toBe('cc-work')
  })

  it('keeps the strip and the tab body together, after the head and the stats', () => {
    const work = frame('board')
    expect(work.children[0].matches('nav.cc-tabs')).toBe(true)
    expect(work.children[1].matches('.cc-body')).toBe(true)
    expect(work.previousElementSibling).not.toBeNull() // the head and the stats scroll away above it
  })

  it('makes the work area the visible height and sticks the strip', () => {
    // 100% of the page is its content box; the page's top padding sits above it.
    expect(css).toMatch(/\.cc-page \{\s*--cc-page-pad-top: 24px;/)
    expect(css).toMatch(/\.cc-work \{[^}]*min-height: calc\(100% \+ var\(--cc-page-pad-top\)\)/)
    expect(css).toMatch(/\.cc-work\.fill \{ height: calc\(100% \+ var\(--cc-page-pad-top\)\); \}/)
    expect(css).toMatch(/\.cc-work\.fill > \.cc-body \{ flex: 1 1 0; overflow-y: auto;/)
    // Flush under the app header: a sticky box keeps clear of the page's top padding.
    expect(css).toMatch(/:is\(\.studio, \.cc-page\) \.cc-tabs \{[^}]*position: sticky; top: calc\(-1 \* var\(--cc-page-pad-top, 0px\)\);/)
    // The body no longer takes only the height the head leaves (flex: 1; min-height: 0).
    expect(css).toMatch(/\.cc-work > \.cc-body \{ flex: 1 0 auto; min-height: 0; overflow: visible; \}/)
  })

  it('leaves the phone layout as it was', () => {
    expect(compact).toContain(':is(.studio, .cc-page) .cc-work { display: contents; }')
    expect(compact).toContain(':is(.studio, .cc-page) .cc-tabs { top: 0; }')
    expect(compact).toMatch(/:is\(\.cc-work, \.cc-work\.fill\) > \.cc-body \{\s*flex: 1; min-height: 0; overflow-y: auto;/)
  })
})

describe('F219 — no server clock in the eyebrow', () => {
  it('renders the eyebrow without a time on the server, so hydration matches', () => {
    activeTab.current = 'summary'
    const html = renderToString(<CommandCenterShell />)
    expect(html).toContain('>Operations</p>')
    expect(html).not.toMatch(/Operations · /)
  })
})
