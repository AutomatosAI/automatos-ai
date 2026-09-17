/** PRD-244 D5 — the rail is mounted in both styles, from existing reads only. */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

const ROOT = path.resolve(__dirname, '..', '..', '..')
const read = (rel: string) => readFileSync(path.join(ROOT, rel), 'utf8')

describe('Auto now mounts', () => {
  it('the Studio shell mounts the rail above the mission section, and its bar control is the pill', () => {
    // PRD-246 US-003 moved the rail's content into a `railPanel` const so the
    // SAME rail can be the grid's third column on a desktop and the pill's
    // sheet below 1280 — one rail, two homes. The order it asserts is inside
    // that panel now, not inside the aside.
    const src = read('components/chatbot/studio-chat-shell.tsx')
    const panel = src.indexOf('const railPanel = (')
    expect(panel).toBeGreaterThan(-1)
    expect(src.indexOf('<AutoNowRail />', panel)).toBeGreaterThan(panel)
    expect(src.indexOf('<MissionSection', panel)).toBeGreaterThan(src.indexOf('<AutoNowRail />', panel))
    expect(src).toContain('<aside className="sh-chat-rail" aria-label="Auto now rail">')
    expect(src).toContain('<AutoNowPill open={railShown} onToggle={toggleRailPanel}')
    expect(src).not.toContain('function MissionRail')
  })

  it('the Classic chat mounts the rail in a right aside with the pill, desktop only', () => {
    const src = read('app/chat/page.tsx')
    expect(src).toContain("useAutoNowOpen('classicChatAutoNowOpen')")
    expect(src).toContain('aria-label="Auto now rail"')
    expect(src).toContain('<AutoNowRail />')
    expect(src).toContain('<AutoNowPill')
  })

  it('the rail reads the floor through the existing hooks only — no new endpoint', () => {
    const hook = read('hooks/use-auto-now.ts')
    expect(hook).not.toMatch(/apiClient\.request|fetch\(/)
    for (const h of ['useActivityStats', 'useFleetState', 'useQuestions', 'useWatches', 'useDecisionsNeeded']) expect(hook).toContain(h)
  })
})
