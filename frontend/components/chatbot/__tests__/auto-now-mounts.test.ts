/** PRD-244 D5 — the rail is mounted in both styles, from existing reads only. */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

const ROOT = path.resolve(__dirname, '..', '..', '..')
const read = (rel: string) => readFileSync(path.join(ROOT, rel), 'utf8')

describe('Auto now mounts', () => {
  it('the Studio shell mounts the rail inside its rail aside, above the mission section, and its bar control is the pill', () => {
    const src = read('components/chatbot/studio-chat-shell.tsx')
    const aside = src.indexOf('<aside className="sh-chat-rail" aria-label="Auto now rail">')
    expect(aside).toBeGreaterThan(-1)
    expect(src.indexOf('<AutoNowRail />', aside)).toBeGreaterThan(aside)
    expect(src.indexOf('<MissionSection', aside)).toBeGreaterThan(src.indexOf('<AutoNowRail />', aside))
    expect(src).toContain('<AutoNowPill open={!railCollapsed} onToggle={toggleRail}')
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
