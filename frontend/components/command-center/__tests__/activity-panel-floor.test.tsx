/**
 * #850 — the Live stream panel (Activity) used `style={{ flex: 1, minHeight: 0 }}`
 * under `.cc-body`, with `.cc-panel { overflow: hidden }`. In a short or narrow
 * window the rows above it (stats strip, wrapped toolbar pills) left it nothing,
 * so it collapsed to 0px — header included — and nothing scrolled to bring it
 * back.
 *
 * jsdom does no layout, so a unit test can't see the panel actually shrink.
 * What it CAN assert: the panel carries the floor class instead of the old
 * inline style, and that class's rule in globals.css defines a real min-height
 * floor (not `0`). A Playwright check at a couple of window sizes (e.g.
 * 1024x600, 390x700) asserting the panel is visible and at least the floor
 * height would be the real proof — see the PR description; this repo has no
 * Playwright setup to hang that off yet.
 */
import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'

const FEED = {
  total: 1,
  items: [
    {
      id: 'evt-1',
      type: 'routine',
      name: 'Morning digest',
      status: 'completed',
      started_at: new Date().toISOString(),
      completed_at: new Date().toISOString(),
      duration_seconds: 4,
      agent: { id: 'a1', name: 'Scout' },
      agents: [],
      summary: 'Sent the digest.',
      source_id: null,
      source_url: null,
      trigger: 'scheduled',
      error_message: null,
    },
  ],
}

vi.mock('@/hooks/use-mobile', () => ({
  useIsMobile: () => false,
  useIsTabletOrBelow: () => false,
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), refresh: vi.fn() }),
}))

vi.mock('@/hooks/use-activity-api', () => ({
  activityQueryKeys: { all: ['activity'] },
  useActivityFeed: () => ({ data: FEED, isLoading: false }),
}))

import { ActivityTab } from '../activity-tab'

const css = readFileSync(
  path.resolve(__dirname, '..', '..', '..', 'app', 'globals.css'),
  'utf8',
)

describe('#850 — the Live stream panel keeps a height floor', () => {
  it('the panel carries .cc-panel-floor, not the old zero-floor inline style', () => {
    const { container } = render(<ActivityTab />)
    const panel = container.querySelector('.cc-panel')
    expect(panel).not.toBeNull()
    expect(panel).toHaveClass('cc-panel-floor')
    // The old collapse-to-zero inline style is gone.
    expect((panel as HTMLElement).style.minHeight).toBe('')
  })

  it('.cc-panel-floor sets a real min-height floor, not 0', () => {
    const rule = css.match(/\.cc-panel-floor\s*\{[^}]*\}/)
    expect(rule).not.toBeNull()
    const body = rule![0]
    expect(body).toMatch(/flex:\s*1\s+0\s+auto/)
    expect(body).toMatch(/min-height:\s*var\(--cc-panel-floor-min/)
    // Never a bare 0 floor — that's the bug this guards against.
    expect(body).not.toMatch(/min-height:\s*0[^a-zA-Z0-9.]/)
  })

  it('the --cc-panel-floor-min token (and the var() fallback) is a real floor, not 0', () => {
    expect(css).toMatch(/--cc-panel-floor-min:\s*clamp\(\s*\d+px/)
    const fallback = css.match(/var\(--cc-panel-floor-min,\s*clamp\(\s*(\d+)px/)
    expect(fallback).not.toBeNull()
    expect(Number(fallback![1])).toBeGreaterThan(0)
  })
})
