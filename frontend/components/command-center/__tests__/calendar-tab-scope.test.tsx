/**
 * The Command Centre calendar renders in two places: the studio shell
 * (`.studio` on <html>) and the classic ActivityPage (dark / light / matte —
 * no `.studio` anywhere). Every `cc-cal-*` rule in globals.css used to be
 * `.studio`-scoped, so the classic Command Centre showed the week grid as a
 * stack of unstyled divs (2026-09-08). These guards keep both mounts styled:
 * the component owns a `.cc-cal-root` scope, and the stylesheet reaches every
 * calendar rule from it.
 */
import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))
vi.mock('@/hooks/use-activity-api', () => ({
  activityQueryKeys: { all: ['activity'] },
  useActivitySchedule: () => ({
    data: { scheduled: [] },
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  }),
  useSchedulerHealth: () => ({ data: { healthy: null, last_fired_at: null } }),
}))
vi.mock('@/hooks/use-heartbeats-api', () => ({
  useToggleHeartbeat: () => ({ mutate: vi.fn() }),
}))
vi.mock('@/hooks/use-scheduled-tasks-api', () => ({
  useUpdateScheduledTaskStatus: () => ({ mutate: vi.fn() }),
}))

import { CalendarTab } from '../calendar-tab'

const css = readFileSync(
  path.resolve(__dirname, '..', '..', '..', 'app', 'globals.css'),
  'utf8',
)

describe('CalendarTab — styled outside the studio theme', () => {
  it('wraps the calendar in its own .cc-cal-root scope', () => {
    const { container } = render(<CalendarTab />)
    const root = container.querySelector('.cc-cal-root')
    expect(root).not.toBeNull()
    expect(root!.querySelector('.cc-cal-toolbar')).not.toBeNull()
    expect(root!.querySelector('.cc-cal-grid')).not.toBeNull()
  })

  it('no calendar rule is reachable only from .studio', () => {
    expect(css).not.toMatch(/^\s*\.studio\s+\.cc-cal-/m)
    expect(css).toMatch(/^\s*:is\(\.studio, \.cc-cal-root\) \.cc-cal-grid \{/m)
    expect(css).toMatch(/^\s*\.cc-cal-root \{/m)
  })

  it('the primitives the calendar uses are reachable from .cc-cal-root', () => {
    for (const cls of ['cc-btn', 'cc-seg', 'cc-panel-empty']) {
      expect(css).toMatch(
        new RegExp(`^\\s*:is\\(\\.studio, \\.cc-cal-root\\) \\.${cls}\\b`, 'm'),
      )
    }
  })

  it('hover and live-dot colours fall back to theme tokens outside studio', () => {
    // CD's cream hover is a studio-only token; a dark-theme event must not
    // flash near-white on hover.
    expect(css).not.toMatch(/\.cc-cal-event:hover \{[^}]*hsl\(38 45% 96%\)/)
    expect(css).toMatch(/^\s*--cc-hover:\s*38 45% 96%;/m)
    expect(css).toMatch(/hsl\(var\(--cc-hover, var\(--muted\)\)\) !important/)
    expect(css).toMatch(/hsl\(var\(--cc-live-dot, var\(--success\)\)\)/)
  })
})
