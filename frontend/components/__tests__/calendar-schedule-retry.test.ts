import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import { scheduleOrThrow } from '@/hooks/use-activity-api'

// PRD-154 S11 — calendar stopgap (real root cause = 1-of-4-workers APScheduler
// state, fixed in PRD-162). A get_schedule 200 carrying scheduler_active=false
// used to be cached and rendered as an EMPTY calendar. Now that response throws
// so React Query (retry: 2) retries, and CalendarTab shows a retry affordance.

const ROOT = path.resolve(__dirname, '..', '..')

describe('S11 schedule — inactive scheduler triggers retry, not empty render', () => {
  it('throws when scheduler_active is false (so React Query retries)', () => {
    expect(() => scheduleOrThrow({ scheduled: [], scheduler_active: false })).toThrow()
  })

  it('returns data unchanged when the scheduler is active', () => {
    const data = { scheduled: [{ id: '1' } as any], scheduler_active: true }
    expect(scheduleOrThrow(data)).toBe(data)
  })

  it('returns data when scheduler_active is absent (legacy 200)', () => {
    const data = { scheduled: [] }
    expect(scheduleOrThrow(data as any)).toBe(data)
  })
})

describe('S11 wiring guards', () => {
  it('the schedule query times out and routes the inactive response through scheduleOrThrow', () => {
    const src = readFileSync(path.join(ROOT, 'hooks', 'use-activity-api.ts'), 'utf8')
    expect(src).toContain('AbortSignal.timeout(15000)')
    expect(src).toContain('scheduleOrThrow(')
    expect(src).toContain('retry: 2')
  })

  it('CalendarTab renders a visible error state with a retry button', () => {
    const src = readFileSync(path.join(ROOT, 'components', 'command-center', 'calendar-tab.tsx'), 'utf8')
    expect(src).toContain('isError')
    expect(src).toContain('refetch')
    expect(src).toContain('Retry')
  })
})
