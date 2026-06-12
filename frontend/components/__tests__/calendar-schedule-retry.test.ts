import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

// PRD-162 S4 — the PRD-154 S11 stopgap is REMOVED. get_schedule is now a
// stateless DB read (identical on every worker), so the throw-on-inactive +
// 15s-abort + worker-hop retry are moot. The schedule query is a plain cached
// query (staleTime + background refetch). These guards stop the stopgap — and
// the deleted classic calendar tree — from creeping back.

const ROOT = path.resolve(__dirname, '..', '..')
const hookSrc = readFileSync(path.join(ROOT, 'hooks', 'use-activity-api.ts'), 'utf8')

describe('PRD-162 S4 — S11 schedule stopgap removed', () => {
  it('no scheduleOrThrow throw-on-inactive helper remains', () => {
    expect(hookSrc).not.toContain('scheduleOrThrow')
    expect(hookSrc).not.toContain('Scheduler is not active')
  })

  it('no 15s abort-timeout + worker-hop retry stopgap remains', () => {
    expect(hookSrc).not.toContain('AbortSignal.timeout(15000)')
    expect(hookSrc).not.toMatch(/retry:\s*2/)
  })

  it('the schedule query is a plain cached query (staleTime + background refetch)', () => {
    expect(hookSrc).toContain('useActivitySchedule')
    expect(hookSrc).toMatch(/staleTime:\s*\d{4,}/)
    expect(hookSrc).toContain('refetchInterval')
    expect(hookSrc).toContain('keepPreviousData')
  })
})

describe('PRD-162 S4 — classic activity/calendar tree deleted (D8)', () => {
  it('the classic calendar directory is gone (studio CalendarTab is canonical)', () => {
    let exists = true
    try {
      readFileSync(path.join(ROOT, 'components', 'activity', 'calendar', 'index.ts'))
    } catch {
      exists = false
    }
    expect(exists).toBe(false)
  })
})
