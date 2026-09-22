/**
 * computeMissionStats — the mission header's numbers, from the server's rows.
 *
 * F062 (night 2): mission 4cb7872e ran 13:14:44 → 14:38:30 on 09-19 (1 h 24 m)
 * and the page showed 26 h 20 m — the clock counted to `now` long after the
 * mission had finished. And night 1: a mission that was 2 verified, 1 failed,
 * 6 skipped read "0/9 tasks", so the server's task_progress now wins.
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { computeMissionStats, type MissionDetailResponse } from '@/types/missions'

function mission(overrides: Partial<MissionDetailResponse> = {}): MissionDetailResponse {
  return {
    tasks: [],
    recent_events: [],
    tokens_used: 0,
    started_at: '2026-09-19T13:14:44Z',
    completed_at: null,
    ...overrides,
  } as unknown as MissionDetailResponse
}

afterEach(() => {
  vi.useRealTimers()
})

describe('computeMissionStats', () => {
  it('stops the clock when a mission completed (F062)', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-20T15:35:00Z')) // the next afternoon, as in the screenshot
    const stats = computeMissionStats(
      mission({ completed_at: '2026-09-19T14:38:30Z' } as Partial<MissionDetailResponse>),
    )
    expect(stats.elapsedMs).toBe((1 * 60 + 23) * 60_000 + 46_000) // 1 h 23 m 46 s, not 26 h
  })

  it('keeps counting while a mission is still running', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-19T13:24:44Z'))
    expect(computeMissionStats(mission()).elapsedMs).toBe(10 * 60_000)
  })

  it("prefers the server's task counts over deriving them", () => {
    const stats = computeMissionStats(
      mission({
        task_progress: {
          total: 9, done: 9, active: 0, verified: 2, failed: 1, skipped: 6, all_terminal: true,
        },
      } as Partial<MissionDetailResponse>),
    )
    expect(stats.taskCount).toBe(9)
    expect(stats.tasksDone).toBe(9) // night 1 showed 0/9 for exactly this mix
    expect(stats.tasksFailed).toBe(1)
  })
})
