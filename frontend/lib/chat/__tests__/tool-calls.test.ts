/**
 * PRD-238 S3 — activity-trail state helpers.
 */
import { describe, it, expect } from 'vitest'
import { completeRunningToolCalls, formatDuration, upsertToolCall } from '@/lib/chat/tool-calls'
import type { ToolCall } from '@/types'

const running = (id: string): ToolCall => ({ toolCallId: id, toolName: 'platform_get_agent', state: 'running' })

describe('upsertToolCall', () => {
  it('appends new calls and merges updates by id without mutating', () => {
    const first = upsertToolCall(undefined, running('a'))
    const second = upsertToolCall(first, running('b'))
    const merged = upsertToolCall(second, { toolCallId: 'a', toolName: 'platform_get_agent', state: 'completed', summary: 'Bob' })
    expect(second).toHaveLength(2)
    expect(merged.map((t) => t.toolCallId)).toEqual(['a', 'b'])
    expect(merged[0]).toMatchObject({ state: 'completed', summary: 'Bob' })
    expect(second[0].state).toBe('running') // input untouched
  })
})

describe('completeRunningToolCalls', () => {
  it('closes every running call at finish and leaves the rest alone', () => {
    const list = [running('a'), { ...running('b'), state: 'error' as const, error: 'x' }]
    const done = completeRunningToolCalls(list, '2026-09-08T10:00:00Z')!
    expect(done[0]).toMatchObject({ state: 'completed', endedAt: '2026-09-08T10:00:00Z' })
    expect(done[1]).toMatchObject({ state: 'error', error: 'x' })
    expect(list[0].state).toBe('running')
  })

  it('returns the same reference when nothing is running', () => {
    const list = [{ ...running('a'), state: 'completed' as const }]
    expect(completeRunningToolCalls(list, 'now')).toBe(list)
    expect(completeRunningToolCalls(undefined, 'now')).toBeUndefined()
  })
})

describe('formatDuration', () => {
  it('formats compactly', () => {
    expect(formatDuration(200)).toBe('0.2 s')
    expect(formatDuration(12_400)).toBe('12 s')
    expect(formatDuration(65_000)).toBe('1m 05s')
    expect(formatDuration(undefined)).toBe('')
    expect(formatDuration(-1)).toBe('')
  })
})
