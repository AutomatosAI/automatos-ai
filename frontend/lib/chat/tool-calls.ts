/**
 * PRD-238 S3 — the activity trail's state helpers (pure).
 *
 * `useChat` folds stream frames into a message's `toolCalls`; the trail
 * renders them. Both sides share these rules so a chip can never spin forever:
 * a `finish` frame closes whatever is still running.
 */
import type { TaskCardData, ToolCall } from '@/types'

/** Insert or merge a tool call by id, preserving order. Never mutates. */
export function upsertToolCall(current: ToolCall[] | undefined, next: ToolCall): ToolCall[] {
  const list = current ? [...current] : []
  const index = list.findIndex((t) => t.toolCallId === next.toolCallId)
  if (index >= 0) {
    return list.map((t, i) => (i === index ? { ...t, ...next } : t))
  }
  return [...list, next]
}

/** The turn ended: anything still "running" completed without a tool-end frame. */
export function completeRunningToolCalls(current: ToolCall[] | undefined, endedAt: string): ToolCall[] | undefined {
  if (!current || !current.some((t) => t.state === 'running')) return current
  return current.map((t) => (t.state === 'running' ? { ...t, state: 'completed', endedAt } : t))
}

/** "0.2 s", "12 s", "1m 05s" — compact, for a trail line. */
export function formatDuration(ms: number | undefined): string {
  if (ms === undefined || !Number.isFinite(ms) || ms < 0) return ''
  if (ms < 1000) return `${(ms / 1000).toFixed(1)} s`
  const seconds = Math.round(ms / 1000)
  if (seconds < 60) return `${seconds} s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${String(seconds % 60).padStart(2, '0')}s`
}

/** PRD-238 S6: one card per ticket id, newest data wins, order preserved. Never mutates. */
export function upsertTaskCard(current: TaskCardData[] | undefined, next: TaskCardData): TaskCardData[] {
  const list = current ? [...current] : []
  const index = list.findIndex((c) => c.id === next.id)
  if (index >= 0) return list.map((c, i) => (i === index ? { ...c, ...next } : c))
  return [...list, next]
}
