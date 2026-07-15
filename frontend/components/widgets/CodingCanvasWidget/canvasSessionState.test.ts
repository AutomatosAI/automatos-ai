/**
 * PRD-170 S3 — canvas session reducer (vitest).
 *
 * Folds a validated canvas event stream into the panel's UI state: ordered
 * turns, live status, the tree-refresh tick that fires on agent file edits, and
 * the pending permission queue S4 renders as approval cards.
 */
import { describe, it, expect } from 'vitest'

import {
  parseCanvasEvent,
  CanvasEventType,
  CANVAS_EVENT_SCHEMA_VERSION,
} from './canvasEvents'
import {
  initialCanvasSessionState,
  reduceCanvasEvent,
  resolvePermission,
  appendUserPrompt,
  type CanvasSessionUiState,
} from './canvasSessionState'

function ev(type: string, data: Record<string, unknown>) {
  const parsed = parseCanvasEvent({
    schema_version: CANVAS_EVENT_SCHEMA_VERSION,
    event_type: type,
    workspace_id: 'ws-1',
    data,
    timestamp: '2026-07-02T00:00:00Z',
  })
  if (!parsed) throw new Error(`fixture did not validate: ${type}`)
  return parsed
}

function fold(events: ReturnType<typeof ev>[]): CanvasSessionUiState {
  return events.reduce(reduceCanvasEvent, initialCanvasSessionState)
}

describe('reduceCanvasEvent', () => {
  it('tracks session status transitions', () => {
    const s = fold([ev(CanvasEventType.SessionStatus, { status: 'running' })])
    expect(s.status).toBe('running')
    expect(s.error).toBeNull()
  })

  it('captures the error on a failed status', () => {
    const s = fold([
      ev(CanvasEventType.SessionStatus, { status: 'failed', error: 'boom' }),
    ])
    expect(s.status).toBe('failed')
    expect(s.error).toBe('boom')
  })

  it('appends assistant text turns in order', () => {
    const s = fold([
      ev(CanvasEventType.AssistantText, { text: 'first' }),
      ev(CanvasEventType.AssistantText, { text: 'second' }),
    ])
    expect(s.turns.map((t) => t.text)).toEqual(['first', 'second'])
    expect(s.turns.every((t) => t.kind === 'text')).toBe(true)
  })

  it('records tool calls with tool name + path', () => {
    const s = fold([
      ev(CanvasEventType.ToolCall, { tool_name: 'Read', path: 'a.py' }),
    ])
    expect(s.turns).toHaveLength(1)
    expect(s.turns[0]).toMatchObject({
      kind: 'tool_call',
      toolName: 'Read',
      path: 'a.py',
    })
  })

  it('bumps the tree-refresh tick on file edits (and only then)', () => {
    const base = fold([
      ev(CanvasEventType.AssistantText, { text: 'hi' }),
      ev(CanvasEventType.ToolCall, { tool_name: 'Read', path: 'a.py' }),
    ])
    expect(base.treeRefreshTick).toBe(0)

    const edited = [
      ev(CanvasEventType.FileEdit, { tool_name: 'Write', path: 'README.md' }),
      ev(CanvasEventType.FileEdit, { tool_name: 'Edit', path: 'a.py' }),
    ].reduce(reduceCanvasEvent, base)
    expect(edited.treeRefreshTick).toBe(2)
    expect(edited.turns.filter((t) => t.kind === 'file_edit')).toHaveLength(2)
  })

  it('queues permission requests for approval cards', () => {
    const s = fold([
      ev(CanvasEventType.PermissionRequest, {
        tool_name: 'Bash',
        path: './run.sh',
        request_id: 'r1',
      }),
    ])
    expect(s.pendingPermissions).toEqual([
      { requestId: 'r1', toolName: 'Bash', path: './run.sh' },
    ])
  })

  it('resolvePermission drops a request once approved/denied', () => {
    const s = fold([
      ev(CanvasEventType.PermissionRequest, {
        tool_name: 'Bash',
        path: null,
        request_id: 'r1',
      }),
      ev(CanvasEventType.PermissionRequest, {
        tool_name: 'Write',
        path: 'x',
        request_id: 'r2',
      }),
    ])
    const after = resolvePermission(s, 'r1')
    expect(after.pendingPermissions.map((p) => p.requestId)).toEqual(['r2'])
  })

  it('counts completed turns on result markers', () => {
    const s = fold([
      ev(CanvasEventType.TurnComplete, { num_turns: 1 }),
      ev(CanvasEventType.TurnComplete, { num_turns: 2 }),
    ])
    expect(s.completedTurns).toBe(2)
  })

  it('is immutable — the initial state is never mutated', () => {
    const before = JSON.stringify(initialCanvasSessionState)
    reduceCanvasEvent(
      initialCanvasSessionState,
      ev(CanvasEventType.FileEdit, { tool_name: 'Write', path: 'z' })
    )
    expect(JSON.stringify(initialCanvasSessionState)).toBe(before)
  })

  it('folds a realistic Create-a-README turn end to end', () => {
    // status.running -> text -> tool.call(Write) + file.edit -> turn.complete -> stopped
    const s = fold([
      ev(CanvasEventType.SessionStatus, { status: 'running' }),
      ev(CanvasEventType.AssistantText, { text: 'Creating the README.' }),
      ev(CanvasEventType.ToolCall, { tool_name: 'Write', path: 'README.md' }),
      ev(CanvasEventType.FileEdit, { tool_name: 'Write', path: 'README.md' }),
      ev(CanvasEventType.TurnComplete, { num_turns: 1 }),
      ev(CanvasEventType.SessionStatus, { status: 'stopped' }),
    ])
    expect(s.status).toBe('stopped')
    expect(s.treeRefreshTick).toBe(1) // one file edit -> one refresh
    expect(s.completedTurns).toBe(1)
    expect(s.turns.map((t) => t.kind)).toEqual([
      'text',
      'tool_call',
      'file_edit',
    ])
  })
})

describe('appendUserPrompt (C·S7 send action)', () => {
  it('appends the user prompt as a user turn, immutably', () => {
    const next = appendUserPrompt(initialCanvasSessionState, 'add validation to X')
    expect(next).not.toBe(initialCanvasSessionState) // new object
    expect(initialCanvasSessionState.turns).toHaveLength(0) // original untouched
    expect(next.turns).toEqual([{ kind: 'user', text: 'add validation to X' }])
  })

  it('trims whitespace and ignores empty prompts', () => {
    const trimmed = appendUserPrompt(initialCanvasSessionState, '   hello   ')
    expect(trimmed.turns).toEqual([{ kind: 'user', text: 'hello' }])

    const noop = appendUserPrompt(initialCanvasSessionState, '   ')
    expect(noop).toBe(initialCanvasSessionState) // no-op returns same state
  })

  it('preserves prior turns and appends after them', () => {
    const first = appendUserPrompt(initialCanvasSessionState, 'first')
    const second = appendUserPrompt(first, 'second')
    expect(second.turns.map((t) => t.text)).toEqual(['first', 'second'])
    expect(second.turns.every((t) => t.kind === 'user')).toBe(true)
  })
})
