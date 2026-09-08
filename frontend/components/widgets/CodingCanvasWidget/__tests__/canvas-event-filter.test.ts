import { describe, it, expect } from 'vitest'
import { acceptsCanvasEvent, type CanvasEventEnvelope } from '../canvasEvents'

const ev = (data: Record<string, unknown>): CanvasEventEnvelope =>
  ({ schema_version: 1, event_type: 'canvas.tool.call', workspace_id: 'ws', data, timestamp: 't' }) as CanvasEventEnvelope

describe('PRD-235 W2 S3 — which events a Canvas shows', () => {
  it('a ticket-rooted Canvas keeps only its own Claude Code session', () => {
    expect(acceptsCanvasEvent(ev({ task_id: 76, source: 'cli' }), '76')).toBe(true)
    expect(acceptsCanvasEvent(ev({ task_id: 71, source: 'cli' }), '76')).toBe(false)
    expect(acceptsCanvasEvent(ev({}), '76')).toBe(false)
  })
  it('a workspace-rooted Canvas shows the SDK session, not tickets', () => {
    expect(acceptsCanvasEvent(ev({}), undefined)).toBe(true)
    expect(acceptsCanvasEvent(ev({ task_id: 76 }), null)).toBe(false)
  })
})
