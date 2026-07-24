/**
 * PRD-170 S3 — canvas event schema drift gate (vitest).
 *
 * Two jobs:
 *  1. VALIDATION: `parseCanvasEvent` accepts well-formed current-schema events
 *     and REJECTS malformed / wrong-version / drifted-name payloads (returns
 *     null instead of rendering a bogus turn).
 *  2. LOCKSTEP: the TS vocabulary + schema version are asserted equal to the
 *     Python source of truth (`services/workspace-worker/canvas_events.py`),
 *     read from disk at test time. A rename or version bump on EITHER side that
 *     isn't reflected on the other fails here — this is the reachability-style
 *     drift guard the PRD requires.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import {
  CANVAS_EVENT_SCHEMA_VERSION,
  CANVAS_EVENT_TYPES,
  CanvasEventType,
  parseCanvasEvent,
  isTreeMutatingEvent,
} from './canvasEvents'

function validEvent(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: CANVAS_EVENT_SCHEMA_VERSION,
    event_type: CanvasEventType.AssistantText,
    workspace_id: 'ws-1',
    data: { text: 'hi' },
    timestamp: '2026-07-02T00:00:00Z',
    ...overrides,
  }
}

describe('parseCanvasEvent validation', () => {
  it('accepts a well-formed current-schema event', () => {
    const ev = parseCanvasEvent(validEvent())
    expect(ev).not.toBeNull()
    expect(ev?.event_type).toBe(CanvasEventType.AssistantText)
  })

  it('accepts every known event type', () => {
    for (const t of CANVAS_EVENT_TYPES) {
      const ev = parseCanvasEvent(validEvent({ event_type: t }))
      expect(ev, `type ${t} should validate`).not.toBeNull()
    }
  })

  it('rejects a drifted / unknown event type', () => {
    expect(parseCanvasEvent(validEvent({ event_type: 'canvas.assistant.txt' }))).toBeNull()
    expect(parseCanvasEvent(validEvent({ event_type: 'assistant.text' }))).toBeNull()
    expect(parseCanvasEvent(validEvent({ event_type: 'bogus' }))).toBeNull()
  })

  it('rejects a wrong (future/legacy) schema version', () => {
    expect(parseCanvasEvent(validEvent({ schema_version: 0 }))).toBeNull()
    expect(parseCanvasEvent(validEvent({ schema_version: 2 }))).toBeNull()
    expect(parseCanvasEvent(validEvent({ schema_version: '1' }))).toBeNull()
  })

  it('rejects structurally malformed payloads', () => {
    expect(parseCanvasEvent(null)).toBeNull()
    expect(parseCanvasEvent('nope')).toBeNull()
    expect(parseCanvasEvent(validEvent({ workspace_id: '' }))).toBeNull()
    expect(parseCanvasEvent(validEvent({ timestamp: '' }))).toBeNull()
    expect(parseCanvasEvent(validEvent({ data: null }))).toBeNull()
    expect(parseCanvasEvent(validEvent({ data: 'x' }))).toBeNull()
  })

  it('flags file-edit events as tree-mutating (and others not)', () => {
    const edit = parseCanvasEvent(validEvent({ event_type: CanvasEventType.FileEdit }))!
    expect(isTreeMutatingEvent(edit)).toBe(true)
    const text = parseCanvasEvent(validEvent())!
    expect(isTreeMutatingEvent(text)).toBe(false)
  })
})

describe('lockstep with the Python source of truth', () => {
  // canvasEvents.ts lives at frontend/components/widgets/CodingCanvasWidget/;
  // the Python schema is at <repo>/services/workspace-worker/canvas_events.py.
  const pySource = readFileSync(
    resolve(__dirname, '../../../../services/workspace-worker/canvas_events.py'),
    'utf-8'
  )

  it('schema version matches the Python literal', () => {
    const m = pySource.match(/CANVAS_EVENT_SCHEMA_VERSION\s*=\s*(\d+)/)
    expect(m, 'CANVAS_EVENT_SCHEMA_VERSION not found in canvas_events.py').not.toBeNull()
    expect(Number(m![1])).toBe(CANVAS_EVENT_SCHEMA_VERSION)
  })

  it('event-type vocabulary matches the Python EVENT_* literals exactly', () => {
    // Extract every  EVENT_XXX = "canvas.â€¦"  string literal from the Python file.
    const pyTypes = new Set<string>()
    const re = /^EVENT_[A-Z_]+\s*=\s*"([^"]+)"/gm
    let match: RegExpExecArray | null
    while ((match = re.exec(pySource)) !== null) {
      pyTypes.add(match[1])
    }
    expect(pyTypes.size, 'no EVENT_* literals parsed from canvas_events.py').toBeGreaterThan(0)

    const tsTypes = new Set(CANVAS_EVENT_TYPES)
    // Symmetric difference must be empty: neither side has an event the other lacks.
    const onlyPy = [...pyTypes].filter((t) => !tsTypes.has(t))
    const onlyTs = [...tsTypes].filter((t) => !pyTypes.has(t))
    expect(onlyPy, `event(s) in Python but not TS: ${onlyPy}`).toEqual([])
    expect(onlyTs, `event(s) in TS but not Python: ${onlyTs}`).toEqual([])
  })
})
