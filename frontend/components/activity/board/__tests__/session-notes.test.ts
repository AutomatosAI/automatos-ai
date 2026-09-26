/**
 * F094 (night 5) — the notes a Claude Code session ticket shows.
 */
import { describe, it, expect } from 'vitest'
import { sessionNotes } from '../session-notes'

const STOPPED_WAITING =
  'The mission stopped waiting for this step after 60 minutes. The session is still working, and its result will land here.'

describe('notes on a session ticket', () => {
  it('reads the notes in order and ignores junk entries', () => {
    const ref = { session_notes: [
      { note: 'reading the brief', by: 'NEWSROOM', at: '2026-09-26T00:10:00Z' },
      null, 'x', { note: 42 }, { note: '  ' },
      { note: STOPPED_WAITING, by: 'the mission' },
    ] }
    expect(sessionNotes(ref)).toEqual([
      { note: 'reading the brief', by: 'NEWSROOM', at: '2026-09-26T00:10:00Z' },
      { note: STOPPED_WAITING, by: 'the mission', at: undefined },
    ])
  })

  it('has nothing to show without notes', () => {
    expect(sessionNotes(null)).toEqual([])
    expect(sessionNotes({})).toEqual([])
    expect(sessionNotes({ session_notes: 'x' })).toEqual([])
  })
})
