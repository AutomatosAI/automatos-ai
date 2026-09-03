/**
 * PRD-234 — the lines a review'd Claude Code session ticket shows.
 */
import { describe, it, expect } from 'vitest'
import { sessionDenials, denialLine, reviewReason } from '../session-denials'

describe('session denials on the ticket', () => {
  it('reads the kept reasons and ignores junk entries', () => {
    const ref = { permission_denials: [
      { tool: 'Bash', stage: 'PreToolUse', reason: "'python3 hello.py' is outside this ticket's Bash allowlist", subject: 'python3 hello.py' },
      null, 'x', { tool: 42 },
    ] }
    const out = sessionDenials(ref)
    expect(out).toHaveLength(2)
    expect(denialLine(out[0])).toBe("Bash · 'python3 hello.py' is outside this ticket's Bash allowlist")
    expect(denialLine(out[1])).toBe('refused by the session policy')
  })

  it('explains review from the count, singular and plural, and stays silent when nothing was refused', () => {
    expect(reviewReason({ denials: 0 })).toBeNull()
    expect(reviewReason(null)).toBeNull()
    expect(reviewReason({ denials: 1 })).toMatch(/refused 1 tool call,/)
    expect(reviewReason({ denials: 2 })).toMatch(/refused 2 tool calls,/)
    expect(reviewReason({ permission_denials: [{ tool: 'Bash' }] })).toMatch(/refused 1 tool call,/)
  })
})
