/**
 * F167 — a session ticket says what the host decided for each tool call.
 */
import { describe, it, expect } from 'vitest'
import { sessionToolCalls, toolCallVerdict, toolCallTitle, toolDecisionsSummary } from '../session-tool-calls'

const UNLISTED = "'google-chrome --headless' is not on this ticket's Bash allowlist; this host runs such commands without asking (--unlisted-bash allow)"

describe('what the host decided, on the ticket', () => {
  it('says a call that ran with nobody asked was not approved by anyone', () => {
    const [chrome] = sessionToolCalls({ recent_tools: [
      { at: '2026-09-25T15:14:56Z', tool: 'Bash', subject: 'google-chrome --headless --print-to-pdf=report.pdf', decision: 'allow', reason: UNLISTED },
    ] })
    expect(toolCallVerdict(chrome)).toBe('ran, nobody was asked')
    expect(toolCallTitle(chrome)).toBe(`google-chrome --headless --print-to-pdf=report.pdf — ran, nobody was asked (${UNLISTED})`)
  })

  it('names a hold by its answer, a refusal as the gate’s, and stays silent for an older host', () => {
    const calls = sessionToolCalls({ recent_tools: [
      { tool: 'Bash', decision: 'ask', answer: 'approved' },
      { tool: 'Bash', decision: 'ask', answer: 'no answer' },
      { tool: 'Bash', decision: 'ask', answer: 'approval not on record' },
      { tool: 'Bash', decision: 'ask' },
      { tool: 'Bash', decision: 'deny', reason: 'never allowed in a session' },
      { tool: 'Edit', subject: 'notes.md' },
      null, 'x',
    ] })
    expect(calls.map(toolCallVerdict)).toEqual([
      'held for the operator: the operator approved it',
      'held for the operator: no answer in time, so it did not run',
      'held for the operator: the host reported an approval that is not on record',
      'held for the operator: no answer recorded',
      'refused by the gate',
      '',
    ])
    expect(toolCallTitle(calls[5])).toBe('notes.md')
  })

  it('counts every decision, and says when nothing was held for the operator', () => {
    expect(toolDecisionsSummary({ tool_decisions: { allow: 3 } }))
      .toBe('3 ran with nobody asked · none held for the operator · 0 refused by the gate')
    expect(toolDecisionsSummary({ tool_decisions: { allow: 1, ask: 2, approved: 1, deny: 4 } }))
      .toBe('1 ran with nobody asked · 2 held for the operator (1 approved) · 4 refused by the gate')
    expect(toolDecisionsSummary({ tool_decisions: { ask: 1, unrecorded: 1 } }))
      .toBe('0 ran with nobody asked · 1 held for the operator (0 approved, 1 approval not on record) · 0 refused by the gate')
    expect(toolDecisionsSummary({})).toBeNull()
    expect(toolDecisionsSummary(null)).toBeNull()
  })
})
