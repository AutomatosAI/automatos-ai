/**
 * approveBody — what the mission page's Approve button sends (F165, night 5).
 *
 * The button sent `modifications` (plan edits the server's approve route never
 * took, and nothing on the page could make) and had no way to set a token
 * budget, which the route does take.
 */
import { describe, expect, it } from 'vitest'
import { approveBody, BUDGET_TOO_SMALL } from '../approve-body'

describe('approveBody', () => {
  it('sends nothing when nothing changed', () => {
    expect(approveBody(null, '', 3)).toEqual({})
    expect(approveBody('3', '  ', 3)).toEqual({})
  })

  it('sends the parallel setting when it differs from the plan', () => {
    expect(approveBody('2', '', 3)).toEqual({ body: { max_concurrent_override: 2 } })
  })

  it('sends a token budget', () => {
    expect(approveBody(null, '200000', 3)).toEqual({ body: { token_budget_override: 200000 } })
    expect(approveBody('1', '5000', 3)).toEqual({ body: { max_concurrent_override: 1, token_budget_override: 5000 } })
  })

  it('refuses a budget that is not a whole number of at least 1,000 tokens', () => {
    expect(approveBody(null, '500', 3)).toEqual({ error: BUDGET_TOO_SMALL })
    expect(approveBody(null, '12.5k', 3)).toEqual({ error: BUDGET_TOO_SMALL })
  })

  it('never sends plan edits', () => {
    expect(JSON.stringify(approveBody('2', '5000', 3))).not.toContain('modifications')
  })
})
