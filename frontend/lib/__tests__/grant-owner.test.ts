import { describe, expect, it } from 'vitest'
import { agentLabel, ownerLine, ticketLabel } from '../grant-owner'

describe('grant owner labels (F091-E1)', () => {
  it('names the agent and the ticket when the list carries them', () => {
    const owner = { agent: { id: 294, name: 'Scout' }, ticket: { id: 612, title: 'Cafe questions' } }
    expect(ownerLine(owner)).toBe('Scout (agent #294) · Ticket #612 · Cafe questions')
  })

  it('falls back to ids, then to "An agent"', () => {
    expect(agentLabel({ agent: { id: 294, name: null } })).toBe('Agent #294')
    expect(agentLabel(null, 7)).toBe('Agent #7')
    expect(agentLabel(null, null)).toBe('An agent')
    expect(ticketLabel({ ticket: { id: 612 } })).toBe('Ticket #612')
  })

  it('says nothing when there is nobody to name', () => {
    expect(ownerLine(null)).toBeNull()
    expect(ownerLine({ agent: null, ticket: null })).toBeNull()
  })
})
