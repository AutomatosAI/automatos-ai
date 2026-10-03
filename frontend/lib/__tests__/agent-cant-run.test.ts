import { describe, expect, it } from 'vitest'

import { cantRunLine, getAgentRoleLine } from '../agent-constants'

// F244 (night 7): every agent read "active" while every run failed for the AI credit.
describe('an agent that cannot run says so on its card', () => {
  it('leads the subtitle with the first sentence of why', () => {
    const agent = {
      category: 'operations',
      job_title: 'Analyst',
      unavailable: "Can't run now: your AI credit ran out. Top up the AI provider account, or give the card to a CLI session agent.",
    }
    expect(cantRunLine(agent)).toBe("Can't run now: your AI credit ran out")
    expect(getAgentRoleLine(agent)).toMatch(/^Can't run now: your AI credit ran out · .+ · Analyst$/)
  })

  it('leaves an agent that can run as it was', () => {
    const agent = { category: 'operations', job_title: 'Analyst', unavailable: null }
    expect(cantRunLine(agent)).toBeNull()
    expect(getAgentRoleLine(agent)).not.toContain("Can't run")
  })
})
