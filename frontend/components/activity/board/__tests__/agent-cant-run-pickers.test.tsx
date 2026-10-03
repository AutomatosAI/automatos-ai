/**
 * F244 (night 7) — the board says which agents can't run before a card goes to one.
 *
 * All eight agents read as working while every run failed for the AI credit, so the
 * owner found the one that still worked by giving #0172 to each in turn. The board's
 * agent list and the new card's agent picker now say "Can't run", with why.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('@/components/shared/premium-icon', () => ({ PremiumIcon: () => null }))

import { AgentRunState } from '../board-agent-sidebar'
import { AgentOption } from '../create-task-steps'

const OUT = "Can't run now: your AI credit ran out. Top up the AI provider account, or give the card to a "
  + 'CLI session agent.'

describe("the board's agent list (F244)", () => {
  it("says an agent can't run, with why on hover", () => {
    render(<AgentRunState agent={{ status: 'active', unavailable: OUT }} />)
    expect(screen.getByText("Can't run")).toBeInTheDocument()
    expect(screen.getByTitle(OUT)).toBeInTheDocument()
  })

  it('reads as before for an agent that can run', () => {
    render(<AgentRunState agent={{ status: 'active', unavailable: null }} />)
    expect(screen.getByText('Working')).toBeInTheDocument()
    expect(screen.queryByText("Can't run")).not.toBeInTheDocument()
  })
})

describe("the new card's agent picker (F244)", () => {
  it("says it beside the agent's name", () => {
    render(<AgentOption agent={{ name: 'Shopify Support Agent', unavailable: OUT }} />)
    expect(screen.getByText("· Can't run now: your AI credit ran out")).toBeInTheDocument()
    expect(screen.getByTitle(OUT)).toBeInTheDocument()
  })

  it('shows only the name for an agent that can run', () => {
    render(<AgentOption agent={{ name: 'Analyst', unavailable: null }} />)
    expect(screen.getByText('Analyst')).toBeInTheDocument()
    expect(screen.queryByText(/Can't run/)).not.toBeInTheDocument()
  })
})
