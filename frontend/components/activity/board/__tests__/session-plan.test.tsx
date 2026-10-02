/**
 * PRD-253 Wave P — a ticket parked on its plan says so, and shows the plan.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { SessionPlanPanel, latestSessionPlan } from '@/components/activity/board/session-plan'

vi.mock('next/link', () => ({ default: ({ children, href }: { children: React.ReactNode; href: string }) => <a href={href}>{children}</a> }))

const waiting = { version: 2, plan: '1. Add hello.txt\n2. Verify with cat', grant_id: 41 }
const answered = { ...waiting, answer: 'Approve', answered_at: '2026-10-02T10:05:00+00:00' }

describe('latestSessionPlan', () => {
  it('reads the latest plan the sessions presented', () => {
    expect(latestSessionPlan({ session_plans: [{ version: 1, plan: 'old', answered_at: 't' }, waiting] })).toEqual({
      round: 2, plan: waiting.plan, waiting: true, answer: null,
    })
    expect(latestSessionPlan({ session_plans: [answered] })?.waiting).toBe(false)
  })

  it('is nothing for a ticket that never planned, or a ledger it cannot read', () => {
    expect(latestSessionPlan(null)).toBeNull()
    expect(latestSessionPlan({})).toBeNull()
    expect(latestSessionPlan({ session_plans: 'nope' })).toBeNull()
    expect(latestSessionPlan({ session_plans: [{ version: 1, plan: '   ' }, null] })).toBeNull()
  })
})

describe('SessionPlanPanel', () => {
  it('says the ticket waits for the plan, shows it and where to answer', () => {
    render(<SessionPlanPanel runtimeRef={{ session_plans: [waiting] }} />)
    const panel = screen.getByTestId('session-plan')
    expect(panel.textContent).toContain('Waiting for your approval of its plan (round 2)')
    expect(panel.textContent).toContain('2. Verify with cat')
    expect(screen.getByRole('link', { name: 'Command Center → Questions' }).getAttribute('href')).toBe(
      '/command-center?tab=questions',
    )
  })

  it('records an answered plan without asking again', () => {
    render(<SessionPlanPanel runtimeRef={{ session_plans: [answered] }} />)
    const panel = screen.getByTestId('session-plan')
    expect(panel.textContent).toContain('Its plan (round 2) — your answer: Approve')
    expect(panel.textContent).not.toContain('Waiting for your approval')
    expect(screen.queryByRole('link')).toBeNull()
  })

  it('renders nothing for a ticket with no plan', () => {
    const { container } = render(<SessionPlanPanel runtimeRef={{ runtime: 'cli' }} />)
    expect(container.innerHTML).toBe('')
  })
})
