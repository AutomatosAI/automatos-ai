/**
 * PRD-244 review batch 2 — "Needs you": everything only a human can move, in one place.
 * PRD-252 R1: each row opens the item itself. R5: one endpoint serves the number and
 * the rows; the header's number is the rows listed. F246: stuck tickets are a family,
 * each saying why, and a mission's plan carries its card's number.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const query = vi.hoisted(() => ({ data: undefined as any, isLoading: false, isError: false }))
vi.mock('@/hooks/use-needs-you', () => ({ useNeedsYou: () => query }))

import { NeedsYouWidget } from '@/components/activity/widgets/needs-you-widget'
import type { NeedsYou } from '@/hooks/use-needs-you'

type Rows = NeedsYou['rows']
const EMPTY: Rows = { review: [], question: [], approval: [], stuck: [], failed: [] }

function waiting(rows: Partial<Rows>, counts?: Partial<NeedsYou['counts']>): NeedsYou {
  const all: Rows = { ...EMPTY, ...rows }
  const c = {
    review: all.review.length, question: all.question.length, approval: all.approval.length,
    stuck: all.stuck.length, failed: all.failed.length, ...counts,
  }
  return { total: c.review + c.question + c.approval + c.stuck + c.failed, counts: c, rows: all }
}

beforeEach(() => {
  query.data = undefined
  query.isLoading = false
  query.isError = false
})
afterEach(cleanup)

describe('NeedsYouWidget', () => {
  it('with nothing waiting, says so and fabricates no count', () => {
    query.data = waiting({})
    render(<NeedsYouWidget />)
    expect(screen.getByText('Nothing on your plate. Grab a tea.')).toBeInTheDocument()
    expect(screen.queryByText(/waiting$/)).toBeNull()
  })

  it('lists every row it counts, so the number is the rows below it', () => {
    query.data = waiting({
      review: [{ ticket_id: 760, title: 'Monday dispatch', agent_name: 'CLUB SECRETARY', mission_id: null, at: null }],
      question: [{ source: 'grant', id: '41', title: '## Which café?', ticket_id: 612, agent_name: 'Rota', at: null }],
      approval: [
        { source: 'grant', id: '11', title: 'Send the price list', ticket_id: 760, agent_name: null, at: null },
        { source: 'mission', id: 'm1', title: 'Ship the invoice run', ticket_id: null, agent_name: null, at: null },
      ],
      failed: [{ ticket_id: 770, title: 'Weekly numbers', agent_name: null, mission_id: null, at: null }],
    })
    render(<NeedsYouWidget />)
    expect(screen.getByText('5 waiting')).toBeInTheDocument()
    expect(screen.getAllByRole('link').filter((a) => !/→$/.test(a.textContent ?? ''))).toHaveLength(5)
    for (const family of ['In review · 1', 'Questions · 1', 'Approvals · 2', 'Failed · 1']) {
      expect(screen.getByText(family)).toBeInTheDocument()
    }
  })

  it('opens each row at the thing itself', () => {
    query.data = waiting({
      review: [{ ticket_id: 760, title: 'Monday dispatch', agent_name: null, mission_id: null, at: null }],
      question: [
        { source: 'grant', id: '41', title: 'Which café?', ticket_id: 612, agent_name: null, at: null },
        { source: 'grant', id: '42', title: 'Which vendor?', ticket_id: null, agent_name: null, at: null },
      ],
      approval: [
        { source: 'grant', id: '11', title: 'Send the price list', ticket_id: 760, agent_name: null, at: null },
        { source: 'mission', id: 'm1', title: 'Ship the invoice run', ticket_id: null, agent_name: null, at: null },
      ],
      failed: [{ ticket_id: 771, title: 'Launch the autumn blend', agent_name: null, mission_id: 'run-9', at: null }],
    })
    render(<NeedsYouWidget />)
    const href = (text: string) => screen.getByText(text).closest('a')?.getAttribute('href')
    // The owner: "it just takes me to the board and I see loads of tickets in review, when one needs me."
    expect(href('Monday dispatch')).toBe('/command-center?tab=board&task_id=760')
    expect(href('Which café?')).toBe('/command-center?tab=board&task_id=612&question=41')
    expect(href('Which vendor?')).toBe('/command-center?tab=questions')     // asked about no ticket
    expect(href('Send the price list')).toBe('/command-center?tab=board&task_id=760&question=11')
    expect(href('Ship the invoice run')).toBe('/missions/m1')                 // at its plan, where Approve is
    expect(href('Launch the autumn blend')).toBe('/missions/run-9')           // a failed mission card: its mission
  })

  it('lists a stuck ticket with why nothing will move it (F246)', () => {
    query.data = waiting({
      stuck: [
        { ticket_id: 1408, number: '#0176.9', title: 'Send the café letters', agent_name: 'Content Creator', mission_id: null, at: null, why: 'mission_ended' },
        { ticket_id: 1290, number: '#0161', title: 'Cash-up check', agent_name: 'Numbers (on my Mac)', mission_id: null, at: null, why: 'no_host' },
        { ticket_id: 1203, number: '#0067', title: 'Order oat milk', agent_name: null, mission_id: null, at: null, why: 'no_agent' },
        { ticket_id: 1380, number: '#0149', title: 'Weekly posts', agent_name: 'Social', mission_id: null, at: null, why: 'not_picked_up' },
      ],
    })
    render(<NeedsYouWidget />)
    expect(screen.getByText('Stuck · 4')).toBeInTheDocument()
    expect(screen.getByText('4 waiting')).toBeInTheDocument()
    const row = (title: string) => screen.getByText(title).closest('a') as HTMLElement
    expect(row('#0176.9 · Send the café letters')).toHaveAttribute('href', '/command-center?tab=board&task_id=1408')
    expect(row('#0176.9 · Send the café letters')).toHaveTextContent('Its mission has ended · Content Creator')
    expect(row('#0161 · Cash-up check')).toHaveTextContent('Waiting for a CLI host that is not online · Numbers (on my Mac)')
    expect(row('#0067 · Order oat milk')).toHaveTextContent('Assigned to no agent · An agent')
    expect(row('#0149 · Weekly posts')).toHaveTextContent('Waiting, but nothing will run it · Social')
  })

  it("names a mission plan's card by its number (F246)", () => {
    query.data = waiting({
      approval: [{ source: 'mission', id: 'run-31', title: 'Autumn menu launch', ticket_id: 1130, ticket_number: '#0031', agent_name: null, at: null }],
    })
    render(<NeedsYouWidget />)
    expect(screen.getByText('#0031 · Autumn menu launch').closest('a')).toHaveAttribute('href', '/missions/run-31')
  })

  it('says how many more a family has than it lists', () => {
    query.data = waiting({ review: [{ ticket_id: 1, title: 'One', agent_name: null, mission_id: null, at: null }] }, { review: 27 })
    render(<NeedsYouWidget />)
    expect(screen.getByText('In review · 27')).toBeInTheDocument()
    expect(screen.getByText('26 more →')).toHaveAttribute('href', '/command-center?tab=board')
  })

  it('hides a family that has nothing', () => {
    query.data = waiting({ approval: [{ source: 'mission', id: 'm1', title: 'Approve me', ticket_id: null, agent_name: null, at: null }] })
    render(<NeedsYouWidget />)
    expect(screen.queryByText(/^Questions ·/)).toBeNull()
    expect(screen.getByText('Approvals · 1')).toBeInTheDocument()
  })

  it('says it could not load instead of "nothing on your plate" (F207)', () => {
    query.isError = true
    render(<NeedsYouWidget />)
    expect(screen.queryByText('Nothing on your plate. Grab a tea.')).toBeNull()
    expect(screen.getByRole('status')).toHaveTextContent('could not be loaded')
  })
})
