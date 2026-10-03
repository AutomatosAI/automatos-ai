/**
 * PRD-244 review batch 2 — "Needs you": everything only a human can move, in one place.
 * PRD-252 R1: each row opens the item itself. R5: one endpoint serves the number and
 * the rows; the header's number is the rows listed.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const query = vi.hoisted(() => ({ data: undefined as any, isLoading: false, isError: false }))
vi.mock('@/hooks/use-needs-you', () => ({ useNeedsYou: () => query }))

import { NeedsYouWidget } from '@/components/activity/widgets/needs-you-widget'
import type { NeedsYou } from '@/hooks/use-needs-you'

type Rows = NeedsYou['rows']
const EMPTY: Rows = { review: [], question: [], approval: [], failed: [] }

function waiting(rows: Partial<Rows>, counts?: Partial<NeedsYou['counts']>): NeedsYou {
  const all: Rows = { ...EMPTY, ...rows }
  const c = { review: all.review.length, question: all.question.length, approval: all.approval.length, failed: all.failed.length, ...counts }
  return { period: '1d', total: c.review + c.question + c.approval + c.failed, counts: c, rows: all }
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
    render(<NeedsYouWidget period="1d" />)
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
    render(<NeedsYouWidget period="1d" />)
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
    render(<NeedsYouWidget period="1d" />)
    const href = (text: string) => screen.getByText(text).closest('a')?.getAttribute('href')
    // The owner: "it just takes me to the board and I see loads of tickets in review, when one needs me."
    expect(href('Monday dispatch')).toBe('/command-center?tab=board&task_id=760')
    expect(href('Which café?')).toBe('/command-center?tab=board&task_id=612&question=41')
    expect(href('Which vendor?')).toBe('/command-center?tab=questions')     // asked about no ticket
    expect(href('Send the price list')).toBe('/command-center?tab=board&task_id=760&question=11')
    expect(href('Ship the invoice run')).toBe('/missions/m1')                 // at its plan, where Approve is
    expect(href('Launch the autumn blend')).toBe('/missions/run-9')           // a failed mission card: its mission
  })

  it('says how many more a family has than it lists', () => {
    query.data = waiting({ review: [{ ticket_id: 1, title: 'One', agent_name: null, mission_id: null, at: null }] }, { review: 27 })
    render(<NeedsYouWidget period="1d" />)
    expect(screen.getByText('In review · 27')).toBeInTheDocument()
    expect(screen.getByText('26 more →')).toHaveAttribute('href', '/command-center?tab=board')
  })

  it('hides a family that has nothing', () => {
    query.data = waiting({ approval: [{ source: 'mission', id: 'm1', title: 'Approve me', ticket_id: null, agent_name: null, at: null }] })
    render(<NeedsYouWidget period="1d" />)
    expect(screen.queryByText(/^Questions ·/)).toBeNull()
    expect(screen.getByText('Approvals · 1')).toBeInTheDocument()
  })

  it('says it could not load instead of "nothing on your plate" (F207)', () => {
    query.isError = true
    render(<NeedsYouWidget period="1d" />)
    expect(screen.queryByText('Nothing on your plate. Grab a tea.')).toBeNull()
    expect(screen.getByRole('status')).toHaveTextContent('could not be loaded')
  })
})
