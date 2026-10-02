/**
 * PRD-252 R2 — Discuss on a ticket opens /chat?ticket=<id>: a new conversation
 * with the ticket in its context, a bar that says so, and "Update ticket and
 * re-queue". A mission's ticket points at the mission instead (D4).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import type { BoardTask } from '@/types/board'

const nav = vi.hoisted(() => ({ query: 'ticket=612', replace: vi.fn() }))
const board = vi.hoisted(() => ({ task: undefined as unknown, isError: false }))
const session = vi.hoisted(() => ({ newDraft: vi.fn() }))

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(nav.query),
  useRouter: () => ({ replace: nav.replace }),
}))
vi.mock('next/link', () => ({ default: ({ href, children, ...rest }: any) => <a href={String(href)} {...rest}>{children}</a> }))
vi.mock('@/hooks/use-board-tasks', () => ({ useBoardTask: () => ({ data: board.task, isError: board.isError }) }))
vi.mock('@/stores/chat-session-store', () => ({
  useChatSessionStore: (pick: (s: unknown) => unknown) => pick({ hydrated: true, newDraft: session.newDraft, session: { activeChatId: null } }),
}))
vi.mock('../rebrief-dialog', () => ({ RebriefDialog: () => <div data-testid="rebrief-dialog" /> }))

import { DiscussionBar } from '../discussion-bar'
import { useDiscussionStore } from '@/stores/discussion-store'

function ticket(over: Partial<BoardTask> = {}): BoardTask {
  return { id: '612', type: 'task', name: 'Welcome email', status: 'review', priority: 'medium', tags: [],
    review_mode: 'human', source_id: '612', number: '#0042', assignee: { agent_id: 3, agent_name: 'Words' }, ...over }
}

beforeEach(() => {
  nav.query = 'ticket=612'
  nav.replace.mockReset()
  session.newDraft.mockReset()
  board.task = ticket()
  board.isError = false
  useDiscussionStore.getState().end()
})
afterEach(cleanup)

describe('DiscussionBar', () => {
  it('starts the discussion in a new conversation and says which ticket it is about', () => {
    render(<DiscussionBar />)
    expect(screen.getByTestId('discussion-bar')).toHaveTextContent('Discussing ticket #0042 · Welcome email')
    expect(screen.getByText('ticket #0042').closest('a')).toHaveAttribute('href', '/command-center?tab=board&task_id=612')
    expect(session.newDraft).toHaveBeenCalledTimes(1)
    expect(useDiscussionStore.getState().discussion).toMatchObject({ ticketId: '612', missionId: null })
    expect(screen.getByText(/To ask Words instead/)).toBeInTheDocument()
  })

  it('opens "Update ticket and re-queue"', () => {
    render(<DiscussionBar />)
    fireEvent.click(screen.getByRole('button', { name: 'Update ticket and re-queue' }))
    expect(screen.getByTestId('rebrief-dialog')).toBeInTheDocument()
  })

  it("points a mission's ticket at the mission, with nothing to re-queue (D4)", () => {
    board.task = ticket({ type: 'mission', mission_id: 'run-9' })
    render(<DiscussionBar />)
    expect(screen.getByText('Decide on the mission →').closest('a')).toHaveAttribute('href', '/missions/run-9')
    expect(screen.queryByRole('button', { name: 'Update ticket and re-queue' })).toBeNull()
  })

  it('ends: the ticket leaves the chat and its link', () => {
    render(<DiscussionBar />)
    fireEvent.click(screen.getByRole('button', { name: 'End the discussion' }))
    expect(useDiscussionStore.getState().discussion).toBeNull()
    expect(nav.replace).toHaveBeenCalledWith('/chat')
  })

  it("leaves a session ticket's canvas link alone, and says when the ticket cannot be opened", () => {
    nav.query = 'repo=projects%2Fsite&ticket=612&runtime=1'
    const { container } = render(<DiscussionBar />)
    expect(container).toBeEmptyDOMElement()
    expect(session.newDraft).not.toHaveBeenCalled()
    cleanup()
    nav.query = 'ticket=999'
    board.task = undefined
    board.isError = true
    render(<DiscussionBar />)
    expect(screen.getByTestId('discussion-bar')).toHaveTextContent('Ticket 999 could not be opened here')
  })
})
