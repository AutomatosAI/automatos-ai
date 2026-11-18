/**
 * Ticket PR fields — the pull request a coding session opened shows on its ticket: a "Pull request"
 * section in the viewer (the link in a new tab with no opener, the branch, the commits, what was
 * tested) and a small PR chip on the card in the board list. Nothing shows for a ticket without one,
 * and only an https address is ever a link: with the install's setting on, the owner's Approve
 * merges the pull request the card shows.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { cleanup, fireEvent, render, renderHook, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React, { type ReactNode } from 'react'
import type { DraggableProvided } from '@hello-pangea/dnd'
import type { BoardTask } from '@/types/board'
import {
  APPROVE_WILL_MERGE,
  BRANCH_LABEL,
  COMMITS_LABEL,
  PR_CHIP,
  PR_CHIP_TITLE,
  PULL_REQUEST_SECTION,
  REMOVE_PULL_REQUEST,
  REMOVE_PULL_REQUEST_CONFIRM,
  REMOVE_PULL_REQUEST_TITLE,
  TESTED_LABEL,
  approvalIdentity,
  pullRequestLinkText,
  pullRequestOf,
  type TicketPullRequest,
} from '@/lib/ticket-pull-request'

vi.mock('@hello-pangea/dnd', () => ({
  Draggable: ({ children }: { children: (provided: unknown, snapshot: unknown) => React.ReactNode }) =>
    children(
      { innerRef: () => {}, draggableProps: { style: {} }, dragHandleProps: {} },
      { isDragging: false },
    ),
  DragDropContext: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Droppable: () => null,
}))
vi.mock('../ticket-actions-menu', () => ({ TicketActionsMenu: () => null }))
vi.mock('@/components/activity/board/board-task-viewer', () => ({ BoardTaskViewer: () => null }))
const requestMock = vi.hoisted(() => vi.fn())
vi.mock('@/lib/api-client', () => ({ apiClient: { request: (...args: unknown[]) => requestMock(...args) } }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { BoardCard } from '../board-card'
import { PullRequestChip, TicketPullRequestSection } from '../ticket-pull-request'
import { useLiveTask } from '../use-live-task'
import { KanbanCardFace } from '@/components/command-center/board-tab'
import { approveBody, useBoardTasks } from '@/hooks/use-board-tasks'

const PR_URL = 'https://github.com/automatos-ai/automatos-ai/pull/1559'
const SHA = '9a9cb36d9e1f4c2b7a0d3e5f6a7b8c9d0e1f2a3b'
const SERVED = {
  url: PR_URL,
  host: 'github.com',
  repository: 'automatos-ai/automatos-ai',
  number: 1559,
  branch: 'feat/ticket-pr-fields',
  commits: [
    { sha: SHA, subject: 'feat(board): the card carries its pull request' },
    { sha: '114d30a', subject: 'test(board): the card’s pull request' },
  ],
  tested: 'py_compile on the changed files; nothing else could be run here.',
  previous_urls: [],
  at: '2026-11-18T09:30:00+00:00',
  merges_on_approve: false,
}

function pr(over: Record<string, unknown> = {}): TicketPullRequest {
  const parsed = pullRequestOf({ ...SERVED, ...over })
  if (!parsed) throw new Error('the test pull request did not parse')
  return parsed
}

function ticket(over: Partial<BoardTask> = {}): BoardTask {
  return { id: '1559', type: 'task', name: 'Ticket PR fields', status: 'review', priority: 'medium',
    tags: [], review_mode: 'human', source_id: '1559', ...over }
}

afterEach(() => cleanup())
beforeEach(() => requestMock.mockReset())

function withQueries() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

const PROVIDED = { innerRef: () => {}, draggableProps: { style: {} }, dragHandleProps: {} } as unknown as DraggableProvided

describe('TicketPullRequestSection', () => {
  it('links the pull request in a new tab with no opener, and names its repository and number', () => {
    render(<TicketPullRequestSection pullRequest={pr()} />)
    const section = screen.getByTestId('ticket-pull-request')
    expect(within(section).getByText(PULL_REQUEST_SECTION)).toBeInTheDocument()
    const link = within(section).getByRole('link', { name: /automatos-ai\/automatos-ai #1559/ })
    expect(link).toHaveAttribute('href', PR_URL)
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('shows the branch, every commit by its short sha and subject, and what was tested', () => {
    render(<TicketPullRequestSection pullRequest={pr()} />)
    expect(screen.getByText(BRANCH_LABEL)).toBeInTheDocument()
    expect(screen.getByText('feat/ticket-pr-fields')).toBeInTheDocument()
    expect(screen.getByText(COMMITS_LABEL)).toBeInTheDocument()
    const commits = within(screen.getByTestId('pull-request-commits')).getAllByRole('listitem')
    expect(commits).toHaveLength(2)
    expect(commits[0]).toHaveTextContent('9a9cb36 feat(board): the card carries its pull request')
    expect(commits[1]).toHaveTextContent('114d30a test(board): the card’s pull request')
    expect(screen.getByText(TESTED_LABEL)).toBeInTheDocument()
    expect(screen.getByTestId('pull-request-tested')).toHaveTextContent(SERVED.tested)
  })

  it('shows nothing for a ticket without a pull request', () => {
    const { container } = render(<TicketPullRequestSection pullRequest={null} />)
    expect(container).toBeEmptyDOMElement()
    cleanup()
    const { container: missing } = render(<TicketPullRequestSection />)
    expect(missing).toBeEmptyDOMElement()
  })

  it('never links an address that is not https', () => {
    for (const url of ['javascript:alert(1)', 'http://github.com/o/r/pull/1', 'data:text/html,x']) {
      const parsed = pullRequestOf({ ...SERVED, url })
      expect(parsed?.url).toBeNull()
      render(<TicketPullRequestSection pullRequest={parsed} />)
      expect(screen.queryByRole('link')).toBeNull()
      expect(screen.getByText('feat/ticket-pr-fields')).toBeInTheDocument()   // the rest still shows
      cleanup()
    }
  })
})

describe('the PR chip', () => {
  it('marks a card in the board list whose session opened a pull request', () => {
    render(<BoardCard task={ticket({ pull_request: pr() })} index={0} onOpen={vi.fn()} />)
    const chip = screen.getByTestId('pull-request-chip')
    expect(chip).toHaveTextContent(PR_CHIP)
    expect(chip).toHaveAttribute('title', PR_CHIP_TITLE)
  })

  it('is absent on a card without one, or with no address', () => {
    render(<BoardCard task={ticket()} index={0} onOpen={vi.fn()} />)
    expect(screen.queryByTestId('pull-request-chip')).toBeNull()
    cleanup()
    const { container } = render(<PullRequestChip pullRequest={pr({ url: null })} />)
    expect(container).toBeEmptyDOMElement()
  })
})

describe('pullRequestOf', () => {
  it('keeps what the API served, checked', () => {
    expect(pullRequestOf(SERVED)).toEqual({
      url: PR_URL, repository: 'automatos-ai/automatos-ai', number: 1559, branch: 'feat/ticket-pr-fields',
      commits: SERVED.commits, tested: SERVED.tested, at: SERVED.at, merges_on_approve: false,
    })
    expect(pullRequestLinkText(pr())).toBe('automatos-ai/automatos-ai #1559')
    expect(pullRequestLinkText(pr({ repository: null }))).toBe(PR_URL)
  })

  it('drops commits that are not a sha, and is null when nothing is left to show', () => {
    expect(pullRequestOf({ ...SERVED, commits: [{ sha: 'not-a-sha', subject: 'x' }, 'abc', null] })?.commits).toEqual([])
    expect(pullRequestOf(null)).toBeNull()
    expect(pullRequestOf('https://github.com/o/r/pull/1')).toBeNull()
    expect(pullRequestOf({ url: 'javascript:alert(1)' })).toBeNull()
  })
})

describe('the wiring', () => {
  it('the board maps the API ticket\'s pull_request onto the card, and null without one', async () => {
    requestMock.mockResolvedValue({ tasks: [
      { id: 1559, title: 'Ticket PR fields', status: 'review', pull_request: SERVED },
      { id: 1560, title: 'Weekly report', status: 'review' },
    ], total: 2 })
    const { result } = renderHook(() => useBoardTasks({}), { wrapper: withQueries() })
    await waitFor(() => expect(result.current.columns.find((c) => c.status === 'review')?.tasks).toHaveLength(2))
    const [withPr, without] = result.current.columns.find((c) => c.status === 'review')!.tasks
    expect(withPr.pull_request).toEqual(pullRequestOf(SERVED))
    expect(without.pull_request).toBeNull()
  })

  it('a running ticket\'s viewer picks up a pull request its session reports mid-run', async () => {
    requestMock.mockResolvedValue({ status: 'in_progress', pull_request: SERVED })
    const running = ticket({ status: 'in_progress', pull_request: null })
    const { result } = renderHook(() => useLiveTask(running), { wrapper: withQueries() })
    await waitFor(() => expect(result.current?.pull_request?.url).toBe(PR_URL))
  })

  it('the viewer shows the section once, for every column, from the ticket\'s pull_request', () => {
    const source = readFileSync(path.resolve(__dirname, '..', 'board-task-viewer.tsx'), 'utf8')
    expect(source.split('<TicketPullRequestSection pullRequest={task.pull_request} taskId={task.id} />')).toHaveLength(2)
    expect(source).toContain("import { useLiveTask } from './use-live-task'")
  })

  it("Studio's board card carries the PR chip too, and not without a pull request", () => {
    render(<KanbanCardFace task={ticket({ pull_request: pr() })} density="comfortable" onOpen={vi.fn()} provided={PROVIDED} isDragging={false} />)
    const chip = screen.getByTestId('pull-request-chip')
    expect(chip).toHaveTextContent(PR_CHIP)
    expect(chip).toHaveAttribute('title', PR_CHIP_TITLE)
    cleanup()
    render(<KanbanCardFace task={ticket()} density="comfortable" onOpen={vi.fn()} provided={PROVIDED} isDragging={false} />)
    expect(screen.queryByTestId('pull-request-chip')).toBeNull()
  })
})

describe('what Approve will do, and taking a wrong pull request off', () => {
  it('says approving will merge it only when the server says it would', () => {
    render(<TicketPullRequestSection pullRequest={pr({ merges_on_approve: true })} />)
    expect(screen.getByTestId('pull-request-merges-on-approve')).toHaveTextContent(APPROVE_WILL_MERGE)
    cleanup()
    render(<TicketPullRequestSection pullRequest={pr()} />)
    expect(screen.queryByTestId('pull-request-merges-on-approve')).toBeNull()
    expect(pullRequestOf({ ...SERVED, url: 'javascript:alert(1)', merges_on_approve: true })?.merges_on_approve).toBe(false)
  })

  it('the owner removes it only after the house confirm, with one DELETE on the ticket', async () => {
    requestMock.mockResolvedValue({ removed: true })
    render(<TicketPullRequestSection pullRequest={pr()} taskId="1559" />, { wrapper: withQueries() })
    fireEvent.click(screen.getByRole('button', { name: new RegExp(`^${REMOVE_PULL_REQUEST}$`) }))
    expect(screen.getByText(REMOVE_PULL_REQUEST_TITLE)).toBeInTheDocument()
    expect(requestMock).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: new RegExp(`^${REMOVE_PULL_REQUEST_CONFIRM}$`) }))
    await waitFor(() => expect(requestMock).toHaveBeenCalledWith('/api/v1/tasks/1559/pull-request', { method: 'DELETE' }))
  })

  it('has no Remove without a ticket to remove it from', () => {
    render(<TicketPullRequestSection pullRequest={pr()} />)
    expect(screen.queryByRole('button', { name: new RegExp(`^${REMOVE_PULL_REQUEST}$`) })).toBeNull()
  })

  it('an Approve sends back the pull request it showed, and nothing when it showed none', () => {
    expect(approvalIdentity(pr())).toEqual({ url: PR_URL, at: SERVED.at })
    expect(approvalIdentity(null)).toBeNull()
    expect(approvalIdentity(pr({ url: null }))).toBeNull()
    expect(approveBody(undefined, approvalIdentity(pr()))).toEqual({ pull_request: { url: PR_URL, at: SERVED.at } })
    expect(approveBody('Ship it.', null)).toEqual({ note: 'Ship it.' })
    expect(approveBody()).toEqual({})
  })
})
