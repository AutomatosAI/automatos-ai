/**
 * PRD-268 §6.11.1, §6.12.3 (US-W04) — the status vocabulary on the web. The needs-you widget, the Missions list
 * (Studio's grouped rows and table, Classic's cards) and the board cards (Classic's and Studio's) each draw a row's
 * glyph, its word and ▸ its next step from the row's own fields, in Studio's pill or Classic's badge; an unknown
 * state draws nothing (the Missions list keeps its run-state badge for a state the vocabulary leaves unnamed); Try
 * again runs the row's retry through the existing routes (a failed Mission card's is its Mission's replan) and opens
 * nothing else; the rows that need the person come first.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, within, type RenderResult } from '@testing-library/react'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const look = vi.hoisted(() => ({ studio: true }))
const calls = vi.hoisted(() => ({
  runTask: vi.fn(() => Promise.resolve({})),
  replan: vi.fn(() => Promise.resolve({})),
  rerun: vi.fn(() => Promise.resolve({})),
  push: vi.fn(),
}))
const data = vi.hoisted(() => ({ needsYou: undefined as unknown, missions: [] as unknown[] }))

vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => look.studio }))
vi.mock('@/hooks/use-needs-you', () => ({
  useNeedsYou: () => ({ data: data.needsYou, isLoading: false, isError: false }),
  needsYouQueryKeys: { all: ['activity', 'needs-you'] },
}))
vi.mock('@/hooks/use-board-tasks', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-board-tasks')>()),
  useRunTask: () => ({ mutateAsync: calls.runTask, isLoading: false }),
}))
vi.mock('@/hooks/use-missions-api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-missions-api')>()),
  useMissions: () => ({ data: { missions: data.missions }, isLoading: false }),
  useReplanMission: () => ({ mutateAsync: calls.replan }),
  useRerunMission: () => ({ mutateAsync: calls.rerun }),
  usePauseMission: () => ({ mutate: vi.fn() }),
  useResumeMission: () => ({ mutate: vi.fn() }),
  useCancelMission: () => ({ mutate: vi.fn() }),
  useDeleteMission: () => ({ mutateAsync: vi.fn() }),
}))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: calls.push }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => '/assignments',
}))
vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: React.ReactNode; href: string }) => <a href={href} {...rest}>{children}</a>,
}))
vi.mock('@/lib/auth-hooks', () => ({ useUser: () => ({ user: { id: 'u-1' } }) }))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => false }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@hello-pangea/dnd', () => ({
  Draggable: ({ children }: { children: (provided: unknown, snapshot: unknown) => React.ReactNode }) =>
    children({ innerRef: () => {}, draggableProps: { style: {} }, dragHandleProps: {} }, { isDragging: false }),
  DragDropContext: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Droppable: () => null,
}))
vi.mock('@/components/activity/board/ticket-actions-menu', () => ({ TicketActionsMenu: () => null }))
vi.mock('@/components/activity/board/board-task-viewer', () => ({ BoardTaskViewer: () => null }))
vi.mock('@/components/activity/board/host-offline-banner', () => ({ HostOfflineBanner: () => null }))
vi.mock('@/hooks/use-agent-api', () => ({ useAssignableAgents: () => ({ data: [] }) }))
vi.mock('@/hooks/use-ticket-deep-link', () => ({ useTicketDeepLink: () => ({}) }))
vi.mock('@/components/workspace-provider', () => ({ useWorkspace: () => ({ canEdit: true }), useWorkspaceOptional: () => null }))
vi.mock('@/components/missions/create-mission-modal', () => ({ CreateMissionModal: () => null }))

import { NeedsYouWidget } from '@/components/activity/widgets/needs-you-widget'
import { MissionsBody } from '@/components/assignments/studio/missions-body'
import { MissionCard } from '@/components/missions/mission-card'
import { MissionList } from '@/components/missions/mission-list'
import { BoardCard } from '@/components/activity/board/board-card'
import { KanbanCardFace } from '@/components/command-center/board-tab'
import { needsYouFirst, rowStatusOf } from '@/lib/proactive/row-status'
import type { NeedsYou } from '@/hooks/use-needs-you'
import type { BoardTask } from '@/types/board'
import type { MissionResponse } from '@/types/missions'
import type { DraggableProvided } from '@hello-pangea/dnd'

const REVIEW_STEP = 'Look over the work: approve it, or send it back with what to change.'
const FAILED_STEP = "It couldn't finish: try again, or close it if it's no longer needed."
const PLAN_STEP = "Approve the Mission's plan, or reject it."
const REPLAN_STEP = "The Mission couldn't finish: replan it to try again, or let it go."

const EMPTY_ROWS: NeedsYou['rows'] = { review: [], question: [], approval: [], stuck: [], failed: [] }

function needsYou(rows: Partial<NeedsYou['rows']>): NeedsYou {
  const all = { ...EMPTY_ROWS, ...rows }
  const counts = { review: all.review.length, question: all.question.length, approval: all.approval.length, stuck: all.stuck.length, failed: all.failed.length }
  return { total: Object.values(counts).reduce((sum, n) => sum + n, 0), counts, rows: all }
}

function mission(over: Partial<MissionResponse> = {}): MissionResponse {
  return {
    id: 'run-1', workspace_id: 'ws-1', goal: 'Ship the autumn menu', state: 'running', plan: null, config: { tone: 'warm' },
    tokens_used: 1200, token_budget_estimate: null, parallel_groups: [], complexity_tier: null, created_by: 'u-1',
    created_at: '2026-10-30T08:00:00Z', updated_at: '2026-10-30T09:00:00Z', started_at: null, completed_at: null,
    ...over,
  } as MissionResponse
}

function ticket(over: Partial<BoardTask> = {}): BoardTask {
  return { id: '1139', type: 'task', name: 'Price list for Tidewater', status: 'in_progress', priority: 'medium', tags: [], review_mode: 'auto', source_id: '1139', ...over }
}

const PROVIDED = { innerRef: () => {}, draggableProps: { style: {} }, dragHandleProps: {} } as unknown as DraggableProvided

/** Renders inside a query client, as the app does: Try again reads the Needs-you number again once it settles. */
function draw(ui: React.ReactElement): RenderResult {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(ui, { wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider> })
}

/** The glyph, the word and ▸ the next step, inside `box`. */
function expectVocabulary(box: HTMLElement, glyph: string, word: string, step?: string): void {
  const scope = within(box)
  expect(scope.getByText(word)).toBeInTheDocument()
  expect(scope.getByText(glyph)).toHaveAttribute('aria-hidden', 'true')
  if (step) expect(scope.getByText(step)).toHaveTextContent(`▸ ${step}`)
}

function needsYouRow(title: string): HTMLElement {
  const link = screen.getByText(title).closest('a')
  if (!link?.parentElement) throw new Error(`no needs-you row for ${title}`)
  return link.parentElement
}

beforeEach(() => {
  look.studio = true
  data.needsYou = undefined
  data.missions = []
  Object.values(calls).forEach((call) => call.mockClear())
})
afterEach(cleanup)

describe('each surface draws the glyph, the word and the next step', () => {
  it('the needs-you widget, one row at a time', () => {
    data.needsYou = needsYou({
      review: [{ ticket_id: 760, number: '#0042', title: 'Monday dispatch', agent_name: null, mission_id: null, at: null, display_state: 'needs_you', next_step: REVIEW_STEP }],
      failed: [{ ticket_id: 770, number: '#0043', title: 'Weekly numbers', agent_name: null, mission_id: null, at: null, display_state: 'failed', next_step: FAILED_STEP, retry: 'retry_ticket' }],
    })
    draw(<NeedsYouWidget />)
    expectVocabulary(needsYouRow('#0042 · Monday dispatch'), '◆', 'Needs you', REVIEW_STEP)
    expectVocabulary(needsYouRow('#0043 · Weekly numbers'), '✕', "Couldn't finish", FAILED_STEP)
    // The status sits beside the link, so the link still opens the ticket and holds no button.
    expect(within(screen.getByText('#0043 · Weekly numbers').closest('a') as HTMLElement).queryByRole('button')).toBeNull()
  })

  it("Studio's grouped rows and table, in Studio's pill", () => {
    data.missions = [mission({ id: 'run-2', goal: 'Approve the menu plan', state: 'awaiting_approval', display_state: 'needs_you', next_step: PLAN_STEP })]
    draw(<MissionsBody />)
    const row = screen.getByText('Approve the menu plan').closest('.mis-row') as HTMLElement
    expectVocabulary(row, '◆', 'Needs you', PLAN_STEP)
    expect(within(row).getByText('Needs you').parentElement).toHaveClass('studio-pill')

    fireEvent.click(screen.getByRole('button', { name: /Table/ }))
    const tableRow = screen.getByText('Approve the menu plan').closest('tr') as HTMLElement
    expectVocabulary(tableRow, '◆', 'Needs you', PLAN_STEP)
  })

  it("Classic's Mission card, in Classic's badge", () => {
    look.studio = false
    draw(<MissionCard mission={mission({ state: 'running', display_state: 'working' })} index={0} />)
    const card = screen.getByText('Ship the autumn menu').closest('.glass-card') as HTMLElement
    expectVocabulary(card, '●', 'Working')
    expect(within(card).getByText('Working').parentElement).not.toHaveClass('studio-pill')
  })

  it("Classic's and Studio's board cards", () => {
    look.studio = false
    const { unmount } = draw(<BoardCard task={ticket({ status: 'done', display_state: 'done' })} index={0} onOpen={vi.fn()} />)
    expectVocabulary(screen.getByRole('button', { name: /Price list for Tidewater/ }), '✓', 'Done')
    unmount()

    look.studio = true
    draw(<KanbanCardFace task={ticket({ display_state: 'working', next_step: '3 of 5 made' })} density="comfortable" onOpen={vi.fn()} provided={PROVIDED} isDragging={false} />)
    expectVocabulary(screen.getByRole('button', { name: /Price list for Tidewater/ }), '●', 'Working', '3 of 5 made')
  })
})

describe('unknown draws nothing', () => {
  it('on a needs-you row and a board card', () => {
    data.needsYou = needsYou({ review: [{ ticket_id: 760, title: 'Monday dispatch', agent_name: null, mission_id: null, at: null, display_state: 'mystery' as never, next_step: REVIEW_STEP }] })
    const { container, unmount } = draw(<NeedsYouWidget />)
    expect(container.querySelector('[data-display-state]')).toBeNull()
    expect(screen.queryByText(REVIEW_STEP)).toBeNull()
    unmount()

    const board = draw(<BoardCard task={ticket({ status: 'inbox', display_state: null, next_step: 'Never drawn' })} index={0} onOpen={vi.fn()} />)
    expect(board.container.querySelector('[data-display-state]')).toBeNull()
    expect(screen.queryByText('Never drawn')).toBeNull()
  })

  it('the Missions list keeps the run-state badge for a state the vocabulary leaves unnamed', () => {
    data.missions = [mission({ goal: 'Paused for the weekend', state: 'paused', display_state: null })]
    draw(<MissionsBody />)
    fireEvent.click(screen.getByRole('button', { name: /Table/ }))
    const tableRow = screen.getByText('Paused for the weekend').closest('tr') as HTMLElement
    expect(tableRow.querySelector('[data-display-state]')).toBeNull()
    expect(within(tableRow).getByText('● PAUSED')).toBeInTheDocument()
    cleanup()

    look.studio = false
    draw(<MissionCard mission={mission({ state: 'paused', display_state: null })} index={0} />)
    expect(document.querySelector('[data-display-state]')).toBeNull()
    expect(screen.getByText('Paused')).toBeInTheDocument()
  })
})

describe("Try again calls the row's retry, and opens nothing else", () => {
  const failedMission = mission({ id: 'run-9', goal: 'Launch the autumn blend', state: 'failed', display_state: 'failed', next_step: REPLAN_STEP, retry: 'replan_mission' })

  it('a failed ticket runs again; a failed Mission card is replanned (the needs-you widget)', () => {
    data.needsYou = needsYou({
      failed: [
        { ticket_id: 770, number: '#0043', title: 'Weekly numbers', agent_name: null, mission_id: null, at: null, display_state: 'failed', next_step: FAILED_STEP, retry: 'retry_ticket' },
        { ticket_id: 771, number: '#0044', title: 'Mission: autumn blend', agent_name: null, mission_id: 'run-9', opens: 'mission', at: null, display_state: 'failed', next_step: REPLAN_STEP, retry: 'replan_mission' },
      ],
    })
    draw(<NeedsYouWidget />)
    fireEvent.click(screen.getByRole('button', { name: 'Try again: #0043 · Weekly numbers' }))
    expect(calls.runTask).toHaveBeenCalledWith({ taskId: '770' })
    fireEvent.click(screen.getByRole('button', { name: 'Try again: #0044 · Mission: autumn blend' }))
    expect(calls.replan).toHaveBeenCalledWith({ id: 'run-9' })
  })

  it("Studio's row and table replan the Mission without opening it", () => {
    data.missions = [failedMission]
    draw(<MissionsBody />)
    fireEvent.click(screen.getByRole('button', { name: 'Try again: Launch the autumn blend' }))
    expect(calls.replan).toHaveBeenCalledWith({ id: 'run-9' })
    fireEvent.click(screen.getByRole('button', { name: /Table/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Try again: Launch the autumn blend' }))
    expect(calls.replan).toHaveBeenCalledTimes(2)
    expect(calls.push).not.toHaveBeenCalled()
  })

  it("Classic's Mission card replans, and reruns from the same goal when that is the retry", () => {
    look.studio = false
    const { unmount } = draw(<MissionCard mission={failedMission} index={0} />)
    fireEvent.click(screen.getByRole('button', { name: 'Try again: Launch the autumn blend' }))
    expect(calls.replan).toHaveBeenCalledWith({ id: 'run-9' })
    unmount()

    draw(<MissionCard mission={{ ...failedMission, retry: 'rerun_mission' }} index={0} />)
    fireEvent.click(screen.getByRole('button', { name: 'Try again: Launch the autumn blend' }))
    expect(calls.rerun).toHaveBeenCalledWith({ goal: 'Launch the autumn blend', config: { tone: 'warm' } })
    expect(calls.push).not.toHaveBeenCalled()
  })

  it('a board card runs its ticket again, by click or key, without opening it', () => {
    const onOpen = vi.fn()
    const failed = ticket({ id: '1201', status: 'failed', display_state: 'failed', next_step: FAILED_STEP, retry: 'retry_ticket' })
    const { unmount } = draw(<BoardCard task={failed} index={0} onOpen={onOpen} />)
    fireEvent.click(screen.getByRole('button', { name: 'Try again: Price list for Tidewater' }))
    expect(calls.runTask).toHaveBeenCalledWith({ taskId: '1201' })
    unmount()

    draw(<KanbanCardFace task={failed} density="comfortable" onOpen={onOpen} provided={PROVIDED} isDragging={false} />)
    const tryAgain = screen.getByRole('button', { name: 'Try again: Price list for Tidewater' })
    fireEvent.keyDown(tryAgain, { key: 'Enter' })
    fireEvent.click(tryAgain)
    expect(calls.runTask).toHaveBeenCalledTimes(2)
    expect(onOpen).not.toHaveBeenCalled()
  })

  it("a failed Mission's own board card replans its Mission, on Classic's board and Studio's", () => {
    const onOpen = vi.fn()
    const card = ticket({ id: '1301', type: 'mission', source_type: 'orchestration', mission_id: 'run-9', status: 'failed', display_state: 'failed', next_step: REPLAN_STEP, retry: 'replan_mission' })
    const { unmount } = draw(<BoardCard task={card} index={0} onOpen={onOpen} />)
    fireEvent.click(screen.getByRole('button', { name: 'Try again: Price list for Tidewater' }))
    expect(calls.replan).toHaveBeenCalledWith({ id: 'run-9' })
    unmount()

    draw(<KanbanCardFace task={card} density="comfortable" onOpen={onOpen} provided={PROVIDED} isDragging={false} />)
    fireEvent.click(screen.getByRole('button', { name: 'Try again: Price list for Tidewater' }))
    expect(calls.replan).toHaveBeenCalledTimes(2)
    expect(calls.runTask).not.toHaveBeenCalled()
    expect(onOpen).not.toHaveBeenCalled()
  })

  it("Studio's Mission row opens on Enter from the row itself, never from its Try again", () => {
    data.missions = [failedMission]
    draw(<MissionsBody />)
    const row = screen.getByText('Launch the autumn blend').closest('.mis-row') as HTMLElement
    expect(row).toHaveAttribute('role', 'button')
    fireEvent.keyDown(screen.getByRole('button', { name: 'Try again: Launch the autumn blend' }), { key: 'Enter' })
    expect(calls.push).not.toHaveBeenCalled()
    fireEvent.keyDown(row, { key: 'Enter' })
    expect(calls.push).toHaveBeenCalledWith('/missions/run-9')
  })

  it('draws no Try again when the row names nothing its retry can act on', () => {
    data.needsYou = needsYou({ failed: [{ ticket_id: 772, title: 'Orphan', agent_name: null, mission_id: null, opens: 'mission', at: null, display_state: 'failed', retry: 'replan_mission' }] })
    draw(<NeedsYouWidget />)
    expect(within(needsYouRow('Orphan')).getByText("Couldn't finish")).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^Try again/ })).toBeNull()
  })
})

describe('the rows that need the person come first', () => {
  it('needsYouFirst keeps every other row where it was', () => {
    const rows = [{ id: 'a', display_state: 'working' }, { id: 'b', display_state: 'needs_you' }, { id: 'c', display_state: null }, { id: 'd', display_state: 'needs_you' }] as const
    expect(needsYouFirst(rows).map((row) => row.id)).toEqual(['b', 'd', 'a', 'c'])
  })

  it("in Studio's table", () => {
    data.missions = [
      mission({ id: 'run-a', goal: 'Weekly report', state: 'running', display_state: 'working' }),
      mission({ id: 'run-b', goal: 'Approve the menu plan', state: 'awaiting_approval', display_state: 'needs_you', next_step: PLAN_STEP }),
    ]
    draw(<MissionsBody />)
    fireEvent.click(screen.getByRole('button', { name: /Table/ }))
    const goals = screen.getAllByRole('row').slice(1).map((row) => within(row).getByText(/Weekly report|Approve the menu plan/).textContent)
    expect(goals).toEqual(['Approve the menu plan', 'Weekly report'])
  })

  it("in Classic's Mission list", () => {
    look.studio = false
    data.missions = [
      mission({ id: 'run-a', goal: 'Weekly report', state: 'running', display_state: 'working' }),
      mission({ id: 'run-b', goal: 'Approve the menu plan', state: 'awaiting_approval', display_state: 'needs_you', next_step: PLAN_STEP }),
    ]
    draw(<MissionList />)
    const titles = screen.getAllByRole('heading', { level: 3 }).map((heading) => heading.textContent)
    expect(titles).toEqual(['Approve the menu plan', 'Weekly report'])
  })

  it('in the needs-you widget, every family that needs the person before the failed one', () => {
    data.needsYou = needsYou({
      failed: [{ ticket_id: 770, title: 'Weekly numbers', agent_name: null, mission_id: null, at: null, display_state: 'failed' }],
      review: [{ ticket_id: 760, title: 'Monday dispatch', agent_name: null, mission_id: null, at: null, display_state: 'needs_you' }],
    })
    const { container } = draw(<NeedsYouWidget />)
    const states = Array.from(container.querySelectorAll('[data-display-state]')).map((node) => node.getAttribute('data-display-state'))
    expect(states).toEqual(['needs_you', 'failed'])
  })
})

describe('rowStatusOf', () => {
  it('keeps each field only when it is what it says', () => {
    expect(rowStatusOf({ display_state: 'failed', next_step: FAILED_STEP, retry: 'retry_ticket' })).toEqual({ display_state: 'failed', next_step: FAILED_STEP, retry: 'retry_ticket' })
    expect(rowStatusOf({ display_state: 'stopped', next_step: { text: 'x' }, retry: 'reboot' })).toEqual({ display_state: null, next_step: null, retry: null })
    expect(rowStatusOf(null)).toEqual({ display_state: null, next_step: null, retry: null })
    expect(rowStatusOf({ display_state: 'working', next_step: '   ' }).next_step).toBeNull()
  })
})
