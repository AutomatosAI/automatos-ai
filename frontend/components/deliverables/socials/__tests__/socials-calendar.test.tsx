/**
 * PRD-251B US-B108 — the Socials calendar on the Command Center's grid.
 *
 * * Month, Week and List show the same posts; a chip carries the time in the post's
 *   timezone and the format (a video with its length), the title, the channel badges
 *   and the status WORD for every state.
 * * The channel filter hides the other channels' posts; Video shows only videos.
 * * Drag: a post waiting for approval dropped on another day keeps its time on the new
 *   day through PUT /slot; so does a scheduled post (the one schedule path keeps the
 *   planned slot and the schedule together); a published post does not drag.
 * * The Today rail lists today's posts, says how many wait for review (into the Queue),
 *   and carries the status key. A click opens the post; Needs you opens the Queue.
 *
 * The clock is fixed on Wed 14 Oct 2026 (only Date is faked); the grid is laid out in
 * the runner's time, the times are read in each post's own zone.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, within, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const state = vi.hoisted(() => ({ go: vi.fn() }))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams(''),
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/lib/api-client', () => {
  const kind = (k: string) => ({ kind: k, available: true, reason: null, needs_public_storage: false })
  const apiClient = {
    listSocialChannels: vi.fn(async () => [
      { toolkit: 'twitter', label: 'X', post_kinds: [kind('image')], verified: true, setup_note: null, copy_limits: { text: 280 } },
      { toolkit: 'linkedin', label: 'LinkedIn', post_kinds: [kind('image')], verified: true, setup_note: null, copy_limits: { text: 3000 } },
    ]),
    setSocialPostSlot: vi.fn(async () => ({})),
    scheduleSocialPost: vi.fn(async () => ({})),
  }
  return { apiClient, default: apiClient }
})
vi.mock('@/components/deliverables/socials/socials-post-list', () => ({
  SocialsPostList: () => <div data-testid="post-list" />,
}))

import { apiClient } from '@/lib/api-client'
import { SocialsCalendar } from '@/components/deliverables/socials/studio/socials-calendar'
import { SOCIAL_DRAG_TYPE } from '@/components/command-center/calendar-social'
import type { SocialsRoute } from '@/components/deliverables/socials/studio/studio-route'

const target = (toolkit: string, published_at: string | null = null) => ({
  id: `t-${toolkit}`, toolkit, post_kind: 'image', options: {}, status: published_at ? 'published' : 'pending',
  attempts: 0, remote_id: null, permalink: null, error: null, published_at,
})

function post(id: string, status: string, slot: string, extra: Record<string, unknown> = {}) {
  return {
    id, title: `Post ${id}`, status, format: 'image', length_seconds: null, timezone: 'UTC',
    planned_for: slot, scheduled_for: null, targets: [target('twitter')], created_at: '2026-10-01T09:00:00Z',
    copy: { base: '' }, variables: {}, sources: {}, media: {}, review_log: [], content_hash: 'h', approved_hash: null,
    ...extra,
  } as any
}

const POSTS = [
  post('planned', 'draft', '2026-10-13T09:00:00Z'),
  post('making', 'rendering', '2026-10-13T11:00:00Z'),
  post('review', 'needs_approval', '2026-10-14T09:00:00Z', { timezone: 'Europe/London', targets: [target('linkedin')] }),
  post('changes', 'changes_requested', '2026-10-15T10:00:00Z'),
  post('scheduled', 'scheduled', null as any, { scheduled_for: '2026-10-14T12:00:00Z' }),
  post('posted', 'published', null as any, {
    scheduled_for: '2026-10-12T12:00:00Z', targets: [target('twitter', '2026-10-12T12:01:00Z'), target('linkedin', '2026-10-12T12:02:00Z')],
  }),
  post('skipped', 'missed', '2026-10-12T15:00:00Z'),
  post('failed', 'failed', null as any, { scheduled_for: '2026-10-16T12:00:00Z' }),
  post('video', 'draft', '2026-10-16T17:00:00Z', { format: 'video', length_seconds: 30, targets: [target('tiktok')] }),
  post('unslotted', 'draft', null as any),
]

const route = (over: Partial<SocialsRoute> = {}): SocialsRoute => ({ view: 'calendar', post: null, plan: null, cal: 'month', ...over })

function renderCalendar(r: SocialsRoute = route()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <SocialsCalendar role="owner" posts={POSTS} route={r} go={state.go} />
    </QueryClientProvider>,
  )
}

const grid = () => screen.getByRole('region', { name: /October 2026|October —/ })
const chip = (id: string) => within(grid()).getByRole('button', { name: new RegExp(`Post ${id}\\b`) })

function dataTransfer() {
  const data: Record<string, string> = {}
  return {
    data,
    effectAllowed: 'none',
    setData(type: string, value: string) { data[type] = value },
    getData(type: string) { return data[type] ?? '' },
    get types() { return Object.keys(data) },
  }
}

function monthCell(container: HTMLElement, day: number): HTMLElement {
  const cells = Array.from(container.querySelectorAll<HTMLElement>('.cc-cal-month-cell:not(.outside)'))
  const cell = cells.find((c) => c.querySelector('.n')?.textContent?.trim() === String(day))
  if (!cell) throw new Error(`no cell for ${day}`)
  return cell
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(new Date('2026-10-14T07:20:00Z'))
  state.go.mockReset()
  vi.mocked(apiClient.setSocialPostSlot).mockClear()
  vi.mocked(apiClient.scheduleSocialPost).mockClear()
})
afterEach(() => { cleanup(); vi.useRealTimers() })

describe('the Socials calendar', () => {
  it('Month: each chip says time · format, the title, the badges and the status word', () => {
    renderCalendar()
    expect(screen.getByRole('heading', { name: 'October 2026' })).toBeInTheDocument()
    const words: Record<string, string> = {
      planned: 'Planned', making: 'Making', review: 'Needs you', changes: 'Needs you', scheduled: 'Scheduled',
      posted: 'Posted', skipped: 'Skipped', failed: 'Failed',
    }
    for (const [id, word] of Object.entries(words)) expect(chip(id)).toHaveTextContent(word)
    expect(chip('review')).toHaveTextContent('10:00 · Image') // 09:00 UTC in London's summer time
    expect(chip('review')).toHaveTextContent('in')
    expect(chip('posted')).toHaveTextContent('12:01 · Image') // at its first publish time
    expect(chip('posted')).toHaveTextContent(/X\s*in/)
    expect(chip('video')).toHaveTextContent('17:00 · Video 0:30')
    expect(chip('video')).toHaveTextContent('TT')
    expect(within(grid()).queryByRole('button', { name: /Post unslotted/ })).toBeNull()
  })

  it('Week and List show the same posts', () => {
    const { unmount } = renderCalendar(route({ cal: 'week' }))
    const weekTitles = within(grid()).getAllByRole('button').map((b) => b.textContent ?? '').filter((t) => t.includes('Post '))
    expect(weekTitles).toHaveLength(9) // every post with a slot falls in Sun 11 – Sat 17 Oct
    unmount()
    renderCalendar(route({ cal: 'list' }))
    expect(screen.getByTestId('post-list')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'All posts' })).toBeInTheDocument()
  })

  it('the channel filter hides the other channels, and Video shows only videos', async () => {
    renderCalendar()
    const filters = screen.getByRole('group', { name: 'Show channels' })
    fireEvent.click(await within(filters).findByRole('button', { name: 'LinkedIn' }))
    const shown = within(grid()).getAllByRole('button').map((b) => b.textContent ?? '').filter((t) => t.includes('Post '))
    expect(shown.map((t) => t.match(/Post (\w+)/)?.[1]).sort()).toEqual(['posted', 'review'])
    fireEvent.click(within(filters).getByRole('button', { name: 'Video' }))
    expect(within(grid()).getAllByRole('button').filter((b) => b.textContent?.includes('Post '))).toHaveLength(1)
    expect(chip('video')).toBeInTheDocument()
  })

  it('dragging a post waiting for approval keeps its time on the new day, through PUT /slot', async () => {
    const { container } = renderCalendar()
    const dt = dataTransfer()
    fireEvent.dragStart(chip('review'), { dataTransfer: dt })
    expect(JSON.parse(dt.data[SOCIAL_DRAG_TYPE])).toMatchObject({ postId: 'review', timezone: 'Europe/London' })
    fireEvent.dragOver(monthCell(container, 16), { dataTransfer: dt })
    fireEvent.drop(monthCell(container, 16), { dataTransfer: dt })
    await waitFor(() => expect(apiClient.setSocialPostSlot).toHaveBeenCalledWith('review', '2026-10-16T09:00:00.000Z', 'Europe/London'))
    expect(apiClient.scheduleSocialPost).not.toHaveBeenCalled()
  })

  it('a scheduled post moves through PUT /slot too; a published one does not drag', async () => {
    const { container } = renderCalendar()
    const dt = dataTransfer()
    fireEvent.dragStart(chip('scheduled'), { dataTransfer: dt })
    fireEvent.drop(monthCell(container, 15), { dataTransfer: dt })
    await waitFor(() => expect(apiClient.setSocialPostSlot).toHaveBeenCalledWith('scheduled', '2026-10-15T12:00:00.000Z', 'UTC'))
    expect(apiClient.scheduleSocialPost).not.toHaveBeenCalled()
    expect(chip('posted')).not.toHaveAttribute('draggable')
    expect(chip('making')).not.toHaveAttribute('draggable')
  })

  it('the Today rail lists today, reviews into the Queue and carries the status key', () => {
    renderCalendar()
    const today = screen.getByRole('region', { name: 'Today' })
    expect(within(today).getByRole('heading', { name: 'Today · Wed 14 Oct' })).toBeInTheDocument()
    const titles = within(today).getAllByRole('button').map((b) => b.textContent ?? '').filter((t) => t.includes('Post '))
    expect(titles.map((t) => t.match(/Post (\w+)/)?.[1])).toEqual(['review', 'scheduled'])
    fireEvent.click(within(today).getByRole('button', { name: 'Review 1 post' }))
    expect(state.go).toHaveBeenCalledWith({ view: 'queue', post: null })

    const key = screen.getByRole('region', { name: 'Status key' })
    for (const [word, meaning] of [
      ['Planned', 'Slot and topic, not made'], ['Making', 'Auto is making it'], ['Needs you', 'Waiting for approval'],
      ['Scheduled', 'Approved, will publish'], ['Posted', 'Published, with receipts'], ['Skipped', 'Not approved in time'],
    ]) {
      expect(within(key).getByText(word)).toBeInTheDocument()
      expect(within(key).getByText(meaning)).toBeInTheDocument()
    }
    expect(screen.getByText('Drag a post to another day to move it. Click a post to open it.')).toBeInTheDocument()
  })

  it('a click opens the post; a post waiting for approval opens in the Queue', () => {
    renderCalendar()
    fireEvent.click(chip('scheduled'))
    expect(state.go).toHaveBeenLastCalledWith({ post: 'scheduled' })
    fireEvent.click(chip('review'))
    expect(state.go).toHaveBeenLastCalledWith({ view: 'queue', post: 'review' })
  })

  it('reuses the Command Center grid, which takes no social drag where none is given', async () => {
    const { MonthGrid } = await import('@/components/command-center/calendar-month-grid')
    const { buildMonthGrid } = await import('@/components/command-center/calendar-model')
    const deps = { navigate: vi.fn(), pauseRoutine: vi.fn(), setScheduledTaskStatus: vi.fn(), rescheduleSocialPost: vi.fn() }
    const { container } = render(<MonthGrid cells={buildMonthGrid(new Date())} events={[]} anchorMonth={9} actionDeps={deps} />)
    expect(container.querySelectorAll('.cc-cal-month-cell')).toHaveLength(42)
    expect(container.querySelector('[draggable]')).toBeNull()
  })

  it('the grid scrolls sideways on a narrow screen and the rail stacks below it', () => {
    renderCalendar()
    const section = grid()
    expect(section.className).toContain('overflow-x-auto')
    expect(section.firstElementChild?.className).toContain('min-w-[840px]')
    expect(section.parentElement?.className).toContain('lg:grid-cols-[minmax(0,1fr)_340px]')
  })
})
