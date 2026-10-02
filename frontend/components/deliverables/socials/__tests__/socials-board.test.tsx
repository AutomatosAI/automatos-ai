/**
 * PRD-251 S2.1 (US-205) — the Socials tab's Board beside its List, and its phone form.
 *
 * * The board and the list read the one posts query: the same posts and the same
 *   count per status (the board keeps every status as a column, an empty one at
 *   0), and switching views fetches nothing.
 * * The board is read-only: no drag handles, no drop targets, no drag library. A
 *   card opens the post, whose detail holds the actions, and back returns to it.
 * * Below 1024 px the list and a post's detail are one view at a time, with a back
 *   control; on a wide screen the two columns hold. jsdom has no layout, and
 *   vitest.setup reports a desktop viewport, so the mobile hooks are mocked.
 * * `?post=` (where global search links) opens that post.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import { render, screen, cleanup, fireEvent, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const state = vi.hoisted(() => ({
  compact: false,
  posts: [] as any[],
  workspace: { id: 'w1', role: 'editor', socials: { available: true, enabled: true } } as any,
}))

vi.mock('next/navigation', () => ({
  usePathname: () => '/deliverables',
  // PRD-251B US-B107: the Studio holds its view in the URL (router.push).
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  // US-B108: the calendar's List is the post list these tests drive.
  useSearchParams: () => new URLSearchParams('tab=socials&cal=list'),
}))
vi.mock('@/hooks/use-mobile', () => ({
  useIsMobile: () => state.compact,
  useIsTabletOrBelow: () => state.compact,
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: state.workspace, isLoading: false, refreshWorkspace: vi.fn() }),
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => {
  const apiClient = {
    listSocialPosts: vi.fn(async () => ({ posts: state.posts, total: state.posts.length })),
    getSocialsUsage: vi.fn(async () => ({ render_minutes: null })),
  }
  return { apiClient, default: apiClient }
})

import { apiClient } from '@/lib/api-client'
import { SocialsTab } from '@/components/deliverables/socials/socials-tab'
import { SOCIAL_STATUS_LABELS, SOCIAL_STATUS_ORDER } from '@/components/deliverables/socials/socials-status'

const SOCIALS_DIR = path.resolve(__dirname, '..')

function seed(title: string, status: string, minute: number) {
  const at = new Date(Date.UTC(2026, 8, 22, 9, minute)).toISOString()
  return {
    id: `post-${title.toLowerCase().replace(/\W+/g, '-')}`, workspace_id: 'w1', created_by: 'user-1', title,
    brief: null, copy: { base: `${title} copy` }, format: null, template_id: null, variables: {}, sources: {},
    media: {}, status, content_hash: `hash-${minute}`, approved_hash: null, approved_by: null, approved_at: null,
    override_unsourced: false, review_log: [], scheduled_for: null, timezone: null, created_at: at, updated_at: at,
  }
}

const POSTS = [
  seed('Harvest teaser', 'needs_approval', 1),
  seed('Harvest recap', 'needs_approval', 2),
  seed('Spring menu', 'draft', 3),
  seed('Chef interview', 'changes_requested', 4),
  seed('Launch day', 'approved', 5),
  seed('Old promo', 'archived', 6),
]

function renderTab(postId: string | null = null) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <SocialsTab postId={postId} />
    </QueryClientProvider>,
  )
}

type Shown = Record<string, { count: number; titles: string[] }>

/** Each status group a view shows: its count badge and its posts' titles, in order. */
function shown(sections: HTMLElement[], suffix: string): Shown {
  return Object.fromEntries(
    sections.map((section) => [
      (section.getAttribute('aria-label') ?? '').replace(suffix, ''),
      {
        count: Number(within(section).getByTestId('socials-status-count').textContent),
        titles: within(section).queryAllByRole('button').map((card) => card.firstElementChild?.textContent ?? ''),
      },
    ]),
  )
}

const listSections = () => screen.queryAllByRole('region', { name: / posts$/ })
const boardColumns = () => screen.queryAllByRole('region', { name: / column$/ })
const board = () => screen.queryByRole('group', { name: 'Posts by status' })

beforeEach(() => {
  state.compact = false
  state.posts = POSTS
  vi.mocked(apiClient.listSocialPosts).mockClear()
})
afterEach(cleanup)

describe('the Board beside the List', () => {
  it('shows the same posts and the same count per status as the list, from the one query', async () => {
    renderTab()
    await screen.findByRole('region', { name: 'Needs approval posts' })
    const list = shown(listSections(), ' posts')
    expect(list['Needs approval']).toEqual({ count: 2, titles: ['Harvest recap', 'Harvest teaser'] })

    fireEvent.click(screen.getByRole('button', { name: 'Board' }))
    expect(screen.getByRole('button', { name: 'Board' })).toHaveAttribute('aria-pressed', 'true')
    expect(listSections()).toHaveLength(0)
    const columns = shown(boardColumns(), ' column')

    // One column per status, in the list's order; the list's groups, post for post.
    expect(boardColumns().map((c) => c.getAttribute('aria-label'))).toEqual(
      SOCIAL_STATUS_ORDER.map((status) => `${SOCIAL_STATUS_LABELS[status]} column`),
    )
    for (const [label, group] of Object.entries(list)) expect(columns[label]).toEqual(group)
    // A status the list leaves out is an empty column.
    for (const [label, column] of Object.entries(columns)) {
      if (!(label in list)) expect(column).toEqual({ count: 0, titles: [] })
    }
    expect(Object.values(columns).reduce((sum, column) => sum + column.count, 0)).toBe(POSTS.length)

    fireEvent.click(screen.getByRole('button', { name: 'List' }))
    expect(shown(listSections(), ' posts')).toEqual(list)
    // Both views read the same query: switching views fetched nothing.
    expect(apiClient.listSocialPosts).toHaveBeenCalledTimes(1)
  })

  it('is read-only: no drag handles, no drop targets, no drag library', async () => {
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: 'Board' }))
    const grid = board()!
    expect(grid).toBeInTheDocument()
    expect(
      grid.querySelectorAll(
        '[draggable="true"], [data-rfd-drag-handle-draggable-id], [data-rfd-draggable-id], [data-rfd-droppable-id]',
      ),
    ).toHaveLength(0)
    for (const file of ['socials-board.tsx', 'socials-post-card.tsx']) {
      const src = readFileSync(path.join(SOCIALS_DIR, file), 'utf8')
      expect(src, file).not.toMatch(/@hello-pangea\/dnd|react-beautiful-dnd|@dnd-kit|draggable|onDrag|onDrop/)
    }
  })

  it("a card opens the post, whose detail holds the actions; back returns to the board", async () => {
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: 'Board' }))
    const column = screen.getByRole('region', { name: 'Needs approval column' })
    fireEvent.click(within(column).getByRole('button', { name: /Harvest teaser/ }))

    const detail = screen.getByRole('article', { name: 'Post: Harvest teaser' })
    expect(within(detail).getByRole('button', { name: 'Approve' })).toBeInTheDocument()
    expect(board()).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Back to board' }))
    expect(board()).toBeInTheDocument()
    expect(screen.queryByRole('article')).toBeNull()
  })
})

describe('the phone form: below 1024 px (the mobile hooks mocked)', () => {
  beforeEach(() => { state.compact = true })

  it('the list stands alone until a post is opened', async () => {
    renderTab()
    await screen.findByRole('region', { name: 'Draft posts' })
    expect(screen.queryByText('Select a post to see its status and actions.')).toBeNull()
    expect(screen.queryByRole('article')).toBeNull()
  })

  it('an opened post takes the list\'s place, full width, with a back control', async () => {
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /Spring menu/ }))
    expect(screen.getByRole('article', { name: 'Post: Spring menu' })).toBeInTheDocument()
    expect(listSections()).toHaveLength(0)

    fireEvent.click(screen.getByRole('button', { name: 'Back to posts' }))
    expect(screen.queryByRole('article')).toBeNull()
    expect(screen.getByRole('region', { name: 'Draft posts' })).toBeInTheDocument()
  })

  it('the board scrolls sideways, one column per status', async () => {
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: 'Board' }))
    const grid = board()!
    expect(grid.className).toContain('socials-board')
    expect(grid.className).toContain('overflow-x-auto')
    expect(boardColumns()).toHaveLength(SOCIAL_STATUS_ORDER.length)
    for (const column of boardColumns()) expect(column.className).toContain('socials-board-col')
  })
})

describe('on a wide screen', () => {
  it('the list keeps the opened post beside it, with no back control', async () => {
    renderTab()
    expect(await screen.findByText('Select a post to see its status and actions.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Spring menu/ }))
    expect(screen.getByRole('article', { name: 'Post: Spring menu' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Draft posts' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^Back to/ })).toBeNull()
  })
})

describe('?post= (global search links a post there)', () => {
  // PRD-251B US-B109: the post opens in the editor, with its way back to the calendar.
  it('opens that post in the editor', async () => {
    renderTab('post-launch-day')
    expect(await screen.findByRole('textbox', { name: 'Title' })).toHaveValue('Launch day')
    expect(screen.getByRole('button', { name: 'Back to calendar' })).toBeInTheDocument()
    expect(listSections()).toHaveLength(0)
  })

  it('on a phone, opens it the same way, in place of the list', async () => {
    state.compact = true
    renderTab('post-launch-day')
    expect(await screen.findByRole('textbox', { name: 'Title' })).toHaveValue('Launch day')
    expect(listSections()).toHaveLength(0)
  })

  it('a published post opens in its read view', async () => {
    state.posts = [...POSTS, seed('Out already', 'published', 7)]
    renderTab('post-out-already')
    expect(await screen.findByRole('article', { name: 'Post: Out already' })).toBeInTheDocument()
  })
})
