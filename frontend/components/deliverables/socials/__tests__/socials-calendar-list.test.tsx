/**
 * PRD-251D US-D202 — the calendar's List is one plain list. It replaces the older post list with
 * its Board, its List | Board toggle and its detail pane; what socials-board.test.tsx proved of
 * them that still holds is proved here.
 *
 * * Every post, by status in the order that puts what needs a person first, each group with its
 *   count and its posts newest first: an archived post and a post with no slot included, which
 *   the Month and Week grids leave out. It reads the one posts query the Studio reads.
 * * A row opens its post as a calendar chip does: the post page (the editor, or the read-only
 *   view for an archived post and for a viewer), in place of the list, with its way back; a post
 *   waiting for approval opens in the Queue.
 * * Nothing in it drags or moves a post.
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
  // PRD-251B US-B107: the Studio holds its view in the URL (router.push), mirrored in state.
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams('tab=socials&cal=list'),
}))
// jsdom has no layout and vitest.setup reports a desktop viewport, so a phone is the mobile hooks mocked.
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
    // The Queue, where a post waiting for approval opens, reads the plans.
    listSocialCampaigns: vi.fn(async () => ({ campaigns: [] })),
  }
  return { apiClient, default: apiClient }
})

import { apiClient } from '@/lib/api-client'
import { SocialsTab } from '@/components/deliverables/socials/socials-tab'
import { SOCIAL_STATUS_LABELS, SOCIAL_STATUS_ORDER } from '@/components/deliverables/socials/socials-status'
import { LOAD_FAILED, NO_POSTS, SocialsCalendarList, START_ONE, TEAM_DRAFTS } from '@/components/deliverables/socials/studio/socials-calendar-list'
import { READ_ONLY_ROLE } from '@/components/deliverables/socials/studio/socials-post-view'

const STUDIO_DIR = path.resolve(__dirname, '..', 'studio')

function seed(title: string, status: string, minute: number, extra: Record<string, unknown> = {}) {
  const at = new Date(Date.UTC(2026, 8, 22, 9, minute)).toISOString()
  return {
    id: `post-${title.toLowerCase().replace(/\W+/g, '-')}`, workspace_id: 'w1', created_by: 'user-1', title,
    brief: null, copy: { base: `${title} copy` }, format: null, template_id: null, variables: {}, sources: {},
    media: {}, status, content_hash: `hash-${minute}`, approved_hash: null, approved_by: null, approved_at: null,
    override_unsourced: false, review_log: [], scheduled_for: null, timezone: null, planned_for: null,
    created_at: at, updated_at: at, ...extra,
  }
}

const POSTS = [
  seed('Harvest teaser', 'needs_approval', 1),
  seed('Harvest recap', 'needs_approval', 2),
  seed('Spring menu', 'draft', 3, { planned_for: '2026-10-16T09:00:00Z', timezone: 'UTC' }),
  seed('Chef interview', 'changes_requested', 4),
  seed('Launch day', 'approved', 5),
  seed('Old promo', 'archived', 6),
  seed('Pantry notes', 'draft', 7),
]

function renderTab(postId: string | null = null) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <SocialsTab postId={postId} />
    </QueryClientProvider>,
  )
}

const listSections = () => screen.queryAllByRole('region', { name: / posts$/ })
/** The titles a status group lists, in order (each row is read aloud title first). */
const titlesIn = (group: HTMLElement) => within(group).getAllByRole('button').map((row) => row.getAttribute('aria-label')?.split(', ')[0])
const current = () => within(screen.getByRole('navigation', { name: 'Socials' })).getAllByRole('button')
  .find((item) => item.getAttribute('aria-current') === 'page')

beforeEach(() => {
  state.compact = false
  state.posts = POSTS
  state.workspace = { id: 'w1', role: 'editor', socials: { available: true, enabled: true } }
  vi.mocked(apiClient.listSocialPosts).mockClear()
})
afterEach(cleanup)

describe('the List: every post by status', () => {
  it('lists an archived post and a post with no slot, each under its status', async () => {
    renderTab()
    const archived = await screen.findByRole('region', { name: 'Archived posts' })
    expect(within(archived).getByTestId('socials-status-count')).toHaveTextContent('1')
    expect(within(archived).getByRole('button', { name: /^Old promo, No slot,/ })).toBeInTheDocument()
    const drafts = screen.getByRole('region', { name: 'Draft posts' })
    expect(within(drafts).getByRole('button', { name: /^Pantry notes, No slot,/ })).toBeInTheDocument()
    // A post with a slot says when, in its own timezone.
    expect(within(drafts).getByRole('button', { name: /^Spring menu, Fri 16 Oct 2026, 09:00/ })).toBeInTheDocument()
  })

  it('groups in the order that puts what needs a person first, with counts, newest first, and lists every post', async () => {
    renderTab()
    await screen.findByRole('region', { name: 'Needs approval posts' })
    const groups = listSections()
    const shown = SOCIAL_STATUS_ORDER.map((status) => SOCIAL_STATUS_LABELS[status])
      .filter((label) => POSTS.some((post) => SOCIAL_STATUS_LABELS[post.status as keyof typeof SOCIAL_STATUS_LABELS] === label))
    expect(groups.map((group) => group.getAttribute('aria-label'))).toEqual(shown.map((label) => `${label} posts`))
    expect(titlesIn(screen.getByRole('region', { name: 'Needs approval posts' }))).toEqual(['Harvest recap', 'Harvest teaser'])
    expect(titlesIn(screen.getByRole('region', { name: 'Draft posts' }))).toEqual(['Pantry notes', 'Spring menu'])
    expect(screen.getByRole('heading', { name: 'Draft 2' })).toBeInTheDocument()
    const counted = groups.reduce((sum, group) => sum + Number(within(group).getByTestId('socials-status-count').textContent), 0)
    expect(counted).toBe(POSTS.length)
    // The List reads the query the Studio reads: one fetch for both.
    expect(apiClient.listSocialPosts).toHaveBeenCalledTimes(1)
  })

  it('with no posts says so, and what to do for a role that drafts', async () => {
    state.posts = []
    renderTab()
    expect(await screen.findByText(NO_POSTS)).toBeInTheDocument()
    expect(screen.getByText(START_ONE)).toBeInTheDocument()
    cleanup()

    state.workspace = { ...state.workspace, role: 'viewer' }
    renderTab()
    expect(await screen.findByText(NO_POSTS)).toBeInTheDocument()
    expect(screen.getByText(TEAM_DRAFTS)).toBeInTheDocument()
  })

  it('drags nothing: no drag handles, no drop targets, no drag library', async () => {
    renderTab()
    await screen.findByRole('region', { name: 'Draft posts' })
    expect(
      document.querySelectorAll('[draggable="true"], [data-rfd-drag-handle-draggable-id], [data-rfd-draggable-id], [data-rfd-droppable-id]'),
    ).toHaveLength(0)
    const src = readFileSync(path.join(STUDIO_DIR, 'socials-calendar-list.tsx'), 'utf8')
    expect(src).not.toMatch(/@hello-pangea\/dnd|react-beautiful-dnd|@dnd-kit|draggable|onDrag|onDrop/)
  })
})

describe('a row opens its post as a calendar chip does', () => {
  it('a draft opens in the post page, in place of the list, and Back to calendar returns to it, on a wide screen and a phone', async () => {
    for (const compact of [false, true]) {
      state.compact = compact
      renderTab()
      fireEvent.click(await screen.findByRole('button', { name: /^Spring menu,/ }))
      expect(await screen.findByRole('textbox', { name: 'Title' })).toHaveValue('Spring menu')
      expect(listSections()).toHaveLength(0)

      fireEvent.click(screen.getByRole('button', { name: 'Back to calendar' }))
      expect(await screen.findByRole('region', { name: 'Draft posts' })).toBeInTheDocument()
      expect(screen.queryByRole('textbox', { name: 'Title' })).toBeNull()
      cleanup()
    }
  })

  it('an archived post opens in its read-only view, with nothing to press but the way back', async () => {
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /^Old promo,/ }))
    const view = screen.getByRole('article', { name: 'Post: Old promo' })
    expect(within(view).getByTestId('socials-post-status')).toHaveTextContent('Archived')
    expect(within(view).getByText('Old promo copy')).toBeInTheDocument()
    expect(within(view).queryAllByRole('button')).toHaveLength(0)
    expect(listSections()).toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: 'Back to calendar' }))
    expect(await screen.findByRole('region', { name: 'Archived posts' })).toBeInTheDocument()
  })

  it('a post waiting for approval opens in the Queue, as from the calendar', async () => {
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /^Harvest teaser,/ }))
    const pane = await screen.findByRole('region', { name: 'Post to approve' })
    expect(within(pane).getByRole('heading', { name: 'Harvest teaser' })).toBeInTheDocument()
    expect(current()).toHaveTextContent(/^Queue/)
  })

  it('a viewer gets a draft in the read-only view, with nothing they may not do', async () => {
    state.workspace = { ...state.workspace, role: 'viewer' }
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /^Spring menu,/ }))
    const view = screen.getByRole('article', { name: 'Post: Spring menu' })
    expect(within(view).getByText(READ_ONLY_ROLE)).toBeInTheDocument()
    expect(within(view).queryAllByRole('button')).toHaveLength(0)
    expect(screen.queryByRole('textbox')).toBeNull()
    for (const name of ['Save draft', 'Render preview', 'Submit for approval', /New post/]) {
      expect(screen.queryByRole('button', { name })).toBeNull()
    }
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
    state.posts = [...POSTS, seed('Out already', 'published', 8)]
    renderTab('post-out-already')
    const view = await screen.findByRole('article', { name: 'Post: Out already' })
    expect(within(view).getByTestId('socials-post-status')).toHaveTextContent('Published')
  })
})

describe('a read of the posts that failed', () => {
  it('keeps the posts the list already has; with none, it says the load failed', async () => {
    vi.mocked(apiClient.listSocialPosts).mockRejectedValueOnce(new Error('offline'))
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const list = (posts: any[]) => (
      <QueryClientProvider client={client}>
        <SocialsCalendarList role="editor" posts={posts} onOpen={vi.fn()} />
      </QueryClientProvider>
    )
    const view = render(list(POSTS))
    expect(await screen.findByRole('region', { name: 'Draft posts' })).toBeInTheDocument()
    expect(screen.queryByRole('alert')).toBeNull()
    view.rerender(list([]))
    expect(await screen.findByRole('alert')).toHaveTextContent(`${LOAD_FAILED} offline`)
  })
})
