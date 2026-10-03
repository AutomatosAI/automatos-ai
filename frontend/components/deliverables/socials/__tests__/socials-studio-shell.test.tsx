/**
 * PRD-251B US-B107 — the Socials Studio shell: the sub-navigation Calendar · Queue (its
 * count) · Plans with New plan and New post (F250: the brand kit is its own Deliverables
 * tab, with no second entry here), and the view held in the URL
 * (/deliverables?tab=socials&view=…&post=…&plan=…). A view change is a router.push, so
 * back returns to it; an unknown view is the calendar; ?post=<id> opens that post.
 *
 * The views themselves have their own tests: here they are markers that show what the
 * shell handed them.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const state = vi.hoisted(() => ({
  search: '',
  push: vi.fn(),
  replace: vi.fn(),
  posts: [] as any[],
  role: 'owner',
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: state.push, replace: state.replace }),
  useSearchParams: () => new URLSearchParams(state.search),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: state.role, socials: { available: true, enabled: true } } }),
}))
vi.mock('@/lib/api-client', () => {
  const apiClient = { listSocialPosts: vi.fn(async () => ({ posts: state.posts, total: state.posts.length })) }
  return { apiClient, default: apiClient }
})
vi.mock('@/components/deliverables/socials/studio/socials-calendar', () => ({
  SocialsCalendar: ({ route }: { route: { post: string | null } }) => (
    <div data-testid="view-calendar">{route.post ?? 'no post'}</div>
  ),
}))
vi.mock('@/components/deliverables/socials/plans/socials-plans-view', () => ({
  // PRD-251B Wave 2: the Plans view (the plans, a plan's page at ?plan=<id>, the Plan form at ?plan=new).
  SocialsPlansView: ({ planId }: { planId: string | null }) => (
    <div data-testid="view-plans">{planId === 'new' ? 'plan form open' : 'plans'}</div>
  ),
}))
vi.mock('@/components/deliverables/socials/studio/socials-post-page', () => ({
  // The editor's own move once a save has made the post (socials-editor.tsx: go({ post })).
  SocialsPostPage: ({ postId, go }: { postId: string; go: (next: { post: string }) => void }) => (
    <div data-testid="post-page">{postId}<button type="button" onClick={() => go({ post: 'p-saved' })}>Saved</button></div>
  ),
}))
vi.mock('@/components/deliverables/socials/studio/queue-pane', () => ({
  QueuePane: ({ post }: { post: any }) => <article aria-label={`Post to approve: ${post.title}`} />,
}))
vi.mock('@/components/deliverables/socials/socials-post-detail', () => ({
  SocialsPostDetail: ({ post }: { post: any }) => <article aria-label={`Post: ${post.title}`} />,
}))

import { SocialsStudio } from '@/components/deliverables/socials/studio/studio-shell'
import { parseSocialsRoute, socialsHref } from '@/components/deliverables/socials/studio/studio-route'

function post(id: string, status: string, title = id) {
  return { id, title, status, format: 'image', planned_for: null, scheduled_for: null, created_at: `2026-10-0${id.length}T09:00:00Z` }
}

function renderStudio(postId: string | null = null) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <SocialsStudio role={state.role as any} postId={postId} />
    </QueryClientProvider>,
  )
}

const nav = () => screen.getByRole('navigation', { name: 'Socials' })
const current = () => within(nav()).getAllByRole('button').find((b) => b.getAttribute('aria-current') === 'page')

beforeEach(() => {
  state.search = ''
  state.role = 'owner'
  state.posts = [post('p1', 'needs_approval', 'Teaser'), post('p22', 'needs_approval', 'Recap'), post('p333', 'draft')]
  state.push.mockReset()
  state.replace.mockReset()
})
afterEach(cleanup)

describe('the Socials Studio shell', () => {
  it('has Calendar, Queue with its count and Plans, and New plan and New post', async () => {
    renderStudio()
    const items = within(nav()).getAllByRole('button').map((b) => b.textContent)
    expect(items).toEqual(['Calendar', 'Queue', 'Plans'])
    expect(await within(nav()).findByText('2 posts need you')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'New plan' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /New post/ })).toBeInTheDocument()
    expect(current()).toHaveTextContent('Calendar')
    expect(screen.getByTestId('view-calendar')).toBeInTheDocument()
  })

  it('opens the view the URL names, and an unknown view is the calendar', async () => {
    state.search = 'tab=socials&view=queue'
    renderStudio()
    expect(current()).toHaveTextContent(/^Queue/)
    // US-B111: no slot is today, so nothing needs you today; both wait under No slot.
    expect(await screen.findByRole('heading', { name: 'All caught up for today' })).toBeInTheDocument()
    expect(within(screen.getByRole('complementary', { name: 'Waiting for approval' })).getAllByRole('button')).toHaveLength(2)
    cleanup()

    state.search = 'tab=socials&view=somewhere'
    renderStudio()
    expect(current()).toHaveTextContent('Calendar')
    expect(screen.getByTestId('view-calendar')).toBeInTheDocument()
  })

  it('a view change is a push (back returns to it), and it opens that view', () => {
    renderStudio()
    fireEvent.click(within(nav()).getByRole('button', { name: 'Plans' }))
    expect(state.push).toHaveBeenCalledWith('/deliverables?tab=socials&view=plans')
    expect(state.replace).not.toHaveBeenCalled()
    expect(screen.getByTestId('view-plans')).toHaveTextContent('plans')
    expect(current()).toHaveTextContent('Plans')
  })

  it('?post=<id> opens that post, from the URL or the page (US-B109: the post page)', () => {
    state.search = 'tab=socials&post=p333'
    renderStudio()
    expect(screen.getByTestId('post-page')).toHaveTextContent('p333')
    cleanup()

    state.search = ''
    renderStudio('p22')
    expect(screen.getByTestId('post-page')).toHaveTextContent('p22')
  })

  it('the old ?view=brand link goes to the Brand kit tab (PRD-251B US-B301)', () => {
    state.search = 'tab=socials&view=brand'
    renderStudio()
    expect(state.replace).toHaveBeenCalledWith('/deliverables?tab=brand')
  })

  it('F250: the brand kit has one entry, the Deliverables tab: none in the Socials nav, for any role', () => {
    renderStudio()
    expect(within(nav()).queryByRole('button', { name: 'Brand kit' })).toBeNull()
    expect(screen.queryByText('Brand kit')).toBeNull()
  })

  it('New plan opens the plan form on Plans; New post opens the editor', () => {
    renderStudio()
    fireEvent.click(screen.getByRole('button', { name: 'New plan' }))
    expect(state.push).toHaveBeenLastCalledWith('/deliverables?tab=socials&view=plans&plan=new')
    expect(screen.getByTestId('view-plans')).toHaveTextContent('plan form open')

    fireEvent.click(screen.getByRole('button', { name: /New post/ }))
    expect(state.push).toHaveBeenLastCalledWith('/deliverables?tab=socials&view=calendar&post=new')
    expect(screen.getByTestId('post-page')).toHaveTextContent('new')
  })

  it('New post from the Queue opens the editor, and the post a save makes stays open there, not in the Queue', () => {
    state.search = 'tab=socials&view=queue'
    renderStudio()
    fireEvent.click(screen.getByRole('button', { name: /New post/ }))
    expect(state.push).toHaveBeenLastCalledWith('/deliverables?tab=socials&view=calendar&post=new')
    // 3 Oct 2026: the upload saved the post, and the Queue (where a draft never shows) opened.
    fireEvent.click(screen.getByRole('button', { name: 'Saved' }))
    expect(state.push).toHaveBeenLastCalledWith('/deliverables?tab=socials&view=calendar&post=p-saved')
    expect(screen.getByTestId('post-page')).toHaveTextContent('p-saved')
  })

  it('a viewer sees the views but neither action nor the brand kit', () => {
    state.role = 'viewer'
    renderStudio()
    expect(within(nav()).getAllByRole('button').map((b) => b.textContent)).toEqual(['Calendar', 'Queue', 'Plans'])
    expect(screen.queryByRole('button', { name: 'New plan' })).toBeNull()
    expect(screen.queryByRole('button', { name: /New post/ })).toBeNull()
  })
})

describe('the route', () => {
  it('parses what it writes and falls back on anything unknown', () => {
    const route = { view: 'queue', post: 'p1', plan: null, cal: 'week' } as const
    expect(socialsHref(route)).toBe('/deliverables?tab=socials&view=queue&cal=week&post=p1')
    expect(parseSocialsRoute(new URLSearchParams(socialsHref(route).split('?')[1]))).toEqual(route)
    expect(parseSocialsRoute(new URLSearchParams('view=nope&cal=year'))).toEqual({
      view: 'calendar', post: null, plan: null, cal: 'month',
    })
  })
})
