/**
 * PRD-251B US-B107 — the Socials Studio shell: the sub-navigation Calendar · Queue (its
 * count) · Plans · Brand kit with New plan and New post, and the view held in the URL
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
vi.mock('@/components/documents/blocks/BrandKitDialog', () => ({
  BrandKitDialog: ({ open }: { open: boolean }) => (open ? <div role="dialog" aria-label="Brand Kit" /> : null),
}))
vi.mock('@/components/deliverables/socials/studio/socials-calendar', () => ({
  SocialsCalendar: ({ route }: { route: { post: string | null } }) => (
    <div data-testid="view-calendar">{route.post ?? 'no post'}</div>
  ),
}))
vi.mock('@/components/deliverables/socials/socials-campaigns-view', () => ({
  SocialsCampaigns: ({ creating }: { creating: boolean }) => (
    <div data-testid="view-plans">{creating ? 'plan form open' : 'plans'}</div>
  ),
}))
vi.mock('@/components/deliverables/socials/socials-composer', () => ({
  SocialsComposer: () => <div role="region" aria-label="Composer" />,
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
  it('has Calendar, Queue with its count, Plans and Brand kit, and New plan and New post', async () => {
    renderStudio()
    const items = within(nav()).getAllByRole('button').map((b) => b.textContent)
    expect(items).toEqual(['Calendar', 'Queue', 'Plans', 'Brand kit'])
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
    expect(await screen.findByRole('heading', { name: '2 posts need you' })).toBeInTheDocument()
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

  it('?post=<id> opens that post, from the URL or the page', () => {
    state.search = 'tab=socials&post=p333'
    renderStudio()
    expect(screen.getByTestId('view-calendar')).toHaveTextContent('p333')
    cleanup()

    state.search = ''
    renderStudio('p22')
    expect(screen.getByTestId('view-calendar')).toHaveTextContent('p22')
  })

  it('Brand kit opens the brand kit dialog in place', () => {
    renderStudio()
    expect(screen.queryByRole('dialog')).toBeNull()
    fireEvent.click(within(nav()).getByRole('button', { name: 'Brand kit' }))
    expect(screen.getByRole('dialog', { name: 'Brand Kit' })).toBeInTheDocument()
    expect(state.push).not.toHaveBeenCalled()
  })

  it('New plan opens the plan form on Plans; New post opens the composer', () => {
    renderStudio()
    fireEvent.click(screen.getByRole('button', { name: 'New plan' }))
    expect(state.push).toHaveBeenLastCalledWith('/deliverables?tab=socials&view=plans&plan=new')
    expect(screen.getByTestId('view-plans')).toHaveTextContent('plan form open')

    fireEvent.click(screen.getByRole('button', { name: /New post/ }))
    expect(state.push).toHaveBeenLastCalledWith('/deliverables?tab=socials&view=plans&post=new')
    expect(screen.getByRole('region', { name: 'Composer' })).toBeInTheDocument()
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
