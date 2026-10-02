/**
 * PRD-251 S2.1 (US-205) — global search and Socials (D1).
 *
 * * The Socials page is listed only while the platform offers Socials
 *   (`socials.available`), and typing its name finds it.
 * * A post is found by its title: while Socials is on for the workspace, the search
 *   asks GET /api/socials/posts with `q` (apiClient.listSocialPosts), at most five
 *   posts, each linking to /deliverables?tab=socials&post=<id>. While Socials is off,
 *   unavailable or there is no workspace, posts are never asked for (the route is 404).
 * * A failed posts search is surfaced, and the other sources still list.
 * * The dialog shows the Socials group; a post found by its brief stays listed (cmdk
 *   filters by an item's value too); choosing a post opens it in the Socials tab.
 */
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from 'vitest'
import { renderHook, act, render, screen, fireEvent, cleanup } from '@testing-library/react'

const nav = vi.hoisted(() => ({ push: vi.fn(), socials: null as null | { available: boolean; enabled: boolean } }))

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: nav.push, replace: vi.fn() }) }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => (nav.socials ? { workspace: { id: 'w1', role: 'owner', socials: nav.socials } } : null),
}))

import { apiClient, type SocialPost } from '@/lib/api-client'
import { useGlobalSearch } from '@/hooks/use-global-search'
import { GlobalSearch } from '@/components/shared/global-search'

const HARVEST = { id: 'p-1', title: 'Harvest Club launch', brief: 'Three weeks to Lisbon', status: 'needs_approval' }

function posts(list: Array<Record<string, unknown>>) {
  return { posts: list as unknown as SocialPost[], total: list.length }
}

/** Tasks, agents and memories answer nothing; the posts list answers `list`. */
function stubSources(list: Array<Record<string, unknown>> = [HARVEST]) {
  vi.spyOn(apiClient, 'request').mockResolvedValue([] as never)
  return vi.spyOn(apiClient, 'listSocialPosts').mockResolvedValue(posts(list))
}

type Search = { current: ReturnType<typeof useGlobalSearch> }

async function search(result: Search, text: string) {
  act(() => result.current.setQuery(text))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300)
  })
}

const labels = (result: Search) => result.current.pages.map((page) => page.label)

describe('global search · Socials (hook)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    stubSources()
  })
  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('lists the Socials page while Socials is available, and only then', () => {
    const available = renderHook(() => useGlobalSearch({ available: true, enabled: false })).result
    expect(labels(available)).toContain('Socials')
    expect(available.current.pages.find((page) => page.label === 'Socials')).toMatchObject({
      id: 'nav-socials', category: 'pages', path: '/deliverables?tab=socials',
    })
    expect(labels(renderHook(() => useGlobalSearch({ available: false, enabled: true })).result)).not.toContain('Socials')
    expect(labels(renderHook(() => useGlobalSearch()).result)).not.toContain('Socials')
    expect(labels(renderHook(() => useGlobalSearch(null)).result)).not.toContain('Socials')
  })

  it('finds the Socials page by its name', () => {
    const { result } = renderHook(() => useGlobalSearch({ available: true, enabled: true }))
    act(() => result.current.setQuery('soci'))
    expect(labels(result)).toEqual(['Socials'])
  })

  it('finds a post by its title through q while Socials is on, linking to it in the tab', async () => {
    const list = stubSources()
    const { result } = renderHook(() => useGlobalSearch({ available: true, enabled: true }))
    await search(result, 'harvest club')

    expect(list).toHaveBeenCalledWith({ q: 'harvest club' })
    expect(result.current.socials).toEqual([
      expect.objectContaining({
        label: 'Harvest Club launch',
        description: 'Needs approval',
        category: 'socials',
        path: '/deliverables?tab=socials&post=p-1',
        keywords: 'Three weeks to Lisbon',
      }),
    ])
    expect(result.current.error).toBeNull()
  })

  it.each([
    ['available but off for this workspace', { available: true, enabled: false }],
    ['unavailable', { available: false, enabled: true }],
    ['outside a workspace', undefined],
  ])('never asks for posts while Socials is %s', async (_name, socials) => {
    const list = stubSources()
    const { result } = renderHook(() => useGlobalSearch(socials))
    await search(result, 'harvest')
    expect(list).not.toHaveBeenCalled()
    expect(result.current.socials).toEqual([])
    expect(result.current.error).toBeNull()
  })

  it('lists at most five posts', async () => {
    const many = Array.from({ length: 7 }, (_, i) => ({ ...HARVEST, id: `p-${i}`, title: `Harvest ${i}` }))
    stubSources(many)
    const { result } = renderHook(() => useGlobalSearch({ available: true, enabled: true }))
    await search(result, 'harvest')
    expect(result.current.socials.map((post) => post.label)).toEqual(['Harvest 0', 'Harvest 1', 'Harvest 2', 'Harvest 3', 'Harvest 4'])
  })

  it('surfaces a failed posts search, and the other sources still list', async () => {
    vi.spyOn(apiClient, 'request').mockImplementation(((url: string) =>
      Promise.resolve(url.startsWith('/api/agents') ? [{ id: 7, name: 'Harvest Agent' }] : [])) as never)
    vi.spyOn(apiClient, 'listSocialPosts').mockRejectedValue(new Error('boom'))
    const { result } = renderHook(() => useGlobalSearch({ available: true, enabled: true }))
    await search(result, 'harvest')
    expect(result.current.agents.map((agent) => agent.label)).toEqual(['Harvest Agent'])
    expect(result.current.socials).toEqual([])
    expect(result.current.error).toBeTruthy()
  })
})

describe('global search · the dialog', () => {
  beforeAll(() => {
    // jsdom has no layout: cmdk's list measures itself, and scrolls its active item into view.
    vi.stubGlobal('ResizeObserver', class {
      observe() {}
      unobserve() {}
      disconnect() {}
    })
    if (!Element.prototype.scrollIntoView) Element.prototype.scrollIntoView = () => {}
  })
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    nav.push.mockClear()
    nav.socials = null
  })

  it('shows a post found by its brief under Socials; choosing it opens the post in the tab', async () => {
    nav.socials = { available: true, enabled: true }
    stubSources([{ id: 'p-9', title: 'Countdown', brief: 'Three weeks to Lisbon', status: 'draft' }])
    render(<GlobalSearch />)
    act(() => {
      window.dispatchEvent(new Event('automatos:global-search-open'))
    })
    fireEvent.change(await screen.findByPlaceholderText(/Search pages/), { target: { value: 'lisbon' } })

    const post = await screen.findByText('Countdown', undefined, { timeout: 3000 })
    expect(screen.getByText('Socials')).toBeInTheDocument()
    expect(screen.getByText('Draft')).toBeInTheDocument()
    fireEvent.click(post)
    expect(nav.push).toHaveBeenCalledWith('/deliverables?tab=socials&post=p-9')
  })

  it('lists the Socials page only where Socials is available', async () => {
    nav.socials = { available: true, enabled: false }
    stubSources()
    render(<GlobalSearch />)
    act(() => {
      window.dispatchEvent(new Event('automatos:global-search-open'))
    })
    fireEvent.change(await screen.findByPlaceholderText(/Search pages/), { target: { value: 'soc' } })
    fireEvent.click(await screen.findByText('Socials'))
    expect(nav.push).toHaveBeenCalledWith('/deliverables?tab=socials')
    expect(apiClient.listSocialPosts).not.toHaveBeenCalled()
  })
})
