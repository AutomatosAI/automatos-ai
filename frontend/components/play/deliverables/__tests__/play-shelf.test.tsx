/**
 * PRD-260 T-007a — Play's Deliverables shelf over a mocked useDeliverables
 * (hooks/use-deliverables-api.ts), ported from Automatos Play's
 * tests/deliverables/{ShelfScreen,ShelfMore}.test.tsx: the cards (title, type,
 * date, the newest in the spotlight), the chips setting artifact_type, Load
 * more asking for the next page, a card's link (?deliverable=<id>), the
 * pictures through the core's authenticated path, the loading, empty and error
 * states (success: false included), Auto's caption, and the page's other
 * sections. One test runs the real hook over a mocked apiClient, to see the
 * next page's offset on the wire. Roles, attributes and text only (CSS
 * modules are proxies under css: false). "Now" is pinned to a local date.
 */
import React from 'react'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Deliverable, DeliverableListResponse } from '@/hooks/use-deliverables-api'

const hook = vi.hoisted(() => ({ real: false, mock: vi.fn() }))
vi.mock('@/hooks/use-deliverables-api', async () => {
  const actual = await vi.importActual<typeof import('@/hooks/use-deliverables-api')>('@/hooks/use-deliverables-api')
  return {
    ...actual,
    useDeliverables: (filters: Parameters<typeof actual.useDeliverables>[0]) =>
      hook.real ? actual.useDeliverables(filters) : hook.mock(filters),
  }
})

const api = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/hooks/use-workspace', () => ({ useWorkspace: () => ({ workspaceId: 'ws-1' }) }))

const workspace = vi.hoisted(() => ({ socials: false }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'ws-1', socials: { available: workspace.socials } }, isLoading: false }),
}))

const pictures = vi.hoisted(() => ({ blobUrl: vi.fn((url: string | undefined) => ({ src: null as string | null, error: null, url })) }))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({ useAuthenticatedBlobUrl: pictures.blobUrl }))

const health = vi.hoisted(() => ({ connection: 'connected' as 'connected' | 'down' | 'checking' }))
vi.mock('@/hooks/use-backend-health', () => ({ useBackendHealth: () => ({ connection: health.connection, check: vi.fn() }) }))

const nav = vi.hoisted(() => ({ push: vi.fn(), replace: vi.fn() }))
vi.mock('next/navigation', () => ({ useRouter: () => nav, usePathname: () => '/deliverables' }))
const store = vi.hoisted(() => ({ newDraft: vi.fn() }))
vi.mock('@/stores/chat-session-store', () => ({
  useChatSessionStore: (select: (state: typeof store) => unknown) => select(store),
}))

vi.mock('@/components/mascot/auto-mascot', () => ({
  AutoMascot: ({ state, caption }: { state: string; caption?: string }) => (
    <div data-testid="auto" data-state={state}>
      {caption}
    </div>
  ),
}))

import { FEED_DEFAULT_FILTERS } from '@/hooks/use-deliverables-api'
import { SEEN_KEY } from '@/lib/play/seen'
import { FILTER_GROUP_LABEL, SECTIONS_LABEL, ShelfScreen } from '../play-shelf'
import { MORE_FAILED, SHELF_FAILED } from '../shelf-body'

const NOW = new Date(2026, 9, 14, 15, 30) // Wed 14 Oct 2026, 15:30

function at(day: number, hour = 9): string {
  return new Date(2026, 9, day, hour).toISOString()
}

function row(id: string, fields: Partial<Deliverable> = {}): Deliverable {
  return {
    id,
    workspace_id: 'ws-1',
    source_type: 'chat',
    source_id: null,
    agent_id: null,
    agent_name: null,
    artifact_type: 'document',
    title: `Deliverable ${id}`,
    summary: null,
    storage_type: 'workspace',
    file_path: `outputs/${id}.md`,
    file_name: null,
    file_type: null,
    file_size_bytes: null,
    preview_url: null,
    preview_type: null,
    thumbnail_url: null,
    extra: {},
    status: 'active',
    created_at: '2026-10-01T09:00:00+00:00',
    updated_at: '2026-10-01T09:00:00+00:00',
    ...fields,
  }
}

const ROWS: readonly Deliverable[] = [
  row('d-laptop', { title: 'Laptop shortlist', artifact_type: 'report', created_at: at(14), thumbnail_url: '/api/deliverables/d-laptop/thumbnail' }),
  row('d-chem', { title: 'Chemistry revision plan', artifact_type: 'document', created_at: at(13) }),
  row('d-poster', { title: 'Bake sale poster', artifact_type: 'image', created_at: at(12), preview_url: '/api/workspaces/files/poster.png' }),
  row('d-slides', { title: 'Climate debate slides', artifact_type: 'slide', created_at: at(4) }),
]

function page(rows: readonly Deliverable[], total = rows.length, offset = 0): DeliverableListResponse {
  return { success: true, deliverables: [...rows], total, limit: 24, offset }
}

const FAILED_PAGE = { success: false, error: 'list failed: relation v_workspace_outputs does not exist', deliverables: [], total: 0, limit: 24, offset: 0 }

interface Result {
  data?: { pages: readonly DeliverableListResponse[] }
  isError?: boolean
  hasNextPage?: boolean
  isFetchingNextPage?: boolean
}

/** One stable result per test, as react-query keeps its data between renders. */
function answer({ data, isError = false, hasNextPage = false, isFetchingNextPage = false }: Result) {
  const result = {
    data,
    isError,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage: vi.fn(async (): Promise<{ isError: boolean; error: unknown }> => ({ isError: false, error: null })),
    refetch: vi.fn(async () => ({})),
  }
  hook.mock.mockReturnValue(result)
  return result
}

function renderShelf() {
  const user = userEvent.setup()
  render(<ShelfScreen now={NOW} />)
  return { user }
}

function cards() {
  return within(screen.getByRole('list', { name: 'Deliverables' })).getAllByRole('link')
}

function lastFilters() {
  return hook.mock.mock.calls[hook.mock.mock.calls.length - 1]?.[0]
}

beforeEach(() => {
  window.localStorage.clear()
  pictures.blobUrl.mockImplementation((url) => ({ src: null, error: null, url }))
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.restoreAllMocks()
  window.localStorage.clear()
  hook.real = false
  workspace.socials = false
  health.connection = 'connected'
})

describe('ShelfScreen — the cards', () => {
  it("shows each Deliverable's title, type and date, the newest in the spotlight, from the feed's filters", () => {
    answer({ data: { pages: [page(ROWS)] } })
    renderShelf()

    expect(screen.getByRole('heading', { level: 1, name: 'Your shelf' })).toBeInTheDocument()
    expect(screen.getByText('4 Deliverables')).toBeInTheDocument()
    const [laptop, chem, poster, slides] = cards()
    expect(laptop).toHaveTextContent('Laptop shortlist')
    expect(laptop).toHaveTextContent('Newest')
    expect(laptop).toHaveTextContent('Report')
    expect(laptop).toHaveTextContent('Today')
    expect(chem).toHaveTextContent('Yesterday')
    expect(chem).not.toHaveTextContent('Newest')
    expect(poster).toHaveTextContent('Image')
    expect(poster).toHaveTextContent('Mon')
    expect(slides).toHaveTextContent('Slides')
    expect(slides).toHaveTextContent('4 Oct')
    expect(hook.mock).toHaveBeenCalledWith({ ...FEED_DEFAULT_FILTERS, artifact_type: null })
    expect(lastFilters()?.source_type_exclude).toEqual(['heartbeat'])
  })

  it('links each card to the core page of that Deliverable (K8)', () => {
    answer({ data: { pages: [page(ROWS)] } })
    renderShelf()
    expect(cards().map((card) => card.getAttribute('href'))).toEqual([
      '/deliverables?deliverable=d-laptop',
      '/deliverables?deliverable=d-chem',
      '/deliverables?deliverable=d-poster',
      '/deliverables?deliverable=d-slides',
    ])
  })

  it("loads each picture through the core's authenticated path: an image's preview, else the thumbnail", () => {
    pictures.blobUrl.mockImplementation((url) => ({ src: url ? `blob:core/${url.split('/').pop()}` : null, error: null, url }))
    answer({ data: { pages: [page(ROWS)] } })
    renderShelf()

    const asked = pictures.blobUrl.mock.calls.map(([url]) => url)
    expect(asked).toContain('/api/deliverables/d-laptop/thumbnail')
    expect(asked).toContain('/api/workspaces/files/poster.png')
    const thumbs = screen.getAllByTestId('shelf-thumb')
    expect(thumbs[0]).toHaveAttribute('data-kind', 'report')
    expect(thumbs[0]?.querySelector('img')).toHaveAttribute('src', 'blob:core/thumbnail')
    expect(thumbs[2]?.querySelector('img')).toHaveAttribute('src', 'blob:core/poster.png')
    expect(thumbs[1]?.querySelector('img')).toBeNull()
  })

  it('shows Auto proud of the count', () => {
    answer({ data: { pages: [page(ROWS)] } })
    renderShelf()
    expect(screen.getByTestId('auto')).toHaveAttribute('data-state', 'idle')
    expect(screen.getByTestId('auto')).toHaveTextContent('4 on the shelf. Nice collection!')
  })

  it('makes Auto celebrate a Deliverable this browser has not seen, and remembers it', () => {
    window.localStorage.setItem(SEEN_KEY, JSON.stringify({ ids: ['d-chem'], newest: at(13) }))
    answer({ data: { pages: [page(ROWS.slice(0, 2))] } })
    renderShelf()
    expect(screen.getByTestId('auto')).toHaveAttribute('data-state', 'celebrate')
    expect(screen.getByTestId('auto')).toHaveTextContent('Your report is ready')
    expect(JSON.parse(window.localStorage.getItem(SEEN_KEY) ?? '{}').ids).toEqual(['d-chem', 'd-laptop'])
  })
})

describe('ShelfScreen — the chips', () => {
  it("sets the list's artifact_type from a chip, and All clears it", async () => {
    answer({ data: { pages: [page(ROWS)] } })
    const { user } = renderShelf()
    const chips = within(screen.getByRole('group', { name: FILTER_GROUP_LABEL }))
    expect(chips.getAllByRole('button').map((chip) => chip.textContent)).toEqual([
      'All',
      'Reports',
      'Documents',
      'Images',
      'Spreadsheets',
      'Slides',
    ])
    expect(chips.getByRole('button', { name: 'All' })).toHaveAttribute('aria-pressed', 'true')

    await user.click(chips.getByRole('button', { name: 'Reports' }))
    expect(lastFilters()).toEqual({ ...FEED_DEFAULT_FILTERS, artifact_type: 'report' })
    expect(chips.getByRole('button', { name: 'Reports' })).toHaveAttribute('aria-pressed', 'true')
    expect(chips.getByRole('button', { name: 'All' })).toHaveAttribute('aria-pressed', 'false')

    await user.click(chips.getByRole('button', { name: 'Slides' }))
    expect(lastFilters()?.artifact_type).toBe('slide')
    await user.click(chips.getByRole('button', { name: 'All' }))
    expect(lastFilters()).toEqual({ ...FEED_DEFAULT_FILTERS, artifact_type: null })
  })

  it('names the type in the empty state of a chip', async () => {
    answer({ data: { pages: [page([])] } })
    const { user } = renderShelf()
    await user.click(screen.getByRole('button', { name: 'Spreadsheets' }))
    expect(screen.getByText('No spreadsheets yet. Ask Auto for one and it lands here.')).toBeInTheDocument()
  })
})

describe('ShelfScreen — Load more', () => {
  it('asks the hook for the next page while there is one', async () => {
    const result = answer({ data: { pages: [page(ROWS, 30)] }, hasNextPage: true })
    const { user } = renderShelf()
    await user.click(screen.getByRole('button', { name: 'Load more' }))
    expect(result.fetchNextPage).toHaveBeenCalledTimes(1)
  })

  it('shows no Load more on the last page, and a disabled one while the next is on its way', () => {
    answer({ data: { pages: [page(ROWS)] } })
    renderShelf()
    expect(screen.queryByRole('button', { name: 'Load more' })).toBeNull()
    cleanup()
    answer({ data: { pages: [page(ROWS, 30)] }, hasNextPage: true, isFetchingNextPage: true })
    renderShelf()
    expect(screen.getByRole('button', { name: 'Loading more…' })).toBeDisabled()
  })

  it('keeps the list and says so when Load more fails', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const result = answer({ data: { pages: [page(ROWS, 30)] }, hasNextPage: true })
    result.fetchNextPage.mockResolvedValue({ isError: true, error: new Error('timeout') })
    const { user } = renderShelf()
    await user.click(screen.getByRole('button', { name: 'Load more' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(MORE_FAILED)
    expect(cards()).toHaveLength(4)
    expect(screen.getByRole('button', { name: 'Load more' })).toBeEnabled()
  })

  it('keeps the list when a later page answers success: false, with Try again', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const result = answer({ data: { pages: [page(ROWS, 30), { ...FAILED_PAGE, offset: 24 }] } })
    const { user } = renderShelf()
    expect(screen.getByRole('alert')).toHaveTextContent(MORE_FAILED)
    expect(cards()).toHaveLength(4)
    expect(screen.getByText('30 Deliverables')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(result.refetch).toHaveBeenCalledTimes(1)
  })

  it('sends the next offset on the wire through the real hook (buildListQuery), with the chip', async () => {
    hook.real = true
    const rows = Array.from({ length: 30 }, (_, index) => row(`d-${index}`, { title: `Deliverable number ${index}`, created_at: at(14) }))
    api.request.mockImplementation(async (path: string) => {
      const params = new URLSearchParams(path.split('?')[1])
      const offset = Number(params.get('offset'))
      return page(rows.slice(offset, offset + 24), 30, offset)
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const user = userEvent.setup()
    render(
      <QueryClientProvider client={client}>
        <ShelfScreen now={NOW} />
      </QueryClientProvider>,
    )
    await screen.findByRole('heading', { name: 'Deliverable number 0' })
    await user.click(screen.getByRole('button', { name: 'Documents' }))
    await screen.findByRole('heading', { name: 'Deliverable number 0' })
    await user.click(screen.getByRole('button', { name: 'Load more' }))
    await screen.findByRole('heading', { name: 'Deliverable number 29' })

    const last = new URLSearchParams(String(api.request.mock.calls[api.request.mock.calls.length - 1]?.[0]).split('?')[1])
    expect(last.get('offset')).toBe('24')
    expect(last.get('limit')).toBe('24')
    expect(last.get('artifact_type')).toBe('document')
    expect(last.get('source_type_exclude')).toBe('heartbeat')
    expect(cards()).toHaveLength(30)
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Load more' })).toBeNull())
  })
})

describe('ShelfScreen — loading, empty and error', () => {
  it('says it is loading before the first page', () => {
    answer({})
    renderShelf()
    expect(screen.getByRole('status')).toHaveTextContent('Loading your shelf')
    expect(screen.getByTestId('auto')).toHaveTextContent('Fetching your shelf')
  })

  it('shows a friendly empty state with a way to start a chat', async () => {
    answer({ data: { pages: [page([])] } })
    const { user } = renderShelf()
    expect(screen.getByText(/Nothing on your shelf yet/)).toBeInTheDocument()
    expect(screen.queryByRole('alert')).toBeNull()
    await user.click(screen.getByRole('button', { name: 'Start a chat' }))
    expect(store.newDraft).toHaveBeenCalledTimes(1)
    expect(nav.push).toHaveBeenCalledWith('/chat')
  })

  it('shows the error state when the list answers success: false, never the empty state or the backend text', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const result = answer({ data: { pages: [FAILED_PAGE] } })
    const { user } = renderShelf()
    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent(SHELF_FAILED)
    expect(alert).not.toHaveTextContent('v_workspace_outputs')
    expect(screen.queryByText(/Nothing on your shelf yet/)).toBeNull()
    expect(console.error).toHaveBeenCalledWith(expect.stringContaining('[shelf]'), expect.stringContaining('v_workspace_outputs'))
    expect(screen.getByTestId('auto')).toHaveAttribute('data-state', 'oops')
    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(result.refetch).toHaveBeenCalledTimes(1)
  })

  it('shows the error state when the request fails', () => {
    answer({ isError: true })
    renderShelf()
    expect(screen.getByRole('alert')).toHaveTextContent(SHELF_FAILED)
  })

  it('lets Auto sleep while the backend does not answer', () => {
    health.connection = 'down'
    answer({ isError: true })
    renderShelf()
    expect(screen.getByTestId('auto')).toHaveAttribute('data-state', 'sleeping')
  })
})

describe("ShelfScreen — the page's other sections", () => {
  it('links to the tabs and the explorer, Socials only while it is offered', () => {
    answer({ data: { pages: [page(ROWS)] } })
    renderShelf()
    const sections = within(screen.getByRole('navigation', { name: SECTIONS_LABEL }))
    expect(sections.getAllByRole('link').map((link) => [link.textContent, link.getAttribute('href')])).toEqual([
      ['Blogs', '/deliverables?tab=blogs'],
      ['Templates', '/deliverables?tab=templates'],
      ['Brand kit', '/deliverables?tab=brand'],
      ['Explorer', '/deliverables/explorer'],
    ])
    cleanup()
    workspace.socials = true
    renderShelf()
    expect(screen.getByRole('link', { name: 'Socials' })).toHaveAttribute('href', '/deliverables?tab=socials')
  })
})
