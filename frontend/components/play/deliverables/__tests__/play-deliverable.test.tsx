/**
 * PRD-260 T-007b, K8 — Play's page for one Deliverable at the core's
 * /deliverables?deliverable=<id>, over a mocked useDeliverable
 * (hooks/use-deliverables-api.ts), ported from Automatos Play's
 * tests/deliverables/DeliverableScreen.test.tsx: a markdown Deliverable on its
 * sheet (no raw HTML), a picture and a video through the core's authenticated
 * path (useAuthenticatedBlobUrl), the file card with the core's download, the
 * side panel's facts, Ask Auto about this (a new chat with the words waiting),
 * the not-found and failed states, the shelf switch (a bad id falls back to the
 * shelf with a notice), and DeliverableDeepLink staying closed under Play. One
 * test runs the real hook over a mocked apiClient, to see the request on the
 * wire. Roles, attributes and text only (CSS modules are proxies under
 * css: false). "Now" is pinned to a local date.
 */
import React from 'react'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Deliverable, DeliverableDetailResponse } from '@/hooks/use-deliverables-api'

const hook = vi.hoisted(() => ({ real: false, detail: vi.fn(), list: vi.fn() }))
vi.mock('@/hooks/use-deliverables-api', async () => {
  const actual = await vi.importActual<typeof import('@/hooks/use-deliverables-api')>('@/hooks/use-deliverables-api')
  return {
    ...actual,
    useDeliverable: (id: string | null, includeContent?: boolean) =>
      hook.real ? actual.useDeliverable(id, includeContent) : hook.detail(id, includeContent),
    useDeliverables: (filters: unknown) => hook.list(filters),
  }
})

const api = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/hooks/use-workspace', () => ({ useWorkspace: () => ({ workspaceId: 'ws-1' }) }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'ws-1', socials: { available: false } }, isLoading: false }),
}))

const media = vi.hoisted(() => ({ blobUrl: vi.fn((url: string | undefined) => ({ src: null as string | null, error: null as string | null, url })) }))
vi.mock('@/components/widgets/FileWidget/FilePreview', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/components/widgets/FileWidget/FilePreview')>()),
  useAuthenticatedBlobUrl: media.blobUrl,
}))
const files = vi.hoisted(() => ({ download: vi.fn(async (_url: string, _filename: string) => undefined) }))
vi.mock('@/hooks/use-api-file-download', () => ({ useApiFileDownload: () => ({ download: files.download, downloading: false }) }))

const nav = vi.hoisted(() => ({ search: '', pathname: '/deliverables', push: vi.fn(), replace: vi.fn() }))
vi.mock('next/navigation', () => ({
  useRouter: () => nav,
  usePathname: () => nav.pathname,
  useSearchParams: () => new URLSearchParams(nav.search),
}))
const style = vi.hoisted(() => ({ play: true, narrow: false }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true, useIsPlay: () => style.play }))
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => style.narrow, useIsMobile: () => style.narrow }))
vi.mock('@/hooks/use-backend-health', () => ({ useBackendHealth: () => ({ connection: 'connected', check: vi.fn() }) }))
vi.mock('@/stores/chat-session-store', () => ({
  useChatSessionStore: (select: (state: { newDraft: () => void }) => unknown) => select({ newDraft: vi.fn() }),
}))
// useStartChat binds the chat session store itself (P260-FX-03): hydrated here.
vi.mock('@/hooks/use-chat-session', () => ({ useChatSessionHydration: () => ({ key: 'k', hydrated: true }) }))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children?: React.ReactNode }) => <div data-testid="frame">{children}</div>,
}))
vi.mock('@/components/mascot/auto-mascot', () => ({
  AutoMascot: ({ state }: { state: string }) => <div data-testid="auto" data-state={state} />,
}))
vi.mock('@/components/deliverables/deliverable-deep-link', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/components/deliverables/deliverable-deep-link')>()),
  DeliverableDeepLink: () => <div data-testid="deep-link" role="dialog" aria-label="Deliverable preview" />,
}))

import DeliverablesLayout from '@/app/deliverables/layout'
import { askAboutText } from '@/lib/play/detail'
import { DRAFT_PARAM } from '@/lib/play/draft'
import { DOWNLOAD_FILE, PICTURE_FAILED, TRUNCATED, UNREADABLE } from '../deliverable-content'
import { ABOUT, ASK_AUTO, DOWNLOAD } from '../detail-panel'
import { NOT_ON_SHELF, OPEN_FAILED, PlayDeliverable, TRY_AGAIN } from '../play-deliverable'
import { BAD_LINK, PlayShelf } from '../play-shelf'

const NOW = new Date(2026, 9, 14, 16, 0) // Wed 14 Oct 2026, 16:00
const MADE = new Date(2026, 9, 14, 15, 42).toISOString()
const RAW_PNG = '/api/workspaces/ws-1/files/raw?path=outputs%2Fposter.png'
const RAW_MP4 = '/api/workspaces/ws-1/files/raw?path=outputs%2Fteaser.mp4'
const RAW_XLSX = '/api/workspaces/ws-1/files/raw?path=outputs%2Fbudget.xlsx'
const TABLE = [
  '# The short answer',
  '',
  'The **Vivobook S 14** is the pick. <script>alert(1)</script> ![a chart](https://evil.example/c.png)',
  '',
  '| Laptop | Price |',
  '|---|---|',
  '| Vivobook S 14 | £679 |',
].join('\n')

function deliverable(fields: Partial<Deliverable> = {}): Deliverable {
  return {
    id: 'd-laptop',
    workspace_id: 'ws-1',
    source_type: 'chat',
    source_id: null,
    agent_id: 7,
    agent_name: 'Researcher',
    artifact_type: 'report',
    title: 'Laptop shortlist',
    summary: null,
    storage_type: 'workspace',
    file_path: 'outputs/laptop-shortlist.md',
    file_name: 'laptop-shortlist.md',
    file_type: 'md',
    file_size_bytes: 2048,
    preview_url: '/api/workspaces/ws-1/files/content?path=outputs%2Flaptop-shortlist.md',
    preview_type: null,
    extra: {},
    status: 'active',
    created_at: MADE,
    updated_at: MADE,
    content: TABLE,
    content_url: '/api/workspaces/ws-1/files/content?path=outputs%2Flaptop-shortlist.md',
    ...fields,
  }
}

interface Answer {
  data?: DeliverableDetailResponse
  isError?: boolean
  error?: unknown
}

function answer({ data, isError = false, error = null }: Answer) {
  const result = { data, isError, error, refetch: vi.fn(async () => ({})) }
  hook.detail.mockReturnValue(result)
  return result
}

const found = (fields: Partial<Deliverable> = {}) => answer({ data: { success: true, deliverable: deliverable(fields) } })

function renderDetail() {
  const user = userEvent.setup()
  const view = render(<PlayDeliverable id="d-laptop" now={NOW} />)
  return { user, ...view }
}

function panel() {
  return within(screen.getByRole('complementary', { name: ABOUT }))
}

function fact(term: string): string | null {
  const dt = panel().getByText(term, { selector: 'dt' })
  return dt.nextElementSibling?.textContent ?? null
}

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
  media.blobUrl.mockImplementation((url) => ({ src: null, error: null, url }))
  hook.list.mockReturnValue({
    data: { pages: [{ success: true, deliverables: [], total: 0, limit: 24, offset: 0 }] },
    isError: false,
    hasNextPage: false,
    isFetchingNextPage: false,
    fetchNextPage: vi.fn(),
    refetch: vi.fn(),
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.restoreAllMocks()
  hook.real = false
  Object.assign(nav, { search: '', pathname: '/deliverables' })
  Object.assign(style, { play: true, narrow: false })
})

describe('PlayDeliverable — the page', () => {
  it('shows the title, type, date and Made by, with the back link to the shelf and no mascot', () => {
    found()
    renderDetail()
    expect(screen.getByRole('heading', { level: 1, name: 'Laptop shortlist' })).toBeInTheDocument()
    const article = screen.getByRole('article', { name: 'Laptop shortlist' })
    expect(article).toHaveTextContent('Report')
    expect(article).toHaveTextContent('Today, 15:42')
    expect(article).toHaveTextContent('Made by Auto with Researcher')
    expect(screen.getByRole('link', { name: /Shelf/ })).toHaveAttribute('href', '/deliverables')
    expect(screen.queryByTestId('auto')).toBeNull()
    expect(hook.detail).toHaveBeenCalledWith('d-laptop', true)
  })

  it("renders a markdown Deliverable through the core's renderer: no raw HTML, images as their alt text", () => {
    found()
    const { container } = renderDetail()
    const article = screen.getByRole('article', { name: 'Laptop shortlist' })
    expect(within(article).getByRole('heading', { level: 1, name: 'The short answer' })).toBeInTheDocument()
    expect(within(article).getByText('Vivobook S 14', { selector: 'strong' })).toBeInTheDocument()
    expect(within(article).getByRole('table')).toHaveTextContent('£679')
    expect(within(article).getByText('a chart')).toBeInTheDocument()
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
  })

  it('shows a text file verbatim in a code block, and says when the file was cut', () => {
    found({ artifact_type: 'code', file_type: 'py', file_name: 'fit.py', content: 'print("<b>hi</b>")', content_truncated: true })
    const { container } = renderDetail()
    expect(container.querySelector('code')).toHaveTextContent('print("<b>hi</b>")')
    expect(container.querySelector('b')).toBeNull()
    expect(screen.getByText(TRUNCATED)).toBeInTheDocument()
  })

  it("shows an image Deliverable through the core's authenticated path", () => {
    media.blobUrl.mockImplementation((url) => ({ src: 'blob:core/poster', error: null, url }))
    found({ artifact_type: 'image', title: 'Bake sale poster', file_type: 'png', content: null, content_url: RAW_PNG, preview_url: RAW_PNG })
    renderDetail()
    expect(media.blobUrl).toHaveBeenCalledWith(RAW_PNG)
    expect(screen.getByRole('img', { name: 'Bake sale poster' })).toHaveAttribute('src', 'blob:core/poster')
  })

  it("says when the picture doesn't load", () => {
    media.blobUrl.mockImplementation((url) => ({ src: null, error: 'Fetch failed: 403', url }))
    found({ artifact_type: 'image', file_type: 'png', content: null, content_url: RAW_PNG })
    renderDetail()
    expect(screen.getByRole('alert')).toHaveTextContent(PICTURE_FAILED)
    expect(screen.queryByRole('img')).toBeNull()
  })

  it("plays a video Deliverable through the core's authenticated path", () => {
    media.blobUrl.mockImplementation((url) => ({ src: 'blob:core/teaser', error: null, url }))
    found({ artifact_type: 'video', title: 'Teaser', file_type: 'mp4', content: null, content_url: RAW_MP4 })
    const { container } = renderDetail()
    expect(media.blobUrl).toHaveBeenCalledWith(RAW_MP4)
    const video = container.querySelector('video')
    expect(video).toHaveAttribute('src', 'blob:core/teaser')
    expect(video).toHaveAttribute('aria-label', 'Teaser')
  })

  it("shows any other type as the file card, with the core's download", async () => {
    found({ artifact_type: 'spreadsheet', file_type: 'xlsx', file_name: 'budget.xlsx', content: 'PK\u0003\u0004', content_url: RAW_XLSX })
    const { user } = renderDetail()
    expect(screen.getByText("Play can't preview this spreadsheet here.")).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: DOWNLOAD_FILE }))
    expect(files.download).toHaveBeenCalledWith(RAW_XLSX, 'budget.xlsx')
  })

  it("says when Automatos couldn't read the file", () => {
    found({ content: null, content_error: 'File not found' })
    renderDetail()
    expect(screen.getByText(UNREADABLE)).toBeInTheDocument()
  })
})

describe('PlayDeliverable — the side panel', () => {
  it('shows the type, when it was made, the agent and where it came from', () => {
    found({ source_type: 'mission' })
    renderDetail()
    expect(fact('Type')).toBe('Report')
    expect(fact('Made')).toBe('Today, 15:42')
    expect(fact('Made by')).toBe('Researcher')
    expect(fact('From')).toBe('A Mission')
  })

  it('says Made by Auto without an agent, and leaves an unknown source out', () => {
    found({ agent_name: null, source_type: '' })
    renderDetail()
    expect(fact('Made by')).toBe('Auto')
    expect(panel().queryByText('From')).toBeNull()
  })

  it('Ask Auto about this opens /chat with a new draft naming the Deliverable', () => {
    found()
    renderDetail()
    const ask = panel().getByRole('link', { name: ASK_AUTO })
    const href = ask.getAttribute('href') ?? ''
    const url = new URL(href, 'http://play.invalid')
    expect(url.pathname).toBe('/chat')
    expect(url.searchParams.get(DRAFT_PARAM)).toBe(askAboutText('Laptop shortlist'))
    expect(url.searchParams.get(DRAFT_PARAM)).toBe('About my Deliverable "Laptop shortlist": ')
  })

  it("Download hands over the file through the core's download", async () => {
    found()
    const { user } = renderDetail()
    await user.click(panel().getByRole('button', { name: DOWNLOAD }))
    expect(files.download).toHaveBeenCalledWith(deliverable().content_url, 'laptop-shortlist.md')
  })

  it('has no Download when there is no file', () => {
    found({ content_url: null, preview_url: null })
    renderDetail()
    expect(panel().queryByRole('button', { name: DOWNLOAD })).toBeNull()
  })
})

describe('PlayDeliverable — loading and failures', () => {
  it('says it is opening while it loads', () => {
    answer({})
    renderDetail()
    expect(screen.getByRole('status')).toHaveTextContent('Opening this Deliverable…')
  })

  it("says a Deliverable the backend doesn't have isn't on the shelf", () => {
    answer({ isError: true, error: Object.assign(new Error('Deliverable not found'), { status: 404 }) })
    renderDetail()
    expect(screen.getByText(NOT_ON_SHELF)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: TRY_AGAIN })).toBeNull()
  })

  it('shows an error with Try again (refetch) when it can not be opened, the backend words in the log only', async () => {
    const result = answer({ isError: true, error: Object.assign(new Error('get failed: boom'), { status: 500 }) })
    const { user } = renderDetail()
    expect(screen.getByRole('alert')).toHaveTextContent(OPEN_FAILED)
    expect(screen.queryByText(/boom/)).toBeNull()
    expect(console.error).toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: TRY_AGAIN }))
    expect(result.refetch).toHaveBeenCalled()
  })

  it('reads an answer of success: false as a failure', () => {
    answer({ data: { success: false, error: 'get failed: relation does not exist' } as DeliverableDetailResponse })
    renderDetail()
    expect(screen.getByRole('alert')).toHaveTextContent(OPEN_FAILED)
  })

  it('asks the real hook for the content on the wire', async () => {
    hook.real = true
    api.request.mockResolvedValue({ success: true, deliverable: deliverable() })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <PlayDeliverable id="d-laptop" now={NOW} />
      </QueryClientProvider>,
    )
    expect(await screen.findByRole('heading', { level: 1, name: 'Laptop shortlist' })).toBeInTheDocument()
    expect(api.request).toHaveBeenCalledWith('/api/deliverables/d-laptop?include_content=true')
  })
})

describe('the shelf switch (PlayShelf)', () => {
  it('shows the Deliverable a ?deliverable= link names instead of the shelf', () => {
    nav.search = 'deliverable=d-laptop'
    found()
    render(<PlayShelf />)
    expect(screen.getByRole('heading', { level: 1, name: 'Laptop shortlist' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Your shelf' })).toBeNull()
  })

  it('falls back to the shelf with a quiet notice for an id that is not one, and never asks for it', () => {
    nav.search = `deliverable=${encodeURIComponent('../../admin?x=1')}`
    render(<PlayShelf />)
    expect(screen.getByRole('heading', { level: 1, name: 'Your shelf' })).toBeInTheDocument()
    expect(screen.getByText(BAD_LINK)).toBeInTheDocument()
    expect(hook.detail).not.toHaveBeenCalled()
  })

  it('shows the shelf without a notice when no Deliverable is named', () => {
    render(<PlayShelf />)
    expect(screen.getByRole('heading', { level: 1, name: 'Your shelf' })).toBeInTheDocument()
    expect(screen.queryByText(BAD_LINK)).toBeNull()
  })
})

describe('DeliverableDeepLink under Play (app/deliverables/layout.tsx)', () => {
  function renderLayout() {
    return render(
      <DeliverablesLayout>
        <div data-testid="page">The Deliverables page</div>
      </DeliverablesLayout>,
    )
  }

  it("stays closed under Play at desktop width: Play's page is the one detail", () => {
    nav.search = 'deliverable=d-laptop'
    found()
    renderLayout()
    expect(screen.getByRole('heading', { level: 1, name: 'Laptop shortlist' })).toBeInTheDocument()
    expect(screen.queryByTestId('deep-link')).toBeNull()
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it.each([
    ['under Classic or Studio', { play: false, narrow: false }],
    ['below 1024 px', { play: true, narrow: true }],
  ])('opens as today %s', async (_label, next) => {
    Object.assign(style, next)
    nav.search = 'deliverable=d-laptop'
    renderLayout()
    await waitFor(() => expect(screen.getByTestId('deep-link')).toBeInTheDocument())
    expect(screen.getByTestId('page')).toBeInTheDocument()
    expect(hook.detail).not.toHaveBeenCalled()
  })
})
