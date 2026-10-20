/**
 * PRD-260 T-007b, K8 — Play's page for one Deliverable at the core's
 * /deliverables?deliverable=<id>, over a mocked useDeliverable
 * (the theme SDK's, lib/theme-sdk/deliverables.ts, since PRD-261 A-14b), ported from Automatos Play's
 * tests/deliverables/DeliverableScreen.test.tsx: a markdown Deliverable on its
 * sheet (no raw HTML), a picture and a video through the core's authenticated
 * path (the SDK's useAuthenticatedBlobUrl), the file card with the core's download, the
 * side panel's facts, Ask Auto about this (a new chat with the words waiting),
 * the not-found and failed states, the shelf switch (a bad id falls back to the
 * shelf with a notice), and DeliverableDeepLink staying closed under Play. One
 * test runs the real hook over a mocked apiClient, to see the request on the
 * wire. Roles, attributes and text only (CSS modules are proxies under
 * css: false). "Now" is pinned to a local date. Opening and "not on your
 * shelf" are one status paragraph, kept from one to the other, so the change
 * is announced (P260-FX-24, M22).
 */
import React from 'react'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Deliverable, DeliverableDetailResponse } from '@/hooks/use-deliverables-api'

const hook = vi.hoisted(() => ({ real: false, detail: vi.fn(), list: vi.fn() }))

const api = vi.hoisted(() => ({ request: vi.fn() }))
// The real raw-file route builder: Download swaps a text file's content route for it (P260-FX-12, M49).
vi.mock('@/lib/api-client', async (importOriginal) => {
  const { apiClient } = await importOriginal<typeof import('@/lib/api-client')>()
  return { apiClient: { request: api.request, getWorkspaceFileRawUrl: apiClient.getWorkspaceFileRawUrl.bind(apiClient) } }
})
vi.mock('@/hooks/use-workspace', () => ({ useWorkspace: () => ({ workspaceId: 'ws-1' }) }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({
    workspace: { id: 'ws-1', name: 'Acme', slug: 'acme', plan: 'pro', socials: { available: false } },
    isLoading: false,
  }),
}))

const media = vi.hoisted(() => ({ blobUrl: vi.fn((url: string | undefined) => ({ src: null as string | null, error: null as string | null, url })) }))
const files = vi.hoisted(() => {
  const download = vi.fn(async (_url: string, _filename: string) => undefined)
  // Each use of the hook, with the toast a failed download shows.
  const hook = vi.fn((_failedMessage?: string) => ({ download, downloading: false }))
  return { download, hook }
})
// PRD-261 A-14b: Play's Deliverable page reads the Deliverable, its media and its download through
// '@automatos/theme-sdk', whose deliverables area (lib/theme-sdk/deliverables.ts) stands in here.
vi.mock('@/lib/theme-sdk/deliverables', async () => {
  const actual = await vi.importActual<typeof import('@/lib/theme-sdk/deliverables')>('@/lib/theme-sdk/deliverables')
  return {
    ...actual,
    useDeliverable: (id: string | null, includeContent?: boolean) =>
      hook.real ? actual.useDeliverable(id, includeContent) : hook.detail(id, includeContent),
    useDeliverables: (filters: unknown) => hook.list(filters),
    useAuthenticatedBlobUrl: media.blobUrl,
    useFileDownload: (failedMessage?: string) => files.hook(failedMessage),
  }
})

const nav = vi.hoisted(() => ({ search: '', pathname: '/deliverables', push: vi.fn(), replace: vi.fn() }))
vi.mock('next/navigation', () => ({
  useRouter: () => nav,
  usePathname: () => nav.pathname,
  useSearchParams: () => new URLSearchParams(nav.search),
}))
const style = vi.hoisted(() => ({ play: true, narrow: false }))
// Play's width source (P260-FX-25): Play's version only under Play at >= 1024 px.
vi.mock('@/hooks/use-studio-theme', async () => {
  const { BUILTIN_PLAY_ENTRY } = await import('@/lib/theme-host/builtin/play-entry')
  return {
    useIsStudio: () => true,
    useActiveTheme: () => (style.play ? BUILTIN_PLAY_ENTRY : null),
    useThemeLayout: () => (style.play && !style.narrow ? 'theme' : 'fallback'),
  }
})
vi.mock('@/lib/theme-sdk/context', async () => ({
  ...(await vi.importActual<typeof import('@/lib/theme-sdk/context')>('@/lib/theme-sdk/context')),
  useHealth: () => ({ connection: 'connected', check: vi.fn() }),
}))
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
import { DOWNLOAD_FAILED } from '@/lib/deliverables/download'
import { askAboutText } from '@/lib/play/detail'
import { DRAFT_PARAM } from '@/lib/play/draft'
import { DOWNLOAD_FILE, PICTURE_FAILED, TRUNCATED, UNREADABLE } from '../deliverable-content'
import { ABOUT, ASK_AUTO, DOWNLOAD, SOURCES } from '../detail-panel'
import { NOT_ON_SHELF, OPENING, OPEN_FAILED, PlayDeliverable, TRY_AGAIN } from '../play-deliverable'
import { BAD_LINK, PlayShelf } from '../play-shelf'

const NOW = new Date(2026, 9, 14, 16, 0) // Wed 14 Oct 2026, 16:00
const MADE = new Date(2026, 9, 14, 15, 42).toISOString()
const RAW_PNG = '/api/workspaces/ws-1/files/raw?path=outputs%2Fposter.png'
const RAW_MP4 = '/api/workspaces/ws-1/files/raw?path=outputs%2Fteaser.mp4'
const RAW_XLSX = '/api/workspaces/ws-1/files/raw?path=outputs%2Fbudget.xlsx'
/** The report's own bytes: its content_url (the content route) answers JSON for the code viewer. */
const RAW_MD = '/api/workspaces/ws-1/files/raw?path=outputs%2Flaptop-shortlist.md'
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

  it("prints a code Deliverable in Play's own <pre> on the sheet, not the core's CodeBlock card (P260-FX-13, M36)", () => {
    found({ artifact_type: 'code', file_type: 'py', file_name: 'fit.py', content: 'print("hi")' })
    renderDetail()
    const article = screen.getByRole('article', { name: 'Laptop shortlist' })
    const pre = article.querySelector('pre')
    expect(pre).toHaveTextContent('print("hi")')
    expect(pre?.querySelector('code')).toHaveTextContent('print("hi")')
    // CodeBlock's <pre> carries Tailwind classes and sits under a Copy header: Play's carries neither.
    expect(pre).not.toHaveAttribute('class')
    expect(within(article).queryByRole('button', { name: /copy/i })).toBeNull()
  })

  it('drops raw HTML from the sheet instead of printing its tags (P260-FX-13, M39)', () => {
    found({ content: `a<br>b\n\n${TABLE}` })
    renderDetail()
    const article = screen.getByRole('article', { name: 'Laptop shortlist' })
    expect(article.textContent).not.toMatch(/<\/?(?:br|script)\b/i)
    expect(within(article).getByText('ab')).toBeInTheDocument()
    expect(article.querySelector('br, script')).toBeNull()
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

  it("the file card downloads a text file's bytes, not its JSON content route, and a failure says try again in a moment", async () => {
    found({ content: null, content_error: 'File not found' })
    const { user } = renderDetail()
    await user.click(screen.getByRole('button', { name: DOWNLOAD_FILE }))
    expect(files.download).toHaveBeenCalledWith(RAW_MD, 'laptop-shortlist.md')
    // The file card and the side panel's Download both say it: "Try again from Deliverables" sent users where they were.
    expect(files.hook.mock.calls.length).toBeGreaterThanOrEqual(2)
    expect(files.hook.mock.calls.every(([failedMessage]) => failedMessage === DOWNLOAD_FAILED)).toBe(true)
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

  it.each([
    ['upload', 'Uploaded', 'An upload', 'Uploaded'],
    ['document', 'Templates', 'A template', 'Made in Templates'],
    ['social_post', 'Socials', 'A social post', 'Made in Socials'],
  ])("never credits Auto with a %s: the core's word instead (P260-FX-13, M104)", (sourceType, maker, from, line) => {
    found({ agent_name: null, source_type: sourceType })
    renderDetail()
    expect(fact('Made by')).toBe(maker)
    expect(fact('From')).toBe(from)
    const article = screen.getByRole('article', { name: 'Laptop shortlist' })
    expect(article).toHaveTextContent(line)
    expect(article).not.toHaveTextContent('Made by Auto')
  })

  it("lists Sources · N from extra.sources, as Play's panel does: a web address as its host (P260-FX-13, M81)", () => {
    const extra = { sources: ['https://www.which.co.uk/reviews/laptops', { url: 'https://rtings.com/x' }, { title: 'Asus spec sheet' }] }
    found({ extra })
    renderDetail()
    expect(panel().getByRole('heading', { level: 2, name: `${SOURCES} · 3` })).toBeInTheDocument()
    expect(panel().getAllByRole('listitem').map((item) => item.textContent)).toEqual(['which.co.uk', 'rtings.com', 'Asus spec sheet'])
  })

  it('has no Sources section when it carries none', () => {
    found({ extra: {} })
    renderDetail()
    expect(panel().queryByRole('heading', { name: new RegExp(SOURCES) })).toBeNull()
    expect(panel().queryAllByRole('listitem')).toHaveLength(0)
  })

  it('Ask Auto about this opens /chat with a new draft naming the Deliverable', () => {
    found()
    renderDetail()
    const ask = panel().getByRole('link', { name: ASK_AUTO })
    const href = ask.getAttribute('href') ?? ''
    const url = new URL(href, 'http://play.invalid')
    expect(url.pathname).toBe('/chat')
    // One plain line (P260-FX-10): the words, their ends trimmed.
    expect(url.searchParams.get(DRAFT_PARAM)).toBe(askAboutText('Laptop shortlist').trimEnd())
    expect(url.searchParams.get(DRAFT_PARAM)).toBe('About my Deliverable "Laptop shortlist":')
  })

  it("Download hands over the file through the core's download: the file's bytes, not the content route's JSON", async () => {
    found()
    const { user } = renderDetail()
    await user.click(panel().getByRole('button', { name: DOWNLOAD }))
    expect(files.download).toHaveBeenCalledWith(RAW_MD, 'laptop-shortlist.md')
    expect(files.hook).toHaveBeenCalledWith(DOWNLOAD_FAILED)
    expect(files.hook).not.toHaveBeenCalledWith(undefined)
  })

  it('Download keeps a raw file route as it is', async () => {
    found({ artifact_type: 'spreadsheet', file_type: 'xlsx', file_name: 'budget.xlsx', content: null, content_url: RAW_XLSX })
    const { user } = renderDetail()
    await user.click(panel().getByRole('button', { name: DOWNLOAD }))
    expect(files.download).toHaveBeenCalledWith(RAW_XLSX, 'budget.xlsx')
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

  it("says a 404 in the page's status line (P260-FX-24, M22)", () => {
    answer({ isError: true, error: Object.assign(new Error('Deliverable not found'), { status: 404 }) })
    renderDetail()
    expect(screen.getByRole('status')).toHaveTextContent(NOT_ON_SHELF)
  })

  it("turns the status line that said it was opening into 'not on your shelf': the same paragraph, so it is announced", () => {
    answer({})
    const { rerender } = renderDetail()
    const status = screen.getByRole('status')
    expect(status).toHaveTextContent(OPENING)

    answer({ isError: true, error: Object.assign(new Error('Deliverable not found'), { status: 404 }) })
    rerender(<PlayDeliverable id="d-laptop" now={NOW} />)
    expect(screen.getByRole('status')).toBe(status)
    expect(status).toHaveTextContent(NOT_ON_SHELF)
    expect(screen.queryByRole('alert')).toBeNull()
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

  it('keeps the open Deliverable on screen when a background refetch fails (react-query keeps the data)', () => {
    answer({
      data: { success: true, deliverable: deliverable() },
      isError: true,
      error: Object.assign(new Error('get failed: boom'), { status: 500 }),
    })
    renderDetail()
    expect(screen.getByRole('heading', { level: 1, name: 'Laptop shortlist' })).toBeInTheDocument()
    expect(screen.queryByRole('alert')).toBeNull()
    expect(screen.queryByText(OPEN_FAILED)).toBeNull()
    expect(screen.queryByText(NOT_ON_SHELF)).toBeNull()
    expect(console.error).toHaveBeenCalled()
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

  it("stays closed under Play at desktop width: Play's page is the one detail", async () => {
    nav.search = 'deliverable=d-laptop'
    found()
    renderLayout()
    // Play's shelf loads through its lazy slot (P260-FX-23, M17).
    expect(await screen.findByRole('heading', { level: 1, name: 'Laptop shortlist' })).toBeInTheDocument()
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
