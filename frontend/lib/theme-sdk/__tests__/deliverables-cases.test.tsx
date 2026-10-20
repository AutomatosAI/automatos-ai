/**
 * PRD-261 A-19e2 — the core behaviour a theme's Deliverables page stands on, on the fixture theme
 * (lib/theme-host/__tests__/fixture-theme-entry.ts, A-19b). Its /deliverables page is written here
 * against '@automatos/theme-sdk' only, as a theme writes it, and app/deliverables/layout.tsx draws it
 * through ThemeSlot. The SDK's hooks are the real ones over the host's own (hooks/use-deliverables-api.ts,
 * hooks/use-api-file-download.ts, lib/deliverables/download.ts); only the backend client, the
 * browser's fetch and the toast are stood in.
 *
 * Replaces, in the core test ledger (plan §2.2.8), the core assertions of
 * components/play/deliverables/__tests__/play-shelf.test.tsx (the next page's offset on the wire with
 * the filter, a next page or a refetch that answers success: false keeping the loaded cards, a first
 * page of success: false read as a failure, the list asked again every refetchInterval ms, a card's
 * ?deliverable=<id> link) and of play-deliverable.test.tsx (the detail asked with include_content on
 * the wire, the ?deliverable= link and the backend's ?tab=outputs&deliverable= link opening the
 * theme's page with the deep link's sheet closed, the core page below the theme's width, and the
 * download URLs: a text file's raw route, a raw route as it is, the caller's token, the file's name,
 * the theme's own words when it fails). Their Play-rendering assertions move to the kit (B-05p,
 * B-05d), which lists these cases in `rehomed`.
 */
import React, { useMemo, useState } from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ThemeLayout } from '@/lib/theme-host/registry-types'

/** The backend client: its requests, the API's base URL and the caller's token headers. */
const api = vi.hoisted(() => ({ request: vi.fn(), base: 'https://api.test', headers: { Authorization: 'Bearer fixture' } }))
vi.mock('@/lib/api-client', async (importOriginal) => {
  const { apiClient } = await importOriginal<typeof import('@/lib/api-client')>()
  return {
    apiClient: {
      request: api.request,
      getBaseUrl: () => api.base,
      getAuthHeaders: async () => api.headers,
      getWorkspaceFileRawUrl: apiClient.getWorkspaceFileRawUrl.bind(apiClient),
    },
  }
})
vi.mock('@/hooks/use-workspace', () => ({ useWorkspace: () => ({ workspaceId: 'ws-1' }) }))
const toast = vi.hoisted(() => ({ error: vi.fn(), success: vi.fn(), info: vi.fn() }))
vi.mock('sonner', async (importOriginal) => ({ ...(await importOriginal<typeof import('sonner')>()), toast }))

const state = vi.hoisted(() => ({ search: '', layout: 'theme' as ThemeLayout }))
vi.mock('next/navigation', async (importOriginal) => ({
  ...(await importOriginal<typeof import('next/navigation')>()),
  usePathname: () => '/deliverables',
  useSearchParams: () => new URLSearchParams(state.search),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
}))
vi.mock('@/hooks/use-studio-theme', async () => {
  const { FIXTURE_THEME_ENTRY } = await import('@/lib/theme-host/__tests__/fixture-theme-entry')
  return { useIsStudio: () => true, useActiveTheme: () => FIXTURE_THEME_ENTRY, useThemeLayout: () => state.layout }
})
// ThemeSlot frames an installed theme's page in MainLayout, full bleed; its chrome has its own tests.
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
}))
vi.mock('@/components/deliverables/deliverable-deep-link', () => ({
  DeliverableDeepLink: () => <div data-testid="deep-link" />,
}))

import {
  DELIVERABLE_FEED_FILTERS,
  DELIVERABLE_PARAM,
  DELIVERABLES_HREF,
  deliverableDownloadUrl,
  deliverableFilename,
  HostLink,
  useDeliverable,
  useDeliverables,
  useFileDownload,
  useHostSearchParams,
  type Deliverable,
  type DeliverableListResponse,
  type DeliverablesQuery,
} from '@automatos/theme-sdk'
import DeliverablesLayout from '@/app/deliverables/layout'
import { ThemeHostContext } from '@/components/theme-host/theme-host-context'
import { API_FILE_DOWNLOAD_FAILED } from '@/hooks/use-api-file-download'
import { FIXTURE_THEME_TEST_IDS, fixtureThemeHost } from '@/lib/theme-host/__tests__/fixture-theme-entry'

/** hooks/use-deliverables-api.ts asks for this many a page (its PAGE_SIZE). */
const PAGE_SIZE = 24
const LIST_PATH = '/api/deliverables?'
/** The fixture theme's own refresh period: it asks the list again this often while it is open. */
const THEME_REFRESH_MS = 30_000
const LIST_LABEL = 'Deliverables'
const DOCUMENTS = 'Documents'
const LOAD_MORE = 'Load more'
const DOWNLOAD = 'Download'
const LIST_FAILED = "The fixture theme's words: the shelf didn't load."
const MORE_FAILED = "The fixture theme's words: the next page didn't load."
const DOWNLOAD_FAILED = "The fixture theme's words: the file didn't download."
const CORE_PAGE = 'core-page'
const CREATED = '2026-10-14T09:00:00+00:00'
const RAW_MD = '/api/workspaces/ws-1/files/raw?path=outputs%2Flaptop-shortlist.md'
const CONTENT_MD = '/api/workspaces/ws-1/files/content?path=outputs%2Flaptop-shortlist.md'
const RAW_XLSX = '/api/workspaces/ws-1/files/raw?path=outputs%2Fbudget.xlsx'
const BACKEND_WORDS = 'list failed: relation v_workspace_outputs does not exist'

/** What the fixture's list last handed its page, for the cases that read the query itself. */
const seen = vi.hoisted(() => ({ list: null as unknown }))

function row(id: string, fields: Partial<Deliverable> = {}): Deliverable {
  return {
    id, workspace_id: 'ws-1', source_type: 'chat', source_id: null, agent_id: null, agent_name: null,
    artifact_type: 'document', title: `Deliverable ${id}`, summary: null, storage_type: 'workspace',
    file_path: `outputs/${id}.md`, file_name: null, file_type: null, file_size_bytes: null, preview_url: null,
    preview_type: null, thumbnail_url: null, extra: {}, status: 'active', created_at: CREATED, updated_at: CREATED,
    ...fields,
  }
}

const numbered = (count: number) => Array.from({ length: count }, (_, i) => row(`d-${i}`, { title: `Deliverable number ${i}` }))
const LAPTOP = row('d-laptop', {
  title: 'Laptop shortlist',
  artifact_type: 'report',
  file_path: 'outputs/laptop-shortlist.md',
  file_name: 'laptop-shortlist.md',
  content: '# The short answer',
  content_url: CONTENT_MD,
  preview_url: CONTENT_MD,
})
const BUDGET = row('d-budget', { title: 'Budget', artifact_type: 'spreadsheet', file_name: 'budget.xlsx', content_url: RAW_XLSX })

function page(rows: readonly Deliverable[], total = rows.length, offset = 0): DeliverableListResponse {
  return { success: true, deliverables: rows, total, limit: PAGE_SIZE, offset }
}

const failedPage = (offset = 0) => ({ success: false, error: BACKEND_WORDS, deliverables: [], total: 0, limit: PAGE_SIZE, offset })

const offsetOf = (path: string) => Number(new URLSearchParams(path.split('?')[1]).get('offset'))

/** The backend's list: a page of `rows` at the offset each ask names. */
function servePages(rows: readonly Deliverable[]) {
  return async (path: string) => {
    const offset = offsetOf(path)
    return page(rows.slice(offset, offset + PAGE_SIZE), rows.length, offset)
  }
}

/** Every list ask the backend client got, in order. */
function listAsks(): string[] {
  return api.request.mock.calls.map(([path]) => String(path)).filter((path) => path.startsWith(LIST_PATH))
}

const queryOf = (path: string | undefined) => new URLSearchParams(String(path).split('?')[1])

/** Whether the page a Load more asked for failed (react-query's answer, which the SDK does not type). */
function pageFailed(result: unknown): boolean {
  return typeof result === 'object' && result !== null && (result as { readonly isError?: unknown }).isError === true
}

function FixtureCard({ deliverable }: { readonly deliverable: Deliverable }) {
  const href = `${DELIVERABLES_HREF}?${DELIVERABLE_PARAM}=${encodeURIComponent(deliverable.id)}`
  return <li><HostLink href={href}>{deliverable.title}</HostLink></li>
}

/** The fixture theme's shelf: the feed, a Documents filter, Load more, its own words for a failure. */
function FixtureList() {
  const [kind, setKind] = useState<string | null>(null)
  const [moreFailed, setMoreFailed] = useState(false)
  const filters = useMemo(() => ({ ...DELIVERABLE_FEED_FILTERS, artifact_type: kind }), [kind])
  const list = useDeliverables(filters, { refetchInterval: THEME_REFRESH_MS })
  seen.list = list
  const rows = list.data?.pages.flatMap((each) => each.deliverables) ?? []
  const loadMore = () => {
    setMoreFailed(false)
    void list.fetchNextPage().then((result) => setMoreFailed(pageFailed(result)))
  }
  return (
    <section data-testid={FIXTURE_THEME_TEST_IDS.deliverables}>
      <button type="button" onClick={() => setKind('document')}>{DOCUMENTS}</button>
      {list.isError && list.data === undefined ? <p role="alert">{LIST_FAILED}</p> : null}
      <ul aria-label={LIST_LABEL}>
        {rows.map((deliverable) => <FixtureCard key={deliverable.id} deliverable={deliverable} />)}
      </ul>
      {moreFailed ? <p role="alert">{MORE_FAILED}</p> : null}
      {list.hasNextPage ? <button type="button" onClick={loadMore}>{LOAD_MORE}</button> : null}
    </section>
  )
}

/** The fixture theme's page for one Deliverable: its title, and Download when it has a file. */
function FixtureDeliverable({ id }: { readonly id: string }) {
  const { data } = useDeliverable(id, true)
  const { download } = useFileDownload(DOWNLOAD_FAILED)
  const deliverable = data?.success ? data.deliverable : null
  if (deliverable === null) return <p role="status">Opening</p>
  const url = deliverableDownloadUrl(deliverable)
  return (
    <article>
      <h1>{deliverable.title}</h1>
      {url ? <button type="button" onClick={() => void download(url, deliverableFilename(deliverable))}>{DOWNLOAD}</button> : null}
    </article>
  )
}

/** The fixture theme's /deliverables page: the Deliverable a ?deliverable= link names, else the shelf. */
function FixtureDeliverablesPage() {
  const asked = useHostSearchParams().get(DELIVERABLE_PARAM)
  return asked ? <FixtureDeliverable key={asked} id={asked} /> : <FixtureList />
}

function renderDeliverables(search = '') {
  state.search = search
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const host = fixtureThemeHost('active', { pages: new Map([[DELIVERABLES_HREF, FixtureDeliverablesPage]]) })
  render(
    <QueryClientProvider client={client}>
      <ThemeHostContext.Provider value={host}>
        <DeliverablesLayout>
          <div data-testid={CORE_PAGE}>The core Deliverables page</div>
        </DeliverablesLayout>
      </ThemeHostContext.Provider>
    </QueryClientProvider>,
  )
  return { client }
}

const cards = () => within(screen.getByRole('list', { name: LIST_LABEL })).queryAllByRole('link')

const lastList = () => seen.list as DeliverablesQuery

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
  api.request.mockReset()
  toast.error.mockClear()
  Object.assign(state, { search: '', layout: 'theme' })
  seen.list = null
})

describe('A-19e2 — the list a theme pages through (the host hook under the SDK)', () => {
  it("asks for the feed's first page, then the next page's offset with the theme's filter, on the wire", async () => {
    api.request.mockImplementation(servePages(numbered(30)))
    renderDeliverables()
    await screen.findByRole('link', { name: 'Deliverable number 0' })
    const first = queryOf(listAsks()[0])
    expect([first.get('offset'), first.get('limit'), first.get('source_type_exclude')]).toEqual(['0', String(PAGE_SIZE), 'heartbeat'])
    expect(first.has('artifact_type')).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: DOCUMENTS }))
    await waitFor(() => expect(listAsks()).toHaveLength(2))
    fireEvent.click(await screen.findByRole('button', { name: LOAD_MORE }))
    await screen.findByRole('link', { name: 'Deliverable number 29' })
    const next = queryOf(listAsks()[2])
    expect([next.get('offset'), next.get('limit')]).toEqual([String(PAGE_SIZE), String(PAGE_SIZE)])
    expect([next.get('artifact_type'), next.get('source_type_exclude')]).toEqual(['document', 'heartbeat'])
    expect(cards()).toHaveLength(30)
    await waitFor(() => expect(screen.queryByRole('button', { name: LOAD_MORE })).toBeNull())
  })

  it('keeps the loaded pages when the next one answers success: false; Load more asks that offset again', async () => {
    const rows = numbered(30)
    api.request.mockImplementation(async (path: string) =>
      offsetOf(path) === 0 ? page(rows.slice(0, PAGE_SIZE), rows.length) : failedPage(offsetOf(path)),
    )
    renderDeliverables()
    fireEvent.click(await screen.findByRole('button', { name: LOAD_MORE }))
    expect(await screen.findByRole('alert')).toHaveTextContent(MORE_FAILED)
    expect(cards()).toHaveLength(PAGE_SIZE)
    fireEvent.click(screen.getByRole('button', { name: LOAD_MORE }))
    await waitFor(() => expect(listAsks()).toHaveLength(3))
    expect(offsetOf(listAsks()[2] ?? '')).toBe(PAGE_SIZE)
  })

  it("keeps the loaded cards when a refetch answers success: false, with the backend's words in the error only", async () => {
    api.request.mockResolvedValueOnce(page([LAPTOP, BUDGET])).mockResolvedValue(failedPage())
    const { client } = renderDeliverables()
    await screen.findByRole('link', { name: LAPTOP.title })
    await act(async () => {
      await client.refetchQueries()
    })
    await waitFor(() => expect(lastList().isError).toBe(true))
    expect(listAsks()).toHaveLength(2)
    expect(cards()).toHaveLength(2)
    expect(String((lastList().error as Error).message)).toContain('v_workspace_outputs')
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('reads a first page of success: false as a failure, never as an empty list', async () => {
    api.request.mockResolvedValue(failedPage())
    renderDeliverables()
    expect(await screen.findByRole('alert')).toHaveTextContent(LIST_FAILED)
    expect(lastList().data).toBeUndefined()
    expect(document.body).not.toHaveTextContent('v_workspace_outputs')
  })

  it("asks the list again every refetchInterval ms while it is open: a new Deliverable lands first", async () => {
    vi.useFakeTimers()
    let rows: readonly Deliverable[] = [BUDGET]
    api.request.mockImplementation(async () => page(rows))
    const elapse = (ms: number) => act(async () => {
      await vi.advanceTimersByTimeAsync(ms)
    })
    renderDeliverables()
    await elapse(1)
    await elapse(1)
    expect(cards().map((card) => card.textContent)).toEqual([BUDGET.title])

    rows = [LAPTOP, BUDGET]
    await elapse(THEME_REFRESH_MS / 2)
    expect(listAsks()).toHaveLength(1)
    await elapse(THEME_REFRESH_MS / 2)
    await elapse(1)
    await elapse(1)
    expect(listAsks()).toHaveLength(2)
    expect(cards().map((card) => card.textContent)).toEqual([LAPTOP.title, BUDGET.title])
  })
})

describe('A-19e2 — the ?deliverable= link', () => {
  it("a card's link names the Deliverable, and following it opens the theme's page for it", async () => {
    api.request.mockImplementation(async (path: string) =>
      path.startsWith(LIST_PATH) ? page([LAPTOP]) : { success: true, deliverable: LAPTOP },
    )
    renderDeliverables()
    const href = (await screen.findByRole('link', { name: LAPTOP.title })).getAttribute('href') ?? ''
    expect(href).toBe(`${DELIVERABLES_HREF}?${DELIVERABLE_PARAM}=d-laptop`)
    cleanup()
    renderDeliverables(href.split('?')[1])
    expect(await screen.findByRole('heading', { level: 1, name: LAPTOP.title })).toBeInTheDocument()
    expect(api.request).toHaveBeenCalledWith('/api/deliverables/d-laptop?include_content=true')
  })

  it.each([`${DELIVERABLE_PARAM}=d-laptop`, `tab=outputs&${DELIVERABLE_PARAM}=d-laptop`])(
    "?%s opens the theme's page for that Deliverable, the deep link's sheet never mounted",
    async (search) => {
      api.request.mockResolvedValue({ success: true, deliverable: LAPTOP })
      renderDeliverables(search)
      expect(await screen.findByRole('heading', { level: 1, name: LAPTOP.title })).toBeInTheDocument()
      expect(api.request).toHaveBeenCalledWith('/api/deliverables/d-laptop?include_content=true')
      expect(screen.queryByTestId('deep-link')).toBeNull()
      expect(screen.queryByTestId(CORE_PAGE)).toBeNull()
    },
  )

  it("below the theme's width keeps the core page and its deep link, and the theme never asks for it", () => {
    state.layout = 'fallback'
    renderDeliverables(`${DELIVERABLE_PARAM}=d-laptop`)
    expect(screen.getByTestId(CORE_PAGE)).toBeInTheDocument()
    expect(screen.getByTestId('deep-link')).toBeInTheDocument()
    expect(api.request).not.toHaveBeenCalled()
  })
})

describe("A-19e2 — Download, with the caller's token", () => {
  const fetchMock = vi.fn()
  const saved: Array<{ href: string; download: string }> = []

  beforeEach(() => {
    saved.length = 0
    fetchMock.mockReset().mockResolvedValue({ ok: true, blob: async () => new Blob(['bytes']) })
    vi.stubGlobal('fetch', fetchMock)
    // jsdom has no object URLs and no downloads.
    Object.defineProperty(URL, 'createObjectURL', { value: vi.fn(() => 'blob:fixture'), configurable: true, writable: true })
    Object.defineProperty(URL, 'revokeObjectURL', { value: vi.fn(), configurable: true, writable: true })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      saved.push({ href: this.getAttribute('href') ?? '', download: this.download })
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  async function downloadOf(deliverable: Deliverable) {
    api.request.mockResolvedValue({ success: true, deliverable })
    renderDeliverables(`${DELIVERABLE_PARAM}=${deliverable.id}`)
    fireEvent.click(await screen.findByRole('button', { name: DOWNLOAD }))
  }

  it("fetches a text file's raw route, not its content route's JSON, from the API with the token, saved under its name", async () => {
    await downloadOf(LAPTOP)
    await waitFor(() => expect(saved).toEqual([{ href: 'blob:fixture', download: 'laptop-shortlist.md' }]))
    expect(fetchMock).toHaveBeenCalledWith(`${api.base}${RAW_MD}`, { headers: api.headers })
  })

  it('hands a raw file route over as it is', async () => {
    await downloadOf(BUDGET)
    await waitFor(() => expect(saved).toEqual([{ href: 'blob:fixture', download: 'budget.xlsx' }]))
    expect(fetchMock).toHaveBeenCalledWith(`${api.base}${RAW_XLSX}`, { headers: api.headers })
  })

  it("says the theme's own words when the download fails, never the host's default, and saves nothing", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 401, blob: async () => new Blob([]) })
    await downloadOf(LAPTOP)
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(DOWNLOAD_FAILED))
    expect(toast.error).not.toHaveBeenCalledWith(API_FILE_DOWNLOAD_FAILED)
    expect(saved).toEqual([])
  })
})
