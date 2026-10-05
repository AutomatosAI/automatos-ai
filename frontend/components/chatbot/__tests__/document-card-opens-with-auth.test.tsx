/**
 * F358 — the chat's document card opens and downloads its file.
 *
 * A generated document's preview_url and download_url are API routes behind auth
 * (/api/documents/generated/<file>). The card framed them with <iframe src>, opened them
 * with window.open and linked them with <a href>: a relative path reaches the frontend's
 * own origin, with no Authorization header, so it failed in both editions. Now the bytes
 * come from the API base URL with the API client's auth headers, as the Deliverables
 * panel fetches them: the preview frames a blob: URL and Download saves the blob.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'

const FILE_ROUTE = '/api/documents/generated/invoice-0042.pdf'
const API_BASE = 'https://api.test'
const AUTH = { Authorization: 'Bearer test' }

const api = vi.hoisted(() => ({
  getBaseUrl: () => 'https://api.test',
  getAuthHeaders: async () => ({ Authorization: 'Bearer test' }),
}))
const toastMock = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn(), info: vi.fn() }))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('sonner', () => ({ toast: toastMock }))
// The widget frame's menu is a Radix dropdown; the test needs only its Download action.
vi.mock('@/components/widgets/WidgetBase', () => ({
  WidgetBase: ({ onDownload, children }: { onDownload?: () => void; children: React.ReactNode }) => (
    <div>
      {onDownload && <button type="button" onClick={onDownload}>Widget download</button>}
      {children}
    </div>
  ),
}))

import { DocumentArtifact } from '../document-artifact'
import { TextArtifact } from '../text-artifact'
import { DocumentWidget } from '@/components/widgets/DocumentWidget'
import { API_FILE_DOWNLOAD_FAILED } from '@/hooks/use-api-file-download'
import type { Artifact } from '@/types'

const fetchMock = vi.fn(async () => ({ ok: true, blob: async () => new Blob(['%PDF'], { type: 'application/pdf' }) }))
const saved: Array<{ href: string; download: string }> = []
const windowOpen = vi.fn()
let anchorClick: { mockRestore: () => void } | null = null
let consoleError: { mockRestore: () => void } | null = null

beforeEach(() => {
  fetchMock.mockClear()
  toastMock.error.mockClear()
  windowOpen.mockClear()
  saved.length = 0
  vi.stubGlobal('fetch', fetchMock)
  vi.stubGlobal('open', windowOpen)
  // jsdom implements no object URLs and no downloads.
  Object.defineProperty(URL, 'createObjectURL', { value: vi.fn(() => 'blob:invoice-1'), configurable: true, writable: true })
  Object.defineProperty(URL, 'revokeObjectURL', { value: vi.fn(), configurable: true, writable: true })
  consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)
  anchorClick = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    saved.push({ href: this.getAttribute('href') ?? '', download: this.download })
  })
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  anchorClick?.mockRestore()
  consoleError?.mockRestore()
})

const PDF_ARTIFACT: Artifact = {
  id: 'a-1',
  kind: 'document',
  title: 'Invoice 0042',
  content: '',
  metadata: { format: 'pdf', size_kb: 12, preview_url: FILE_ROUTE, download_url: FILE_ROUTE, filename: 'invoice-0042.pdf' },
}

function expectFetchedWithAuth() {
  expect(fetchMock).toHaveBeenCalledWith(`${API_BASE}${FILE_ROUTE}`, { headers: AUTH })
}

function expectSavedAs(filename: string) {
  expect(saved).toEqual([{ href: 'blob:invoice-1', download: filename }])
  expect(windowOpen).not.toHaveBeenCalled()
}

describe('F358 the chat document card reaches its file through the API, with auth', () => {
  it('the artifact viewer frames the PDF from a blob: URL, never the relative route', async () => {
    render(<DocumentArtifact artifact={PDF_ARTIFACT} />)
    await waitFor(() => expect(document.querySelector('iframe')).not.toBeNull())
    expect(document.querySelector('iframe')).toHaveAttribute('src', 'blob:invoice-1')
    expectFetchedWithAuth()
  })

  it("the artifact viewer's Download saves the file instead of opening the route", async () => {
    render(<DocumentArtifact artifact={{ ...PDF_ARTIFACT, metadata: { ...PDF_ARTIFACT.metadata, preview_url: undefined } }} />)
    fireEvent.click(screen.getByRole('button', { name: /Download PDF/ }))
    await waitFor(() => expectSavedAs('invoice-0042.pdf'))
    expectFetchedWithAuth()
  })

  it('a text artifact offers Download as a button, not a link to the route', async () => {
    render(<TextArtifact content="Report" metadata={{ download_url: FILE_ROUTE }} />)
    expect(document.querySelector(`a[href="${FILE_ROUTE}"]`)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Download/ }))
    await waitFor(() => expectSavedAs('invoice-0042.pdf'))
    expectFetchedWithAuth()
  })

  it("the chat's generated-document widget downloads through the API", async () => {
    render(
      <DocumentWidget
        id="w-1"
        title="Invoice 0042"
        data={{ content: '*Document generated: invoice-0042.pdf*', format: 'markdown', filename: 'invoice-0042.pdf', downloadUrl: FILE_ROUTE }}
        metadata={{ source: { type: 'tool', name: 'generate_document', provider: 'document_generation' }, createdAt: new Date() }}
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Widget download' }))
    await waitFor(() => expectSavedAs('invoice-0042.pdf'))
    expectFetchedWithAuth()
  })

  it('a refused download says so instead of failing silently', async () => {
    fetchMock.mockResolvedValueOnce({ ok: false, status: 401, blob: async () => new Blob([]) } as never)
    render(<DocumentArtifact artifact={{ ...PDF_ARTIFACT, metadata: { ...PDF_ARTIFACT.metadata, preview_url: undefined } }} />)
    fireEvent.click(screen.getByRole('button', { name: /Download PDF/ }))
    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith(API_FILE_DOWNLOAD_FAILED))
    expect(saved).toEqual([])
  })
})
