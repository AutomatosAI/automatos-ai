/**
 * F358 — the Deliverables preview downloads through the shared, authenticated helper.
 *
 * Its private downloadViaApi duplicated downloadGeneratedFile, and on any failure it fell
 * back to window.open on `${base}${url}`, a request without the Authorization header that
 * cannot succeed in the hosted edition. Download now goes through useApiFileDownload: the
 * file is fetched from the API base URL with the auth headers and saved, and a failed
 * fetch shows the error toast instead of opening anything.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'

const FILE_ROUTE = '/api/documents/generated/invoice-0042.pdf'

const api = vi.hoisted(() => ({
  getBaseUrl: () => 'https://api.test',
  getAuthHeaders: async () => ({ Authorization: 'Bearer test' }),
}))
const toastMock = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn(), info: vi.fn() }))
// The file could not be read for the preview, so the page shows the fallback with its
// own Download as well as the header's; no preview fetch muddies what Download does.
const detail = vi.hoisted(() => ({
  id: 'd-pdf',
  title: 'Invoice 0042',
  agent_name: 'Scribe',
  artifact_type: 'document',
  file_path: 'documents/invoice-0042.pdf',
  file_name: 'invoice-0042.pdf',
  file_type: 'pdf',
  summary: null,
  preview_url: '/api/documents/generated/invoice-0042.pdf',
  content: null,
  content_url: '/api/documents/generated/invoice-0042.pdf',
  content_error: 'storage unavailable',
  created_at: '2026-10-05T09:00:00Z',
}))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('sonner', () => ({ toast: toastMock }))
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ open, children }: { open: boolean; children: React.ReactNode }) => (open ? <div>{children}</div> : null),
  SheetContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
}))
vi.mock('@/hooks/use-deliverables-api', () => ({
  useDeliverable: (id: string | null) => ({ data: id ? { success: true, deliverable: detail } : undefined, isLoading: false, isError: false }),
  useDeleteDeliverable: () => ({ mutate: vi.fn(), isLoading: false }),
}))

// F354's Add to Knowledge button sits in the same action row; its own test covers it.
vi.mock('../add-to-knowledge-button', () => ({ AddToKnowledgeButton: () => null }))

import { DeliverablePreview } from '../deliverable-preview'
import { API_FILE_DOWNLOAD_FAILED } from '@/hooks/use-api-file-download'

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

function downloadButtons() {
  return screen.getAllByRole('button', { name: /Download/ })
}

describe('F358 the Deliverables preview downloads with auth', () => {
  it.each([0, 1])('Download #%i fetches the file with the auth headers and saves it', async (index) => {
    render(<DeliverablePreview deliverableId="d-pdf" open onOpenChange={() => {}} />)
    expect(downloadButtons()).toHaveLength(2)
    fireEvent.click(downloadButtons()[index])
    await waitFor(() => expect(saved).toEqual([{ href: 'blob:invoice-1', download: 'invoice-0042.pdf' }]))
    expect(fetchMock).toHaveBeenCalledWith(
      `https://api.test${FILE_ROUTE}`,
      expect.objectContaining({ headers: expect.objectContaining({ Authorization: 'Bearer test' }) }),
    )
  })

  it.each([0, 1])('a refused Download #%i shows the error and opens nothing', async (index) => {
    fetchMock.mockResolvedValueOnce({ ok: false, status: 401, blob: async () => new Blob([]) } as never)
    render(<DeliverablePreview deliverableId="d-pdf" open onOpenChange={() => {}} />)
    fireEvent.click(downloadButtons()[index])
    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith(API_FILE_DOWNLOAD_FAILED))
    expect(windowOpen).not.toHaveBeenCalled()
    expect(saved).toEqual([])
  })
})
