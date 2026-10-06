/**
 * An email attachment's Download:
 * - opens its link only when the mail tool gave an http(s) URL (Outlook's contentLocation),
 *   in a new tab with no opener; a script URL gets no Download button and opens nothing;
 * - for a Gmail attachment (no link), fetches the platform's Gmail attachment route from the
 *   API with the API client's auth headers and saves it under the attachment's name. Only
 *   that route is fetched: any other path in a tool result is never sent the token.
 */
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const API_BASE = 'https://api.test'
const AUTH = { Authorization: 'Bearer test' }

const api = vi.hoisted(() => ({
  getBaseUrl: () => 'https://api.test',
  getAuthHeaders: async () => ({ Authorization: 'Bearer test' }),
}))
const toastMock = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn(), info: vi.fn() }))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('sonner', () => ({ toast: toastMock }))

import { AttachmentCard, EMAIL_ATTACHMENT_DOWNLOAD_FAILED } from '@/components/widgets/EmailWidget/EmailViewer'

const ATTACHMENT = { id: 'a1', filename: 'quote.pdf', mimeType: 'application/pdf', size: 2048 }
const GMAIL_PATH = '/api/emails/attachments/gmail/18f2a9c4d1e0b7a3/ANGjdJ8x-Wq_3vLk0?filename=Quote%2042.pdf'

const fetchMock = vi.fn(async () => ({ ok: true, blob: async () => new Blob(['%PDF'], { type: 'application/pdf' }) }))
const saved: Array<{ href: string; download: string }> = []

beforeEach(() => {
  fetchMock.mockClear()
  toastMock.error.mockClear()
  saved.length = 0
  vi.stubGlobal('fetch', fetchMock)
  // jsdom implements no object URLs and no downloads.
  Object.defineProperty(URL, 'createObjectURL', { value: vi.fn(() => 'blob:quote-1'), configurable: true, writable: true })
  Object.defineProperty(URL, 'revokeObjectURL', { value: vi.fn(), configurable: true, writable: true })
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    saved.push({ href: this.getAttribute('href') ?? '', download: this.download })
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('AttachmentCard', () => {
  it('opens an http(s) download link in a new tab with no opener', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    render(<AttachmentCard attachment={{ ...ATTACHMENT, downloadUrl: 'https://files.example/quote.pdf' }} />)

    fireEvent.click(screen.getByTitle('Download'))

    expect(open).toHaveBeenCalledTimes(1)
    expect(open).toHaveBeenCalledWith('https://files.example/quote.pdf', '_blank', 'noopener,noreferrer')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('offers no download for a script link, and a click opens nothing', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    render(<AttachmentCard attachment={{ ...ATTACHMENT, downloadUrl: 'javascript:alert(document.cookie)' }} />)

    expect(screen.queryByTitle('Download')).toBeNull()
    fireEvent.click(screen.getByText('quote.pdf'))
    expect(open).not.toHaveBeenCalled()
  })

  it('offers no download when the mail tool gave neither a link nor an API path', () => {
    render(<AttachmentCard attachment={ATTACHMENT} />)
    expect(screen.queryByTitle('Download')).toBeNull()
  })

  it('downloads a Gmail attachment from the API route with auth, under its own name', async () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    render(<AttachmentCard attachment={{ ...ATTACHMENT, filename: 'Quote 42.pdf', downloadPath: GMAIL_PATH }} />)

    fireEvent.click(screen.getByTitle('Download'))

    await waitFor(() => expect(saved).toEqual([{ href: 'blob:quote-1', download: 'Quote 42.pdf' }]))
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE}${GMAIL_PATH}`,
      expect.objectContaining({ headers: expect.objectContaining(AUTH) }),
    )
    expect(open).not.toHaveBeenCalled()
  })

  it('says so when the Gmail download is refused', async () => {
    fetchMock.mockResolvedValueOnce({ ok: false, status: 404, blob: async () => new Blob([]) } as never)
    render(<AttachmentCard attachment={{ ...ATTACHMENT, downloadPath: GMAIL_PATH }} />)

    fireEvent.click(screen.getByTitle('Download'))

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith(EMAIL_ATTACHMENT_DOWNLOAD_FAILED))
    expect(saved).toEqual([])
  })

  it.each([
    '/api/documents/generated/secret.pdf',
    '/api/emails/attachments/gmail/../../admin/users',
    '//evil.example/api/emails/attachments/gmail/a/b',
    'https://evil.example/api/emails/attachments/gmail/a/b',
  ])('never sends the token to a path that is not the Gmail attachment route: %s', (downloadPath) => {
    render(<AttachmentCard attachment={{ ...ATTACHMENT, downloadPath }} />)

    expect(screen.queryByTitle('Download')).toBeNull()
    fireEvent.click(screen.getByText('quote.pdf'))
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
