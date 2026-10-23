/**
 * PRD-265 W2: apiClient.requestBlob, for a binary answer (an invoice PDF from the Billing tab).
 * It is still the one HTTP client: the same base URL and headers as `request`, the bytes as a
 * Blob on a 2xx, and the same ApiHttpError (status and parsed body) on anything else.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type ApiHttpError } from '@/lib/api-client'

const PDF_PATH = '/api/v1/billing/invoices/in_1/pdf'

function stubFetch(response: Response) {
  const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => response)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('apiClient.requestBlob', () => {
  it('answers the bytes as a Blob, from the API path, with the JSON client headers', async () => {
    const fetchMock = stubFetch(new Response('%PDF-1.7', { status: 200, headers: { 'Content-Type': 'application/pdf' } }))

    const blob = await apiClient.requestBlob(PDF_PATH)

    expect(blob.type).toBe('application/pdf')
    expect(blob.size).toBe('%PDF-1.7'.length) // jsdom's Blob has no text(); the bytes arrived whole
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url).endsWith(PDF_PATH)).toBe(true)
    expect((init?.headers as Record<string, string>)['Content-Type']).toBe('application/json')
  })

  it('throws the ApiHttpError, with its status and body, for a refusal', async () => {
    stubFetch(new Response(JSON.stringify({ detail: 'Invoice not found.' }), { status: 404, headers: { 'Content-Type': 'application/json' } }))

    const error = await apiClient.requestBlob(PDF_PATH).then(
      () => null,
      (caught: ApiHttpError) => caught,
    )

    expect(error?.status).toBe(404)
    expect(error?.message).toBe('Invoice not found.')
    expect(error?.body).toEqual({ detail: 'Invoice not found.' })
  })
})
