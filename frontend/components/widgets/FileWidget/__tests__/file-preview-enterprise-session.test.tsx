/**
 * PRD-277 US-502 — in enterprise a file preview read from the API goes with the session: the
 * session cookie (credentials) and, for these GETs, no cross-site header. An absolute or presigned
 * address keeps exactly what it sent before. The saas and local requests are pinned by
 * f352-pdf-blob-is-a-pdf.test.tsx.
 *
 *   - The bytes read (authenticatedArrayBufferFetch) names its credentials itself
 *     (lib/api-file-url.ts apiFileCredentials): in saas and local it sends none to the API.
 *   - The framed file (useAuthenticatedBlobUrl) sends what it always sent; in enterprise the
 *     session fetch the provider puts over `window.fetch` (lib/enterprise/session-fetch.ts) adds
 *     the credentials, as it does for every request to the page's own `/api`.
 */
import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const base = vi.hoisted(() => ({ url: 'https://automatos.example.test' }))

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))
vi.mock('@/lib/api-client', () => {
  const apiClient = { getBaseUrl: () => base.url, getAuthHeaders: vi.fn(async () => ({})) }
  return { apiClient, default: apiClient }
})

import { authenticatedArrayBufferFetch, useAuthenticatedBlobUrl } from '@/components/widgets/FileWidget/FilePreview'
import { installSessionFetch } from '@/lib/enterprise/session-fetch'
import { SESSION_CSRF_HEADER } from '@/lib/enterprise/session-transport'

const pageFetch = window.fetch

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response('bytes')))
  URL.createObjectURL = vi.fn(() => 'blob:app/1') as typeof URL.createObjectURL
  URL.revokeObjectURL = vi.fn()
})

afterEach(() => {
  base.url = 'https://automatos.example.test'
  vi.unstubAllGlobals()
  window.fetch = pageFetch
})

describe('file previews in enterprise', () => {
  it('reads an API file with the session', async () => {
    await authenticatedArrayBufferFetch('/api/workspaces/files/raw?path=a.docx')
    expect(fetch).toHaveBeenCalledWith('https://automatos.example.test/api/workspaces/files/raw?path=a.docx', {
      headers: {},
      credentials: 'include',
    })
  })

  it('frames an API file from bytes read with the session (the session fetch, the API on the page host)', async () => {
    base.url = ''
    const network = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>(async () => new Response('bytes'))
    window.fetch = network as unknown as typeof fetch
    const restore = installSessionFetch(window, true)
    const { result } = renderHook(() => useAuthenticatedBlobUrl('/api/documents/generated/x.png'))
    await waitFor(() => expect(result.current.src).toBe('blob:app/1'))
    restore()

    const [input, init] = network.mock.calls[0]
    expect(String(input)).toBe(`${window.location.origin}/api/documents/generated/x.png`)
    expect(init?.credentials).toBe('include')
    expect(new Headers(init?.headers).has(SESSION_CSRF_HEADER)).toBe(false)
  })

  it('keeps an absolute (presigned) address exactly as before', async () => {
    await authenticatedArrayBufferFetch('https://bucket.example.test/x.docx?sig=1')
    expect(fetch).toHaveBeenCalledWith('https://bucket.example.test/x.docx?sig=1', { headers: {}, credentials: 'include' })
  })
})
