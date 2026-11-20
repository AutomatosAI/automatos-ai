/**
 * PRD-277 US-502 — in enterprise a file preview read from the API goes with the session: the
 * session cookie (credentials) and, for this GET, no cross-site header. An absolute or presigned
 * address keeps exactly what it sent before. The saas and local requests are pinned by
 * f352-pdf-blob-is-a-pdf.test.tsx.
 */
import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))
vi.mock('@/lib/api-client', () => {
  const apiClient = { getBaseUrl: () => 'https://automatos.example.test', getAuthHeaders: vi.fn(async () => ({})) }
  return { apiClient, default: apiClient }
})

import { authenticatedArrayBufferFetch, useAuthenticatedBlobUrl } from '@/components/widgets/FileWidget/FilePreview'

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response('bytes')))
  URL.createObjectURL = vi.fn(() => 'blob:app/1') as typeof URL.createObjectURL
  URL.revokeObjectURL = vi.fn()
})

afterEach(() => vi.unstubAllGlobals())

describe('file previews in enterprise', () => {
  it('reads an API file with the session', async () => {
    await authenticatedArrayBufferFetch('/api/workspaces/files/raw?path=a.docx')
    expect(fetch).toHaveBeenCalledWith('https://automatos.example.test/api/workspaces/files/raw?path=a.docx', {
      headers: {},
      credentials: 'include',
    })
  })

  it('frames an API file from bytes read with the session', async () => {
    const { result } = renderHook(() => useAuthenticatedBlobUrl('/api/documents/generated/x.png'))
    await waitFor(() => expect(result.current.src).toBe('blob:app/1'))
    expect(fetch).toHaveBeenCalledWith('https://automatos.example.test/api/documents/generated/x.png', {
      headers: {},
      credentials: 'include',
    })
  })

  it('keeps an absolute (presigned) address exactly as before', async () => {
    await authenticatedArrayBufferFetch('https://bucket.example.test/x.docx?sig=1')
    expect(fetch).toHaveBeenCalledWith('https://bucket.example.test/x.docx?sig=1', { headers: {}, credentials: 'include' })
  })
})
