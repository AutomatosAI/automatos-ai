/**
 * F352: a PDF preview is framed as a blob: URL, which runs as the app's own origin. Its bytes
 * are typed as a PDF whatever the server sent, so a file that is really HTML never runs as the
 * app. F358: a path that would leave the API is never fetched with the token.
 */
import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/api-client', () => {
  const apiClient = {
    getBaseUrl: () => 'https://api.test',
    getAuthHeaders: vi.fn(async () => ({ Authorization: 'Bearer t' })),
  }
  return { apiClient, default: apiClient }
})

import { useAuthenticatedBlobUrl } from '@/components/widgets/FileWidget/FilePreview'

const made: Blob[] = []

beforeEach(() => {
  made.length = 0
  vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(['<script>x()</script>'], { type: 'text/html' }))))
  URL.createObjectURL = vi.fn((blob: Blob) => {
    made.push(blob)
    return 'blob:app/1'
  }) as typeof URL.createObjectURL
  URL.revokeObjectURL = vi.fn()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useAuthenticatedBlobUrl', () => {
  it('types a PDF preview as a PDF, whatever the server sent', async () => {
    const { result } = renderHook(() => useAuthenticatedBlobUrl('/api/documents/generated/x.pdf', 'application/pdf'))

    await waitFor(() => expect(result.current.src).toBe('blob:app/1'))
    expect(made[0].type).toBe('application/pdf')
    expect(fetch).toHaveBeenCalledWith('https://api.test/api/documents/generated/x.pdf', {
      headers: { Authorization: 'Bearer t' },
    })
  })

  it('keeps the served type when no type is forced (an image, a video)', async () => {
    const { result } = renderHook(() => useAuthenticatedBlobUrl('/api/documents/generated/x.png'))

    await waitFor(() => expect(result.current.src).toBe('blob:app/1'))
    expect(made[0].type).toBe('text/html')
  })

  it('never sends the token to a path that would leave the API', async () => {
    const { result } = renderHook(() => useAuthenticatedBlobUrl('.evil.example/steal', 'application/pdf'))

    await waitFor(() => expect(result.current.error).toBeTruthy())
    expect(fetch).not.toHaveBeenCalled()
    expect(result.current.src).toBeNull()
  })
})
