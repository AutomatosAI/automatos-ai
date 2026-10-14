/**
 * The bytes of a file route on the API, fetched with the API client's auth headers.
 *
 * A stored file (a brand logo or font, a skin's font) sits behind auth that a plain
 * <img src>, @font-face or CSS url() cannot send in SaaS, so it is fetched here and handed
 * to the caller as a Blob: an object URL for an image, an ArrayBuffer for the FontFace API.
 * The path must be a path on the API's own origin (apiFileUrl, F358), so the token never
 * leaves it. null when the path is not one, or the route answers with an error.
 */
import { apiClient } from '@/lib/api-client'
import { apiFileUrl } from '@/lib/api-file-url'

export async function fetchApiFile(path: string): Promise<Blob | null> {
  const url = apiFileUrl(apiClient.getBaseUrl(), path)
  if (!url) return null
  const headers = await apiClient.getAuthHeaders()
  const resp = await fetch(url, { headers })
  if (!resp.ok) return null
  return resp.blob()
}
