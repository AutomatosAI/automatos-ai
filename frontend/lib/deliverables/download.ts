import type { Deliverable } from '@/hooks/use-deliverables-api'
import { apiClient } from '@/lib/api-client'

/**
 * A Deliverable's file as the Deliverables panel hands it over (F358): the file
 * route the detail answered with (content_url, with include_content=true), else
 * its preview_url. Both are API paths, fetched with the API client's auth
 * (useApiFileDownload, useAuthenticatedBlobUrl), never a plain link. Shared by
 * the core's preview (DeliverablePreview) and Play's Deliverable page.
 *
 * Download takes the file's bytes (P260-FX-12, M49): the backend names a text
 * file (.md, .csv, .py…) by its content route, which answers JSON for the code
 * viewer, so Download swaps it for the same file's raw route. What a page shows
 * is still the file route as the detail answered it.
 */

/** The name a file saves under when the Deliverable names none. */
export const DEFAULT_FILENAME = 'download'

/** What a failed Download says on a Deliverable's page (Play's): the user is already where they would try again. */
export const DOWNLOAD_FAILED = "Couldn't download the file. Try again in a moment."

/** A workspace file's content route (JSON): the workspace id, then the path as it is encoded in the query. */
const CONTENT_ROUTE = /^\/api\/workspaces\/([^/]+)\/files\/content\?path=([^&#]*)/

/** The API path of the Deliverable's file, or null when it has none. */
export function deliverableFileUrl(deliverable: Pick<Deliverable, 'content_url' | 'preview_url'>): string | null {
  return deliverable.content_url || deliverable.preview_url || null
}

/** The path a query carries, decoded once; null when it isn't valid percent-encoding. */
function decodedPath(encoded: string): string | null {
  try {
    return decodeURIComponent(encoded)
  } catch (error) {
    console.warn("[deliverables] a file route's path isn't valid percent-encoding, so Download keeps the route:", error)
    return null
  }
}

/** The route Download fetches for a file route: a content route becomes the file's raw route; any other is unchanged. */
export function fileDownloadUrl(url: string): string {
  const match = CONTENT_ROUTE.exec(url)
  const path = match ? decodedPath(match[2]) : null
  return match && path !== null ? apiClient.getWorkspaceFileRawUrl(match[1], path) : url
}

/** What Download fetches for the Deliverable: its file's bytes, never the content route's JSON; null when it has no file. */
export function deliverableDownloadUrl(deliverable: Pick<Deliverable, 'content_url' | 'preview_url'>): string | null {
  const url = deliverableFileUrl(deliverable)
  return url ? fileDownloadUrl(url) : null
}

/** The file's own name, else the last segment of its path. */
export function deliverableFilename(deliverable: Pick<Deliverable, 'file_name' | 'file_path'>): string {
  return deliverable.file_name || deliverable.file_path?.split('/').pop() || DEFAULT_FILENAME
}
