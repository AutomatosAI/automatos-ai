import type { Deliverable } from '@/hooks/use-deliverables-api'

/**
 * A Deliverable's file as the Deliverables panel hands it over (F358): the file
 * route the detail answered with (content_url, with include_content=true), else
 * its preview_url. Both are API paths, fetched with the API client's auth
 * (useApiFileDownload, useAuthenticatedBlobUrl), never a plain link. Shared by
 * the core's preview (DeliverablePreview) and Play's Deliverable page.
 */

/** The name a file saves under when the Deliverable names none. */
export const DEFAULT_FILENAME = 'download'

/** The API path of the Deliverable's file, or null when it has none. */
export function deliverableFileUrl(deliverable: Pick<Deliverable, 'content_url' | 'preview_url'>): string | null {
  return deliverable.content_url || deliverable.preview_url || null
}

/** The file's own name, else the last segment of its path. */
export function deliverableFilename(deliverable: Pick<Deliverable, 'file_name' | 'file_path'>): string {
  return deliverable.file_name || deliverable.file_path?.split('/').pop() || DEFAULT_FILENAME
}
