/**
 * PRD-261 A-12a — SDK area `deliverables`: the Deliverables list, one Deliverable and its files, the
 * feed's filters, the page's tabs and their labels, and a file the backend serves only with the
 * caller's token (docs/theme-host/sdk-surface-map.md, A-10c).
 *
 * The hooks are the host's own (hooks/use-deliverables-api.ts, hooks/use-api-file-download.ts,
 * FilePreview's useAuthenticatedBlobUrl), so every read goes through the backend client with the
 * workspace and the token, which a theme never holds. Each hands back react-query's result typed as
 * the SDK's read-only view of it (DeliverablesQuery, DeliverableQuery). The constants are frozen
 * copies, so a theme cannot change the host's own. The types are declared again for themes in
 * packages/theme-sdk/areas/deliverables.d.ts.
 */
import { deliverableLabel as hostLabel, isDeliverableType } from '@/components/icons/deliverable-icon'
import { useAuthenticatedBlobUrl as useHostBlobUrl } from '@/components/widgets/FileWidget/FilePreview'
import { useApiFileDownload } from '@/hooks/use-api-file-download'
import {
  DEFAULT_FILTERS,
  FEED_DEFAULT_FILTERS,
  useDeliverable as useHostDeliverable,
  useDeliverables as useHostDeliverables,
  type FilterState,
} from '@/hooks/use-deliverables-api'
import { isDeliverableId } from '@/lib/chat/deliverables'
import { TAB_LABELS, visibleDeliverableTabs as hostVisibleTabs } from '@/lib/deliverables/tabs'

export { deliverableDownloadUrl, deliverableFilename, deliverableFileUrl, fileDownloadUrl } from '@/lib/deliverables/download'

export type DeliverableTab = 'outputs' | 'blogs' | 'templates' | 'brand' | 'socials'
export type DeliverableDateRange = 'all' | 'today' | 'week' | 'month'

export interface Deliverable {
  readonly id: string
  readonly workspace_id: string
  readonly source_type: string
  readonly source_id: string | null
  readonly agent_id: number | null
  readonly agent_name: string | null
  readonly artifact_type: string
  readonly title: string
  readonly summary: string | null
  readonly storage_type: string
  readonly file_path: string
  readonly file_name: string | null
  readonly file_type: string | null
  readonly file_size_bytes: number | null
  readonly preview_url: string | null
  readonly preview_type: string | null
  readonly thumbnail_url?: string | null
  readonly extra: Readonly<Record<string, unknown>>
  readonly status: string
  readonly created_at: string
  readonly updated_at: string
  /** With `includeContent`: the text of a text, code or report file. */
  readonly content?: string | null
  /** With `includeContent`: the API path of the file. */
  readonly content_url?: string | null
  readonly content_error?: string
  /** True when `content` is the first part only (the file is over the inline cap). */
  readonly content_truncated?: boolean
  readonly knowledge_document_id?: number | null
  readonly tags?: readonly string[]
}

export interface DeliverableListResponse {
  readonly success: boolean
  readonly deliverables: readonly Deliverable[]
  readonly total: number
  readonly limit: number
  readonly offset: number
  readonly error?: string
}

export interface DeliverableDetailResponse {
  readonly success: boolean
  readonly deliverable: Deliverable
  readonly error?: string
}

export interface DeliverableFilters {
  readonly artifact_type: string | null
  readonly source_type: string | null
  readonly source_type_exclude: readonly string[] | null
  /** One originating Mission, task or heartbeat. */
  readonly source_id: string | null
  readonly agent_id: number | null
  readonly date_range: DeliverableDateRange
  readonly search: string
  /** Only Deliverables carrying this tag (any case). */
  readonly tag?: string | null
}

export interface DeliverablesOptions {
  /** Ask for the loaded pages again every so many ms while the tab is visible; never when unset. */
  readonly refetchInterval?: number
}

/** The list, a page at a time: react-query's infinite result, read only. */
export interface DeliverablesQuery {
  readonly data: { readonly pages: readonly DeliverableListResponse[]; readonly pageParams: readonly unknown[] } | undefined
  readonly isLoading: boolean
  readonly isError: boolean
  readonly error: unknown
  readonly hasNextPage?: boolean
  readonly isFetchingNextPage: boolean
  readonly fetchNextPage: () => Promise<unknown>
  readonly refetch: () => Promise<unknown>
}

/** One Deliverable: react-query's result, read only. */
export interface DeliverableQuery {
  readonly data: DeliverableDetailResponse | undefined
  readonly isLoading: boolean
  readonly isError: boolean
  readonly error: unknown
  readonly refetch: () => Promise<unknown>
}

export interface FileDownload {
  /** Fetches `url` (an API path) with the caller's token and saves it as `filename`. */
  readonly download: (url: string, filename: string) => Promise<void>
  readonly downloading: boolean
}

export interface AuthenticatedBlobUrl {
  /** A blob: URL of the file, once loaded. */
  readonly src: string | null
  readonly error: string | null
}

/** The fields a file's URL is read from. */
export type DeliverableFileRef = Pick<Deliverable, 'content_url' | 'preview_url'>
/** The fields a file's name is read from. */
export type DeliverableNameRef = Pick<Deliverable, 'file_name' | 'file_path'>

/** The host's filters as a copy the host may keep. */
function hostFilters(filters: DeliverableFilters): FilterState {
  const exclude = filters.source_type_exclude
  return { ...filters, source_type_exclude: exclude === null ? null : [...exclude] }
}

/** A frozen copy of the host's filters. */
function frozenFilters(filters: FilterState): DeliverableFilters {
  const exclude = filters.source_type_exclude
  return Object.freeze({ ...filters, source_type_exclude: exclude === null ? null : Object.freeze([...exclude]) })
}

/** The feed's default filters: every Deliverable but a heartbeat's (the host's feed hides those). */
export const DELIVERABLE_FEED_FILTERS: DeliverableFilters = frozenFilters(FEED_DEFAULT_FILTERS)

/** Each Deliverables tab's name, as the host's tab strip shows it. */
export const DELIVERABLE_TAB_LABELS: Readonly<Record<DeliverableTab, string>> = Object.freeze({ ...TAB_LABELS })

/** The Deliverables tabs this workspace shows: Socials only while the platform offers it. */
export function visibleDeliverableTabs(socialsAvailable: boolean): DeliverableTab[] {
  return hostVisibleTabs(socialsAvailable)
}

/** A kind's name ("Reports"); a kind the host has no name for is named as it is. */
export function deliverableLabel(kind: string): string {
  return isDeliverableType(kind) ? hostLabel(kind) : kind
}

/** This workspace's Deliverables, a page at a time (every Deliverable when `filters` is omitted). */
export function useDeliverables(filters: DeliverableFilters = DEFAULT_FILTERS, options?: DeliverablesOptions): DeliverablesQuery {
  return useHostDeliverables(hostFilters(filters), options)
}

/**
 * The id the host may ask the backend for: a Deliverable id (the core's pattern, isDeliverableId),
 * else null. A theme often hands over a ?deliverable= link's value as it came; one that is not an id
 * ('../../admin?x=1') never becomes a request path.
 */
function askableId(id: unknown): string | null {
  return typeof id === 'string' && isDeliverableId(id) ? id : null
}

/**
 * One Deliverable; with `includeContent`, its text and its file's API path. Idle while `id` is null,
 * and for an id that is not a Deliverable's, which never reaches the backend.
 */
export function useDeliverable(id: string | null, includeContent = false): DeliverableQuery {
  return useHostDeliverable(askableId(id), includeContent)
}

/** Downloads an API file with the caller's token; a failed download shows `failedMessage` as a notice. */
export function useFileDownload(failedMessage?: string): FileDownload {
  return useApiFileDownload(failedMessage)
}

/** A file the backend serves only with the caller's token, as a blob: URL (`forcedType` types its bytes). */
export function useAuthenticatedBlobUrl(url: string | undefined, forcedType?: string): AuthenticatedBlobUrl {
  return useHostBlobUrl(url, forcedType)
}
