/**
 * PRD-261 A-12a — SDK area `deliverables` (frontend/lib/theme-sdk/deliverables.ts): the Deliverables list, one
 * Deliverable and its files, the feed's filters, the page's tabs and their labels, and a file the backend serves
 * only with the caller's token. The hooks are the host's own, so every read carries the workspace and the token,
 * which a theme never holds; each hands back react-query's result, read only. The constants are frozen.
 */

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

/** The feed's default filters: every Deliverable but a heartbeat's. Frozen. */
export declare const DELIVERABLE_FEED_FILTERS: DeliverableFilters

/** Each Deliverables tab's name, as the host's tab strip shows it. Frozen. */
export declare const DELIVERABLE_TAB_LABELS: Readonly<Record<DeliverableTab, string>>

/** This workspace's Deliverables, a page at a time (every Deliverable when `filters` is omitted). */
export declare function useDeliverables(filters?: DeliverableFilters, options?: DeliverablesOptions): DeliverablesQuery

/**
 * One Deliverable; with `includeContent`, its text and its file's API path. Idle while `id` is null,
 * and for an id that is not a Deliverable's (letters, digits, `-` and `_`, 1 to 128 of them): a
 * ?deliverable= value handed over as it came that is not one never reaches the backend.
 */
export declare function useDeliverable(id: string | null, includeContent?: boolean): DeliverableQuery

/** Downloads an API file with the caller's token; a failed download shows `failedMessage` as a notice. */
export declare function useFileDownload(failedMessage?: string): FileDownload

/** A file the backend serves only with the caller's token, as a blob: URL (`forcedType` types its bytes). */
export declare function useAuthenticatedBlobUrl(url: string | undefined, forcedType?: string): AuthenticatedBlobUrl

/** The API path of the Deliverable's file, or null when it has none. */
export declare function deliverableFileUrl(deliverable: DeliverableFileRef): string | null

/** What Download fetches for the Deliverable: its file's bytes, never a content route's JSON; null with no file. */
export declare function deliverableDownloadUrl(deliverable: DeliverableFileRef): string | null

/** The route Download fetches for a file route: a content route becomes the file's raw route. */
export declare function fileDownloadUrl(url: string): string

/** The file's own name, else the last segment of its path. */
export declare function deliverableFilename(deliverable: DeliverableNameRef): string

/** The Deliverables tabs this workspace shows: Socials only while the platform offers it. */
export declare function visibleDeliverableTabs(socialsAvailable: boolean): DeliverableTab[]

/** A kind's name ("Reports"); a kind the host has no name for is named as it is. */
export declare function deliverableLabel(kind: string): string
