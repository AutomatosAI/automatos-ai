/**
 * PRD-261 A-12f — SDK area `pages/documents` (frontend/lib/theme-sdk/pages/documents.ts): the data the Knowledge
 * page (/documents) reads, for a theme's own or cloned knowledge page. Each hook wraps the core hook the page reads,
 * so it asks the same endpoints with the same cache, toasts and invalidation, and hands back a small frozen copy:
 * a query is `{ data, isLoading, isFetching, error, refetch }` (data null until it answers), an action is
 * `{ run, runAsync, isRunning, error, result, reset }`, an error is its message. The query and action hooks need
 * the host's QueryClientProvider above them, as every page the host draws has. What a theme hands an action is
 * checked first: a refused `run` throws a TypeError, a refused `runAsync` or database call rejects with one.
 */

/** One knowledge document, as GET /api/documents/ lists it. */
export interface KnowledgeDocument {
  readonly id: number
  readonly filename: string
  readonly original_filename?: string | null
  readonly file_type?: string | null
  readonly file_size?: number | null
  readonly status: string
  readonly chunk_count?: number | null
  readonly tags?: readonly string[] | null
  readonly description?: string | null
  readonly team_access?: readonly string[] | null
  readonly upload_date: string
  readonly processed_date?: string | null
  readonly source_type?: string | null
  /** How much of the file's text the knowledge base holds (F086), and whether that is under the threshold. */
  readonly kept_pct?: number | null
  readonly partial?: boolean
}

/** The knowledge totals (GET /api/documents/analytics); agent outputs are counted apart, never summed in. */
export interface KnowledgeStats {
  readonly total_documents?: number
  readonly status_distribution?: Readonly<Record<string, number>>
  readonly total_storage_bytes?: number
  readonly total_storage_mb?: number
  readonly file_type_distribution?: Readonly<Record<string, number>>
  readonly total_chunks?: number
  readonly average_chunks_per_document?: number
  readonly recent_uploads_24h?: number
  readonly processing_success_rate?: number
  readonly agent_outputs?: number
  readonly last_updated?: string
}

export interface KnowledgeTeam {
  readonly id: number
  readonly name: string
  readonly normalized_name: string
}

/** Documents per team (keyed by normalized name), untagged ones and the total. */
export interface KnowledgeTeamCounts {
  readonly counts: Readonly<Record<string, number>>
  readonly untagged: number
  readonly total: number
}

export interface KnowledgeCloudConnection {
  readonly id: number
  readonly app_name: string
  readonly status: string
  readonly connected_at?: string
  readonly sync_enabled: boolean
  readonly total_documents_synced: number
  readonly last_successful_sync?: string
  readonly root_folder_path?: string
}

export interface KnowledgeSyncJob {
  readonly job_id: number
  readonly status: string
  readonly message: string
}

export interface KnowledgeUploadInput {
  readonly file: File
  readonly description?: string
  readonly tags?: readonly string[]
  /** The teams that may read it (normalized names); none means the whole workspace. */
  readonly teamAccess?: readonly string[]
}

export interface KnowledgeRootFolderInput {
  readonly connectionId: number
  readonly rootFolderPath: string
}

/** A read: null data until it answers; `isLoading` is the first load only. */
export interface KnowledgeQuery<T> {
  readonly data: T | null
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly error: string | null
  readonly refetch: () => void
}

/** A write: `run` reports failure in `error`; `runAsync` rejects with it. */
export interface KnowledgeAction<V, R> {
  readonly run: (variables: V) => void
  readonly runAsync: (variables: V) => Promise<R>
  readonly isRunning: boolean
  readonly error: string | null
  readonly result: R | null
  readonly reset: () => void
}

export interface KnowledgeDatabaseSource {
  readonly id: number
  readonly name: string
  readonly dialect: string
  readonly status: string
  readonly tables_count: number
  readonly last_synced: string
  readonly credential_id: number | null
}

export interface KnowledgeQueryTemplate {
  readonly id: number
  readonly name: string
  readonly natural_language: string
  readonly category: string
  readonly sql_template: string
  readonly parameters: readonly unknown[]
  readonly visualization_type: string
}

export interface KnowledgeDatabaseSourceInput {
  readonly name: string
  readonly credentialId: number
  readonly description?: string
}

export interface KnowledgeDatabaseQueryOptions {
  readonly validate?: boolean
  readonly useCache?: boolean
}

export interface KnowledgeDatabaseQueryResult {
  readonly sql: string
  readonly results: readonly unknown[]
  readonly validation: unknown
  readonly execution_time: number
}

export interface KnowledgeCacheStats {
  readonly schema_cache_hits: number
  readonly schema_cache_misses: number
  readonly query_cache_hits: number
  readonly query_cache_misses: number
  readonly schema_hit_rate: number
  readonly query_hit_rate: number
}

/** The workspace's database sources and query templates, and what the Knowledge page does with them. */
export interface KnowledgeDatabase {
  readonly sources: readonly KnowledgeDatabaseSource[]
  readonly templates: readonly KnowledgeQueryTemplate[]
  readonly isLoading: boolean
  readonly error: string | null
  readonly refresh: () => Promise<void>
  readonly createSource: (input: KnowledgeDatabaseSourceInput) => Promise<unknown>
  readonly deleteSource: (sourceId: number) => Promise<void>
  readonly syncSchema: (sourceId: number) => Promise<void>
  readonly executeQuery: (
    sourceId: number,
    question: string,
    options?: KnowledgeDatabaseQueryOptions,
  ) => Promise<KnowledgeDatabaseQueryResult>
  readonly getSchemaMetadata: (sourceId: number) => Promise<unknown>
  readonly getCacheStats: () => Promise<KnowledgeCacheStats>
}

/** The workspace's knowledge documents, or one team's (`team` is a team's normalized name). */
export declare function useDocuments(team?: string | null): KnowledgeQuery<readonly KnowledgeDocument[]>
/** The knowledge totals the page's header reads. */
export declare function useDocumentStats(): KnowledgeQuery<KnowledgeStats>
/** Upload one file into the knowledge base; the list and totals refresh when it lands. */
export declare function useUploadDocument(): KnowledgeAction<KnowledgeUploadInput, unknown>
/** Delete a document; `run(documentId)` (one path segment). The list and totals refresh. */
export declare function useDeleteDocument(): KnowledgeAction<string | number, unknown>
/** The workspace's teams (the page's team filter). */
export declare function useTeams(): KnowledgeQuery<readonly KnowledgeTeam[]>
/** How many documents each team can read. */
export declare function useDocumentTeamCounts(): KnowledgeQuery<KnowledgeTeamCounts>
/** The connected cloud folders and their sync state. */
export declare function useCloudConnections(): KnowledgeQuery<readonly KnowledgeCloudConnection[]>
/** Sync a cloud connection now; `run(connectionId)` (a positive whole number). */
export declare function useTriggerSync(): KnowledgeAction<number, KnowledgeSyncJob>
/** Choose the folder a cloud connection syncs from. */
export declare function useSelectRootFolder(): KnowledgeAction<KnowledgeRootFolderInput, unknown>
/** The workspace's database sources and query templates, loaded on mount, and the page's calls on them. */
export declare function useDatabaseKnowledge(): KnowledgeDatabase
