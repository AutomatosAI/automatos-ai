/**
 * PRD-261 A-12f — SDK area `pages/documents`: the data the Knowledge page (/documents) reads, for a
 * theme's own (or cloned) knowledge page (PRD §4.3 item 7). The names are A-10c's
 * (surface/pages-documents.json, docs/theme-host/sdk-surface-map.md).
 *
 * Each hook wraps the core hook DocumentManagement reads today (hooks/use-document-api.ts,
 * hooks/use-teams.ts, hooks/use-database-knowledge.ts, hooks/use-cloud-storage.ts), so a theme's
 * page asks the same endpoints, through the same backend client, with the same cache keys, toasts
 * and invalidation as the core page; no core hook is added or changed. `usePageAPI` stays host-only
 * (the map's reason: the host's page layout tags the page's requests).
 *
 * What comes back is a small, frozen, theme-neutral copy, never react-query's own result or its
 * cached objects: a query is `{ data, isLoading, isFetching, error, refetch }` (data null until it
 * answers; each row a frozen copy of the fields the type names), an action is
 * `{ run, runAsync, isRunning, error, result, reset }`, and an error is its message. The query and
 * action hooks read react-query, so they need the host's QueryClientProvider above them, as every
 * page and theme the host draws has (components/providers.tsx).
 *
 * A theme is outside code, so what it hands an action is checked before it reaches a URL or a
 * form: a document id is one path segment, a connection or database source id a positive whole
 * number, an upload a File with string metadata. A refused `run` throws a TypeError (as the
 * navigation area does for a bad href); a refused `runAsync` or database call rejects with it.
 */
import { useMemo, useRef } from 'react'

import {
  useCloudConnections as useCoreCloudConnections,
  useSelectRootFolder as useCoreSelectRootFolder,
  useTriggerSync as useCoreTriggerSync,
} from '@/hooks/use-cloud-storage'
import { useDatabaseKnowledge as useCoreDatabaseKnowledge } from '@/hooks/use-database-knowledge'
import {
  useDeleteDocument as useCoreDeleteDocument,
  useDocumentStats as useCoreDocumentStats,
  useDocuments as useCoreDocuments,
  useUploadDocument as useCoreUploadDocument,
} from '@/hooks/use-document-api'
import { useDocumentTeamCounts as useCoreDocumentTeamCounts, useTeams as useCoreTeams } from '@/hooks/use-teams'

import {
  checkedId,
  checkedText,
  listOf,
  recordOf,
  refuse,
  rowsOf,
  textList,
  useActionView,
  useQueryView,
  type Prepare,
} from '../internal/documents-view'

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

interface CoreUploadVariables {
  readonly file: File
  readonly metadata: { readonly description: string; readonly tags: string[]; readonly team_access: string[] }
}

type CoreDatabase = ReturnType<typeof useCoreDatabaseKnowledge>

/** A document id is one path segment (DELETE /api/documents/<id>). */
const DOCUMENT_ID = /^[A-Za-z0-9_-]{1,128}$/
const DOCUMENT_FIELDS = [
  'id', 'filename', 'original_filename', 'file_type', 'file_size', 'status', 'chunk_count', 'tags', 'description',
  'team_access', 'upload_date', 'processed_date', 'source_type', 'kept_pct', 'partial',
]
const STATS_FIELDS = [
  'total_documents', 'status_distribution', 'total_storage_bytes', 'total_storage_mb', 'file_type_distribution',
  'total_chunks', 'average_chunks_per_document', 'recent_uploads_24h', 'processing_success_rate', 'agent_outputs',
  'last_updated',
]
const TEAM_FIELDS = ['id', 'name', 'normalized_name']
const TEAM_COUNT_FIELDS = ['counts', 'untagged', 'total']
const CONNECTION_FIELDS = [
  'id', 'app_name', 'status', 'connected_at', 'sync_enabled', 'total_documents_synced', 'last_successful_sync',
  'root_folder_path',
]
const SOURCE_FIELDS = ['id', 'name', 'dialect', 'status', 'tables_count', 'last_synced', 'credential_id']
const TEMPLATE_FIELDS = [
  'id', 'name', 'natural_language', 'category', 'sql_template', 'parameters', 'visualization_type',
]

const shapeDocuments = listOf<KnowledgeDocument>(DOCUMENT_FIELDS)
const shapeStats = recordOf<KnowledgeStats>(STATS_FIELDS)
const shapeTeams = listOf<KnowledgeTeam>(TEAM_FIELDS)
const shapeTeamCounts = recordOf<KnowledgeTeamCounts>(TEAM_COUNT_FIELDS)
const shapeConnections = listOf<KnowledgeCloudConnection>(CONNECTION_FIELDS)

const prepareDocumentId: Prepare<string | number, string> = (documentId) => {
  const id = typeof documentId === 'number' ? String(documentId) : documentId
  return typeof id === 'string' && DOCUMENT_ID.test(id) ? id : refuse(`${String(documentId)} is not a document id.`)
}

const prepareUpload: Prepare<KnowledgeUploadInput, CoreUploadVariables> = (input) => {
  if (typeof File === 'undefined' || !(input?.file instanceof File)) refuse('an upload needs a File.')
  const description = input.description === undefined ? '' : input.description
  if (typeof description !== 'string') refuse('description must be text.')
  const metadata = { description, tags: textList(input.tags, 'tags'), team_access: textList(input.teamAccess, 'teamAccess') }
  return { file: input.file, metadata }
}

const prepareConnectionId: Prepare<number, number> = (connectionId) => checkedId(connectionId, 'connection')

const prepareRootFolder: Prepare<KnowledgeRootFolderInput, KnowledgeRootFolderInput> = (input) => ({
  connectionId: checkedId(input?.connectionId, 'connection'),
  rootFolderPath: checkedText(input?.rootFolderPath, 'rootFolderPath'),
})

/** The workspace's knowledge documents, or one team's (`team` is a team's normalized name). */
export function useDocuments(team?: string | null): KnowledgeQuery<readonly KnowledgeDocument[]> {
  const filter = typeof team === 'string' && team.length > 0 ? team : undefined
  return useQueryView(useCoreDocuments(filter), shapeDocuments)
}

/** The knowledge totals the page's header reads. */
export function useDocumentStats(): KnowledgeQuery<KnowledgeStats> {
  return useQueryView(useCoreDocumentStats(), shapeStats)
}

/** Upload one file into the knowledge base; the list and totals refresh when it lands. */
export function useUploadDocument(): KnowledgeAction<KnowledgeUploadInput, unknown> {
  return useActionView(useCoreUploadDocument(), prepareUpload)
}

/** Delete a document; `run(documentId)`. The list and totals refresh. */
export function useDeleteDocument(): KnowledgeAction<string | number, unknown> {
  return useActionView(useCoreDeleteDocument(), prepareDocumentId)
}

/** The workspace's teams (the page's team filter). */
export function useTeams(): KnowledgeQuery<readonly KnowledgeTeam[]> {
  return useQueryView(useCoreTeams(), shapeTeams)
}

/** How many documents each team can read. */
export function useDocumentTeamCounts(): KnowledgeQuery<KnowledgeTeamCounts> {
  return useQueryView(useCoreDocumentTeamCounts(), shapeTeamCounts)
}

/** The connected cloud folders (Drive, Dropbox, …) and their sync state. */
export function useCloudConnections(): KnowledgeQuery<readonly KnowledgeCloudConnection[]> {
  return useQueryView(useCoreCloudConnections(), shapeConnections)
}

/** Sync a cloud connection now; `run(connectionId)`. */
export function useTriggerSync(): KnowledgeAction<number, KnowledgeSyncJob> {
  return useActionView(useCoreTriggerSync(), prepareConnectionId)
}

/** Choose the folder a cloud connection syncs from. */
export function useSelectRootFolder(): KnowledgeAction<KnowledgeRootFolderInput, unknown> {
  return useActionView(useCoreSelectRootFolder(), prepareRootFolder)
}

/** The query flags the core hook sends: only the ones the theme set, and only as booleans. */
function queryFlags(options: KnowledgeDatabaseQueryOptions | undefined): { validate?: boolean; use_cache?: boolean } {
  const validate = typeof options?.validate === 'boolean' ? { validate: options.validate } : {}
  const useCache = typeof options?.useCache === 'boolean' ? { use_cache: options.useCache } : {}
  return { ...validate, ...useCache }
}

/** The database calls, stable for the hook's life: each reads the core hook's latest render. */
function databaseActions(latest: { readonly current: CoreDatabase }) {
  return {
    refresh: async () => latest.current.fetchSources(),
    createSource: async (input: KnowledgeDatabaseSourceInput): Promise<unknown> => {
      const name = checkedText(input?.name, 'name')
      const credential_id = checkedId(input?.credentialId, 'credential')
      const description = typeof input?.description === 'string' ? input.description : undefined
      return latest.current.createSource({ name, credential_id, description })
    },
    deleteSource: async (sourceId: number) => latest.current.deleteSource(checkedId(sourceId, 'database source')),
    syncSchema: async (sourceId: number) => latest.current.syncSchema(checkedId(sourceId, 'database source')),
    executeQuery: async (
      sourceId: number,
      question: string,
      options?: KnowledgeDatabaseQueryOptions,
    ): Promise<KnowledgeDatabaseQueryResult> => {
      const id = checkedId(sourceId, 'database source')
      return latest.current.executeQuery(id, checkedText(question, 'question'), queryFlags(options))
    },
    getSchemaMetadata: async (sourceId: number): Promise<unknown> =>
      latest.current.getSchemaMetadata(checkedId(sourceId, 'database source')),
    getCacheStats: async (): Promise<KnowledgeCacheStats> => latest.current.getCacheStats(),
  }
}

/** The workspace's database sources and query templates, loaded on mount, and the page's calls on them. */
export function useDatabaseKnowledge(): KnowledgeDatabase {
  const core = useCoreDatabaseKnowledge()
  const latest = useRef(core)
  latest.current = core
  const actions = useMemo(() => databaseActions(latest), [])
  const { sources, templates, loading, error } = core
  return useMemo(
    () =>
      Object.freeze({
        ...actions,
        sources: rowsOf<KnowledgeDatabaseSource>(sources, SOURCE_FIELDS),
        templates: rowsOf<KnowledgeQueryTemplate>(templates, TEMPLATE_FIELDS),
        isLoading: loading,
        error,
      }),
    [actions, sources, templates, loading, error],
  )
}
