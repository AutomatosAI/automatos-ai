/**
 * PRD-261 A-12f — the SDK's pages/documents area wraps the core hooks the Knowledge page
 * (/documents, DocumentManagement) reads: the same endpoints through the host's backend client,
 * handed to a theme as small frozen copies (never react-query's result or its cached objects),
 * with a theme's inputs checked before they reach a URL or a form. The backend client is mocked;
 * react-query and the core hooks are real.
 */
import type { ReactNode } from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const client = vi.hoisted(() => ({
  request: vi.fn(),
  get: vi.fn(),
  post: vi.fn(),
  getDocuments: vi.fn(),
  getDocumentAnalytics: vi.fn(),
  getTeams: vi.fn(),
  getDocumentTeamCounts: vi.fn(),
  uploadDocument: vi.fn(),
  deleteDocument: vi.fn(),
}))

vi.mock('@/lib/api-client', () => ({ apiClient: client, default: client }))

import * as Sdk from '@automatos/theme-sdk'
import {
  useCloudConnections,
  useDatabaseKnowledge,
  useDeleteDocument,
  useDocumentStats,
  useDocumentTeamCounts,
  useDocuments,
  useSelectRootFolder,
  useTeams,
  useTriggerSync,
  useUploadDocument,
} from '../pages/documents'

/** A document as the SDK hands it on: only the fields KnowledgeDocument names. */
const NAMED_DOCUMENT = {
  id: 41,
  filename: 'handbook.pdf',
  status: 'completed',
  chunk_count: 12,
  team_access: ['finance'],
  upload_date: '2026-10-19T09:00:00',
  kept_pct: 100,
  partial: false,
}
/** The same document as the backend lists it, with a field the SDK does not name. */
const DOCUMENT = { ...NAMED_DOCUMENT, content_hash: 'not-a-field-the-sdk-names' }
const STATS = { total_documents: 1, status_distribution: { completed: 1 }, agent_outputs: 2, internal_cost: 9 }
const TEAMS = [{ id: 3, name: 'Finance', normalized_name: 'finance' }]
const TEAM_COUNTS = { counts: { finance: 1 }, untagged: 0, total: 1 }
const CONNECTIONS = [{ id: 7, app_name: 'googledrive', status: 'active', sync_enabled: true, total_documents_synced: 4 }]
const SOURCES_PATH = '/api/knowledge/sources/database/'
const TEMPLATES_PATH = '/api/knowledge/sources/database/templates/list'
const RAW_SOURCE = { id: 5, name: 'Sales DB', dialect: 'postgresql', status: 'active', schema_tables_count: 8, credential_id: 2 }
const QUERY_VIEW_KEYS = ['data', 'error', 'isFetching', 'isLoading', 'refetch']
const ACTION_VIEW_KEYS = ['error', 'isRunning', 'reset', 'result', 'run', 'runAsync']
const SDK_REFUSAL = /^Automatos theme SDK: /
const SILENT = { log: () => undefined, warn: () => undefined, error: () => undefined }

function wrapper() {
  const queryClient = new QueryClient({ logger: SILENT, defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return function Providers({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  }
}

/** The database endpoints: the sources and the templates on mount; DELETE and the query answered too. */
function answerDatabase(url: string, options?: { method?: string }) {
  if (options?.method === 'DELETE') return Promise.resolve(null)
  if (options?.method === 'POST') return Promise.resolve({ sql: 'select 1', results: [], validation: null, execution_time: 3 })
  if (url === SOURCES_PATH) return Promise.resolve([RAW_SOURCE])
  if (url === TEMPLATES_PATH) return Promise.resolve([])
  return Promise.reject(new Error(`unexpected ${url}`))
}

beforeEach(() => {
  Object.values(client).forEach((mock) => mock.mockReset())
})

describe('PRD-261 A-12f — the area is the one the SDK exports', () => {
  it('exports the same hooks by the SDK name', () => {
    expect(Sdk.useDocuments).toBe(useDocuments)
    expect(Sdk.useDeleteDocument).toBe(useDeleteDocument)
    expect(Sdk.useDatabaseKnowledge).toBe(useDatabaseKnowledge)
  })
})

describe('PRD-261 A-12f — queries', () => {
  it('reads the documents through the backend client and hands back frozen copies of the named fields', async () => {
    client.getDocuments.mockResolvedValue([DOCUMENT])
    const { result } = renderHook(() => useDocuments(), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.data).not.toBeNull())
    expect(client.getDocuments).toHaveBeenCalledWith(undefined)
    expect(Object.keys(result.current).sort()).toEqual(QUERY_VIEW_KEYS)
    expect(Object.isFrozen(result.current)).toBe(true)
    const rows = result.current.data ?? []
    expect(rows).toEqual([NAMED_DOCUMENT])
    expect(rows[0]).not.toBe(DOCUMENT)
    expect(Object.isFrozen(rows)).toBe(true)
    expect(Object.isFrozen(rows[0])).toBe(true)
    expect(rows[0].team_access).not.toBe(DOCUMENT.team_access)
    expect(Object.isFrozen(rows[0].team_access)).toBe(true)
  })

  it("asks for one team's documents, and ignores a team that is not text", async () => {
    client.getDocuments.mockResolvedValue([])
    const { result } = renderHook(() => useDocuments('finance'), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toEqual([]))
    expect(client.getDocuments).toHaveBeenCalledWith('finance')

    const odd = renderHook(() => useDocuments({ team: 'x' } as unknown as string), { wrapper: wrapper() })
    await waitFor(() => expect(odd.result.current.data).toEqual([]))
    expect(client.getDocuments).toHaveBeenLastCalledWith(undefined)
  })

  it('reads an answer that is not a list as an empty list', async () => {
    client.getDocuments.mockResolvedValue({ detail: 'odd' })
    const { result } = renderHook(() => useDocuments(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toEqual([]))
  })

  it('reads the knowledge totals, keeping only the fields the type names', async () => {
    client.getDocumentAnalytics.mockResolvedValue(STATS)
    const { result } = renderHook(() => useDocumentStats(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toEqual({ total_documents: 1, status_distribution: { completed: 1 }, agent_outputs: 2 }))
    expect(Object.isFrozen(result.current.data?.status_distribution)).toBe(true)
  })

  it('reads the teams and the per-team counts', async () => {
    client.getTeams.mockResolvedValue(TEAMS)
    client.getDocumentTeamCounts.mockResolvedValue(TEAM_COUNTS)
    const teams = renderHook(() => useTeams(), { wrapper: wrapper() })
    const counts = renderHook(() => useDocumentTeamCounts(), { wrapper: wrapper() })
    await waitFor(() => expect(teams.result.current.data).toEqual(TEAMS))
    await waitFor(() => expect(counts.result.current.data).toEqual(TEAM_COUNTS))
  })

  it("hands back a failed read's message, not the error object", async () => {
    client.getTeams.mockRejectedValue(new Error('Teams are down'))
    const { result } = renderHook(() => useTeams(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.error).toBe('Teams are down'))
    expect(result.current.data).toBeNull()
  })

  it('reads the cloud connections from their endpoint', async () => {
    client.get.mockResolvedValue(CONNECTIONS)
    const { result } = renderHook(() => useCloudConnections(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toEqual(CONNECTIONS))
    expect(client.get).toHaveBeenCalledWith('/api/cloud-documents/connections')
  })
})

describe('PRD-261 A-12f — document actions', () => {
  it('deletes a document by its id through the backend client', async () => {
    client.deleteDocument.mockResolvedValue(null)
    const { result } = renderHook(() => useDeleteDocument(), { wrapper: wrapper() })
    expect(Object.keys(result.current).sort()).toEqual(ACTION_VIEW_KEYS)

    await act(async () => {
      await result.current.runAsync(41)
    })
    expect(client.deleteDocument).toHaveBeenCalledWith('41')
  })

  it('refuses a document id that is not one path segment, and sends nothing', async () => {
    const { result } = renderHook(() => useDeleteDocument(), { wrapper: wrapper() })
    expect(() => result.current.run('../teams')).toThrow(TypeError)
    expect(() => result.current.run('41?all=1')).toThrow(SDK_REFUSAL)
    await expect(result.current.runAsync('')).rejects.toThrow(TypeError)
    expect(client.deleteDocument).not.toHaveBeenCalled()
  })

  it('uploads a File with only text metadata, as the core page sends it', async () => {
    client.uploadDocument.mockResolvedValue({ id: 42, status: 'processing' })
    const file = new File(['hello'], 'notes.txt', { type: 'text/plain' })
    const { result } = renderHook(() => useUploadDocument(), { wrapper: wrapper() })

    await act(async () => {
      await result.current.runAsync({ file, tags: ['notes'], teamAccess: ['finance'] })
    })
    expect(client.uploadDocument).toHaveBeenCalledWith(file, { description: '', tags: ['notes'], team_access: ['finance'] })
    await waitFor(() => expect(result.current.result).toEqual({ id: 42, status: 'processing' }))
  })

  it('refuses an upload that is not a File or whose metadata is not text', () => {
    const file = new File(['hello'], 'notes.txt')
    const { result } = renderHook(() => useUploadDocument(), { wrapper: wrapper() })
    expect(() => result.current.run({ file: 'notes.txt' } as unknown as Sdk.KnowledgeUploadInput)).toThrow(TypeError)
    expect(() => result.current.run({ file, tags: [1] } as unknown as Sdk.KnowledgeUploadInput)).toThrow(SDK_REFUSAL)
    expect(() => result.current.run({ file, description: 7 } as unknown as Sdk.KnowledgeUploadInput)).toThrow(TypeError)
    expect(client.uploadDocument).not.toHaveBeenCalled()
  })
})

describe('PRD-261 A-12f — cloud actions', () => {
  it('syncs a connection by its id, and refuses one that is not a positive whole number', async () => {
    client.post.mockResolvedValue({ job_id: 9, status: 'queued', message: 'Sync started' })
    const { result } = renderHook(() => useTriggerSync(), { wrapper: wrapper() })

    expect(() => result.current.run('7/../8' as unknown as number)).toThrow(TypeError)
    expect(() => result.current.run(-7)).toThrow(TypeError)
    expect(client.post).not.toHaveBeenCalled()
    await act(async () => {
      await result.current.runAsync(7)
    })
    expect(client.post).toHaveBeenCalledWith('/api/cloud-documents/connections/7/sync')
    await waitFor(() => expect(result.current.result).toEqual({ job_id: 9, status: 'queued', message: 'Sync started' }))
  })

  it('chooses the folder a connection syncs from', async () => {
    client.post.mockResolvedValue({ ok: true })
    const { result } = renderHook(() => useSelectRootFolder(), { wrapper: wrapper() })

    expect(() => result.current.run({ connectionId: 7, rootFolderPath: '' })).toThrow(TypeError)
    await act(async () => {
      await result.current.runAsync({ connectionId: 7, rootFolderPath: '/Team Docs' })
    })
    expect(client.post).toHaveBeenCalledTimes(1)
    expect(client.post).toHaveBeenCalledWith('/api/cloud-documents/connections/7/select-folder', { root_folder_path: '/Team Docs' })
  })
})

describe('PRD-261 A-12f — database knowledge', () => {
  it('loads the sources on mount and hands back frozen copies', async () => {
    client.request.mockImplementation(answerDatabase)
    const { result } = renderHook(() => useDatabaseKnowledge())

    await waitFor(() => expect(result.current.sources).toHaveLength(1))
    expect(result.current.sources[0]).toEqual({
      id: 5, name: 'Sales DB', dialect: 'postgresql', status: 'active', tables_count: 8, last_synced: 'Never', credential_id: 2,
    })
    expect(Object.isFrozen(result.current)).toBe(true)
    expect(Object.isFrozen(result.current.sources[0])).toBe(true)
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('keeps its calls the same across renders', async () => {
    client.request.mockImplementation(answerDatabase)
    const { result, rerender } = renderHook(() => useDatabaseKnowledge())
    await waitFor(() => expect(result.current.sources).toHaveLength(1))
    const { deleteSource, executeQuery } = result.current
    rerender()
    expect(result.current.deleteSource).toBe(deleteSource)
    expect(result.current.executeQuery).toBe(executeQuery)
  })

  it('deletes a source by its id, and rejects an id that is not a positive whole number', async () => {
    client.request.mockImplementation(answerDatabase)
    const { result } = renderHook(() => useDatabaseKnowledge())
    await waitFor(() => expect(result.current.sources).toHaveLength(1))

    await expect(result.current.deleteSource('5/../1' as unknown as number)).rejects.toThrow(TypeError)
    await act(async () => {
      await result.current.deleteSource(5)
    })
    const deletes = client.request.mock.calls.filter(([, options]) => options?.method === 'DELETE')
    expect(deletes).toEqual([['/api/knowledge/sources/database/5', { method: 'DELETE' }]])
  })

  it('asks a question of a source, sending only the flags the theme set as booleans', async () => {
    client.request.mockImplementation(answerDatabase)
    const { result } = renderHook(() => useDatabaseKnowledge())
    await waitFor(() => expect(result.current.sources).toHaveLength(1))

    const options = { validate: true, useCache: 'yes' } as unknown as Sdk.KnowledgeDatabaseQueryOptions
    let answer: Sdk.KnowledgeDatabaseQueryResult | null = null
    await act(async () => {
      answer = await result.current.executeQuery(5, 'How many orders?', options)
    })
    expect(answer).toEqual({ sql: 'select 1', results: [], validation: null, execution_time: 3 })
    expect(client.request).toHaveBeenCalledWith('/api/knowledge/sources/database/5/query', {
      method: 'POST',
      body: { source_id: 5, query: 'How many orders?', validate: true },
    })
    await expect(result.current.executeQuery(5, '   ')).rejects.toThrow(SDK_REFUSAL)
  })
})
