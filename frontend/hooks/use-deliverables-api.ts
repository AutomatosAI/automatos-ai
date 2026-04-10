/**
 * Deliverables API hooks (PRD-129: Workspace Outputs Hub)
 * =========================================================
 *
 * React Query hooks for the consumer-facing Gallery view.
 *
 * Endpoints:
 *   GET    /api/deliverables              list (paginated via offset)
 *   GET    /api/deliverables/stats        aggregate counts
 *   GET    /api/deliverables/{id}         single deliverable (?include_content=)
 *   DELETE /api/deliverables/{id}         soft delete
 *
 * Query keys are scoped by workspaceId so switching workspaces clears cache.
 */

import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import { toast } from 'react-hot-toast'

import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/hooks/use-workspace'

// ============= TYPES =============

export type ArtifactType =
  | 'report'
  | 'image'
  | 'document'
  | 'code'
  | 'slide'
  | 'spreadsheet'
  | 'archive'
  | 'audio'
  | 'video'

export type SourceType =
  | 'chat'
  | 'task'
  | 'mission'
  | 'heartbeat'
  | 'playbook'
  | 'trigger'
  | 'upload'

export type DateRange = 'all' | 'today' | 'week' | 'month'

export interface Deliverable {
  id: string
  workspace_id: string
  source_type: SourceType | string
  source_id: string | null
  agent_id: number | null
  agent_name: string | null
  artifact_type: ArtifactType | string
  title: string
  summary: string | null
  storage_type: string
  file_path: string
  file_name: string | null
  file_type: string | null
  file_size_bytes: number | null
  preview_url: string | null
  preview_type: string | null
  extra: Record<string, any>
  status: string
  created_at: string
  updated_at: string
  // Populated when fetched with include_content=true
  content?: string | null
  content_url?: string | null
  content_error?: string
}

export interface DeliverableListResponse {
  success: boolean
  deliverables: Deliverable[]
  total: number
  limit: number
  offset: number
  error?: string
}

export interface DeliverableDetailResponse {
  success: boolean
  deliverable: Deliverable
  error?: string
}

export interface DeliverableStats {
  success: boolean
  total: number
  by_type: Record<string, number>
  by_agent: Array<{
    agent_id: number | null
    agent_name: string
    count: number
  }>
  error?: string
}

export interface FilterState {
  artifact_type: string | null
  source_type: string | null
  agent_id: number | null
  date_range: DateRange
  search: string
}

export const DEFAULT_FILTERS: FilterState = {
  artifact_type: null,
  source_type: null,
  agent_id: null,
  date_range: 'all',
  search: '',
}

// ============= HELPERS =============

const PAGE_SIZE = 24

/**
 * Translate a DateRange into an ISO timestamp for `date_from`.
 * Returns null for 'all'.
 */
function dateRangeToFrom(range: DateRange): string | null {
  if (range === 'all') return null
  const now = new Date()
  const from = new Date(now)
  if (range === 'today') {
    from.setHours(0, 0, 0, 0)
  } else if (range === 'week') {
    from.setDate(now.getDate() - 7)
  } else if (range === 'month') {
    from.setMonth(now.getMonth() - 1)
  }
  return from.toISOString()
}

function buildListQuery(filters: FilterState, offset: number): string {
  const params = new URLSearchParams()
  params.set('limit', String(PAGE_SIZE))
  params.set('offset', String(offset))
  if (filters.artifact_type) params.set('artifact_type', filters.artifact_type)
  if (filters.source_type) params.set('source_type', filters.source_type)
  if (filters.agent_id !== null) params.set('agent_id', String(filters.agent_id))
  if (filters.search.trim()) params.set('search', filters.search.trim())
  const dateFrom = dateRangeToFrom(filters.date_range)
  if (dateFrom) params.set('date_from', dateFrom)
  return params.toString()
}

// ============= QUERY KEYS =============

export const deliverableQueryKeys = {
  all: (workspaceId: string | null) => ['deliverables', workspaceId] as const,
  list: (workspaceId: string | null, filters: FilterState) =>
    ['deliverables', workspaceId, 'list', filters] as const,
  detail: (workspaceId: string | null, id: string, includeContent: boolean) =>
    ['deliverables', workspaceId, 'detail', id, includeContent] as const,
  stats: (workspaceId: string | null) =>
    ['deliverables', workspaceId, 'stats'] as const,
}

// ============= QUERY HOOKS =============

/**
 * Fetch a paginated (infinite) list of deliverables for the current workspace.
 * Use `fetchNextPage()` / `hasNextPage` for Load More UX.
 */
export function useDeliverables(filters: FilterState = DEFAULT_FILTERS) {
  const { workspaceId } = useWorkspace()

  // React Query v4: pageParam defaults to undefined; seed with 0 via destructuring.
  return useInfiniteQuery<DeliverableListResponse>({
    queryKey: deliverableQueryKeys.list(workspaceId, filters),
    enabled: !!workspaceId,
    queryFn: ({ pageParam = 0 }) => {
      const qs = buildListQuery(filters, pageParam as number)
      return apiClient.request<DeliverableListResponse>(
        `/api/deliverables?${qs}`,
      )
    },
    getNextPageParam: (lastPage) => {
      const fetchedSoFar = lastPage.offset + lastPage.deliverables.length
      return fetchedSoFar < lastPage.total ? fetchedSoFar : undefined
    },
    staleTime: 15_000,
  })
}

/**
 * Fetch a single deliverable — optionally with inline content (text/code/report).
 */
export function useDeliverable(
  deliverableId: string | null,
  includeContent = false,
) {
  const { workspaceId } = useWorkspace()

  return useQuery<DeliverableDetailResponse>({
    queryKey: deliverableQueryKeys.detail(
      workspaceId,
      deliverableId ?? '',
      includeContent,
    ),
    enabled: !!workspaceId && !!deliverableId,
    queryFn: () => {
      const qs = includeContent ? '?include_content=true' : ''
      return apiClient.request<DeliverableDetailResponse>(
        `/api/deliverables/${deliverableId}${qs}`,
      )
    },
    staleTime: 30_000,
  })
}

/**
 * Fetch aggregate stats for the gallery (total, by_type, by_agent).
 */
export function useDeliverableStats() {
  const { workspaceId } = useWorkspace()

  return useQuery<DeliverableStats>({
    queryKey: deliverableQueryKeys.stats(workspaceId),
    enabled: !!workspaceId,
    queryFn: () =>
      apiClient.request<DeliverableStats>('/api/deliverables/stats'),
    staleTime: 30_000,
  })
}

// ============= MUTATION HOOKS =============

/**
 * Soft-delete a deliverable. Invalidates list + stats so the Gallery updates.
 */
export function useDeleteDeliverable() {
  const { workspaceId } = useWorkspace()
  const queryClient = useQueryClient()

  return useMutation<
    { success: boolean; deliverable_id: string; error?: string },
    Error,
    string
  >({
    mutationFn: (deliverableId: string) =>
      apiClient.request(`/api/deliverables/${deliverableId}`, {
        method: 'DELETE',
      }),
    onSuccess: (data) => {
      if (data?.success === false) {
        toast.error(data.error || 'Failed to delete')
        return
      }
      queryClient.invalidateQueries({
        queryKey: deliverableQueryKeys.all(workspaceId),
      })
      toast.success('Deliverable deleted')
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to delete deliverable')
    },
  })
}
