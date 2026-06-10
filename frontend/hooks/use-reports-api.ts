/**
 * Reports API hooks (PRD-76)
 * React Query integration for agent reports: list, get, grade, stats.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'

// ============= TYPES =============

export interface AgentReport {
  id: string
  agent_id: number
  agent_name: string
  heartbeat_result_id: number | null
  report_type: 'standup' | 'research' | 'incident' | 'summary' | 'delivery' | 'audit'
  title: string
  summary: string | null
  status: 'ok' | 'warning' | 'critical' | 'info'
  file_path: string
  file_type: string
  file_size_bytes: number | null
  metrics: Record<string, any>
  attachments: Array<{
    title: string
    file_path: string
    file_type: string
  }>
  content?: string | null
  content_error?: string
  grade: number | null
  grade_notes: string | null
  graded_by: number | null
  graded_at: string | null
  created_at: string
}

export interface ReportListResponse {
  success: boolean
  reports: AgentReport[]
  total: number
  limit: number
  offset: number
}

export interface ReportDetailResponse {
  success: boolean
  report: AgentReport
}

export interface ReportStats {
  success: boolean
  total: number
  ungraded_count: number
  avg_grade: number | null
  by_type: Record<string, number>
  by_status: Record<string, number>
  period: string
}

export interface ReportFilters {
  agent_id?: number
  report_type?: string
  status?: string
  graded?: boolean
  period?: string
  limit?: number
  offset?: number
}

// ============= QUERY KEYS =============

export const reportQueryKeys = {
  all: ['reports'] as const,
  list: (filters: ReportFilters) => ['reports', 'list', filters] as const,
  detail: (id: string) => ['reports', 'detail', id] as const,
  stats: (period: string) => ['reports', 'stats', period] as const,
  agent: (agentId: number, filters?: ReportFilters) => ['reports', 'agent', agentId, filters] as const,
}

// ============= QUERY HOOKS =============

/**
 * Fetch reports list with filters.
 */
export function useReports(filters: ReportFilters = {}) {
  const params = new URLSearchParams()
  if (filters.agent_id) params.set('agent_id', String(filters.agent_id))
  if (filters.report_type) params.set('report_type', filters.report_type)
  if (filters.status) params.set('status', filters.status)
  if (filters.graded !== undefined) params.set('graded', String(filters.graded))
  if (filters.period) params.set('period', filters.period)
  if (filters.limit) params.set('limit', String(filters.limit))
  if (filters.offset) params.set('offset', String(filters.offset))

  const qs = params.toString()

  return useQuery<ReportListResponse>({
    queryKey: reportQueryKeys.list(filters),
    queryFn: () => apiClient.request<ReportListResponse>(`/api/reports${qs ? `?${qs}` : ''}`),
    staleTime: 15000,
    refetchInterval: 30000,
  })
}

/**
 * Fetch a single report with content.
 */
export function useReport(reportId: string | null, options?: { enabled?: boolean }) {
  return useQuery<ReportDetailResponse>({
    queryKey: reportQueryKeys.detail(reportId!),
    queryFn: () => apiClient.request<ReportDetailResponse>(`/api/reports/${reportId}`),
    enabled: !!reportId && (options?.enabled !== false),
    staleTime: 30000,
  })
}

/**
 * Fetch report stats for the workspace.
 */
export function useReportStats(period: string = '7d') {
  return useQuery<ReportStats>({
    queryKey: reportQueryKeys.stats(period),
    queryFn: () => apiClient.request<ReportStats>(`/api/reports/stats?period=${period}`),
    staleTime: 30000,
    refetchInterval: 60000,
  })
}

/**
 * Fetch reports for a specific agent.
 */
export function useAgentReports(agentId: number | null, filters: ReportFilters = {}) {
  const params = new URLSearchParams()
  if (filters.report_type) params.set('report_type', filters.report_type)
  if (filters.period) params.set('period', filters.period || '30d')
  if (filters.limit) params.set('limit', String(filters.limit))
  if (filters.offset) params.set('offset', String(filters.offset))

  const qs = params.toString()

  return useQuery<ReportListResponse>({
    queryKey: reportQueryKeys.agent(agentId!, filters),
    queryFn: () =>
      apiClient.request<ReportListResponse>(`/api/reports/agent/${agentId}${qs ? `?${qs}` : ''}`),
    enabled: !!agentId,
    staleTime: 15000,
  })
}

// ============= MUTATION HOOKS =============

/**
 * Grade a report (1-5 stars + optional notes).
 */
export function useGradeReport() {
  const queryClient = useQueryClient()

  return useMutation<
    { success: boolean; report_id: string; grade: number },
    Error,
    { reportId: string; grade: number; grade_notes?: string }
  >({
    mutationFn: ({ reportId, grade, grade_notes }) =>
      apiClient.request(`/api/reports/${reportId}/grade`, {
        method: 'PATCH',
        body: JSON.stringify({ grade, grade_notes }),
      }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: reportQueryKeys.all })
      toast.success(`Report graded ${data.grade}/5`)
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to grade report')
    },
  })
}
