/**
 * KPI API hooks — Command Centre Dashboard Widgets
 * React Query integration for cost tracker, agent performance, playbook metrics, approval gates.
 */

import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// ============= TYPES =============

export interface CostTrackerData {
  total_cost: number
  change_pct: number
  daily_trend: Array<{ date: string; cost: number }>
  top_agents: Array<{ name: string; cost: number }>
  period: string
}

export interface AgentPerformanceData {
  agents: Array<{
    agent_id: number
    name: string
    success_rate: number
    tasks_completed: number
    avg_completion_seconds: number
  }>
  total_agents: number
  period: string
}

export interface PlaybookMetricsData {
  playbooks: Array<{
    recipe_id: number
    name: string
    runs: number
    success_pct: number
    avg_duration_seconds: number
  }>
  total: number
  period: string
}

export interface ApprovalGatesData {
  pending_count: number
  pending_missions: Array<{
    id: string
    goal: string
    created_at: string | null
    waiting_since: string | null
  }>
  avg_approval_seconds: number
  recent_count: number
  period: string
}

export interface DecisionsNeededItem {
  kind: 'report' | 'mission'
  id: string
  title: string
  summary: string | null
  status: string | null
  escalation_level: number | null
  agent_name?: string | null
  created_at: string | null
  updated_at?: string | null
}

export interface DecisionsNeededData {
  total: number
  reports_count: number
  missions_count: number
  items: DecisionsNeededItem[]
}

// ============= QUERY KEYS =============

export const kpiQueryKeys = {
  all: ['kpi'] as const,
  costTracker: (period: string) => ['kpi', 'cost-tracker', period] as const,
  agentPerformance: (period: string) => ['kpi', 'agent-performance', period] as const,
  playbookMetrics: (period: string) => ['kpi', 'playbook-metrics', period] as const,
  approvalGates: (period: string) => ['kpi', 'approval-gates', period] as const,
  decisionsNeeded: (limit: number) => ['kpi', 'decisions-needed', limit] as const,
}

// ============= HOOKS =============

export function useCostTracker(period: string = '30d') {
  return useQuery<CostTrackerData>({
    queryKey: kpiQueryKeys.costTracker(period),
    queryFn: () => apiClient.request<CostTrackerData>(`/api/kpi/cost-tracker?period=${period}`),
    staleTime: 30_000,
    refetchInterval: 60_000,
  })
}

export function useAgentPerformance(period: string = '30d') {
  return useQuery<AgentPerformanceData>({
    queryKey: kpiQueryKeys.agentPerformance(period),
    queryFn: () => apiClient.request<AgentPerformanceData>(`/api/kpi/agent-performance?period=${period}`),
    staleTime: 30_000,
    refetchInterval: 60_000,
  })
}

export function usePlaybookMetrics(period: string = '30d') {
  return useQuery<PlaybookMetricsData>({
    queryKey: kpiQueryKeys.playbookMetrics(period),
    queryFn: () => apiClient.request<PlaybookMetricsData>(`/api/kpi/playbook-metrics?period=${period}`),
    staleTime: 30_000,
    refetchInterval: 60_000,
  })
}

export function useApprovalGates(period: string = '30d') {
  return useQuery<ApprovalGatesData>({
    queryKey: kpiQueryKeys.approvalGates(period),
    queryFn: () => apiClient.request<ApprovalGatesData>(`/api/kpi/approval-gates?period=${period}`),
    staleTime: 15_000,
    refetchInterval: 30_000,
  })
}

export function useDecisionsNeeded(limit: number = 10) {
  return useQuery<DecisionsNeededData>({
    queryKey: kpiQueryKeys.decisionsNeeded(limit),
    queryFn: () => apiClient.request<DecisionsNeededData>(`/api/kpi/decisions-needed?limit=${limit}`),
    staleTime: 15_000,
    refetchInterval: 30_000,
  })
}
