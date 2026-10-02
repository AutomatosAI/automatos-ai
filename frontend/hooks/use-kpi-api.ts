/**
 * KPI API hooks — Command Centre Dashboard Widgets
 * React Query integration for cost tracker, agent performance and playbook metrics.
 * PRD-252 R5: approvals and decisions are part of the one Needs-you number (use-needs-you.ts).
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

// ============= QUERY KEYS =============

export const kpiQueryKeys = {
  all: ['kpi'] as const,
  costTracker: (period: string) => ['kpi', 'cost-tracker', period] as const,
  agentPerformance: (period: string) => ['kpi', 'agent-performance', period] as const,
  playbookMetrics: (period: string) => ['kpi', 'playbook-metrics', period] as const,
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
