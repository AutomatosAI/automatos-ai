import { useQuery, useMutation } from '@tanstack/react-query'
import { apiClient } from "@/lib/api-client'

export const analyticsQueryKeys = {
  successRate: ['analytics', 'success-rate'],
  completionTime: ['analytics', 'completion-time'],
  loadTrend: ['analytics', 'load-trend'],
  errorRate: ['analytics', 'error-rate'],
  queueDepth: ['analytics', 'queue-depth'],
  efficiency: ['analytics', 'efficiency'],
  costPerExecution: ['analytics', 'cost-per-execution'],
  peakHours: ['analytics', 'peak-hours'],
  bottlenecks: ['analytics', 'bottlenecks'],
  alerts: ['analytics', 'alerts'],
  ranking: ['analytics', 'ranking'],
  sla: ['analytics', 'sla'],
  allMetrics: ['analytics', 'all-metrics'],
  allEnhancements: ['analytics', 'all-enhancements']
}

export function useAnalyticsSuccessRate() {
  return useQuery({
    queryKey: analyticsQueryKeys.successRate,
    queryFn: () => apiClient.getAnalyticsSuccessRate()
  })
}

export function useTaskCompletionTime() {
  return useQuery({
    queryKey: analyticsQueryKeys.completionTime,
    queryFn: () => apiClient.getTaskCompletionTime()
  })
}

export function useSystemLoadTrend() {
  return useQuery({
    queryKey: analyticsQueryKeys.loadTrend,
    queryFn: () => apiClient.getSystemLoadTrend()
  })
}

export function useErrorRateByType() {
  return useQuery({
    queryKey: analyticsQueryKeys.errorRate,
    queryFn: () => apiClient.getErrorRateByType()
  })
}

export function useAllMetrics() {
  return useQuery({
    queryKey: analyticsQueryKeys.allMetrics,
    queryFn: () => apiClient.getAllMetrics()
  })
}
