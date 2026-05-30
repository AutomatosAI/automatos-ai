import { useQuery, useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type {
  ActivationMetric,
  MissionSuccessRateMetric,
  ErrorsBySubsystemMetric,
  WidgetEngagementMetric,
} from '@/lib/api-client'

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

// PRD-142 Wave 0 — "Is it working?" dashboard tiles
export function useActivationMetrics() {
  return useQuery<ActivationMetric>({
    queryKey: ['analytics', 'activation'],
    queryFn: () => apiClient.getActivationMetrics()
  })
}

export function useMissionSuccessRate() {
  return useQuery<MissionSuccessRateMetric>({
    queryKey: ['analytics', 'mission-success-rate'],
    queryFn: () => apiClient.getMissionSuccessRate()
  })
}

export function useErrorsBySubsystem(window: string = '24h') {
  return useQuery<ErrorsBySubsystemMetric>({
    queryKey: ['analytics', 'errors-by-subsystem', window],
    queryFn: () => apiClient.getErrorsBySubsystem(window)
  })
}

export function useWidgetEngagement(window: string = '7d') {
  return useQuery<WidgetEngagementMetric>({
    queryKey: ['analytics', 'widget-engagement', window],
    queryFn: () => apiClient.getWidgetEngagement(window)
  })
}

export function useCostAnalysis() {
  return useQuery({
    queryKey: ['analytics', 'cost-analysis'],
    queryFn: () => apiClient.getCostAnalysis()
  })
}

export function usePerformanceEnhancements() {
  return useQuery({
    queryKey: analyticsQueryKeys.allEnhancements,
    queryFn: () => apiClient.getPerformanceEnhancements()
  })
}

export function usePeakUsageAnalysis() {
  return useQuery({
    queryKey: analyticsQueryKeys.peakHours,
    queryFn: () => apiClient.getPeakUsageAnalysis()
  })
}

export function useBottleneckDetection() {
  return useQuery({
    queryKey: analyticsQueryKeys.bottlenecks,
    queryFn: () => apiClient.getBottleneckDetection()
  })
}

export function usePredictiveAlerts() {
  return useQuery({
    queryKey: analyticsQueryKeys.alerts,
    queryFn: () => apiClient.getPredictiveAlerts()
  })
}

export function useAgentRanking() {
  return useQuery({
    queryKey: analyticsQueryKeys.ranking,
    queryFn: () => apiClient.getAgentRanking()
  })
}

export function useSLACompliance() {
  return useQuery({
    queryKey: analyticsQueryKeys.sla,
    queryFn: () => apiClient.getSLACompliance()
  })
}
