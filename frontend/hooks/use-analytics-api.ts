import { useQuery, useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type {
  ActivationMetric,
  MissionSuccessRateMetric,
  ErrorsBySubsystemMetric,
  WidgetEngagementMetric,
  PrimitiveHealthMetric,
  SlosMetric,
  SubstrateHealthMetric,
  WorkspaceActivationMetric,
  DeliverableFreshnessMetric,
  CommerceIntegrityMetric,
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

export function usePrimitiveHealth() {
  return useQuery<PrimitiveHealthMetric>({
    queryKey: ['analytics', 'primitive-health'],
    queryFn: () => apiClient.getPrimitiveHealth()
  })
}

// PRD-185 S12 — own-workspace cockpit tiles (workspace-admin reachable)
export function useSLOs(window: string = '24h') {
  return useQuery<SlosMetric>({
    queryKey: ['analytics', 'slos', window],
    queryFn: () => apiClient.getSLOs(window)
  })
}

// PRD-197 S4 — per-seam retrieval substrate health (documents / memory / field)
export function useSubstrateHealth(window: string = '24h') {
  return useQuery<SubstrateHealthMetric>({
    queryKey: ['analytics', 'substrate-health', window],
    queryFn: () => apiClient.getSubstrateHealth(window)
  })
}

export function useWorkspaceActivation() {
  return useQuery<WorkspaceActivationMetric>({
    queryKey: ['analytics', 'workspace-activation'],
    queryFn: () => apiClient.getWorkspaceActivation()
  })
}

export function useDeliverableFreshness() {
  return useQuery<DeliverableFreshnessMetric>({
    queryKey: ['analytics', 'deliverable-freshness'],
    queryFn: () => apiClient.getDeliverableFreshness()
  })
}

// PRD-189 S2 — cross-sell persistence integrity (the Commerce tile).
export function useCommerceIntegrity() {
  return useQuery<CommerceIntegrityMetric>({
    queryKey: ['analytics', 'commerce-integrity'],
    queryFn: () => apiClient.getCommerceIntegrity()
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
