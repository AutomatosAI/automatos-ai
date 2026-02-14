/**
 * Unified Analytics Hooks
 * Consolidated React Query hooks for the unified analytics page.
 * Reuses existing API endpoints and adds new ones for costs, plans, and recommendations.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// ============= QUERY KEYS =============
export const unifiedAnalyticsKeys = {
  overview: (days: number) => ['unified-analytics', 'overview', days] as const,
  agents: (days: number) => ['unified-analytics', 'agents', days] as const,
  workflows: (days: number) => ['unified-analytics', 'workflows', days] as const,
  documents: (days: number) => ['unified-analytics', 'documents', days] as const,
  costs: (days: number) => ['unified-analytics', 'costs', days] as const,
  recommendations: ['unified-analytics', 'recommendations'] as const,
  planUsage: ['unified-analytics', 'plan-usage'] as const,
  memory: ['unified-analytics', 'memory'] as const,
  adminWorkspaces: (days: number) => ['unified-analytics', 'admin', 'workspaces', days] as const,
  openrouterCredits: ['unified-analytics', 'openrouter', 'credits'] as const,
  openrouterKeyInfo: ['unified-analytics', 'openrouter', 'key-info'] as const,
  composioApps: (days: number) => ['unified-analytics', 'composio', 'apps', days] as const,
  composioActions: (days: number) => ['unified-analytics', 'composio', 'actions', days] as const,
  composioAgentTools: (days: number) => ['unified-analytics', 'composio', 'agent-tools', days] as const,
  chartPresets: ['unified-analytics', 'charts', 'presets'] as const,
  modelComparison: (modelIds: string[], period: string) => ['unified-analytics', 'llm', 'comparison', modelIds, period] as const,
  costProjections: (period: string) => ['unified-analytics', 'llm', 'projections', period] as const,
}

// ============= OVERVIEW =============
export function useAnalyticsOverview(days: number = 30) {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.overview(days),
    queryFn: async () => {
      // Aggregate from existing endpoints
      const [agents, costData, workflowStats, docStats] = await Promise.all([
        apiClient.getAgents().catch(() => []),
        apiClient.getCostAnalysis().catch(() => null),
        apiClient.getWorkflowStatsDashboard().catch(() => null),
        apiClient.getAnalyticsOverview().catch(() => null),
      ])

      const agentList = Array.isArray(agents) ? agents : []

      return {
        agents: {
          total: agentList.length,
          active: agentList.filter((a: any) => a.status === 'active').length,
        },
        workflows: {
          total: (workflowStats as any)?.overview?.total_workflows || 0,
          active: (workflowStats as any)?.overview?.running_executions || 0,
          successRate: (workflowStats as any)?.today?.success_rate_today || 0,
        },
        documents: {
          total: (docStats as any)?.total_documents || 0,
          storageMb: (docStats as any)?.storage_mb || 0,
        },
        cost: {
          currentPeriod: (costData as any)?.total_cost || 0,
          previousPeriod: (costData as any)?.previous_cost || 0,
        },
        system: {},
      }
    },
    staleTime: 60000,
  })
}

// ============= AGENT ANALYTICS =============
export function useAgentAnalytics(days: number = 30) {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.agents(days),
    queryFn: async () => {
      const [agents, stats, ranking] = await Promise.all([
        apiClient.getAgents().catch(() => []),
        apiClient.getSystemAgentStatistics().catch(() => null),
        apiClient.getAgentRanking().catch(() => null),
      ])

      const agentList = Array.isArray(agents) ? agents : []

      return {
        agents: agentList.map((agent: any) => ({
          id: agent.id,
          name: agent.name,
          status: agent.status,
          agentType: agent.agent_type,
          successRate: agent.performance_metrics?.success_rate || 0,
          avgRunTime: agent.performance_metrics?.avg_execution_time || 0,
          tokensUsed: agent.model_usage_stats?.total_tokens || 0,
          cost: agent.model_usage_stats?.total_cost || 0,
          llmModel: agent.model_config?.model_id || 'unknown',
          totalRequests: agent.model_usage_stats?.total_requests || 0,
          lastUsed: agent.model_usage_stats?.last_used_at || agent.updated_at,
        })),
        summary: {
          totalAgents: agentList.length,
          activeAgents: agentList.filter((a: any) => a.status === 'active').length,
          avgSuccessRate: stats?.average_performance || 0,
          totalTokens: agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_tokens || 0), 0),
          totalCost: agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_cost || 0), 0),
        },
        ranking: ranking || [],
      }
    },
    staleTime: 60000,
  })
}

// ============= WORKFLOW ANALYTICS =============
export function useWorkflowAnalytics(days: number = 30) {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.workflows(days),
    queryFn: async () => {
      const [workflows, stats] = await Promise.all([
        apiClient.getWorkflows().catch(() => []),
        apiClient.getWorkflowStatsDashboard().catch(() => null),
      ])

      const workflowList = Array.isArray(workflows) ? workflows : []

      return {
        workflows: workflowList.map((wf: any) => ({
          id: wf.id,
          name: wf.name,
          status: wf.status,
          totalRuns: wf.execution_count || 0,
          successRate: wf.success_rate || 0,
          avgDuration: wf.avg_duration || '0s',
          tokensUsed: wf.total_tokens || 0,
          cost: wf.total_cost || 0,
          lastRun: wf.last_executed_at || wf.updated_at,
        })),
        summary: {
          totalWorkflows: workflowList.length,
          totalExecutions: (stats as any)?.overview?.total_executions || 0,
          successRate: (stats as any)?.today?.success_rate_today || 0,
          avgDuration: (stats as any)?.today?.avg_duration_today || '0s',
        },
        stats,
      }
    },
    staleTime: 60000,
  })
}

// ============= DOCUMENT ANALYTICS =============
export function useDocumentAnalyticsUnified(days: number = 30) {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.documents(days),
    queryFn: async () => {
      const period = days <= 1 ? '24h' : days <= 7 ? '7d' : days <= 30 ? '30d' : '90d'

      const [documents, usage] = await Promise.all([
        apiClient.getDocuments().catch(() => []),
        fetch(`/api/documents/analytics/usage?period=${period}`)
          .then(r => r.ok ? r.json() : null)
          .catch(() => null),
      ])

      const docList = Array.isArray(documents) ? documents : []

      return {
        documents: docList.map((doc: any) => ({
          id: doc.id,
          name: doc.filename || doc.name,
          type: doc.file_type || 'unknown',
          size: doc.file_size || doc.size || 0,
          uploaded: doc.created_at || doc.upload_date,
          lastAccessed: doc.last_accessed || null,
          ragQueries: doc.rag_query_count || 0,
          confidenceScore: doc.avg_confidence || 0,
          status: doc.status,
          tags: doc.tags || [],
        })),
        summary: {
          totalDocuments: docList.length,
          totalStorageMb: docList.reduce((sum: number, d: any) => sum + (d.file_size || d.size || 0), 0) / (1024 * 1024),
          totalRagQueries: usage?.total_events || 0,
          avgConfidence: 0,
        },
        usage: usage || null,
        neverAccessed: docList.filter((d: any) => !d.last_accessed).length,
      }
    },
    staleTime: 60000,
  })
}

// ============= LLM & COSTS =============
export function useCostAnalyticsUnified(days: number = 30) {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.costs(days),
    queryFn: async () => {
      const [costData, agents] = await Promise.all([
        apiClient.getCostAnalysis().catch(() => null),
        apiClient.getAgents().catch(() => []),
      ])

      const agentList = Array.isArray(agents) ? agents : []

      // Group costs by LLM model
      const modelCosts: Record<string, { requests: number; inputTokens: number; outputTokens: number; cost: number; agents: Set<string> }> = {}

      agentList.forEach((agent: any) => {
        const model = agent.model_config?.model_id || 'unknown'
        if (!modelCosts[model]) {
          modelCosts[model] = { requests: 0, inputTokens: 0, outputTokens: 0, cost: 0, agents: new Set() }
        }
        modelCosts[model].requests += agent.model_usage_stats?.total_requests || 0
        modelCosts[model].inputTokens += (agent.model_usage_stats?.total_tokens || 0) * 0.7 // estimate input
        modelCosts[model].outputTokens += (agent.model_usage_stats?.total_tokens || 0) * 0.3 // estimate output
        modelCosts[model].cost += agent.model_usage_stats?.total_cost || 0
        modelCosts[model].agents.add(agent.name)
      })

      const totalTokens = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_tokens || 0), 0)
      const totalCost = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_cost || 0), 0)
      const totalRequests = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_requests || 0), 0)

      // Find most expensive agent
      const sortedByCost = [...agentList].sort((a: any, b: any) =>
        (b.model_usage_stats?.total_cost || 0) - (a.model_usage_stats?.total_cost || 0)
      )
      const mostExpensive = sortedByCost[0]

      return {
        summary: {
          totalTokens,
          totalCost,
          totalRequests,
          costPerTask: totalRequests > 0 ? totalCost / totalRequests : 0,
          mostExpensiveAgent: mostExpensive ? {
            name: mostExpensive.name,
            cost: mostExpensive.model_usage_stats?.total_cost || 0,
            model: mostExpensive.model_config?.model_id || 'unknown',
          } : null,
        },
        byModel: Object.entries(modelCosts).map(([model, data]) => ({
          model,
          requests: data.requests,
          inputTokens: Math.round(data.inputTokens),
          outputTokens: Math.round(data.outputTokens),
          totalCost: data.cost,
          avgCostPerRequest: data.requests > 0 ? data.cost / data.requests : 0,
          agentCount: data.agents.size,
        })),
        byAgent: agentList.map((agent: any) => ({
          id: agent.id,
          name: agent.name,
          model: agent.model_config?.model_id || 'unknown',
          tokens: agent.model_usage_stats?.total_tokens || 0,
          cost: agent.model_usage_stats?.total_cost || 0,
          requests: agent.model_usage_stats?.total_requests || 0,
        })).sort((a: any, b: any) => b.cost - a.cost),
        costTrend: (costData as any)?.monthly_data || [],
      }
    },
    staleTime: 60000,
  })
}

// ============= PLAN USAGE =============
export function usePlanUsage() {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.planUsage,
    queryFn: async () => {
      // For now, return placeholder limits (pilot phase — limits TBD)
      const [agents, workflows, documents] = await Promise.all([
        apiClient.getAgents().catch(() => []),
        apiClient.getWorkflows().catch(() => []),
        apiClient.getDocuments().catch(() => []),
      ])

      const agentList = Array.isArray(agents) ? agents : []
      const workflowList = Array.isArray(workflows) ? workflows : []
      const docList = Array.isArray(documents) ? documents : []

      const totalTokens = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_tokens || 0), 0)
      const totalRequests = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_requests || 0), 0)
      const storageMb = docList.reduce((sum: number, d: any) => sum + (d.file_size || d.size || 0), 0) / (1024 * 1024)

      return {
        planName: 'Pilot',
        planTier: 'pilot',
        usage: {
          agents: { used: agentList.length, limit: null as number | null, label: 'Agents' },
          workflows: { used: workflowList.length, limit: null as number | null, label: 'Workflows' },
          documents: { used: docList.length, limit: null as number | null, label: 'Documents' },
          storageGb: { used: parseFloat((storageMb / 1024).toFixed(2)), limit: null as number | null, label: 'Storage (GB)' },
          apiCalls: { used: totalRequests, limit: null as number | null, label: 'API Calls' },
          tokens: { used: totalTokens, limit: null as number | null, label: 'Tokens' },
        },
      }
    },
    staleTime: 120000,
  })
}

// ============= RECOMMENDATIONS (AI-powered, cached daily) =============
export function useRecommendations() {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.recommendations,
    queryFn: async () => {
      // Build recommendations from data analysis
      const agents = await apiClient.getAgents().catch(() => [])
      const agentList = Array.isArray(agents) ? agents : []

      const recommendations: Array<{
        id: string
        type: 'cost' | 'performance' | 'document' | 'quota'
        title: string
        description: string
        impact: string
        action?: string
      }> = []

      // Analyze agent LLM usage
      agentList.forEach((agent: any) => {
        const model = agent.model_config?.model_id || ''
        const cost = agent.model_usage_stats?.total_cost || 0
        const tokens = agent.model_usage_stats?.total_tokens || 0
        const avgTokens = agent.model_usage_stats?.total_requests > 0
          ? tokens / agent.model_usage_stats.total_requests
          : 0

        // Flag expensive models on simple tasks (low token usage = simple task)
        if (avgTokens < 500 && avgTokens > 0 && (model.includes('gpt-4') || model.includes('opus') || model.includes('claude-3'))) {
          recommendations.push({
            id: `cost-${agent.id}`,
            type: 'cost',
            title: `${agent.name} could use a cheaper model`,
            description: `This agent averages only ${Math.round(avgTokens)} tokens per request — it's handling simple tasks. Consider switching from ${model} to a more cost-effective model like gpt-4o-mini or claude-3-haiku.`,
            impact: `Estimated savings: $${(cost * 0.7).toFixed(2)}/month`,
          })
        }

        // Flag agents with low success rates
        const successRate = agent.performance_metrics?.success_rate || 0
        if (successRate > 0 && successRate < 80) {
          recommendations.push({
            id: `perf-${agent.id}`,
            type: 'performance',
            title: `${agent.name} has a low success rate (${successRate.toFixed(0)}%)`,
            description: `This agent is failing on ${(100 - successRate).toFixed(0)}% of tasks. Check the error logs and consider adjusting the agent's configuration or skills.`,
            impact: 'Improved reliability',
          })
        }
      })

      // Sort by impact (cost savings first)
      return recommendations.slice(0, 5)
    },
    staleTime: 24 * 60 * 60 * 1000, // Cache for 24 hours
  })
}

// ============= WORKSPACE MEMORY =============
export function useWorkspaceMemory() {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.memory,
    queryFn: async () => {
      const memoryData = await apiClient.request('/api/v1/memory/stats/real').catch(() => null)

      return {
        items: (memoryData as any)?.user_preferences || [],
        totalMemories: (memoryData as any)?.system_stats?.total_memories || 0,
        hitRate: (memoryData as any)?.access_metrics?.hit_rate || 0,
      }
    },
    staleTime: 60000,
  })
}

// ============= OPENROUTER ANALYTICS =============

interface OpenRouterCreditsData {
  total_credits: number
  total_usage: number
}

interface OpenRouterKeyInfoData {
  limit: number | null
  limit_remaining: number | null
  usage_daily: number
  usage_weekly: number
  usage_monthly: number
  is_free_tier: boolean
  rate_limit: Record<string, any>
}

interface OpenRouterSyncResult {
  synced: number
  skipped: number
  error: string | null
}

export function useOpenRouterCredits() {
  return useQuery<OpenRouterCreditsData | null>({
    queryKey: unifiedAnalyticsKeys.openrouterCredits,
    queryFn: async () => {
      const data = await apiClient.request<OpenRouterCreditsData>(
        '/api/analytics/llm/openrouter/credits'
      ).catch((err: any) => {
        // 404 = no OpenRouter key configured — return null gracefully
        if (err?.message?.includes('404')) return null
        throw err
      })
      return data
    },
    staleTime: 300000, // 5 minutes
  })
}

export function useOpenRouterKeyInfo() {
  return useQuery<OpenRouterKeyInfoData | null>({
    queryKey: unifiedAnalyticsKeys.openrouterKeyInfo,
    queryFn: async () => {
      const data = await apiClient.request<OpenRouterKeyInfoData>(
        '/api/analytics/llm/openrouter/key-info'
      ).catch((err: any) => {
        if (err?.message?.includes('404')) return null
        throw err
      })
      return data
    },
    staleTime: 60000, // 1 minute
  })
}

export function useTriggerOpenRouterSync() {
  const queryClient = useQueryClient()
  return useMutation<OpenRouterSyncResult, Error>({
    mutationFn: async () => {
      return apiClient.request<OpenRouterSyncResult>(
        '/api/analytics/llm/openrouter/sync',
        { method: 'POST' }
      )
    },
    onSuccess: () => {
      // Invalidate related queries so they refetch with fresh data
      queryClient.invalidateQueries({ queryKey: unifiedAnalyticsKeys.openrouterCredits })
      queryClient.invalidateQueries({ queryKey: unifiedAnalyticsKeys.openrouterKeyInfo })
      queryClient.invalidateQueries({ queryKey: ['unified-analytics', 'costs'] })
    },
  })
}

// ============= COMPOSIO ANALYTICS =============

interface ComposioAppStats {
  app_name: string
  status: string
  total_actions_used: number
  agent_count: number
  documents_synced: number
  last_used_at: string | null
}

interface ComposioActionEntry {
  action_name: string
  app_name: string
  total_usage_count: number
  agent_count: number
  last_used_at: string | null
}

interface ComposioAgentToolEntry {
  tool_name: string
  app_name: string
  usage_count: number
  enabled: boolean
}

interface ComposioAgentToolMapping {
  agent_id: number
  agent_name: string
  tools: ComposioAgentToolEntry[]
}

export function useComposioApps(days: number = 30) {
  return useQuery<ComposioAppStats[]>({
    queryKey: unifiedAnalyticsKeys.composioApps(days),
    queryFn: async () => {
      return apiClient.request<ComposioAppStats[]>(
        `/api/analytics/composio/apps?days=${days}`
      )
    },
    staleTime: 60000, // 1 minute
  })
}

export function useComposioActions(days: number = 30) {
  return useQuery<ComposioActionEntry[]>({
    queryKey: unifiedAnalyticsKeys.composioActions(days),
    queryFn: async () => {
      return apiClient.request<ComposioActionEntry[]>(
        `/api/analytics/composio/actions?days=${days}`
      )
    },
    staleTime: 60000, // 1 minute
  })
}

export function useComposioAgentTools(days: number = 30) {
  return useQuery<ComposioAgentToolMapping[]>({
    queryKey: unifiedAnalyticsKeys.composioAgentTools(days),
    queryFn: async () => {
      return apiClient.request<ComposioAgentToolMapping[]>(
        `/api/analytics/composio/agent-tools?days=${days}`
      )
    },
    staleTime: 60000, // 1 minute
  })
}

// ============= PANDASAI CHARTS =============

interface ChartGenerateResponse {
  summary: string
  charts: string[]
  data: Record<string, any>[]
}

interface ChartPreset {
  id: string
  title: string
  description: string
  query: string
  chart_type: string
}

export function useAnalyticsChart() {
  return useMutation<ChartGenerateResponse, Error, { query: string; chartType?: string }>({
    mutationFn: async ({ query, chartType }) => {
      return apiClient.request<ChartGenerateResponse>(
        '/api/analytics/charts/generate',
        {
          method: 'POST',
          body: JSON.stringify({
            query,
            chart_type: chartType || 'auto',
          }),
        }
      )
    },
  })
}

export function useChartPresets() {
  return useQuery<ChartPreset[]>({
    queryKey: unifiedAnalyticsKeys.chartPresets,
    queryFn: async () => {
      return apiClient.request<ChartPreset[]>(
        '/api/analytics/charts/presets'
      )
    },
    staleTime: 300000, // 5 minutes — presets rarely change
  })
}

// ============= MODEL COMPARISON =============

interface ModelComparisonItem {
  model_id: string
  display_name: string
  provider: string
  input_cost_per_1k: number | null
  output_cost_per_1k: number | null
  context_window: number | null
  capabilities: Record<string, any>
  total_requests: number
  total_tokens: number
  total_cost: number
  avg_latency_ms: number | null
  error_rate: number
  success_rate: number
}

export function useModelComparison(modelIds: string[], period: string = '30d') {
  return useQuery<ModelComparisonItem[]>({
    queryKey: unifiedAnalyticsKeys.modelComparison(modelIds, period),
    queryFn: async () => {
      const ids = modelIds.join(',')
      return apiClient.request<ModelComparisonItem[]>(
        `/api/analytics/llm/comparison?model_ids=${encodeURIComponent(ids)}&period=${period}`
      )
    },
    enabled: modelIds.length > 0,
    staleTime: 60000, // 1 minute
  })
}

// ============= COST PROJECTIONS =============

interface ProjectedItem {
  key: string
  projected_monthly_cost: number
  current_period_cost: number
}

interface CostProjectionData {
  current_period_cost: number
  daily_average: number
  projected_monthly: number
  change_percent: number | null
  projected_by_model: ProjectedItem[]
  projected_by_provider: ProjectedItem[]
}

export function useCostProjections(period: string = '30d') {
  return useQuery<CostProjectionData>({
    queryKey: unifiedAnalyticsKeys.costProjections(period),
    queryFn: async () => {
      return apiClient.request<CostProjectionData>(
        `/api/analytics/llm/projections?period=${period}`
      )
    },
    staleTime: 60000, // 1 minute
  })
}

// ============= ADMIN: COST ANALYTICS =============

interface AdminWorkspaceCostEntry {
  workspace_id: string
  workspace_name: string
  plan: string
  total_cost: number
  total_tokens: number
  total_requests: number
  top_model: string | null
}

interface AdminCostBreakdown {
  key: string
  input_cost: number
  output_cost: number
  total_cost: number
  request_count: number
}

interface AdminDailyCostTrend {
  date: string
  cost: number
  requests: number
}

interface AdminCostAnalyticsData {
  total_platform_cost: number
  total_tokens: number
  total_requests: number
  cost_by_workspace: AdminWorkspaceCostEntry[]
  cost_by_provider: AdminCostBreakdown[]
  daily_cost_trend: AdminDailyCostTrend[]
}

export function useAdminCostAnalytics(period: string = '30d') {
  return useQuery<AdminCostAnalyticsData | null>({
    queryKey: ['unified-analytics', 'admin', 'costs', period],
    queryFn: async () => {
      const data = await apiClient.request<AdminCostAnalyticsData>(
        `/api/admin/analytics/costs?period=${period}`
      ).catch((err: any) => {
        // 403 = not admin, 401 = unauthenticated
        if (err?.message?.includes('403') || err?.message?.includes('401')) return null
        throw err
      })
      return data
    },
    staleTime: 120000, // 2 minutes
  })
}

// ============= ADMIN: CROSS-WORKSPACE =============
export function useAdminWorkspaceAnalytics(days: number = 30) {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.adminWorkspaces(days),
    queryFn: async () => {
      // Admin-only endpoint — will need backend support
      // For now, return current workspace data as placeholder
      const [agents] = await Promise.all([
        apiClient.getAgents().catch(() => []),
      ])

      const agentList = Array.isArray(agents) ? agents : []
      const totalTokens = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_tokens || 0), 0)
      const totalCost = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_cost || 0), 0)
      const totalRequests = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_requests || 0), 0)

      return {
        platformSummary: {
          totalWorkspaces: 1,
          totalUsers: 1,
          totalApiCalls: totalRequests,
          totalTokens,
          totalCost,
        },
        workspaces: [
          {
            id: 'current',
            name: 'Current Workspace',
            plan: 'Pilot',
            users: 1,
            agents: agentList.length,
            workflows: 0,
            apiCalls: totalRequests,
            tokens: totalTokens,
            cost: totalCost,
            status: 'active',
          },
        ],
      }
    },
    staleTime: 120000,
  })
}
