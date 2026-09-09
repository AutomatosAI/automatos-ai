/**
 * Unified Analytics Hooks
 * Consolidated React Query hooks for the unified analytics page.
 * Reuses existing API endpoints and adds new ones for costs, plans, and recommendations.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, getAdminWorkspaceOverride } from '@/lib/api-client'
import {
  agentRuntimeFacts,
  cacheShare,
  mergeAgentUsage,
  meteredCost,
  routeLabel,
  type ProviderUsage,
  type UsageGroup,
} from '@/lib/analytics-usage'

/** The user's RAG documents: every workspace document that is not an agent output. */
const KNOWLEDGE_DOCUMENTS_PATH = '/api/documents/?exclude_source_type=agent_output&limit=1000'

/** The backend period for a day count (the page's 7/30/90-day selector). */
function periodFor(days: number): string {
  return days <= 1 ? '24h' : days <= 7 ? '7d' : days <= 30 ? '30d' : '90d'
}

// Each call wrapped so a synchronous throw never breaks the Promise.all
const safeRequest = <T,>(fn: () => Promise<T>, fallback: T): Promise<T> =>
  Promise.resolve().then(fn).catch((err) => {
    console.warn('[Analytics] API call failed:', err?.message || err)
    return fallback
  })

// Workspace scope for cache correctness: when admin switches workspace,
// cached data for workspace A must not bleed into workspace B.
function wsScope() {
  return getAdminWorkspaceOverride() || 'own'
}

// ============= QUERY KEYS =============
// All keys are functions (or thunks) so wsScope() is called at query time, not module load time.
export const unifiedAnalyticsKeys = {
  overview: (days: number) => ['unified-analytics', wsScope(), 'overview', days] as const,
  agents: (days: number) => ['unified-analytics', wsScope(), 'agents', days] as const,
  workflows: (days: number) => ['unified-analytics', wsScope(), 'workflows', days] as const,
  documents: (days: number) => ['unified-analytics', wsScope(), 'documents', days] as const,
  costs: (days: number) => ['unified-analytics', wsScope(), 'costs', days] as const,
  recommendations: () => ['unified-analytics', wsScope(), 'recommendations'] as const,
  planUsage: () => ['unified-analytics', wsScope(), 'plan-usage'] as const,
  adminWorkspaces: (days: number) => ['unified-analytics', wsScope(), 'admin', 'workspaces', days] as const,
  openrouterCredits: () => ['unified-analytics', wsScope(), 'openrouter', 'credits'] as const,
  openrouterKeyInfo: () => ['unified-analytics', wsScope(), 'openrouter', 'key-info'] as const,
  composioApps: (days: number) => ['unified-analytics', wsScope(), 'composio', 'apps', days] as const,
  composioActions: (days: number) => ['unified-analytics', wsScope(), 'composio', 'actions', days] as const,
  composioAgentTools: (days: number) => ['unified-analytics', wsScope(), 'composio', 'agent-tools', days] as const,
  composioExecStats: (days: number) => ['unified-analytics', wsScope(), 'composio', 'exec-stats', days] as const,
  composioPerformance: (days: number) => ['unified-analytics', wsScope(), 'composio', 'performance', days] as const,
  composioDailyVolume: (days: number) => ['unified-analytics', wsScope(), 'composio', 'daily-volume', days] as const,
  composioRecentExecs: () => ['unified-analytics', wsScope(), 'composio', 'recent-execs'] as const,
  composioErrors: (days: number) => ['unified-analytics', wsScope(), 'composio', 'errors', days] as const,
  modelComparison: (modelIds: string[], period: string) => ['unified-analytics', wsScope(), 'llm', 'comparison', modelIds, period] as const,
  costProjections: (period: string) => ['unified-analytics', wsScope(), 'llm', 'projections', period] as const,
  dailyCostByModel: (period: string) => ['unified-analytics', wsScope(), 'llm', 'daily-by-model', period] as const,
  adminDashboard: (period: string) => ['unified-analytics', wsScope(), 'admin', 'dashboard', period] as const,
}

// ============= OVERVIEW =============
export function useAnalyticsOverview(days: number = 30) {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.overview(days),
    queryFn: async () => {
      const period = periodFor(days)

      const [agents, llmSummary, projections, workflowStats, docStats, missionStats] = await Promise.all([
        safeRequest(() => apiClient.getAgents(), []),
        safeRequest(() => apiClient.request<any>(`/api/analytics/llm/summary?period=${period}`), null),
        safeRequest(() => apiClient.request<any>(`/api/analytics/llm/projections?period=${period}`), null),
        safeRequest(() => apiClient.getWorkflowStatsDashboard(), null),
        safeRequest(() => apiClient.getAnalyticsOverview(), null),
        // PRD-125 Phase 2: Fetch mission stats alongside workflow stats
        safeRequest(() => apiClient.request<any>(`/api/missions/stats?period=${period}`), null),
      ])

      const agentList = Array.isArray(agents) ? agents : []

      // Cost is what llm_usage booked for the period — never the cumulative
      // per-agent blob, which ignores the selector and misses sessions.
      const totalCost = llmSummary?.total_cost || 0
      const providers: ProviderUsage[] = Array.isArray(llmSummary?.by_provider) ? llmSummary.by_provider : []

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
        // PRD-125 Phase 2: Mission stats
        missions: {
          total: (missionStats as any)?.total_missions || 0,
          completed: Math.round(((missionStats as any)?.total_missions || 0) * ((missionStats as any)?.success_rate || 0)),
          successRate: ((missionStats as any)?.success_rate || 0) * 100,
          avgDurationMinutes: (missionStats as any)?.avg_duration_minutes || null,
          avgTokensUsed: (missionStats as any)?.avg_tokens_used || 0,
        },
        documents: {
          total: (docStats as any)?.total_documents || 0,
          storageMb: (docStats as any)?.total_storage_mb || 0,
        },
        cost: {
          currentPeriod: totalCost,
          projectedMonthly: projections?.projected_monthly || 0,
          changePercent: projections?.change_percent ?? null,
          meteredCost: meteredCost(providers),
          requests: llmSummary?.total_requests || 0,
          providers,
        },
        system: {},
      }
    },
    staleTime: 60000,
  })
}

// ============= AGENT ANALYTICS =============

interface AgentMemoryStats {
  agent_id: number
  memory_count: number
  avg_importance: number
  total_accesses: number
  memory_types: Record<string, number>
  memory_levels: Record<string, number>
  last_memory_at: string | null
}

export function useAgentAnalytics(days: number = 30) {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.agents(days),
    queryFn: async () => {
      const period = periodFor(days)
      const [agents, stats, memoryStats, usageByAgent] = await Promise.all([
        safeRequest(() => apiClient.getAgents(), []),
        safeRequest(() => apiClient.getSystemAgentStatistics(), null),
        safeRequest(() => apiClient.request<AgentMemoryStats[]>('/api/v1/memory/stats/agents'), []),
        // The period's spend per agent from llm_usage — the cumulative
        // model_usage_stats blob ignored the 7/30/90-day selector entirely.
        safeRequest(() => apiClient.request<UsageGroup[]>(`/api/analytics/llm/usage?period=${period}&group_by=agent`), []),
      ])

      const agentList = Array.isArray(agents) ? agents : []
      const memoryList = Array.isArray(memoryStats) ? memoryStats : []
      const usageRows = mergeAgentUsage(agentList, Array.isArray(usageByAgent) ? usageByAgent : [])
      const usageById = new Map(usageRows.map((row) => [row.id, row]))

      const memoryMap = new Map<number, AgentMemoryStats>()
      memoryList.forEach((m) => memoryMap.set(m.agent_id, m))

      return {
        agents: agentList.map((agent: any) => {
          const mem = memoryMap.get(agent.id)
          const usage = usageById.get(agent.id)
          const facts = agentRuntimeFacts(agent)
          return {
            id: agent.id,
            name: agent.name,
            status: agent.status,
            agentType: agent.agent_type,
            successRate: (agent.performance_metrics?.success_rate || 0) * 100,
            avgRunTime: agent.performance_metrics?.avg_execution_time || 0,
            tokensUsed: usage?.tokens || 0,
            cost: usage?.cost || 0,
            cacheReadTokens: usage?.cacheReadTokens || 0,
            errors: usage?.errors || 0,
            llmModel: facts.label,
            runtime: facts.runtime,
            billing: facts.billing,
            totalRequests: usage?.requests || 0,
            lastUsed: agent.model_usage_stats?.last_used_at || agent.updated_at,
            // Memory data
            memoryCount: mem?.memory_count || 0,
            avgImportance: mem?.avg_importance || 0,
            totalAccesses: mem?.total_accesses || 0,
            memoryTypes: mem?.memory_types || {},
            memoryLevels: mem?.memory_levels || {},
            lastMemoryAt: mem?.last_memory_at || null,
          }
        }),
        summary: (() => {
          // Calculate avg success rate from per-agent data (not workflow executions)
          const agentsWithRate = agentList.filter((a: any) => (a.performance_metrics?.success_rate || 0) > 0)
          const avgRate = agentsWithRate.length > 0
            ? (agentsWithRate.reduce((sum: number, a: any) => sum + (a.performance_metrics?.success_rate || 0), 0) / agentsWithRate.length) * 100
            : (stats as any)?.average_performance || 0

          return {
            totalAgents: agentList.length,
            activeAgents: agentList.filter((a: any) => a.status === 'active').length,
            avgSuccessRate: avgRate,
            totalTokens: usageRows.reduce((sum, row) => sum + row.tokens, 0),
            totalCost: usageRows.reduce((sum, row) => sum + row.cost, 0),
            sessionAgents: agentList.filter((a: any) => agentRuntimeFacts(a).runtime === 'cli').length,
          }
        })(),
        ranking: [],
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
      const period = periodFor(days)

      const [workflows, stats, recipesResp, recipeStats, missionsResp, missionStats, usageByExecution] = await Promise.all([
        safeRequest(() => apiClient.getWorkflows(), []),
        safeRequest(() => apiClient.getWorkflowStatsDashboard(), null),
        safeRequest(() => apiClient.listWorkflowRecipes({ limit: 100 }), null),
        safeRequest(() => apiClient.request<any>('/api/workflow-recipes/stats/dashboard'), null),
        safeRequest(() => apiClient.request<any>(`/api/missions?limit=100`), null),
        safeRequest(() => apiClient.request<any>(`/api/missions/stats?period=${period}`), null),
        // What each mission's tasks actually spent (llm_usage rows tagged mission:<id>)
        safeRequest(() => apiClient.request<UsageGroup[]>(`/api/analytics/llm/usage?period=${period}&group_by=execution`), []),
      ])

      const workflowList = Array.isArray(workflows) ? workflows : []
      const spendByMission = new Map<string, UsageGroup>()
      for (const group of Array.isArray(usageByExecution) ? usageByExecution : []) {
        if (group.key.startsWith('mission:')) spendByMission.set(group.key.slice('mission:'.length), group)
      }

      // Normalize missions into same shape as workflows for the table
      const missionList = (missionsResp?.missions || []).map((m: any) => {
        const isCompleted = m.state === 'completed'
        const isFailed = m.state === 'failed'
        const durationMs = m.started_at && m.completed_at
          ? new Date(m.completed_at).getTime() - new Date(m.started_at).getTime()
          : 0
        const durationStr = durationMs > 0 ? `${Math.round(durationMs / 1000)}s` : '0s'
        const spend = spendByMission.get(String(m.id))
        return {
          id: `mission-${m.id}`,
          name: m.goal || 'Untitled Mission',
          status: m.state,
          totalRuns: 1,
          successRate: isCompleted ? 100 : isFailed ? 0 : -1,
          avgDuration: durationStr,
          tokensUsed: spend?.total_tokens || m.tokens_used || 0,
          cost: spend?.total_cost || 0,
          lastRun: m.completed_at || m.started_at || m.created_at,
          source: 'mission' as const,
        }
      })

      // Combine both sources
      const combined = [
        ...workflowList.map((wf: any) => ({
          id: wf.id,
          name: wf.name,
          status: wf.status,
          totalRuns: wf.execution_count || 0,
          successRate: wf.success_rate || 0,
          avgDuration: wf.avg_duration || '0s',
          tokensUsed: wf.total_tokens || 0,
          cost: wf.total_cost || 0,
          lastRun: wf.last_executed_at || wf.updated_at,
          source: 'workflow' as const,
        })),
        ...missionList,
      ]

      // Normalize recipes
      const rawRecipes = (recipesResp as any)?.items || (Array.isArray(recipesResp) ? recipesResp : [])
      const recipes = rawRecipes.map((r: any) => ({
        id: r.id,
        templateId: r.template_id,
        name: r.name,
        description: r.description || '',
        useCount: r.use_count || 0,
        successRate: r.success_rate || 0,
        qualityScore: r.quality_score,
        stepsCount: r.steps?.length || 0,
        tags: r.tags || [],
        lastUsedAt: r.last_used_at,
        createdAt: r.created_at,
        isSystem: r.is_system || false,
        isFeatured: r.is_featured || false,
      }))

      const recipeSummary = {
        totalRecipes: recipeStats?.overview?.total_recipes ?? recipes.length,
        totalExecutions: recipeStats?.overview?.total_executions ?? 0,
        avgQualityScore: recipeStats?.overview?.avg_quality_score ?? 0,
        avgSuccessRate: recipeStats?.overview?.avg_success_rate ?? 0,
        statusBreakdown: recipeStats?.status_breakdown ?? null,
      }

      const totalItems = combined.length
      const missionSuccessRate = missionStats?.success_rate ?? 0
      const legacySuccessRate = (stats as any)?.today?.success_rate_today || 0
      const blendedSuccessRate = missionList.length > 0 && workflowList.length > 0
        ? (legacySuccessRate * workflowList.length + missionSuccessRate * missionList.length) / totalItems
        : missionList.length > 0 ? missionSuccessRate : legacySuccessRate

      return {
        workflows: combined,
        summary: {
          totalWorkflows: totalItems,
          totalExecutions: ((stats as any)?.overview?.total_executions || 0) + (missionStats?.total_missions || 0),
          successRate: blendedSuccessRate,
          avgDuration: (stats as any)?.today?.avg_duration_today || '0s',
          sources: { workflows: workflowList.length, missions: missionList.length },
        },
        stats,
        recipes,
        recipeSummary,
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
      const period = periodFor(days)

      const [documents, usage] = await Promise.all([
        // The user's RAG documents only — agent outputs (reports, digests, mission
        // syntheses) live in the Deliverables explorer, not in the knowledge base.
        safeRequest(() => apiClient.request<any[]>(KNOWLEDGE_DOCUMENTS_PATH), []),
        safeRequest(() => apiClient.request<any>(`/api/documents/analytics/usage?period=${period}`), null),
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
          totalRagQueries: (usage?.event_counts?.rag_query || 0) + (usage?.event_counts?.document_searched || 0),
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
      const period = periodFor(days)

      const [summary, usageByRoute, usageByProvider, usageByAgent, usageByLane, agents] = await Promise.all([
        safeRequest(() => apiClient.request<any>(`/api/analytics/llm/summary?period=${period}`), null),
        safeRequest(() => apiClient.request<UsageGroup[]>(`/api/analytics/llm/usage?period=${period}&group_by=route`), []),
        safeRequest(() => apiClient.request<UsageGroup[]>(`/api/analytics/llm/usage?period=${period}&group_by=provider`), []),
        safeRequest(() => apiClient.request<UsageGroup[]>(`/api/analytics/llm/usage?period=${period}&group_by=agent`), []),
        safeRequest(() => apiClient.request<UsageGroup[]>(`/api/analytics/llm/usage?period=${period}&group_by=request_type`), []),
        safeRequest(() => apiClient.getAgents(), []),
      ])

      const agentList = Array.isArray(agents) ? agents : []
      const routeRows: UsageGroup[] = Array.isArray(usageByRoute) ? usageByRoute : []
      const providerRows: UsageGroup[] = Array.isArray(usageByProvider) ? usageByProvider : []
      const laneRows: UsageGroup[] = Array.isArray(usageByLane) ? usageByLane : []
      const byAgent = mergeAgentUsage(agentList, Array.isArray(usageByAgent) ? usageByAgent : [])
      const providers: ProviderUsage[] = Array.isArray(summary?.by_provider) ? summary.by_provider : []

      const totalTokens = summary?.total_tokens || 0
      const totalCost = summary?.total_cost || 0
      const totalRequests = summary?.total_requests || 0

      const byModel = routeRows.map((row) => ({
        key: row.key,
        model: row.model_id || row.key,
        label: routeLabel(row),
        provider: row.provider || 'unknown',
        providerLabel: row.provider_label || row.provider || 'unknown',
        billing: row.billing || 'unknown',
        requests: row.request_count || 0,
        inputTokens: row.input_tokens || 0,
        outputTokens: row.output_tokens || 0,
        cacheReadTokens: row.cache_read_tokens || 0,
        totalCost: row.total_cost || 0,
        errors: row.error_count || 0,
        avgLatencyMs: row.avg_latency_ms ?? null,
        avgCostPerRequest: row.request_count > 0 ? (row.total_cost || 0) / row.request_count : 0,
      }))

      const byProvider = providerRows.map((row) => ({
        provider: row.provider || row.key,
        label: row.provider_label || row.label || row.key,
        billing: row.billing || 'unknown',
        requests: row.request_count || 0,
        tokens: row.total_tokens || 0,
        cacheReadTokens: row.cache_read_tokens || 0,
        cost: row.total_cost || 0,
        errors: row.error_count || 0,
        avgLatencyMs: row.avg_latency_ms ?? null,
        share: totalCost > 0 ? (row.total_cost || 0) / totalCost : 0,
        routes: byModel.filter((m) => m.provider === (row.provider || row.key)).length,
      }))

      const byLane = laneRows.map((row) => ({
        lane: row.key,
        requests: row.request_count || 0,
        tokens: row.total_tokens || 0,
        cost: row.total_cost || 0,
        errors: row.error_count || 0,
        avgLatencyMs: row.avg_latency_ms ?? null,
      }))

      const topSpender = byAgent[0]

      return {
        summary: {
          totalTokens,
          totalCost,
          totalRequests,
          meteredCost: meteredCost(providers),
          cacheReadTokens: summary?.cache_read_tokens || 0,
          cacheShare: cacheShare(routeRows),
          errorRate: summary?.error_rate || 0,
          avgLatencyMs: summary?.avg_latency_ms ?? null,
          costPerTask: totalRequests > 0 ? totalCost / totalRequests : 0,
          mostExpensiveAgent: topSpender && (topSpender.cost > 0 || topSpender.tokens > 0) ? {
            name: topSpender.name,
            cost: topSpender.cost,
            tokens: topSpender.tokens,
            model: topSpender.modelLabel,
            billing: topSpender.billing,
          } : null,
        },
        byModel,
        byProvider,
        byLane,
        byAgent,
        costTrend: (summary?.cost_trend || []).map((t: any) => ({
          date: t.date,
          total_cost: t.cost,
        })),
      }
    },
    staleTime: 60000,
  })
}

// ============= PLAN USAGE =============
export function usePlanUsage() {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.planUsage(),
    queryFn: async () => {
      // For now, return placeholder limits (pilot phase — limits TBD)
      const [agents, missions, documents, llmSummary] = await Promise.all([
        apiClient.getAgents().catch(() => []),
        apiClient.request<any>('/api/missions?limit=100').catch(() => ({ items: [] })),
        apiClient.request<any[]>(KNOWLEDGE_DOCUMENTS_PATH).catch(() => []),
        // 30-day LLM calls and tokens from llm_usage — the per-agent blob undercounted both
        apiClient.request<any>('/api/analytics/llm/summary?period=30d').catch(() => null),
      ])

      const agentList = Array.isArray(agents) ? agents : []
      const missionItems = missions?.missions || missions?.items || (Array.isArray(missions) ? missions : [])
      const docList = Array.isArray(documents) ? documents : []

      const totalTokens = llmSummary?.total_tokens || 0
      const totalRequests = llmSummary?.total_requests || 0
      const storageMb = docList.reduce((sum: number, d: any) => sum + (d.file_size || d.size || 0), 0) / (1024 * 1024)

      return {
        planName: 'Pilot',
        planTier: 'pilot',
        usage: {
          agents: { used: agentList.length, limit: null as number | null, label: 'Agents' },
          missions: { used: missionItems.length, limit: null as number | null, label: 'Missions' },
          documents: { used: docList.length, limit: null as number | null, label: 'Documents' },
          storageGb: { used: parseFloat((storageMb / 1024).toFixed(2)), limit: null as number | null, label: 'Storage (GB)' },
          apiCalls: { used: totalRequests, limit: null as number | null, label: 'LLM Calls (30 days)' },
          tokens: { used: totalTokens, limit: null as number | null, label: 'Tokens (30 days)' },
        },
      }
    },
    staleTime: 120000,
  })
}

// ============= RECOMMENDATIONS (AI-powered, cached daily) =============
export function useRecommendations() {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.recommendations(),
    queryFn: async () => {
      // Fetch real data in parallel: backend LLM recommendations + agent list + LLM summary
      const [backendRecs, agents, llmSummary] = await Promise.all([
        safeRequest(() => apiClient.request<any[]>('/api/analytics/llm/recommendations'), []),
        safeRequest(() => apiClient.getAgents(), []),
        safeRequest(() => apiClient.request<any>('/api/analytics/llm/summary?period=30d'), null),
      ])

      const agentList = Array.isArray(agents) ? agents : []
      const recommendations: Array<{
        id: string
        type: 'cost' | 'performance' | 'document' | 'quota'
        title: string
        description: string
        impact: string
        action?: string
      }> = []

      // Include backend LLM cost optimization recommendations
      const backendRecList = Array.isArray(backendRecs) ? backendRecs : []
      backendRecList.forEach((rec: any, idx: number) => {
        if (rec.type === 'info') return // Skip "no suggestions" placeholder
        const agentName = rec.affected_agent_ids?.[0]
          ? agentList.find((a: any) => a.id === rec.affected_agent_ids[0])?.name || `Agent ${rec.affected_agent_ids[0]}`
          : ''
        recommendations.push({
          id: `llm-${idx}`,
          type: 'cost',
          title: agentName ? rec.title.replace(`Agent ${rec.affected_agent_ids[0]}`, agentName) : rec.title,
          description: rec.description,
          impact: rec.potential_savings ? `Potential savings: $${rec.potential_savings}/month` : 'Cost optimization',
        })
      })

      // Real LLM summary stats
      const totalCost = llmSummary?.total_cost || 0
      const totalTokens = llmSummary?.total_tokens || 0
      const totalRequests = llmSummary?.total_requests || 0
      const activeAgents = agentList.filter((a: any) => a.status === 'active').length

      // Agent-level insights from agent data
      const agentsWithNoSkills = agentList.filter((a: any) => a.status === 'active' && (!a.skills || a.skills.length === 0))
      const agentsWithNoTools = agentList.filter((a: any) => a.status === 'active' && (!a.tools || a.tools.length === 0))

      if (agentsWithNoSkills.length > 0) {
        recommendations.push({
          id: 'no-skills',
          type: 'performance',
          title: `${agentsWithNoSkills.length} active agent${agentsWithNoSkills.length > 1 ? 's have' : ' has'} no skills assigned`,
          description: `${agentsWithNoSkills.map((a: any) => a.name).slice(0, 3).join(', ')}${agentsWithNoSkills.length > 3 ? ` and ${agentsWithNoSkills.length - 3} more` : ''} — assign skills to improve routing accuracy.`,
          impact: 'Better agent routing',
        })
      }

      if (agentsWithNoTools.length > 0 && agentsWithNoTools.length < agentList.length) {
        recommendations.push({
          id: 'no-tools',
          type: 'performance',
          title: `${agentsWithNoTools.length} agent${agentsWithNoTools.length > 1 ? 's have' : ' has'} no connected tools`,
          description: `${agentsWithNoTools.map((a: any) => a.name).slice(0, 3).join(', ')} — connect Composio tools so agents can take real actions.`,
          impact: 'Enable tool usage',
        })
      }

      // Cost summary if available
      if (totalCost > 0) {
        const costPerRequest = totalRequests > 0 ? totalCost / totalRequests : 0
        recommendations.push({
          id: 'cost-summary',
          type: 'cost',
          title: `$${totalCost.toFixed(2)} spent across ${totalRequests.toLocaleString()} LLM requests`,
          description: `Average $${costPerRequest.toFixed(4)}/request. ${totalTokens.toLocaleString()} tokens used across ${activeAgents} active agent${activeAgents !== 1 ? 's' : ''}.`,
          impact: 'Cost overview',
        })
      }

      // Getting started hint if no LLM usage
      if (totalRequests === 0 && agentList.length > 0) {
        recommendations.push({
          id: 'no-usage',
          type: 'quota',
          title: 'No LLM usage tracked yet',
          description: `${agentList.length} agents configured. Chat with an agent or run a recipe to start tracking usage and costs.`,
          impact: 'Getting started',
        })
      }

      const typePriority: Record<string, number> = { cost: 0, performance: 1, document: 2, quota: 3 }
      recommendations.sort((a, b) => (typePriority[a.type] ?? 9) - (typePriority[b.type] ?? 9))
      return recommendations.slice(0, 5)
    },
    staleTime: 5 * 60 * 1000, // 5 min (was 24h — too stale)
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
    queryKey: unifiedAnalyticsKeys.openrouterCredits(),
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
    queryKey: unifiedAnalyticsKeys.openrouterKeyInfo(),
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
      queryClient.invalidateQueries({ queryKey: unifiedAnalyticsKeys.openrouterCredits() })
      queryClient.invalidateQueries({ queryKey: unifiedAnalyticsKeys.openrouterKeyInfo() })
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

// ============= COMPOSIO API MONITORING =============

export interface ComposioExecStats {
  total_executions: number
  success_count: number
  error_count: number
  timeout_count: number
  success_rate: number
  error_rate: number
  avg_latency_ms: number
  p50_latency_ms: number | null
  p95_latency_ms: number | null
  max_latency_ms: number | null
  cache_hit_rate: number
  unique_actions: number
  unique_apps: number
}

export interface ComposioActionPerformance {
  action_name: string
  app_name: string
  total_calls: number
  success_count: number
  error_count: number
  error_rate: number
  avg_latency_ms: number
  max_latency_ms: number
  cache_hit_rate: number
  last_executed: string | null
}

export interface ComposioDailyVolume {
  date: string
  total: number
  successes: number
  errors: number
  avg_latency_ms: number
}

export interface ComposioRecentExecution {
  id: number
  agent_name: string | null
  app_name: string
  action_name: string
  status: string
  execution_time_ms: number | null
  error_message: string | null
  cache_hit: boolean
  executed_at: string | null
}

export interface ComposioErrorBreakdown {
  error_code: string | null
  error_message: string
  count: number
  last_seen: string | null
  app_name: string
  action_name: string
}

export function useComposioExecStats(days: number = 30) {
  return useQuery<ComposioExecStats>({
    queryKey: unifiedAnalyticsKeys.composioExecStats(days),
    queryFn: () => apiClient.request<ComposioExecStats>(`/api/analytics/composio/execution-stats?days=${days}`),
    staleTime: 60000,
  })
}

export function useComposioPerformance(days: number = 30) {
  return useQuery<ComposioActionPerformance[]>({
    queryKey: unifiedAnalyticsKeys.composioPerformance(days),
    queryFn: () => apiClient.request<ComposioActionPerformance[]>(`/api/analytics/composio/performance-by-action?days=${days}`),
    staleTime: 60000,
  })
}

export function useComposioDailyVolume(days: number = 30) {
  return useQuery<ComposioDailyVolume[]>({
    queryKey: unifiedAnalyticsKeys.composioDailyVolume(days),
    queryFn: () => apiClient.request<ComposioDailyVolume[]>(`/api/analytics/composio/daily-volume?days=${days}`),
    staleTime: 60000,
  })
}

export function useComposioRecentExecs() {
  return useQuery<ComposioRecentExecution[]>({
    queryKey: unifiedAnalyticsKeys.composioRecentExecs(),
    queryFn: () => apiClient.request<ComposioRecentExecution[]>('/api/analytics/composio/recent-executions?limit=20'),
    staleTime: 30000, // 30s for live-ish monitoring
  })
}

export function useComposioErrors(days: number = 30) {
  return useQuery<ComposioErrorBreakdown[]>({
    queryKey: unifiedAnalyticsKeys.composioErrors(days),
    queryFn: () => apiClient.request<ComposioErrorBreakdown[]>(`/api/analytics/composio/error-breakdown?days=${days}`),
    staleTime: 60000,
  })
}

// ============= MODEL COMPARISON =============

interface ModelComparisonItem {
  model_id: string
  display_name: string
  provider: string
  provider_label?: string | null
  billing?: string | null
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
  current_period_tokens?: number
  label?: string | null
  model_id?: string | null
  provider?: string | null
  provider_label?: string | null
  billing?: string | null
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

// ============= DAILY COST BY MODEL (multi-line chart) =============

export interface DailyRouteFacts {
  key: string
  model_id: string
  provider: string
  provider_label: string
  billing: string
  label: string
  total_cost: number
  total_tokens: number
  request_count: number
}

interface DailyCostByModelData {
  /** Series keys: ``<model>@<provider>`` — one line per ROUTE */
  models: string[]
  routes: DailyRouteFacts[]
  series: Record<string, any>[]
}

export function useDailyCostByModel(period: string = '30d') {
  return useQuery<DailyCostByModelData | null>({
    queryKey: unifiedAnalyticsKeys.dailyCostByModel(period),
    queryFn: async () => {
      return apiClient.request<DailyCostByModelData>(
        `/api/analytics/llm/costs/daily-by-model?period=${period}`
      ).catch(() => null)
    },
    staleTime: 60000,
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
    queryKey: ['unified-analytics', wsScope(), 'admin', 'costs', period],
    queryFn: async () => {
      // Fetch from backend + agent data as fallback
      const [backendData, agents] = await Promise.all([
        safeRequest(() => apiClient.request<AdminCostAnalyticsData>(
          `/api/admin/analytics/costs?period=${period}`
        ), null),
        safeRequest(() => apiClient.getAgents(), []),
      ])

      // If backend returned real data with actual costs, use it
      if (backendData && backendData.total_platform_cost > 0) {
        return backendData
      }

      // Fallback: build from agent model_usage_stats
      const agentList = Array.isArray(agents) ? agents : []
      const totalCost = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_cost || 0), 0)
      const totalTokens = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_tokens || 0), 0)
      const totalRequests = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_requests || 0), 0)

      // Build provider breakdown from agent model configs
      const providerMap: Record<string, { cost: number; requests: number }> = {}
      agentList.forEach((a: any) => {
        const provider = a.agent_model_config?.provider || 'unknown'
        if (!providerMap[provider]) providerMap[provider] = { cost: 0, requests: 0 }
        providerMap[provider].cost += a.model_usage_stats?.total_cost || 0
        providerMap[provider].requests += a.model_usage_stats?.total_requests || 0
      })

      return {
        total_platform_cost: totalCost,
        total_tokens: totalTokens,
        total_requests: totalRequests,
        cost_by_workspace: [],
        cost_by_provider: Object.entries(providerMap)
          .filter(([, d]) => d.cost > 0)
          .map(([key, d]) => ({
            key,
            input_cost: 0,
            output_cost: 0,
            total_cost: d.cost,
            request_count: d.requests,
          })),
        daily_cost_trend: [],
      } as AdminCostAnalyticsData
    },
    staleTime: 120000, // 2 minutes
  })
}

// ============= ADMIN: COMPREHENSIVE DASHBOARD =============

interface AdminDashboardData {
  overview: {
    total_cost: number
    total_tokens: number
    total_requests: number
    total_workspaces: number
    daily_average: number
    projected_monthly: number
  }
  byok_split: {
    platform_cost: number
    platform_requests: number
    byok_cost: number
    byok_requests: number
  }
  workspaces: Array<{
    id: string
    name: string
    plan: string
    is_personal: boolean
    created_at: string | null
    agents: number
    recipes: number
    executions: number
    cost: number
    tokens: number
    requests: number
  }>
  models: Array<{
    key: string
    model_id: string
    provider: string
    provider_label: string
    billing: string
    label: string
    cost: number
    tokens: number
    requests: number
    workspace_count: number
  }>
  daily_by_provider: {
    providers: string[]
    labels: Record<string, string>
    series: Record<string, any>[]
  }
}

export function useAdminDashboard(period: string = '30d') {
  return useQuery<AdminDashboardData | null>({
    queryKey: unifiedAnalyticsKeys.adminDashboard(period),
    queryFn: async () => {
      return apiClient.request<AdminDashboardData>(
        `/api/admin/analytics/dashboard?period=${period}`
      ).catch((err) => {
        console.warn('[Analytics] Admin dashboard failed:', err?.message || err)
        return null
      })
    },
    staleTime: 60000,
  })
}

// ============= ADMIN: CROSS-WORKSPACE =============
export function useAdminWorkspaceAnalytics(days: number = 30) {
  const period = periodFor(days)
  return useQuery({
    queryKey: unifiedAnalyticsKeys.adminWorkspaces(days),
    queryFn: async () => {
      const data = await apiClient.request<{
        overview: {
          total_cost: number
          total_tokens: number
          total_requests: number
          total_workspaces: number
          daily_average: number
          projected_monthly: number
        }
        workspaces: Array<{
          id: string
          name: string
          plan: string
          is_personal: boolean
          created_at: string | null
          agents: number
          recipes: number
          executions: number
          cost: number
          tokens: number
          requests: number
        }>
      }>(`/api/admin/analytics/dashboard?period=${period}`)

      const overview = data.overview
      return {
        platformSummary: {
          totalWorkspaces: overview.total_workspaces,
          totalUsers: overview.total_workspaces,
          totalApiCalls: overview.total_requests,
          totalTokens: overview.total_tokens,
          totalCost: overview.total_cost,
          dailyAverage: overview.daily_average,
          projectedMonthly: overview.projected_monthly,
        },
        workspaces: data.workspaces.map((ws) => ({
          id: ws.id,
          name: ws.name,
          plan: ws.plan,
          users: 1,
          agents: ws.agents,
          workflows: ws.recipes,
          executions: ws.executions,
          apiCalls: ws.requests,
          tokens: ws.tokens,
          cost: ws.cost,
          status: 'active' as const,
        })),
      }
    },
    staleTime: 120000,
  })
}
