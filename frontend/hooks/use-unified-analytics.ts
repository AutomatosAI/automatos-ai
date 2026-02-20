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
  dailyCostByModel: (period: string) => ['unified-analytics', 'llm', 'daily-by-model', period] as const,
  adminDashboard: (period: string) => ['unified-analytics', 'admin', 'dashboard', period] as const,
}

// ============= OVERVIEW =============
export function useAnalyticsOverview(days: number = 30) {
  return useQuery({
    queryKey: unifiedAnalyticsKeys.overview(days),
    queryFn: async () => {
      const period = days <= 1 ? '24h' : days <= 7 ? '7d' : days <= 30 ? '30d' : '90d'

      // Each call wrapped in try/catch to prevent synchronous TypeError from breaking Promise.all
      const safeRequest = <T,>(fn: () => Promise<T>, fallback: T): Promise<T> =>
        Promise.resolve().then(fn).catch((err) => {
          console.warn('[Analytics] API call failed:', err?.message || err)
          return fallback
        })

      const [agents, llmSummary, workflowStats, docStats] = await Promise.all([
        safeRequest(() => apiClient.getAgents(), []),
        safeRequest(() => apiClient.request<any>(`/api/analytics/llm/summary?period=${period}`), null),
        safeRequest(() => apiClient.getWorkflowStatsDashboard(), null),
        safeRequest(() => apiClient.getAnalyticsOverview(), null),
      ])

      const agentList = Array.isArray(agents) ? agents : []

      // Cost: prefer llm_usage table, fallback to agent model_usage_stats
      const llmCost = llmSummary?.total_cost || 0
      const agentCost = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_cost || 0), 0)
      const totalCost = llmCost > 0 ? llmCost : agentCost

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
          currentPeriod: totalCost,
          previousPeriod: 0,
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
      const safeRequest = <T,>(fn: () => Promise<T>, fallback: T): Promise<T> =>
        Promise.resolve().then(fn).catch((err) => {
          console.warn('[Analytics] API call failed:', err?.message || err)
          return fallback
        })

      const [agents, stats, memoryStats] = await Promise.all([
        safeRequest(() => apiClient.getAgents(), []),
        safeRequest(() => apiClient.getSystemAgentStatistics(), null),
        safeRequest(() => apiClient.request<AgentMemoryStats[]>('/api/v1/memory/stats/agents'), []),
      ])

      const agentList = Array.isArray(agents) ? agents : []
      const memoryList = Array.isArray(memoryStats) ? memoryStats : []

      // Build lookup map by agent_id
      const memoryMap = new Map<number, AgentMemoryStats>()
      memoryList.forEach((m) => memoryMap.set(m.agent_id, m))

      return {
        agents: agentList.map((agent: any) => {
          const mem = memoryMap.get(agent.id)
          return {
            id: agent.id,
            name: agent.name,
            status: agent.status,
            agentType: agent.agent_type,
            successRate: agent.performance_metrics?.success_rate || 0,
            avgRunTime: agent.performance_metrics?.avg_execution_time || 0,
            tokensUsed: agent.model_usage_stats?.total_tokens || 0,
            cost: agent.model_usage_stats?.total_cost || 0,
            llmModel: agent.agent_model_config?.model_id || 'unknown',
            totalRequests: agent.model_usage_stats?.total_requests || 0,
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
        summary: {
          totalAgents: agentList.length,
          activeAgents: agentList.filter((a: any) => a.status === 'active').length,
          avgSuccessRate: (stats as any)?.average_performance || 0,
          totalTokens: agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_tokens || 0), 0),
          totalCost: agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_cost || 0), 0),
        },
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
      const safeRequest = <T,>(fn: () => Promise<T>, fallback: T): Promise<T> =>
        Promise.resolve().then(fn).catch((err) => {
          console.warn('[Analytics] API call failed:', err?.message || err)
          return fallback
        })

      const [workflows, stats, recipesResp, recipeStats] = await Promise.all([
        safeRequest(() => apiClient.getWorkflows(), []),
        safeRequest(() => apiClient.getWorkflowStatsDashboard(), null),
        safeRequest(() => apiClient.listWorkflowRecipes({ limit: 100 }), null),
        safeRequest(() => apiClient.request<any>('/api/workflow-recipes/stats/dashboard'), null),
      ])

      const workflowList = Array.isArray(workflows) ? workflows : []

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
      const period = days <= 1 ? '24h' : days <= 7 ? '7d' : days <= 30 ? '30d' : '90d'

      const safeRequest = <T,>(fn: () => Promise<T>, fallback: T): Promise<T> =>
        Promise.resolve().then(fn).catch((err) => {
          console.warn('[Analytics] API call failed:', err?.message || err)
          return fallback
        })

      const [documents, usage] = await Promise.all([
        safeRequest(() => apiClient.getDocuments(), []),
        safeRequest(() => fetch(`/api/documents/analytics/usage?period=${period}`)
          .then(r => r.ok ? r.json() : null), null),
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
      const period = days <= 1 ? '24h' : days <= 7 ? '7d' : days <= 30 ? '30d' : '90d'

      const safeRequest = <T,>(fn: () => Promise<T>, fallback: T): Promise<T> =>
        Promise.resolve().then(fn).catch((err) => {
          console.warn('[Analytics] API call failed:', err?.message || err)
          return fallback
        })

      // Fetch from backend LLM analytics + agent data as fallback
      const [summary, usageByModel, agents] = await Promise.all([
        safeRequest(() => apiClient.request<any>(`/api/analytics/llm/summary?period=${period}`), null),
        safeRequest(() => apiClient.request<any[]>(`/api/analytics/llm/usage?period=${period}&group_by=model`), []),
        safeRequest(() => apiClient.getAgents(), []),
      ])

      const agentList = Array.isArray(agents) ? agents : []
      const modelRows = Array.isArray(usageByModel) ? usageByModel : []

      // llm_usage totals
      const llmTokens = summary?.total_tokens || 0
      const llmCost = summary?.total_cost || 0
      const llmRequests = summary?.total_requests || 0

      // agent model_usage_stats totals (fallback)
      const agentTokens = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_tokens || 0), 0)
      const agentCost = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_cost || 0), 0)
      const agentRequests = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_requests || 0), 0)

      // Use llm_usage data when available, fallback to agent stats
      const hasLlmData = llmTokens > 0 || llmCost > 0
      const totalTokens = hasLlmData ? llmTokens : agentTokens
      const totalCost = hasLlmData ? llmCost : agentCost
      const totalRequests = hasLlmData ? llmRequests : agentRequests

      // Build byModel: prefer backend breakdown, fallback to agent-derived grouping
      let byModel: any[]
      if (modelRows.length > 0) {
        byModel = modelRows.map((row: any) => ({
          model: row.key || 'unknown',
          requests: row.request_count || 0,
          inputTokens: row.input_tokens || 0,
          outputTokens: row.output_tokens || 0,
          totalCost: row.total_cost || 0,
          avgCostPerRequest: row.request_count > 0 ? (row.total_cost || 0) / row.request_count : 0,
          agentCount: 0,
        }))
      } else {
        // Derive from agent model_usage_stats
        const modelMap: Record<string, { requests: number; inputTokens: number; outputTokens: number; cost: number; agents: Set<string> }> = {}
        agentList.forEach((agent: any) => {
          const model = agent.agent_model_config?.model_id || 'unknown'
          if (!modelMap[model]) modelMap[model] = { requests: 0, inputTokens: 0, outputTokens: 0, cost: 0, agents: new Set() }
          modelMap[model].requests += agent.model_usage_stats?.total_requests || 0
          modelMap[model].inputTokens += agent.model_usage_stats?.input_tokens || 0
          modelMap[model].outputTokens += agent.model_usage_stats?.output_tokens || 0
          modelMap[model].cost += agent.model_usage_stats?.total_cost || 0
          modelMap[model].agents.add(agent.name)
        })
        byModel = Object.entries(modelMap).map(([model, data]) => ({
          model,
          requests: data.requests,
          inputTokens: data.inputTokens,
          outputTokens: data.outputTokens,
          totalCost: data.cost,
          avgCostPerRequest: data.requests > 0 ? data.cost / data.requests : 0,
          agentCount: data.agents.size,
        })).filter(m => m.requests > 0 || m.totalCost > 0)
      }

      // Most expensive agent
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
          mostExpensiveAgent: mostExpensive?.model_usage_stats?.total_cost > 0 ? {
            name: mostExpensive.name,
            cost: mostExpensive.model_usage_stats.total_cost,
            model: mostExpensive.agent_model_config?.model_id || 'unknown',
          } : null,
        },
        byModel,
        byAgent: agentList.map((agent: any) => ({
          id: agent.id,
          name: agent.name,
          model: agent.agent_model_config?.model_id || 'unknown',
          tokens: agent.model_usage_stats?.total_tokens || 0,
          cost: agent.model_usage_stats?.total_cost || 0,
          requests: agent.model_usage_stats?.total_requests || 0,
        })).sort((a: any, b: any) => b.cost - a.cost),
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
      const totalCost = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_cost || 0), 0)
      const totalTokens = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_tokens || 0), 0)
      const totalRequests = agentList.reduce((sum: number, a: any) => sum + (a.model_usage_stats?.total_requests || 0), 0)
      const activeAgents = agentList.filter((a: any) => a.status === 'active').length
      const unusedAgents = agentList.filter((a: any) => !a.model_usage_stats?.total_requests || a.model_usage_stats.total_requests === 0)

      agentList.forEach((agent: any) => {
        const model = agent.agent_model_config?.model_id || ''
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

      // Summary insight: unused agents
      if (unusedAgents.length > 0 && agentList.length > 1) {
        recommendations.push({
          id: 'unused-agents',
          type: 'performance',
          title: `${unusedAgents.length} agent${unusedAgents.length > 1 ? 's have' : ' has'} never been used`,
          description: `${unusedAgents.map((a: any) => a.name).slice(0, 3).join(', ')}${unusedAgents.length > 3 ? ` and ${unusedAgents.length - 3} more` : ''} — consider assigning them to workflows or removing unused ones.`,
          impact: 'Cleaner workspace',
        })
      }

      // Summary insight: total spend
      if (totalCost > 0) {
        const costPerRequest = totalRequests > 0 ? totalCost / totalRequests : 0
        recommendations.push({
          id: 'cost-summary',
          type: 'cost',
          title: `$${totalCost.toFixed(2)} spent across ${totalRequests.toLocaleString()} requests`,
          description: `Average cost per request: $${costPerRequest.toFixed(4)}. ${totalTokens.toLocaleString()} tokens used across ${activeAgents} active agent${activeAgents !== 1 ? 's' : ''}.`,
          impact: 'Cost overview',
        })
      }

      // Insight: no usage data yet
      if (totalRequests === 0 && agentList.length > 0) {
        recommendations.push({
          id: 'no-usage',
          type: 'quota',
          title: `${agentList.length} agents configured but no LLM usage tracked yet`,
          description: 'Start a chat conversation or run a recipe to begin tracking token usage and costs automatically.',
          impact: 'Getting started',
        })
      }

      // Sort: cost first, then performance, then others
      const typePriority: Record<string, number> = { cost: 0, performance: 1, document: 2, quota: 3 }
      recommendations.sort((a, b) => (typePriority[a.type] ?? 9) - (typePriority[b.type] ?? 9))
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
      const safeRequest = <T,>(fn: () => Promise<T>, fallback: T): Promise<T> =>
        Promise.resolve().then(fn).catch((err) => {
          console.warn('[Analytics] API call failed:', err?.message || err)
          return fallback
        })

      const [memoryData, recentMemories] = await Promise.all([
        safeRequest(() => apiClient.request('/api/v1/memory/stats/real'), null),
        safeRequest(() => apiClient.request<any[]>('/api/v1/memory/stats/recent?limit=8'), []),
      ])

      const items = Array.isArray(recentMemories) ? recentMemories.map((mem: any) => ({
        key: mem.memory_type || 'memory',
        value: mem.content || `${mem.memory_level || 'memory'} (importance: ${mem.importance ?? 'N/A'})`,
        source: mem.agent_id ? 'learned' : 'system',
        updated_at: mem.created_at,
      })) : []

      return {
        items,
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

// ============= DAILY COST BY MODEL (multi-line chart) =============

interface DailyCostByModelData {
  models: string[]
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
    queryKey: ['unified-analytics', 'admin', 'costs', period],
    queryFn: async () => {
      const safeRequest = <T,>(fn: () => Promise<T>, fallback: T): Promise<T> =>
        Promise.resolve().then(fn).catch((err) => {
          console.warn('[Analytics] API call failed:', err?.message || err)
          return fallback
        })

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
    model_id: string
    provider: string
    cost: number
    tokens: number
    requests: number
    workspace_count: number
  }>
  daily_by_provider: {
    providers: string[]
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
  return useQuery({
    queryKey: unifiedAnalyticsKeys.adminWorkspaces(days),
    queryFn: async () => {
      const safeRequest = <T,>(fn: () => Promise<T>, fallback: T): Promise<T> =>
        Promise.resolve().then(fn).catch((err) => {
          console.warn('[Analytics] API call failed:', err?.message || err)
          return fallback
        })

      const [agents] = await Promise.all([
        safeRequest(() => apiClient.getAgents(), []),
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
