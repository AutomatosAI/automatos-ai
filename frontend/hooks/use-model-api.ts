/**
 * Model API Hooks (PRD-15)
 * ========================
 * 
 * React hooks for interacting with the model management API.
 * Provides model discovery, recommendations, and cost estimation.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/hooks/use-workspace'

// Types
export interface ModelInfo {
  id: number
  provider: string
  /** PRD-236 W1: the route that serves this row (openrouter, nvidia, openai…). */
  serving_provider?: string
  serving_provider_label?: string
  route_label?: string
  is_free?: boolean
  price_tier?: string
  key_available?: boolean
  sourcing?: string
  model_id: string
  display_name: string
  model_family: string
  context_window: number
  max_output_tokens: number
  input_cost_per_1k: number
  output_cost_per_1k: number
  capabilities: Record<string, string>
  recommended_for: string[]
  supports_functions: boolean
  supports_vision: boolean
  supports_streaming: boolean
  status: string
  description?: string
  tier?: string
  category?: string
  tags?: string[]
  is_featured?: boolean
  is_default?: boolean
}

export interface ProviderInfo {
  name: string
  model_count: number
  avg_input_cost: number
  avg_output_cost: number
  models: Array<{ model_id: string; display_name: string }>
}

export interface ModelRecommendation {
  model_id: string
  display_name: string
  provider: string
  reason: string
  model: ModelInfo
}

export interface CostEstimate {
  model_id: string
  input_tokens: number
  output_tokens: number
  input_cost: number
  output_cost: number
  total_cost: number
  currency: string
  model_details: {
    display_name: string
    provider: string
    input_cost_per_1k: number
    output_cost_per_1k: number
  }
}

/**
 * Fetch all available models (global catalog)
 */
export function useModels(provider?: string, status: string = 'active', options?: { enabled?: boolean }) {
  return useQuery<ModelInfo[]>({
    queryKey: ['models', provider, status],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (provider) params.append('provider', provider)
      if (status) params.append('status', status)

      const response = await apiClient.request(`/api/models/?${params.toString()}`)
      return response
    },
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    enabled: options?.enabled ?? true,
  })
}

/**
 * PRD-54: Fetch workspace-installed models (for agent model selector).
 * Falls back to defaults if workspace has no installs yet.
 */
export function useWorkspaceModels(options?: { enabled?: boolean }) {
  const { workspaceId } = useWorkspace()
  return useQuery<ModelInfo[]>({
    queryKey: ['workspace-models', workspaceId],
    queryFn: async () => {
      const response = await apiClient.get('/api/marketplace/llm/installed')
      return response
    },
    staleTime: 5 * 60 * 1000,
    enabled: (options?.enabled ?? true) && !!workspaceId,
  })
}

/**
 * Fetch a specific model by ID
 */
export function useModel(modelId: string | null) {
  return useQuery<ModelInfo>({
    queryKey: ['model', modelId],
    queryFn: async () => {
      if (!modelId) throw new Error('Model ID is required')
      const response = await apiClient.request(`/api/models/${modelId}`)
      return response
    },
    enabled: !!modelId,
    staleTime: 10 * 60 * 1000, // Cache for 10 minutes
  })
}

/**
 * Fetch all providers with their models
 */
export function useProviders() {
  return useQuery<ProviderInfo[]>({
    queryKey: ['providers'],
    queryFn: async () => {
      const response = await apiClient.request('/api/models/providers/')
      return response
    },
    staleTime: 10 * 60 * 1000,
  })
}

/**
 * Get model recommendations based on requirements
 */
export function useModelRecommendation() {
  return useMutation<ModelRecommendation, Error, Record<string, any>>({
    mutationFn: async (requirements) => {
      const response = await apiClient.request('/api/models/recommend', {
        method: 'POST',
        body: JSON.stringify(requirements),
      })
      return response
    },
  })
}

/**
 * Estimate cost for model usage
 */
export function useEstimateCost() {
  return useMutation<CostEstimate, Error, { model_id: string; input_tokens: number; output_tokens: number }>({
    mutationFn: async (request) => {
      const response = await apiClient.request('/api/models/estimate-cost', {
        method: 'POST',
        body: JSON.stringify(request),
      })
      return response
    },
  })
}

/**
 * Get models recommended for a specific task type
 */
export function useModelsByTask(taskType: string | null) {
  return useQuery<ModelInfo[]>({
    queryKey: ['models-by-task', taskType],
    queryFn: async () => {
      if (!taskType) throw new Error('Task type is required')
      const response = await apiClient.request(`/api/models/by-task/${taskType}`)
      return response
    },
    enabled: !!taskType,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Get overall model statistics
 */
export function useModelStats() {
  return useQuery({
    queryKey: ['model-stats'],
    queryFn: async () => {
      const response = await apiClient.request('/api/models/stats/')
      return response
    },
    staleTime: 10 * 60 * 1000,
  })
}

/**
 * Get agent's model configuration
 */
export function useAgentModelConfig(agentId: number | null) {
  return useQuery({
    queryKey: ['agent-model-config', agentId],
    queryFn: async () => {
      if (!agentId) throw new Error('Agent ID is required')
      const response = await apiClient.request(`/api/agents/${agentId}/model-config`)
      return response
    },
    enabled: !!agentId,
  })
}

/**
 * Update agent's model configuration
 */
export function useUpdateAgentModelConfig() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ agentId, modelConfig }: { agentId: number; modelConfig: Record<string, any> }) => {
      const response = await apiClient.request(`/api/agents/${agentId}/model-config`, {
        method: 'PUT',
        body: JSON.stringify(modelConfig),
      })
      return response
    },
    onSuccess: (_, variables) => {
      // Invalidate agent model config cache
      queryClient.invalidateQueries({ queryKey: ['agent-model-config', variables.agentId] })
      queryClient.invalidateQueries({ queryKey: ['agents'] })
    },
  })
}

/**
 * Get agent's model usage statistics
 */
export function useAgentModelUsage(agentId: number | null) {
  return useQuery({
    queryKey: ['agent-model-usage', agentId],
    queryFn: async () => {
      if (!agentId) throw new Error('Agent ID is required')
      const response = await apiClient.request(`/api/agents/${agentId}/model-usage`)
      return response
    },
    enabled: !!agentId,
    refetchInterval: 30000, // Refetch every 30 seconds for live stats
  })
}

/**
 * Switch agent to a different model
 */
export function useSwitchAgentModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({
      agentId,
      modelId,
      temperature,
      maxTokens
    }: {
      agentId: number
      modelId: string
      temperature?: number
      maxTokens?: number
    }) => {
      const response = await apiClient.request(`/api/agents/${agentId}/switch-model`, {
        method: 'POST',
        body: JSON.stringify({
          model_id: modelId,
          temperature,
          max_tokens: maxTokens,
        }),
      })
      return response
    },
    onSuccess: (_, variables) => {
      // Invalidate related caches
      queryClient.invalidateQueries({ queryKey: ['agent-model-config', variables.agentId] })
      queryClient.invalidateQueries({ queryKey: ['agents'] })
    },
  })
}

