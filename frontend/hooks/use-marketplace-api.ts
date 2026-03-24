'use client'

/**
 * Marketplace API hooks for agent marketplace functionality
 * Provides React Query integration for marketplace operations
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'

// Query keys for React Query
export const marketplaceQueryKeys = {
  items: (filters?: any) => ['marketplace', 'items', filters] as const,
  item: (id: number) => ['marketplace', 'items', id] as const,
  featured: ['marketplace', 'featured'] as const,
  updates: ['marketplace', 'updates'] as const,
}

// Marketplace item interface
export interface MarketplaceItem {
  id: number
  type: string
  name: string
  description: string
  creator_name: string
  icon?: string
  category?: string
  tags: string[]
  install_count: number
  is_featured: boolean
  version: string
  metadata: Record<string, any>
  created_at: string
  updated_at: string
  dependencies?: Record<string, any>
}

export interface SubmitToMarketplaceRequest {
  item_type: string
  name?: string
  description?: string
  category?: string
  tags?: string[]
  metadata: {
    agent_id: number
    icon?: string
  }
}

export interface SubmitToMarketplaceResponse {
  success: boolean
  message: string
  item_id: number
  auto_approved: boolean
}

/**
 * Hook to submit an agent to the marketplace
 */
export function useSubmitToMarketplace() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (request: SubmitToMarketplaceRequest) => {
      const response = await apiClient.submitToMarketplace(request)
      return response as SubmitToMarketplaceResponse
    },
    onSuccess: (data) => {
      // Invalidate relevant queries
      queryClient.invalidateQueries({ queryKey: marketplaceQueryKeys.items() })
      queryClient.invalidateQueries({ queryKey: marketplaceQueryKeys.featured })

      // Show success message
      if (data.auto_approved) {
        toast.success('Agent published to marketplace!', {
          description: 'Your agent is now available in the marketplace.'
        })
      } else {
        toast.success('Agent submitted for approval!', {
          description: 'Your agent will be reviewed before appearing in the marketplace.'
        })
      }
    },
    onError: (error: any) => {
      toast.error('Failed to submit agent', {
        description: error?.message || 'An error occurred while submitting to the marketplace.'
      })
      console.error('Marketplace submission error:', error)
    }
  })
}

/**
 * Hook to get marketplace items with optional filters
 */
export function useMarketplaceItems(filters?: {
  type?: string
  category?: string
  search?: string
  featured?: boolean
  limit?: number
  offset?: number
}) {
  return useQuery({
    queryKey: marketplaceQueryKeys.items(filters),
    queryFn: () => apiClient.getMarketplaceItems(filters),
    staleTime: 1000 * 60 * 5, // 5 minutes
  })
}

/**
 * Hook to get a single marketplace item
 */
export function useMarketplaceItem(itemId: number | null) {
  return useQuery({
    queryKey: marketplaceQueryKeys.item(itemId!),
    queryFn: () => apiClient.getMarketplaceItem(itemId!),
    enabled: itemId !== null,
    staleTime: 1000 * 60 * 5, // 5 minutes
  })
}

/**
 * Hook to get featured marketplace items
 */
export function useFeaturedItems(limit: number = 8) {
  return useQuery({
    queryKey: marketplaceQueryKeys.featured,
    queryFn: () => apiClient.getFeaturedMarketplaceItems(limit),
    staleTime: 1000 * 60 * 10, // 10 minutes
  })
}

/**
 * Hook to install a marketplace item
 */
export function useInstallMarketplaceItem() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (itemId: number) => {
      return await apiClient.installMarketplaceItem(itemId)
    },
    onSuccess: (data) => {
      // Invalidate agents list to show the new cloned agent
      queryClient.invalidateQueries({ queryKey: ['agents'] })

      toast.success('Agent installed successfully!', {
        description: data.message || 'The agent has been added to your workspace.'
      })
    },
    onError: (error: any) => {
      toast.error('Failed to install agent', {
        description: error?.message || 'An error occurred while installing the agent.'
      })
      console.error('Installation error:', error)
    }
  })
}

/**
 * Hook to check for marketplace updates
 */
export function useMarketplaceUpdates() {
  return useQuery({
    queryKey: marketplaceQueryKeys.updates,
    queryFn: () => apiClient.checkMarketplaceUpdates(),
    staleTime: 1000 * 60 * 15, // 15 minutes
    refetchOnWindowFocus: false,
  })
}


// ===================================================================
// Agent Catalog (PRD-120) — pre-built agent templates from SKILL.md
// ===================================================================

export interface AgentCatalogTemplate {
  id: number
  slug: string
  name: string
  category: string
  description: string
  recommended_model?: string
  recommended_tools: string[]
  tags: string[]
  icon?: string
  tier: string
}

export interface AgentCatalogTemplateDetail extends AgentCatalogTemplate {
  persona?: string
  skill_slug?: string
  is_active: boolean
  created_at?: string
  updated_at?: string
}

export interface AgentCatalogCategory {
  category: string
  count: number
  icon?: string
}

export interface AgentCatalogDeployResponse {
  success: boolean
  agent_id: number
  agent_name: string
  message: string
}

export const agentCatalogQueryKeys = {
  templates: (filters?: { category?: string; search?: string }) =>
    ['agentCatalog', 'templates', filters] as const,
  template: (slug: string) => ['agentCatalog', 'templates', slug] as const,
  categories: ['agentCatalog', 'categories'] as const,
}

/**
 * Browse agent catalog templates with optional category and search filters
 */
export function useAgentCatalogTemplates(filters?: {
  category?: string
  search?: string
  limit?: number
  offset?: number
}) {
  const params = new URLSearchParams()
  if (filters?.category) params.set('category', filters.category)
  if (filters?.search) params.set('search', filters.search)
  if (filters?.limit) params.set('limit', String(filters.limit))
  if (filters?.offset) params.set('offset', String(filters.offset))
  const qs = params.toString()

  return useQuery({
    queryKey: agentCatalogQueryKeys.templates(filters),
    queryFn: () =>
      apiClient.request<AgentCatalogTemplate[]>(
        `/api/marketplace/agents${qs ? `?${qs}` : ''}`
      ),
    staleTime: 1000 * 60 * 5,
  })
}

/**
 * Get agent catalog categories with counts
 */
export function useAgentCatalogCategories() {
  return useQuery({
    queryKey: agentCatalogQueryKeys.categories,
    queryFn: () =>
      apiClient.request<AgentCatalogCategory[]>('/api/marketplace/agents/categories'),
    staleTime: 1000 * 60 * 10,
  })
}

/**
 * Get single agent catalog template detail by slug
 */
export function useAgentCatalogTemplate(slug: string | null) {
  return useQuery({
    queryKey: agentCatalogQueryKeys.template(slug!),
    queryFn: () =>
      apiClient.request<AgentCatalogTemplateDetail>(
        `/api/marketplace/agents/${slug}`
      ),
    enabled: slug !== null,
    staleTime: 1000 * 60 * 5,
  })
}

/**
 * Deploy an agent catalog template to the user's workspace
 */
export function useDeployAgentCatalogTemplate() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (slug: string) => {
      return await apiClient.request<AgentCatalogDeployResponse>(
        `/api/marketplace/agents/${slug}/deploy`,
        { method: 'POST' }
      )
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['agents'] })
      toast.success(`${data.agent_name} added to workspace`, {
        description: data.message,
      })
    },
    onError: (error: any) => {
      toast.error('Failed to deploy agent', {
        description: error?.message || 'An error occurred while deploying the agent template.',
      })
    },
  })
}
