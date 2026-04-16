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
      // Cascade installs tools to workspace — refresh tools cache so agent config sees them
      queryClient.invalidateQueries({ queryKey: ['tools'] })

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
 * Hook to toggle featured status on a marketplace item (admin only)
 */
export function useToggleFeatured() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (itemId: number) => {
      return await apiClient.toggleMarketplaceFeatured(itemId)
    },
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: marketplaceQueryKeys.items() })
      queryClient.invalidateQueries({ queryKey: marketplaceQueryKeys.featured })
      toast.success(data.is_featured ? 'Item featured' : 'Item unfeatured')
    },
    onError: (error: any) => {
      toast.error('Failed to toggle featured', {
        description: error?.message || 'An error occurred.'
      })
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
