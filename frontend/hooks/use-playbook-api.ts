import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// Query key factory
export const playbookKeys = {
  all: ['playbooks'] as const,
  lists: () => [...playbookKeys.all, 'list'] as const,
  list: (params?: any) => [...playbookKeys.lists(), params] as const,
  details: () => [...playbookKeys.all, 'detail'] as const,
  detail: (id: string) => [...playbookKeys.details(), id] as const,
  featured: () => [...playbookKeys.all, 'featured'] as const,
  categories: () => [...playbookKeys.all, 'categories'] as const,
  suggestions: (id: string) => [...playbookKeys.all, 'suggestions', id] as const,
  executions: (id: string, params?: any) => [...playbookKeys.all, 'executions', id, params] as const,
  execution: (playbookId: string, executionId: string) => [...playbookKeys.all, 'execution', playbookId, executionId] as const,
}

// List playbooks with filtering
export function useWorkflowPlaybooks(params?: {
  category?: string
  difficulty?: string
  is_featured?: boolean
  is_public?: boolean
  search?: string
  skip?: number
  limit?: number
  sort_by?: string
}) {
  return useQuery({
    queryKey: playbookKeys.list(params),
    queryFn: () => apiClient.listWorkflowRecipes(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

// Get single playbook
export function useWorkflowPlaybook(playbookId: string | undefined) {
  return useQuery({
    queryKey: playbookKeys.detail(playbookId || ''),
    queryFn: () => apiClient.getWorkflowRecipeById(playbookId!),
    enabled: !!playbookId,
  })
}

// Get featured playbooks
export function useFeaturedPlaybooks(limit?: number) {
  return useQuery({
    queryKey: playbookKeys.featured(),
    queryFn: () => apiClient.getFeaturedRecipes(limit),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

// Get playbook categories
export function usePlaybookCategories() {
  return useQuery({
    queryKey: playbookKeys.categories(),
    queryFn: () => apiClient.getRecipeCategories(),
    staleTime: 30 * 60 * 1000, // 30 minutes
  })
}

// Create playbook mutation
export function useCreatePlaybook() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (playbookData: any) => apiClient.createWorkflowRecipe(playbookData),
    onSuccess: () => {
      // Invalidate all playbook lists
      queryClient.invalidateQueries({ queryKey: playbookKeys.lists() })
      queryClient.invalidateQueries({ queryKey: playbookKeys.featured() })
    },
  })
}

// Update playbook mutation
export function useUpdatePlaybook() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ playbookId, playbookData }: { playbookId: string; playbookData: any }) =>
      apiClient.updateWorkflowRecipe(playbookId, playbookData),
    onSuccess: (_, variables) => {
      // Invalidate the specific playbook and all lists
      queryClient.invalidateQueries({ queryKey: playbookKeys.detail(variables.playbookId) })
      queryClient.invalidateQueries({ queryKey: playbookKeys.lists() })
      queryClient.invalidateQueries({ queryKey: playbookKeys.featured() })
    },
  })
}

// Delete playbook mutation
export function useDeletePlaybook() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (playbookId: string) => apiClient.deleteWorkflowRecipe(playbookId),
    onSuccess: () => {
      // Invalidate all playbook lists
      queryClient.invalidateQueries({ queryKey: playbookKeys.lists() })
      queryClient.invalidateQueries({ queryKey: playbookKeys.featured() })
    },
  })
}

// Execute playbook directly (creates workflow + launches execution)
export function useExecutePlaybook() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ playbookId, inputData }: { playbookId: string; inputData?: Record<string, any> }) =>
      apiClient.executeRecipe(playbookId, inputData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: playbookKeys.lists() })
    },
  })
}

// Record playbook usage mutation
export function useRecordPlaybookUsage() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (playbookId: string) => apiClient.recordRecipeUsage(playbookId),
    onSuccess: (_, playbookId) => {
      // Invalidate the specific playbook to refresh use_count
      queryClient.invalidateQueries({ queryKey: playbookKeys.detail(playbookId) })
      queryClient.invalidateQueries({ queryKey: playbookKeys.lists() })
    },
  })
}

// Submit playbook to marketplace
export function useSubmitPlaybookToMarketplace() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: { recipe_id: string; category?: string; icon?: string }) =>
      apiClient.submitRecipeToMarketplace(params),
    onSuccess: () => {
      // Invalidate marketplace items to show the new playbook
      queryClient.invalidateQueries({ queryKey: ['marketplaceRecipes'] })
    },
  })
}

// Install playbook from marketplace
export function useInstallPlaybookFromMarketplace() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (playbookId: number) => apiClient.installRecipeFromMarketplace(playbookId),
    onSuccess: () => {
      // Invalidate workspace playbooks to show the newly installed playbook
      queryClient.invalidateQueries({ queryKey: playbookKeys.lists() })
      // Cascade installs agents + tools + models — refresh all caches
      queryClient.invalidateQueries({ queryKey: ['agents'] })
      queryClient.invalidateQueries({ queryKey: ['tools'] })
      queryClient.invalidateQueries({ queryKey: ['workspace-models'] })
    },
  })
}

// Get playbook suggestions from learning_data
export function usePlaybookSuggestions(playbookId: string | undefined) {
  return useQuery({
    queryKey: playbookKeys.suggestions(playbookId || ''),
    queryFn: () => apiClient.getRecipeSuggestions(playbookId!),
    enabled: !!playbookId,
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

// Get playbook executions list
export function usePlaybookExecutions(playbookId: string | undefined, params?: { status?: string; skip?: number; limit?: number }) {
  return useQuery({
    queryKey: playbookKeys.executions(playbookId || '', params),
    queryFn: () => apiClient.getRecipeExecutions(playbookId!, params),
    enabled: !!playbookId,
    staleTime: 1 * 60 * 1000, // 1 minute
  })
}

// Get single playbook execution detail with polling while running
export function usePlaybookExecution(playbookId: string | undefined, executionId: string | undefined) {
  return useQuery({
    queryKey: playbookKeys.execution(playbookId || '', executionId || ''),
    queryFn: () => apiClient.getRecipeExecution(playbookId!, executionId!),
    enabled: !!playbookId && !!executionId,
    refetchInterval: (data: any) => {
      // Poll every 3s while execution is running or pending
      const status = data?.status
      if (status === 'running' || status === 'pending') {
        return 3000
      }
      return false // Stop polling when completed/failed
    },
  })
}
