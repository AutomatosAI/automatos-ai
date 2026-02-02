import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// Query key factory
export const recipeKeys = {
  all: ['workflow-recipes'] as const,
  lists: () => [...recipeKeys.all, 'list'] as const,
  list: (params?: any) => [...recipeKeys.lists(), params] as const,
  details: () => [...recipeKeys.all, 'detail'] as const,
  detail: (id: string) => [...recipeKeys.details(), id] as const,
  featured: () => [...recipeKeys.all, 'featured'] as const,
  categories: () => [...recipeKeys.all, 'categories'] as const,
  suggestions: (id: string) => [...recipeKeys.all, 'suggestions', id] as const,
  executions: (id: string, params?: any) => [...recipeKeys.all, 'executions', id, params] as const,
}

// List recipes with filtering
export function useWorkflowRecipes(params?: {
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
    queryKey: recipeKeys.list(params),
    queryFn: () => apiClient.listWorkflowRecipes(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

// Get single recipe
export function useWorkflowRecipe(recipeId: string | undefined) {
  return useQuery({
    queryKey: recipeKeys.detail(recipeId || ''),
    queryFn: () => apiClient.getWorkflowRecipeById(recipeId!),
    enabled: !!recipeId,
  })
}

// Get featured recipes
export function useFeaturedRecipes(limit?: number) {
  return useQuery({
    queryKey: recipeKeys.featured(),
    queryFn: () => apiClient.getFeaturedRecipes(limit),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

// Get recipe categories
export function useRecipeCategories() {
  return useQuery({
    queryKey: recipeKeys.categories(),
    queryFn: () => apiClient.getRecipeCategories(),
    staleTime: 30 * 60 * 1000, // 30 minutes
  })
}

// Create recipe mutation
export function useCreateRecipe() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (recipeData: any) => apiClient.createWorkflowRecipe(recipeData),
    onSuccess: () => {
      // Invalidate all recipe lists
      queryClient.invalidateQueries({ queryKey: recipeKeys.lists() })
      queryClient.invalidateQueries({ queryKey: recipeKeys.featured() })
    },
  })
}

// Update recipe mutation
export function useUpdateRecipe() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ recipeId, recipeData }: { recipeId: string; recipeData: any }) =>
      apiClient.updateWorkflowRecipe(recipeId, recipeData),
    onSuccess: (_, variables) => {
      // Invalidate the specific recipe and all lists
      queryClient.invalidateQueries({ queryKey: recipeKeys.detail(variables.recipeId) })
      queryClient.invalidateQueries({ queryKey: recipeKeys.lists() })
      queryClient.invalidateQueries({ queryKey: recipeKeys.featured() })
    },
  })
}

// Delete recipe mutation
export function useDeleteRecipe() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (recipeId: string) => apiClient.deleteWorkflowRecipe(recipeId),
    onSuccess: () => {
      // Invalidate all recipe lists
      queryClient.invalidateQueries({ queryKey: recipeKeys.lists() })
      queryClient.invalidateQueries({ queryKey: recipeKeys.featured() })
    },
  })
}

// Execute recipe directly (creates workflow + launches execution)
export function useExecuteRecipe() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ recipeId, inputData }: { recipeId: string; inputData?: Record<string, any> }) =>
      apiClient.executeRecipe(recipeId, inputData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: recipeKeys.lists() })
    },
  })
}

// Record recipe usage mutation
export function useRecordRecipeUsage() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (recipeId: string) => apiClient.recordRecipeUsage(recipeId),
    onSuccess: (_, recipeId) => {
      // Invalidate the specific recipe to refresh use_count
      queryClient.invalidateQueries({ queryKey: recipeKeys.detail(recipeId) })
      queryClient.invalidateQueries({ queryKey: recipeKeys.lists() })
    },
  })
}

// Submit recipe to marketplace
export function useSubmitRecipeToMarketplace() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: { recipe_id: string; category?: string; icon?: string }) =>
      apiClient.submitRecipeToMarketplace(params),
    onSuccess: () => {
      // Invalidate marketplace items to show the new recipe
      queryClient.invalidateQueries({ queryKey: ['marketplaceRecipes'] })
    },
  })
}

// Install recipe from marketplace
export function useInstallRecipeFromMarketplace() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (recipeId: number) => apiClient.installRecipeFromMarketplace(recipeId),
    onSuccess: () => {
      // Invalidate workspace recipes to show the newly installed recipe
      queryClient.invalidateQueries({ queryKey: recipeKeys.lists() })
    },
  })
}

// Get recipe suggestions from learning_data
export function useRecipeSuggestions(recipeId: string | undefined) {
  return useQuery({
    queryKey: recipeKeys.suggestions(recipeId || ''),
    queryFn: () => apiClient.getRecipeSuggestions(recipeId!),
    enabled: !!recipeId,
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

// Get recipe executions list
export function useRecipeExecutions(recipeId: string | undefined, params?: { status?: string; skip?: number; limit?: number }) {
  return useQuery({
    queryKey: recipeKeys.executions(recipeId || '', params),
    queryFn: () => apiClient.getRecipeExecutions(recipeId!, params),
    enabled: !!recipeId,
    staleTime: 1 * 60 * 1000, // 1 minute
  })
}

