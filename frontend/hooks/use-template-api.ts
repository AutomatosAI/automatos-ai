import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// Query key factory
export const templateKeys = {
  all: ['workflow-templates'] as const,
  lists: () => [...templateKeys.all, 'list'] as const,
  list: (params?: any) => [...templateKeys.lists(), params] as const,
  details: () => [...templateKeys.all, 'detail'] as const,
  detail: (id: string) => [...templateKeys.details(), id] as const,
  featured: () => [...templateKeys.all, 'featured'] as const,
  categories: () => [...templateKeys.all, 'categories'] as const,
}

// List templates with filtering
export function useWorkflowTemplates(params?: {
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
    queryKey: templateKeys.list(params),
    queryFn: () => apiClient.listWorkflowTemplates(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

// Get single template
export function useWorkflowTemplate(templateId: string | undefined) {
  return useQuery({
    queryKey: templateKeys.detail(templateId || ''),
    queryFn: () => apiClient.getWorkflowTemplateById(templateId!),
    enabled: !!templateId,
  })
}

// Get featured templates
export function useFeaturedTemplates(limit?: number) {
  return useQuery({
    queryKey: templateKeys.featured(),
    queryFn: () => apiClient.getFeaturedTemplates(limit),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

// Get template categories
export function useTemplateCategories() {
  return useQuery({
    queryKey: templateKeys.categories(),
    queryFn: () => apiClient.getTemplateCategories(),
    staleTime: 30 * 60 * 1000, // 30 minutes
  })
}

// Create template mutation
export function useCreateTemplate() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (templateData: any) => apiClient.createWorkflowTemplate(templateData),
    onSuccess: () => {
      // Invalidate all template lists
      queryClient.invalidateQueries({ queryKey: templateKeys.lists() })
      queryClient.invalidateQueries({ queryKey: templateKeys.featured() })
    },
  })
}

// Update template mutation
export function useUpdateTemplate() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ templateId, templateData }: { templateId: string; templateData: any }) =>
      apiClient.updateWorkflowTemplate(templateId, templateData),
    onSuccess: (_, variables) => {
      // Invalidate the specific template and all lists
      queryClient.invalidateQueries({ queryKey: templateKeys.detail(variables.templateId) })
      queryClient.invalidateQueries({ queryKey: templateKeys.lists() })
      queryClient.invalidateQueries({ queryKey: templateKeys.featured() })
    },
  })
}

// Delete template mutation
export function useDeleteTemplate() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (templateId: string) => apiClient.deleteWorkflowTemplate(templateId),
    onSuccess: () => {
      // Invalidate all template lists
      queryClient.invalidateQueries({ queryKey: templateKeys.lists() })
      queryClient.invalidateQueries({ queryKey: templateKeys.featured() })
    },
  })
}

// Record template usage mutation
export function useRecordTemplateUsage() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (templateId: string) => apiClient.recordTemplateUsage(templateId),
    onSuccess: (_, templateId) => {
      // Invalidate the specific template to refresh use_count
      queryClient.invalidateQueries({ queryKey: templateKeys.detail(templateId) })
      queryClient.invalidateQueries({ queryKey: templateKeys.lists() })
    },
  })
}

