/**
 * Blog API hooks
 * React Query integration for blog post management.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from '@/lib/api-client'

// ============= TYPES =============

export interface BlogPost {
  id: string
  workspace_id: string
  author_agent_id: number | null
  author_name: string
  title: string
  slug: string
  excerpt: string | null
  content?: string
  cover_image_url: string | null
  tags: string[]
  category: string | null
  status: 'draft' | 'published' | 'scheduled' | 'archived'
  published_at: string | null
  scheduled_for: string | null
  seo_title: string | null
  seo_description: string | null
  reading_time_minutes: number
  view_count: number
  created_at: string
  updated_at: string
}

export interface BlogListResponse {
  posts: BlogPost[]
  total: number
  page: number
  per_page: number
  total_pages: number
}

export interface BlogFilters {
  status?: string
  category?: string
  tag?: string
  page?: number
  per_page?: number
}

// ============= QUERY KEYS =============

export const blogQueryKeys = {
  all: ['blog-posts'] as const,
  list: (filters: BlogFilters) => ['blog-posts', 'list', filters] as const,
  detail: (id: string) => ['blog-posts', 'detail', id] as const,
}

// ============= QUERY HOOKS =============

export function useBlogPosts(filters: BlogFilters = {}) {
  const params = new URLSearchParams()
  if (filters.status) params.set('status', filters.status)
  if (filters.category) params.set('category', filters.category)
  if (filters.tag) params.set('tag', filters.tag)
  if (filters.page) params.set('page', String(filters.page))
  if (filters.per_page) params.set('per_page', String(filters.per_page))

  const qs = params.toString()

  return useQuery<BlogListResponse>({
    queryKey: blogQueryKeys.list(filters),
    queryFn: () => apiClient.request<BlogListResponse>(`/api/blog/posts${qs ? `?${qs}` : ''}`),
    staleTime: 15000,
    refetchInterval: 30000,
  })
}

export function useBlogPost(postId: string | null) {
  return useQuery<BlogPost>({
    queryKey: blogQueryKeys.detail(postId!),
    queryFn: () => apiClient.request<BlogPost>(`/api/blog/posts/${postId}`),
    enabled: !!postId,
    staleTime: 30000,
  })
}

// ============= MUTATION HOOKS =============

export function usePublishPost() {
  const queryClient = useQueryClient()

  return useMutation<BlogPost, Error, string>({
    mutationFn: (postId) =>
      apiClient.request(`/api/blog/posts/${postId}/publish`, { method: 'POST' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: blogQueryKeys.all })
      toast.success('Post published')
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to publish post')
    },
  })
}

export function useUnpublishPost() {
  const queryClient = useQueryClient()

  return useMutation<BlogPost, Error, string>({
    mutationFn: (postId) =>
      apiClient.request(`/api/blog/posts/${postId}/unpublish`, { method: 'POST' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: blogQueryKeys.all })
      toast.success('Post unpublished')
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to unpublish post')
    },
  })
}

export function useDeletePost() {
  const queryClient = useQueryClient()

  return useMutation<{ success: boolean }, Error, string>({
    mutationFn: (postId) =>
      apiClient.request(`/api/blog/posts/${postId}`, { method: 'DELETE' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: blogQueryKeys.all })
      toast.success('Post archived')
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to delete post')
    },
  })
}

export function useCreatePost() {
  const queryClient = useQueryClient()

  return useMutation<BlogPost, Error, Partial<BlogPost>>({
    mutationFn: (data) =>
      apiClient.request('/api/blog/posts', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: blogQueryKeys.all })
      toast.success('Post created')
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to create post')
    },
  })
}

export function useUpdatePost() {
  const queryClient = useQueryClient()

  return useMutation<BlogPost, Error, { postId: string; data: Partial<BlogPost> }>({
    mutationFn: ({ postId, data }) =>
      apiClient.request(`/api/blog/posts/${postId}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: blogQueryKeys.all })
      toast.success('Post updated')
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to update post')
    },
  })
}

export interface BlogMissionResult {
  success: boolean
  mission_id: number
  state: string
  topic: string
  category: string
  task_count: number
  message: string
}

export function useCreateBlogMission() {
  const queryClient = useQueryClient()

  return useMutation<BlogMissionResult, Error, { topic: string; category?: string }>({
    mutationFn: (data) =>
      apiClient.request<BlogMissionResult>('/api/blog/missions', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: blogQueryKeys.all })
      toast.success(`Blog mission started — ${result.task_count} tasks queued`)
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to start blog mission')
    },
  })
}

export interface CoverImageUploadResult {
  image_id: string
  cover_image_url: string
  size_bytes: number
  content_type: string
}

export function useUploadCoverImage() {
  return useMutation<CoverImageUploadResult, Error, File>({
    mutationFn: async (file: File) => {
      const fd = new FormData()
      fd.append('file', file)
      // apiClient.request auto-handles FormData and adds workspace + auth headers
      return apiClient.request<CoverImageUploadResult>('/api/blog/cover-image/upload', {
        method: 'POST',
        body: fd,
      })
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to upload image')
    },
  })
}
