/**
 * Notifications API hooks (PRD-128 US-008)
 * React Query v4 hooks backing the bell dropdown + settings page.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// ============= TYPES =============

export interface NotificationRow {
  id: string
  workspace_id: string
  user_id: number | null
  event_type: string
  title: string
  message: string | null
  link_type: string | null
  link_id: string | null
  agent_id: number | null
  agent_name: string | null
  status: string
  read_at: string | null
  dismissed_at: string | null
  created_at: string
}

export interface NotificationListResponse {
  success: boolean
  notifications: NotificationRow[]
  total: number
  limit: number
  offset: number
}

export interface UnreadCountResponse {
  success: boolean
  count: number
}

export interface PreferenceRow {
  event_type: string
  destination: string
  enabled: boolean
  channel_connection_id: string | null
}

export interface PreferenceListResponse {
  success: boolean
  preferences: PreferenceRow[]
}

export interface PreferenceBulkUpdateResponse {
  success: boolean
  updated_count: number
}

// ============= QUERY KEYS =============

export const notificationQueryKeys = {
  all: ['notifications'] as const,
  list: (limit: number, offset: number, unreadOnly: boolean) =>
    ['notifications', 'list', { limit, offset, unreadOnly }] as const,
  unreadCount: ['notifications', 'unread-count'] as const,
  preferences: ['notification-preferences'] as const,
}

// ============= QUERIES =============

/**
 * Polls /api/notifications/unread-count every 30s for the bell badge.
 */
export function useUnreadNotificationCount() {
  return useQuery<UnreadCountResponse>({
    queryKey: notificationQueryKeys.unreadCount,
    queryFn: () =>
      apiClient.request<UnreadCountResponse>('/api/notifications/unread-count'),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

/**
 * Fetches the most recent notifications. Only enabled when the caller opts in
 * (e.g. when the popover is opened) to avoid wasted requests.
 */
export function useNotifications(
  options: { limit?: number; offset?: number; unreadOnly?: boolean; enabled?: boolean } = {}
) {
  const { limit = 20, offset = 0, unreadOnly = false, enabled = true } = options

  return useQuery<NotificationListResponse>({
    queryKey: notificationQueryKeys.list(limit, offset, unreadOnly),
    queryFn: () => {
      const params = new URLSearchParams()
      params.set('limit', String(limit))
      params.set('offset', String(offset))
      if (unreadOnly) params.set('unread_only', 'true')
      return apiClient.request<NotificationListResponse>(
        `/api/notifications?${params.toString()}`
      )
    },
    enabled,
    staleTime: 15000,
  })
}

/**
 * GET merged notification preferences for the settings page.
 */
export function useNotificationPreferences(enabled: boolean = true) {
  return useQuery<PreferenceListResponse>({
    queryKey: notificationQueryKeys.preferences,
    queryFn: () =>
      apiClient.request<PreferenceListResponse>('/api/notification-preferences'),
    enabled,
    staleTime: 30000,
  })
}

// ============= MUTATIONS =============

export function useMarkNotificationRead() {
  const queryClient = useQueryClient()

  return useMutation<{ success: boolean }, Error, string>({
    mutationFn: (id: string) =>
      apiClient.request<{ success: boolean }>(
        `/api/notifications/${id}/read`,
        { method: 'POST' }
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: notificationQueryKeys.all })
    },
  })
}

export function useMarkAllNotificationsRead() {
  const queryClient = useQueryClient()

  return useMutation<{ success: boolean; marked_count: number }, Error, void>({
    mutationFn: () =>
      apiClient.request<{ success: boolean; marked_count: number }>(
        '/api/notifications/read-all',
        { method: 'POST' }
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: notificationQueryKeys.all })
    },
  })
}

export function useDismissNotification() {
  const queryClient = useQueryClient()

  return useMutation<{ success: boolean }, Error, string>({
    mutationFn: (id: string) =>
      apiClient.request<{ success: boolean }>(
        `/api/notifications/${id}/dismiss`,
        { method: 'POST' }
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: notificationQueryKeys.all })
    },
  })
}

export function useUpdateNotificationPreferences() {
  const queryClient = useQueryClient()

  return useMutation<
    PreferenceBulkUpdateResponse,
    Error,
    PreferenceRow[]
  >({
    mutationFn: (preferences: PreferenceRow[]) =>
      apiClient.request<PreferenceBulkUpdateResponse>(
        '/api/notification-preferences',
        { method: 'PUT', body: JSON.stringify({ preferences }) }
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: notificationQueryKeys.preferences })
      queryClient.invalidateQueries({ queryKey: notificationQueryKeys.all })
    },
  })
}
