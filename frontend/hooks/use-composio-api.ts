/**
 * Composio API Hooks (PRD-36)
 * ===========================
 * 
 * React Query hooks for Composio integration:
 * - App discovery and management
 * - OAuth connection handling
 * - Feature toggles
 * - Trigger subscriptions
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useAuth } from '@/lib/auth-hooks'
import { apiClient } from '@/lib/api-client'

function shouldRetry(error: any) {
    const msg = String(error?.message || '')
    if (msg.includes('HTTP 401')) return false
    return true
}

// Types
export interface ComposioApp {
    name: string
    display_name: string
    description?: string
    logo_url?: string
    categories: string[]
    is_connected: boolean
    action_count: number
    auth_schemes?: string[]
    trigger_count?: number
    triggers?: Array<{
        name?: string
        display_name?: string
        description?: string
        trigger_name?: string
    }>
}

export interface ComposioAction {
    name: string
    display_name: string
    description?: string
    app_name: string
    parameters: Record<string, any>
    enabled: boolean
}

export interface ComposioConnection {
    app_name: string
    status: string
    connected_at?: string
    connection_id?: string
}

export interface FeatureToggle {
    action_name: string
    enabled: boolean
}

export interface TriggerSubscription {
    id: number
    trigger_name: string
    agent_id?: number
    workflow_id?: number
    created_at: string
}

// ============================================================================
// App Discovery
// ============================================================================

/**
 * Fetch all available Composio apps
 */
export function useAvailableApps(category?: string, opts?: { enabled?: boolean }) {
    const { isLoaded, isSignedIn } = useAuth()

    console.log('🔍 [useAvailableApps] Hook called:', {
        category,
        isLoaded,
        isSignedIn,
        enabled: opts?.enabled ?? true,
        optsEnabled: opts?.enabled
    })

    const result = useQuery({
        queryKey: ['composio', 'apps', category],
        queryFn: async () => {
            console.log('🚀 [useAvailableApps] queryFn executing - Fetching apps...', { category })
            const params = category ? `?category=${encodeURIComponent(category)}` : ''
            console.log('🌐 [useAvailableApps] Calling API:', `/api/composio/apps${params}`)

            try {
                const response = await apiClient.get<ComposioApp[]>(`/api/composio/apps${params}`)
                console.log('✅ [useAvailableApps] Response received!', {
                    type: Array.isArray(response) ? 'array' : typeof response,
                    count: response?.length,
                    firstApp: response?.[0]?.name || 'N/A'
                })
                return response as ComposioApp[]
            } catch (error) {
                console.error('❌ [useAvailableApps] Error fetching apps:', error)
                throw error
            }
        },
        staleTime: 5 * 60 * 1000, // Cache for 5 minutes
        enabled: opts?.enabled ?? true,  // ← REMOVED Clerk dependency - apps list doesn't need auth
        retry: (failureCount, error) => shouldRetry(error) && failureCount < 1,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
    })

    console.log('📊 [useAvailableApps] Query result:', {
        status: result.status,
        fetchStatus: result.fetchStatus,
        isLoading: result.isLoading,
        isFetching: result.isFetching,
        dataLength: result.data?.length || 0,
        error: result.error ? String(result.error) : null
    })

    return result
}

/**
 * Get actions for a specific app
 */
export function useAppActions(appName: string, opts?: { enabled?: boolean }) {
    const { isLoaded, isSignedIn } = useAuth()
    return useQuery({
        queryKey: ['composio', 'apps', appName, 'actions'],
        queryFn: async () => {
            const response = await apiClient.get(`/api/composio/apps/${appName}/actions`)
            return response.data as ComposioAction[]
        },
        enabled: (opts?.enabled ?? true) && !!appName && isLoaded && isSignedIn,
        retry: (failureCount, error) => shouldRetry(error) && failureCount < 1,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
    })
}

// ============================================================================
// Connection Management
// ============================================================================

/**
 * Fetch connected apps for the workspace
 */
export function useConnectedApps(opts?: { enabled?: boolean }) {
    const { isLoaded, isSignedIn } = useAuth()
    return useQuery({
        queryKey: ['composio', 'connections'],
        queryFn: async () => {
            const response = await apiClient.get<ComposioConnection[]>('/api/composio/connections')
            return response as ComposioConnection[]
        },
        staleTime: 30 * 1000,
        enabled: (opts?.enabled ?? true) && isLoaded && isSignedIn,
        retry: (failureCount, error) => shouldRetry(error) && failureCount < 1,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
    })
}

/**
 * Initiate OAuth connection for an app
 */
export function useInitiateConnection() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: async ({ appName, callbackUrl }: { appName: string; callbackUrl?: string }) => {
            const response = await apiClient.post(`/api/composio/connect/${appName}`, {
                callback_url: callbackUrl,
            })
            // apiClient.post() returns parsed JSON directly, not wrapped in { data: ... }
            return response as { redirect_url: string; app_name: string }
        },
        onSuccess: () => {
            // Invalidate connections to refresh after OAuth
            queryClient.invalidateQueries({ queryKey: ['composio', 'connections'] })
        },
    })
}

/**
 * Handle OAuth callback
 */
export function useConnectionCallback() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: async ({
            appName,
            connectionId,
            status
        }: {
            appName: string
            connectionId: string
            status: string
        }) => {
            const response = await apiClient.post(
                `/api/composio/connect/${appName}/callback?connection_id=${connectionId}&status=${status}`
            )
            return response.data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['composio', 'connections'] })
            queryClient.invalidateQueries({ queryKey: ['composio', 'apps'] })
        },
    })
}

/**
 * Disconnect an app
 */
export function useDisconnectApp() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: async (appName: string) => {
            const response = await apiClient.delete(`/api/composio/connections/${appName}`)
            return response.data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['composio', 'connections'] })
            queryClient.invalidateQueries({ queryKey: ['composio', 'apps'] })
        },
    })
}

// ============================================================================
// Agent Feature Management
// ============================================================================

/**
 * Get enabled features for an agent-app combination
 */
export function useAgentAppFeatures(agentId: number, appName: string) {
    const { isLoaded, isSignedIn } = useAuth()
    return useQuery({
        queryKey: ['composio', 'agents', agentId, 'apps', appName, 'features'],
        queryFn: async () => {
            const response = await apiClient.get(`/api/composio/agents/${agentId}/apps/${appName}/features`)
            return response.data as ComposioAction[]
        },
        enabled: !!agentId && !!appName && isLoaded && isSignedIn,
        retry: (failureCount, error) => shouldRetry(error) && failureCount < 1,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
    })
}

/**
 * Update feature toggles for an agent-app combination
 */
export function useUpdateAgentFeatures() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: async ({
            agentId,
            appName,
            features
        }: {
            agentId: number
            appName: string
            features: FeatureToggle[]
        }) => {
            const response = await apiClient.put(
                `/api/composio/agents/${agentId}/apps/${appName}/features`,
                { features }
            )
            return response.data
        },
        onSuccess: (_, { agentId, appName }) => {
            queryClient.invalidateQueries({
                queryKey: ['composio', 'agents', agentId, 'apps', appName, 'features']
            })
        },
    })
}

/**
 * Enable all features for an agent-app combination
 */
export function useEnableAllFeatures() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: async ({ agentId, appName }: { agentId: number; appName: string }) => {
            const response = await apiClient.post(
                `/api/composio/agents/${agentId}/apps/${appName}/enable-all`
            )
            return response.data
        },
        onSuccess: (_, { agentId, appName }) => {
            queryClient.invalidateQueries({
                queryKey: ['composio', 'agents', agentId, 'apps', appName, 'features']
            })
        },
    })
}

/**
 * Disable all features for an agent-app combination
 */
export function useDisableAllFeatures() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: async ({ agentId, appName }: { agentId: number; appName: string }) => {
            const response = await apiClient.post(
                `/api/composio/agents/${agentId}/apps/${appName}/disable-all`
            )
            return response.data
        },
        onSuccess: (_, { agentId, appName }) => {
            queryClient.invalidateQueries({
                queryKey: ['composio', 'agents', agentId, 'apps', appName, 'features']
            })
        },
    })
}

// ============================================================================
// Trigger Management
// ============================================================================

/**
 * Get trigger subscriptions for the workspace
 */
export function useTriggerSubscriptions() {
    const { isLoaded, isSignedIn } = useAuth()
    return useQuery({
        queryKey: ['composio', 'triggers'],
        queryFn: async () => {
            const response = await apiClient.get('/api/composio/triggers')
            return response.data as TriggerSubscription[]
        },
        enabled: isLoaded && isSignedIn,
        retry: (failureCount, error) => shouldRetry(error) && failureCount < 1,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
    })
}

/**
 * Subscribe to a trigger
 */
export function useSubscribeToTrigger() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: async ({
            triggerName,
            agentId,
            workflowId
        }: {
            triggerName: string
            agentId?: number
            workflowId?: number
        }) => {
            const response = await apiClient.post('/api/composio/triggers/subscribe', {
                trigger_name: triggerName,
                agent_id: agentId,
                workflow_id: workflowId,
            })
            return response.data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['composio', 'triggers'] })
        },
    })
}

/**
 * Unsubscribe from a trigger
 */
export function useUnsubscribeTrigger() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: async (subscriptionId: number) => {
            const response = await apiClient.delete(`/api/composio/triggers/${subscriptionId}`)
            return response.data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['composio', 'triggers'] })
        },
    })
}
