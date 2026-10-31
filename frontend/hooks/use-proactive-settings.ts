'use client'

/**
 * PRD-268 §6.9, FR-13 — Proactive Auto's switches in Settings → Orchestrator.
 *
 * - `useProactiveSettings` reads GET /api/proactive/settings, the workspace switch and its caps (owners and admins
 *   only: pass `enabled: false` for anyone else, who reads the switch from the workspace instead).
 * - `useChangeProactiveSettings` is PUT /api/proactive/settings `{enabled?, caps?}`.
 * - `useMyProactiveSettings` and `useChangeMyProactiveSettings` are GET and PUT /api/proactive/me, the person's own
 *   "Pause suggestions" (any member).
 *
 * Both changes are optimistic: the cache shows the change at once, the server's answer replaces it, and a refusal
 * puts back what was there before and reads it again (the mutation's `error` carries the server's sentence,
 * `refusedLine`). A change
 * that lands asks the open suggestions and the day's brief for again, so the Command Center, the pinned card and the
 * thread's cards follow the switch without a reload. The keys carry the workspace, so another workspace's answer is
 * never shown.
 */
import { useMutation, useQuery, useQueryClient, type QueryClient, type QueryKey } from '@tanstack/react-query'
import { useWorkspaceOptional } from '@/components/workspace-provider'
import { briefQueryKeys } from '@/hooks/use-brief'
import { suggestionQueryKeys } from '@/hooks/use-suggestions'
import { apiClient } from '@/lib/api-client'
import {
  proactiveMeOf,
  proactiveSettingsOf,
  withChange,
  type ProactiveMe,
  type ProactiveMeChange,
  type ProactiveSettings,
  type ProactiveSettingsChange,
} from '@/lib/proactive/settings'

const SETTINGS_PATH = '/api/proactive/settings'
const ME_PATH = '/api/proactive/me'
const SETTINGS_STALE_MS = 60_000

export const proactiveSettingsQueryKeys = {
  all: ['proactive', 'settings'] as const,
  workspace: (workspaceId: string | null) => ['proactive', 'settings', 'workspace', workspaceId] as const,
  me: (workspaceId: string | null) => ['proactive', 'settings', 'me', workspaceId] as const,
}

interface Rollback<T> {
  readonly previous: T | undefined
}

function useWorkspaceId(): string | null {
  return useWorkspaceOptional()?.workspace?.id ?? null
}

/** The surfaces a switch or a pause changes: the open suggestions and the day's brief, asked for again. */
function refreshProactiveSurfaces(queryClient: QueryClient): void {
  void queryClient.invalidateQueries({ queryKey: suggestionQueryKeys.all }, { cancelRefetch: false })
  void queryClient.invalidateQueries({ queryKey: briefQueryKeys.all }, { cancelRefetch: false })
}

/** Show `next` in place of what the cache holds, and keep what it held to put back on a refusal. */
async function showAtOnce<T>(queryClient: QueryClient, key: QueryKey, next: (current: T) => T): Promise<Rollback<T>> {
  await queryClient.cancelQueries({ queryKey: key })
  const previous = queryClient.getQueryData<T>(key)
  if (previous !== undefined) queryClient.setQueryData<T>(key, next(previous))
  return { previous }
}

/**
 * Put back what the cache held, then ask the server again: another change that landed meanwhile (the switch and the
 * caps save apart) is read back, not overwritten by this one's older snapshot.
 */
function putBack<T>(queryClient: QueryClient, key: QueryKey, rollback: Rollback<T> | undefined): void {
  if (rollback?.previous !== undefined) queryClient.setQueryData<T>(key, rollback.previous)
  void queryClient.invalidateQueries({ queryKey: key })
}

/** The workspace switch and its caps; `enabled` false asks nothing (a member reads the switch from the workspace). */
export function useProactiveSettings({ enabled = true }: { readonly enabled?: boolean } = {}) {
  const workspaceId = useWorkspaceId()
  return useQuery<ProactiveSettings>({
    queryKey: proactiveSettingsQueryKeys.workspace(workspaceId),
    queryFn: async () => proactiveSettingsOf(await apiClient.request<unknown>(SETTINGS_PATH)),
    enabled: enabled && workspaceId !== null,
    staleTime: SETTINGS_STALE_MS,
  })
}

/** Switch the workspace's Proactive Auto on or off, or change its caps: what is left out stays. */
export function useChangeProactiveSettings() {
  const queryClient = useQueryClient()
  const key = proactiveSettingsQueryKeys.workspace(useWorkspaceId())
  return useMutation<ProactiveSettings, Error, ProactiveSettingsChange, Rollback<ProactiveSettings>>({
    mutationFn: async (change) => proactiveSettingsOf(await apiClient.put<unknown>(SETTINGS_PATH, change)),
    onMutate: (change) => showAtOnce<ProactiveSettings>(queryClient, key, (current) => withChange(current, change)),
    onError: (_error, _change, rollback) => putBack(queryClient, key, rollback),
    onSuccess: (settings) => {
      queryClient.setQueryData(key, settings)
      refreshProactiveSurfaces(queryClient)
    },
  })
}

/** The signed-in person's own Proactive Auto settings in this workspace. */
export function useMyProactiveSettings() {
  const workspaceId = useWorkspaceId()
  return useQuery<ProactiveMe>({
    queryKey: proactiveSettingsQueryKeys.me(workspaceId),
    queryFn: async () => proactiveMeOf(await apiClient.request<unknown>(ME_PATH)),
    enabled: workspaceId !== null,
    staleTime: SETTINGS_STALE_MS,
  })
}

/** Pause or resume the person's suggestions in this workspace. */
export function useChangeMyProactiveSettings() {
  const queryClient = useQueryClient()
  const key = proactiveSettingsQueryKeys.me(useWorkspaceId())
  return useMutation<ProactiveMe, Error, ProactiveMeChange, Rollback<ProactiveMe>>({
    mutationFn: async (change) => proactiveMeOf(await apiClient.put<unknown>(ME_PATH, change)),
    onMutate: (change) => showAtOnce<ProactiveMe>(queryClient, key, (current) => ({ ...current, paused: change.paused })),
    onError: (_error, _change, rollback) => putBack(queryClient, key, rollback),
    onSuccess: (mine) => {
      queryClient.setQueryData(key, mine)
      refreshProactiveSurfaces(queryClient)
    },
  })
}
