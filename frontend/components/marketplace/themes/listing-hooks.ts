/**
 * PRD-261 C-04b — the Themes tab's listings and "Add to my workspace" (C-04a's routes, plan §2.2).
 *
 * React Query over the apiClient methods of the W-C2 opening commit. The keys sit under
 * `themeQueryKeys.all(workspaceId)` (hooks/use-installed-themes.ts), so every theme write A-07d
 * makes (update, rollback, uninstall, choose) also refreshes the cards' "in this workspace"
 * state, and an install from a listing refreshes A-07d's installed list.
 *
 * Install is `manage` + who installs + step-up on the server: it goes through `useStepUp`
 * (lib/theme-host/api.ts), so SaaS shows Clerk's "confirm it's you" and runs the call once more;
 * the local edition calls straight through. A refusal shows the server's own sentence.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { useWorkspace } from '@/components/workspace-provider'
import { isStaleThemeRefusal, themeQueryKeys } from '@/hooks/use-installed-themes'
import { apiClient } from '@/lib/api-client'
import type { ThemeListing, ThemeListingDetail, ThemeListingInstallResult } from '@/lib/api-client'
import { themeErrorMessage, useStepUp } from '@/lib/theme-host/api'

import { ADDED_TO_WORKSPACE, ALREADY_IN_WORKSPACE } from './listing-format'

const LISTING_STALE_MS = 30_000

export const themeListingKeys = {
  list: (workspaceId: string | null) => [...themeQueryKeys.all(workspaceId), 'listings'] as const,
  detail: (workspaceId: string | null, listingId: string) =>
    [...themeQueryKeys.all(workspaceId), 'listing', listingId] as const,
}

function useWorkspaceId(): string | null {
  return useWorkspace().workspace?.id ?? null
}

/** Every approved theme, one card each, with this workspace's install of it. */
export function useThemeListings() {
  const workspaceId = useWorkspaceId()
  return useQuery<ThemeListing[]>({
    queryKey: themeListingKeys.list(workspaceId),
    enabled: !!workspaceId,
    queryFn: async () => (await apiClient.getThemeListings()).listings,
    staleTime: LISTING_STALE_MS,
  })
}

/** One listing's detail; fetched only while a listing is open. */
export function useThemeListing(listingId: string | null) {
  const workspaceId = useWorkspaceId()
  return useQuery<ThemeListingDetail>({
    queryKey: themeListingKeys.detail(workspaceId, listingId ?? ''),
    enabled: !!workspaceId && !!listingId,
    queryFn: () => apiClient.getThemeListing(listingId ?? ''),
    staleTime: LISTING_STALE_MS,
  })
}

export interface InstallListingVars {
  readonly listingId: string
  /** A version from the versions map; omitted, the server installs the newest one this host runs. */
  readonly packageId?: string
}

/** Add a listed version to this workspace by reference (manage + who installs + step-up). */
export function useInstallThemeListing() {
  const workspaceId = useWorkspaceId()
  const queryClient = useQueryClient()
  const install = useStepUp(apiClient.installThemeListing.bind(apiClient))
  const invalidate = () => queryClient.invalidateQueries({ queryKey: themeQueryKeys.all(workspaceId) })
  return useMutation<ThemeListingInstallResult, unknown, InstallListingVars>({
    mutationFn: ({ listingId, packageId }) => install(listingId, packageId),
    onSuccess: async (result) => {
      toast.success(result.created ? ADDED_TO_WORKSPACE : ALREADY_IN_WORKSPACE)
      await invalidate()
    },
    onError: async (error) => {
      toast.error(themeErrorMessage(error))
      if (isStaleThemeRefusal(error)) await invalidate()
    },
  })
}
