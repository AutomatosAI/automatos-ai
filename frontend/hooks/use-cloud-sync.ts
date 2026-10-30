/**
 * PRD-270 US-270-05a (§5.5, K18) — Sync with Automatos Cloud for the Marketplace header: the Sync's
 * status, and Sync now.
 *
 * React Query over lib/marketplace/cloud-sync.ts. `enabled` false asks nothing: the callers pass it on
 * the hosted edition and for a member, where the routes answer 404 and 403. A Sync writes listings and
 * may move installs to a newer Cloud version (US-270-04b), so after one every theme query of this
 * workspace (the listings, the installed list) is read again, without holding the button busy while
 * they load, and its answer's status replaces the cached one. A refused or failed Sync shows the
 * server's own sentence and reads the status again, which then carries the error it recorded.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { syncedMessage, updatesFailedMessage } from '@/components/marketplace/cloud/copy'
import { useWorkspace } from '@/components/workspace-provider'
import { themeQueryKeys } from '@/hooks/use-installed-themes'
import { getCloudSyncStatus, runCloudSync } from '@/lib/marketplace/cloud-sync'
import type { CloudSyncResult, CloudSyncStatus } from '@/lib/marketplace/cloud-sync'
import { themeErrorMessage } from '@/lib/theme-host/api'

const STATUS_STALE_MS = 30_000

export const cloudSyncKeys = {
  status: (workspaceId: string | null) => ['marketplace', 'cloud-sync', workspaceId, 'status'] as const,
}

function useWorkspaceId(): string | null {
  return useWorkspace().workspace?.id ?? null
}

/** The Sync's status: on or off, the last success and error, the themes held from Automatos Cloud. */
export function useCloudSyncStatus(enabled: boolean) {
  const workspaceId = useWorkspaceId()
  return useQuery<CloudSyncStatus>({
    queryKey: cloudSyncKeys.status(workspaceId),
    enabled: enabled && !!workspaceId,
    queryFn: getCloudSyncStatus,
    staleTime: STATUS_STALE_MS,
  })
}

/** One Sync now; the toast says what it brought, and which themes' updates failed. */
export function useCloudSync() {
  const workspaceId = useWorkspaceId()
  const client = useQueryClient()
  const refreshThemes = () => client.invalidateQueries({ queryKey: themeQueryKeys.all(workspaceId) })
  return useMutation<CloudSyncResult, unknown, void>({
    mutationFn: () => runCloudSync(),
    onSuccess: (result) => {
      client.setQueryData(cloudSyncKeys.status(workspaceId), result.status)
      toast.success(syncedMessage(result.counts))
      if (result.updatesFailed > 0) toast.error(updatesFailedMessage(result.updatesFailed))
      void refreshThemes()
    },
    onError: async (error) => {
      toast.error(themeErrorMessage(error))
      await client.invalidateQueries({ queryKey: cloudSyncKeys.status(workspaceId) })
    },
  })
}
