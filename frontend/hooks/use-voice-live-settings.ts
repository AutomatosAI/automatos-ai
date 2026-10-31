'use client'

/**
 * Auto Live's workspace switch (Settings → Voice → Auto Live), on the
 * `useSetWorkspaceSocials` pattern (hooks/use-socials-api.ts): one mutation over
 * `PUT /api/workspaces/current/voice-live`. The current state (`workspace.voiceLive`)
 * comes from `GET /api/workspaces/current` already, so a flip only needs to refresh
 * the workspace context — no separate query, no second round trip.
 */
import { useMutation } from '@tanstack/react-query'
import { toast } from 'sonner'

import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/components/workspace-provider'

export function useSetWorkspaceVoiceLive() {
  const { refreshWorkspace } = useWorkspace()
  return useMutation<unknown, Error, boolean>({
    mutationFn: (enabled) => apiClient.setWorkspaceVoiceLiveEnabled(enabled),
    onSuccess: async (_result, enabled) => {
      await refreshWorkspace()
      toast.success(enabled ? 'Auto Live is on for this workspace' : 'Auto Live is off for this workspace')
    },
    onError: (error, enabled) => {
      toast.error(error.message || (enabled ? 'Could not turn on Auto Live' : 'Could not turn off Auto Live'))
    },
  })
}
