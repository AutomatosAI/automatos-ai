'use client'

/**
 * F372 (night 10c) — when the workspace's brand kit last changed (`updated_at`, stamped by the
 * kit's one writer on every save, by any route: the page, Auto, the Brand designer, the API).
 *
 * The brand board on the Brand kit page is keyed on it. The page used to draw the board again
 * only after a save made on the page, so a kit restored through the API still showed the old
 * board. The stamp is read on load, when the window regains focus, every KIT_STAMP_POLL_MS
 * while the page is open, and again whenever `changes` (the page's own stored changes) moves.
 */
import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'

import { templateBlocksApi } from '@/components/documents/blocks/api'
import { useWorkspaceOptional } from '@/components/workspace-provider'

export const KIT_STAMP_POLL_MS = 30_000

export const brandKitStampKey = (workspaceId: string | null) => ['brand-kit', workspaceId, 'updated-at'] as const

/** The kit's `updated_at` ('' for a kit not saved since); undefined until it is first read. */
export function useBrandKitStamp(changes: number): string | undefined {
  const workspaceId = useWorkspaceOptional()?.workspace?.id ?? null
  const query = useQuery({
    queryKey: brandKitStampKey(workspaceId),
    queryFn: async () => (await templateBlocksApi.getBrandKit()).updated_at ?? '',
    refetchOnWindowFocus: true,
    refetchInterval: KIT_STAMP_POLL_MS,
  })
  const { refetch } = query
  useEffect(() => {
    if (changes > 0) void refetch()
  }, [changes, refetch])
  return query.data
}
