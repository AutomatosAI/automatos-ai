'use client'

import { useCallback } from 'react'
import useLocalStorageState from 'use-local-storage-state'

export const MAX_PINNED = 6

export function usePinnedAgents(workspaceId: string) {
  const [pinnedIds, setPinnedIds] = useLocalStorageState<number[]>(
    `pinned_chat_agents_${workspaceId}`,
    { defaultValue: [] }
  )

  const pin = useCallback(
    (agentId: number) => {
      setPinnedIds((prev) => {
        if (prev.length >= MAX_PINNED) return prev
        if (prev.includes(agentId)) return prev
        return [...prev, agentId]
      })
    },
    [setPinnedIds]
  )

  const unpin = useCallback(
    (agentId: number) => {
      setPinnedIds((prev) => prev.filter((id) => id !== agentId))
    },
    [setPinnedIds]
  )

  const isPinned = useCallback(
    (agentId: number): boolean => pinnedIds.includes(agentId),
    [pinnedIds]
  )

  const reorder = useCallback(
    (newOrder: number[]) => {
      setPinnedIds(newOrder)
    },
    [setPinnedIds]
  )

  return { pinnedIds, pin, unpin, isPinned, reorder }
}
