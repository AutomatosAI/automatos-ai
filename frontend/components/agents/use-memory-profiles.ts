'use client'

/**
 * PRD-274 Wave 4 — the memory profiles this workspace can pick (`GET /api/v1/memory/profiles`),
 * for the memory profile choice of an agent's configuration modal.
 *
 * Through the shared lazy `apiClient` import, as the Runtime group's other reads are
 * (`lazy-api-client.ts`), not react-query: the Runtime group's tests render it without a
 * QueryClientProvider. Never throws: a failed read is a state, and `reload` asks again.
 */

import { useCallback, useEffect, useState } from 'react'
import { loadApiClient } from './lazy-api-client'
import { MEMORY_PROFILES_PATH, normalizeMemoryProfiles, type MemoryProfileList } from './memory-profile-model'

export type MemoryProfilesRead =
  | { status: 'loading' }
  | { status: 'failed' }
  | { status: 'ready'; list: MemoryProfileList }

const LOADING: MemoryProfilesRead = { status: 'loading' }
const FAILED: MemoryProfilesRead = { status: 'failed' }

export function useMemoryProfiles(): { read: MemoryProfilesRead; reload: () => void } {
  const [read, setRead] = useState<MemoryProfilesRead>(LOADING)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    let cancelled = false
    setRead(LOADING)
    void (async () => {
      try {
        const apiClient = await loadApiClient()
        const body = await apiClient.request<unknown>(MEMORY_PROFILES_PATH)
        if (!cancelled) setRead({ status: 'ready', list: normalizeMemoryProfiles(body) })
      } catch (error) {
        console.error('Failed to load the memory profiles:', error)
        if (!cancelled) setRead(FAILED)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [attempt])
  const reload = useCallback(() => setAttempt((count) => count + 1), [])
  return { read, reload }
}
