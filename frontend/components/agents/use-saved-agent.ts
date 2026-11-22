'use client'

/**
 * PRD-274 — the saved agent (`GET /api/agents/{id}`), read once for the profile choices of its
 * configuration modal: the runtime profile reads `is_system_agent` and
 * `configuration.runtime_profile` from it, the memory profile `configuration.memory_profile`.
 *
 * Through the shared lazy `apiClient` import, as the Runtime group's other reads are
 * (`lazy-api-client.ts`), not react-query: the Runtime group's tests render it without a
 * QueryClientProvider. Never throws: a failed read is a state the fields show.
 */

import { useEffect, useState } from 'react'
import { loadApiClient } from './lazy-api-client'
import { agentDetailPath } from './session-tool-groups-model'

export type SavedAgentRead =
  | { status: 'loading' }
  | { status: 'failed' }
  | { status: 'ready'; detail: unknown }

const LOADING: SavedAgentRead = { status: 'loading' }
const FAILED: SavedAgentRead = { status: 'failed' }

/** The saved agent's detail, read once per agent. */
export function useSavedAgent(agentId: number): SavedAgentRead {
  const [read, setRead] = useState<SavedAgentRead>(LOADING)
  useEffect(() => {
    let cancelled = false
    setRead(LOADING)
    void (async () => {
      try {
        const apiClient = await loadApiClient()
        const detail = await apiClient.request<unknown>(agentDetailPath(agentId, null))
        if (!cancelled) setRead({ status: 'ready', detail })
      } catch (error) {
        console.error('Failed to load the agent for its profile settings:', error)
        if (!cancelled) setRead(FAILED)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [agentId])
  return read
}
