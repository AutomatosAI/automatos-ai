'use client'

/**
 * #942 — what this agent's sessions get, and what they would miss, for the selection on screen.
 *
 * The saved agent is read once (`GET /api/agents/{id}`). While the owner ticks boxes, or when a
 * saved api agent is being switched to cli (its detail carries no gaps), the same call with
 * `?groups=` asks the backend for the gaps of the selection, debounced. Never throws: on a failed
 * call the page keeps what it last knew, and an older backend simply returns no groups.
 */

import { useEffect, useState } from 'react'
import type { SessionToolGapEntry, SessionToolGroups } from '@/types/session-tools'
import { agentDetailPath, normalizeSessionToolGroups } from './session-tool-groups-model'

export interface SessionToolPreview {
  /** the saved agent's groups; null on a backend older than #942 */
  groups: SessionToolGroups | null
  /** the gaps of the selection on screen, else of the saved agent; null when unknown */
  gaps: SessionToolGapEntry[] | null
}

const UNKNOWN: SessionToolPreview = { groups: null, gaps: null }

export const PREVIEW_DEBOUNCE_MS = 400

async function readAgentTools(path: string): Promise<SessionToolPreview> {
  const { apiClient } = await import('@/lib/api-client')
  const detail = await apiClient.request<{ session_tool_groups?: unknown; session_tool_gaps?: unknown }>(path)
  const gaps = detail?.session_tool_gaps
  return {
    groups: normalizeSessionToolGroups(detail?.session_tool_groups),
    gaps: Array.isArray(gaps) ? (gaps as SessionToolGapEntry[]) : null,
  }
}

/** The selection to preview: the owner's edit, else the saved one when the saved agent reports no gaps; null = nothing to ask. */
function previewKey(draft: string[] | undefined, saved: SessionToolPreview): string | null {
  if (!saved.groups) return null
  if (draft) return draft.join(',')
  return saved.gaps === null ? saved.groups.enabled.join(',') : null
}

export function useSessionToolPreview(
  agentId: number | null,
  draft: string[] | undefined,
  enabled: boolean,
): SessionToolPreview {
  const [saved, setSaved] = useState<SessionToolPreview>(UNKNOWN)
  const [previewGaps, setPreviewGaps] = useState<SessionToolGapEntry[] | null>(null)
  const active = enabled && agentId !== null

  useEffect(() => {
    if (!active || agentId === null) return
    let cancelled = false
    readAgentTools(agentDetailPath(agentId, null)).then(
      (view) => !cancelled && setSaved(view),
      () => !cancelled && setSaved(UNKNOWN),
    )
    return () => {
      cancelled = true
    }
  }, [active, agentId])

  const key = active ? previewKey(draft, saved) : null
  useEffect(() => {
    if (key === null || agentId === null) {
      setPreviewGaps(null)
      return
    }
    let cancelled = false
    const timer = setTimeout(() => {
      readAgentTools(agentDetailPath(agentId, key.split(',').filter(Boolean))).then(
        (view) => !cancelled && setPreviewGaps(view.gaps),
        () => undefined, // keep the last answer; the saved agent's gaps still show
      )
    }, PREVIEW_DEBOUNCE_MS)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [key, agentId])

  return active ? { groups: saved.groups, gaps: previewGaps ?? saved.gaps } : UNKNOWN
}
