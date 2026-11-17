'use client'

/**
 * PRD-274 — the "Runtime profile" choice of a saved agent, Full or Lean, on the Model tab of its
 * configuration modal, under the Runtime group. Both editions: the Runtime group above it is
 * local only, this is not.
 *
 * The pick rides the form as `runtime_profile` (undefined until the person chooses), so the
 * modal's existing save sends `configuration.runtime_profile` only when they did, through
 * `PUT /api/agents/{id}`, which merges it into the stored configuration.
 *
 * The saved profile, and whether the agent is a system agent, come from the agent detail, read
 * once. The choice stays disabled until that read answers, when it fails, and for a system agent.
 */

import { useEffect, useState } from 'react'
import { Layers } from 'lucide-react'
import { Label } from '@/components/ui/label'
import { loadApiClient } from './lazy-api-client'
import {
  PROFILE_READ_FAILED_REASON,
  RUNTIME_PROFILE_HELP,
  RUNTIME_PROFILE_LABEL,
  RUNTIME_PROFILE_OPTIONS,
  SYSTEM_AGENT_PROFILE_REASON,
  savedRuntimeProfile,
  type RuntimeProfile,
  type SavedRuntimeProfile,
} from './runtime-profile-model'
import { agentDetailPath } from './session-tool-groups-model'

const LABEL_ID = 'runtime-profile-label'
const RADIO_NAME = 'runtime-profile'

type SavedProfileRead =
  | { status: 'loading' }
  | { status: 'failed' }
  | { status: 'ready'; saved: SavedRuntimeProfile }

const LOADING: SavedProfileRead = { status: 'loading' }
const FAILED: SavedProfileRead = { status: 'failed' }

/** The saved agent's profile, read once per agent. Never throws: a failed read is a state. */
function useSavedRuntimeProfile(agentId: number): SavedProfileRead {
  const [read, setRead] = useState<SavedProfileRead>(LOADING)
  useEffect(() => {
    let cancelled = false
    setRead(LOADING)
    void (async () => {
      try {
        const apiClient = await loadApiClient()
        const detail = await apiClient.request<unknown>(agentDetailPath(agentId, null))
        if (!cancelled) setRead({ status: 'ready', saved: savedRuntimeProfile(detail) })
      } catch (error) {
        console.error('Failed to load the agent runtime profile:', error)
        if (!cancelled) setRead(FAILED)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [agentId])
  return read
}

/** Why the choice cannot change right now; null while loading or when it can. */
function lockedReason(read: SavedProfileRead): string | null {
  if (read.status === 'failed') return PROFILE_READ_FAILED_REASON
  if (read.status === 'ready' && read.saved.systemAgent) return SYSTEM_AGENT_PROFILE_REASON
  return null
}

/** The profile to show as chosen: the person's pick over the saved one; none until the read answers. */
function shownProfile(read: SavedProfileRead, draft: RuntimeProfile | undefined): RuntimeProfile | null {
  if (read.status !== 'ready') return null
  if (read.saved.systemAgent) return read.saved.profile
  return draft ?? read.saved.profile
}

interface RuntimeProfileFieldProps {
  agentId: number
  /** the person's pick on the form; undefined until they choose */
  value: RuntimeProfile | undefined
  onChange: (profile: RuntimeProfile) => void
}

export function RuntimeProfileField({ agentId, value, onChange }: RuntimeProfileFieldProps) {
  const read = useSavedRuntimeProfile(agentId)
  const editable = read.status === 'ready' && !read.saved.systemAgent
  const shown = shownProfile(read, value)
  const reason = lockedReason(read)
  const choose = (profile: RuntimeProfile) => {
    if (editable) onChange(profile)
  }
  return (
    <div className="space-y-3 rounded-lg border border-border/40 p-4" data-testid="runtime-profile">
      <div className="flex items-center gap-2">
        <Layers className="h-4 w-4 text-[hsl(var(--agent))]" aria-hidden />
        <Label id={LABEL_ID} className="text-sm font-medium">{RUNTIME_PROFILE_LABEL}</Label>
      </div>
      <div className="flex flex-wrap gap-4 text-xs" role="radiogroup" aria-labelledby={LABEL_ID}>
        {RUNTIME_PROFILE_OPTIONS.map(({ id, label }) => (
          <label key={id} className="flex items-center gap-2 text-foreground">
            <input
              type="radio"
              name={RADIO_NAME}
              value={id}
              checked={shown === id}
              disabled={!editable}
              onChange={() => choose(id)}
            />
            <span>{label}</span>
          </label>
        ))}
      </div>
      {reason && (
        <p className="text-xs text-muted-foreground" data-testid="runtime-profile-locked">
          {reason}
        </p>
      )}
      <p className="text-xs text-muted-foreground">{RUNTIME_PROFILE_HELP}</p>
    </div>
  )
}

export default RuntimeProfileField
