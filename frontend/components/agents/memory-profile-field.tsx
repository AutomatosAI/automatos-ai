'use client'

/**
 * PRD-274 Wave 4 — the "Memory profile" choice of a saved agent: what its distillation keeps as a
 * fact, from the profiles this workspace can pick. On the Model tab of its configuration modal,
 * under the runtime profile, in both editions.
 *
 * The pick rides the form as `memory_profile` (undefined until the person chooses), so the
 * modal's existing save sends `configuration.memory_profile` only when they did, through
 * `PUT /api/agents/{id}`, which merges it into the stored configuration. "Workspace default" is
 * always offered: it is what an agent with no profile of its own shows, and choosing it on an
 * agent that has one saves `memory_profile: null`, which the backend reads as no profile of the
 * agent's own. Nothing on the backend treats a system agent differently here, so the choice is
 * open for one too.
 *
 * A failed list shows the shared error state with a retry; the runtime profile above it keeps
 * working.
 */

import { Brain } from 'lucide-react'
import { ErrorState } from '@/components/shared/error-state'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  AGENT_LOAD_FAILED_TEXT,
  MEMORY_PROFILE_LABEL,
  PROFILES_LOAD_FAILED_TEXT,
  PROFILES_LOAD_FAILED_TITLE,
  PROFILES_OFF_NOTE,
  memoryProfileOptions,
  pickFor,
  savedMemoryProfile,
  shownMemoryProfile,
  type MemoryProfilePick,
} from './memory-profile-model'
import { useMemoryProfiles } from './use-memory-profiles'
import type { SavedAgentRead } from './use-saved-agent'

const FIELD_ID = 'memory-profile'
/** The shared error state, sized for one setting rather than a page. */
const SETTING_ERROR_CLASS = 'py-4'
const NOTE_CLASS = 'text-xs text-muted-foreground'

interface MemoryProfileFieldProps {
  /** the saved agent's read, shared with the runtime profile choice */
  agent: SavedAgentRead
  /** the person's pick on the form: a slug, null to clear a saved one, undefined until they choose */
  value: MemoryProfilePick
  onChange: (pick: MemoryProfilePick) => void
}

export function MemoryProfileField({ agent, value, onChange }: MemoryProfileFieldProps) {
  const { read, reload } = useMemoryProfiles()
  const saved = agent.status === 'ready' ? savedMemoryProfile(agent.detail) : null
  const ready = read.status === 'ready' && agent.status === 'ready'
  const options = read.status === 'ready' && ready ? memoryProfileOptions(read.list, saved) : []
  const shown = shownMemoryProfile(value, saved)
  const help = options.find((option) => option.slug === shown)?.description ?? ''
  const choose = (next: string) => {
    if (ready) onChange(pickFor(next, saved))
  }
  return (
    <div className="space-y-3 rounded-lg border border-border/40 p-4" data-testid="memory-profile">
      <div className="flex items-center gap-2">
        <Brain className="h-4 w-4 text-[hsl(var(--agent))]" aria-hidden />
        <Label htmlFor={FIELD_ID} className="text-sm font-medium">{MEMORY_PROFILE_LABEL}</Label>
      </div>
      {read.status === 'failed' ? (
        <ErrorState
          title={PROFILES_LOAD_FAILED_TITLE}
          description={PROFILES_LOAD_FAILED_TEXT}
          onRetry={reload}
          className={SETTING_ERROR_CLASS}
        />
      ) : (
        <>
          <Select value={shown} onValueChange={choose} disabled={!ready}>
            <SelectTrigger id={FIELD_ID}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {options.map((option) => (
                <SelectItem key={option.slug} value={option.slug}>
                  {option.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {help && (
            <p className={NOTE_CLASS} data-testid="memory-profile-help">
              {help}
            </p>
          )}
          {agent.status === 'failed' && <p className={NOTE_CLASS}>{AGENT_LOAD_FAILED_TEXT}</p>}
          {read.status === 'ready' && !read.list.enabled && (
            <p className={NOTE_CLASS} data-testid="memory-profiles-off">
              {PROFILES_OFF_NOTE}
            </p>
          )}
        </>
      )}
    </div>
  )
}

export default MemoryProfileField
