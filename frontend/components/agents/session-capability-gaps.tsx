'use client'

/**
 * #942 — what this workspace has that the agent's sessions could not use with the groups ticked.
 *
 * F329 (4 Oct): a Business Analyst moved to a CLI session told the owner "There is no database
 * query tool here … someone needs to run a SELECT" — the page had said nothing before the switch.
 * Each gap is the backend's own sentence, with a one-click "Turn on <group>" that ticks the group
 * that covers it. The fix only edits the form; nothing is saved until the owner saves.
 */

import { AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { SessionToolGroups, WorkspaceCapabilityGap } from '@/types/session-tools'
import { gapFixGroup } from './session-tool-groups-model'

const GAPS_TITLE = "This agent's sessions can't use everything this workspace has"

interface SessionCapabilityGapsProps {
  gaps: WorkspaceCapabilityGap[]
  /** null on a backend without groups: the messages show, with no fix button */
  groups: SessionToolGroups | null
  /** the groups ticked on screen */
  selection: string[]
  onEnable: (groupId: string) => void
}

/** The group a gap's fix turns on, when the picker offers it and it is not ticked yet. */
function fixFor(gap: WorkspaceCapabilityGap, groups: SessionToolGroups | null, selection: string[]) {
  const id = gapFixGroup(gap)
  if (!id || selection.includes(id)) return null
  return groups?.available.find((g) => g.id === id) ?? null
}

export function SessionCapabilityGaps({ gaps, groups, selection, onEnable }: SessionCapabilityGapsProps) {
  if (!gaps.length) return null
  return (
    <div
      role="alert"
      className="space-y-2 rounded-md border border-[hsl(var(--warning))]/40 bg-[hsl(var(--warning))]/10 p-3 text-xs"
      data-testid="session-capability-gaps"
    >
      <p className="flex items-center gap-2 font-medium text-foreground">
        <AlertTriangle className="h-4 w-4 text-[hsl(var(--warning))]" />
        {GAPS_TITLE}
      </p>
      <ul className="space-y-2">
        {gaps.map((gap, index) => {
          const fix = fixFor(gap, groups, selection)
          return (
            <li key={`${gap.capability}-${index}`} className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-muted-foreground">{gap.message}</span>
              {fix && (
                <Button type="button" size="sm" variant="outline" className="h-7 text-xs" onClick={() => onEnable(fix.id)}>
                  Turn on {fix.label}
                </Button>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default SessionCapabilityGaps
