'use client'

/**
 * The Mission approval card's plan: each task with the agent that runs it, which the owner may
 * change before approving (PRD-163 S4/Q57), and who would run it and why (PRD-164 S2). Moved out
 * of MissionApprovalCard as it was (PRD-268 US-C20c), so the card stays within the component
 * limit as it gains the approval view.
 */

import type { MissionApprovalTask } from '../types'

export interface MissionTaskListProps {
  readonly tasks: readonly MissionApprovalTask[]
  /** A call is in flight or the plan is decided: the agent boxes are read-only. */
  readonly disabled: boolean
  /** Told when an agent box loses focus, with the task, its sequence, the value and the original. */
  readonly onCommitRole: (task: MissionApprovalTask, seq: number, value: string, original: string) => void
}

export function MissionTaskList({ tasks, disabled, onCommitRole }: MissionTaskListProps) {
  if (tasks.length === 0) return null
  return (
    <ol className="space-y-1 max-h-48 overflow-y-auto">
      {tasks.map((t, i) => {
        const seq = t.sequence ?? i + 1
        const role = t.agent_role || ''
        return (
          <li key={i} className="text-sm">
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground w-5 shrink-0">{seq}.</span>
              <span className="flex-1 truncate">{t.title}</span>
              <input
                type="text"
                aria-label={`Agent for task ${seq}`}
                defaultValue={role}
                disabled={disabled}
                placeholder="agent"
                onBlur={(e) => onCommitRole(t, seq, e.target.value, role)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
                }}
                className="w-24 shrink-0 rounded border border-border bg-background px-1.5 py-0.5 text-[10px] text-muted-foreground outline-none focus:border-primary focus:text-foreground disabled:opacity-50"
              />
            </div>
            {/* PRD-164 S2: who would run this task and why (match preview) */}
            {t.match_agent && (
              <p
                className="ml-7 mt-0.5 truncate text-[10px] text-muted-foreground"
                title={t.match_reason || undefined}
              >
                → {t.match_agent}
                {t.match_is_override ? ' (your pick)' : ''}
                {t.match_reason ? ` — ${t.match_reason}` : ''}
              </p>
            )}
          </li>
        )
      })}
    </ol>
  )
}
