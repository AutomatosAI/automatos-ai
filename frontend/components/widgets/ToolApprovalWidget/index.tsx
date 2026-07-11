'use client'

/**
 * ToolApprovalWidget (PRD-193 S3, P2-12)
 *
 * In-chat card for a confirmation-gated tool call awaiting approval: the
 * action, a human-readable params digest, permission level, AI-Act oversight
 * tier + rationale, and Approve / Deny wired to the PRD-181 approval-grant
 * API (its first frontend consumer). Cloned from MissionApprovalWidget —
 * shared presentation, different behaviour (a grant is granted/denied; a
 * mission is PATCHed) — deliberately NOT generalised (revisit on a third card).
 *
 * Approving resumes the interrupted execution server-side (PRD-193 S4); the
 * card reflects the executed result honestly — a failed resume is shown as a
 * failure, never a fake success.
 */

import { useState } from 'react'
import { Check, ShieldAlert, ShieldCheck, X } from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import { Button } from '@/components/ui/button'
import { useDenyApproval, useGrantApproval } from '@/hooks/use-approval-grants-api'
import type { ToolApprovalWidgetData, WidgetBaseProps, WidgetDefinition } from '../types'
import { toast } from 'sonner'

/**
 * Human-readable label for an oversight tier (clone of the mission card's —
 * presentation only, kept local per the no-premature-abstraction call).
 * Exported for tests.
 */
export function oversightTierLabel(tier?: string): string {
  switch (tier) {
    case 'monitor':
      return 'Monitored'
    case 'human_on_the_loop':
      return 'Human on the loop'
    case 'human_in_the_loop':
      return 'Human approval required'
    default:
      return 'Human approval required'
  }
}

interface ExecutedResult {
  success?: boolean
  error?: string | null
  requires_confirmation?: boolean
  resumed_via?: string
}

/** Post-approve outcome line — honest about what actually happened (S4). */
export function executedOutcomeLine(executed?: ExecutedResult | null): string {
  if (!executed) return 'Approved.'
  if (executed.resumed_via === 'board_task_requeue') {
    return 'Approved — the blocked task was re-queued and will complete under this grant.'
  }
  if (executed.requires_confirmation) {
    return 'Approved, but the action asked for confirmation again — check the pending approvals.'
  }
  if (executed.success) return 'Approved — the action ran.'
  return `Approved, but execution failed: ${executed.error || 'unknown error'}`
}

export function ToolApprovalWidget({
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
  onRefresh,
}: WidgetBaseProps<ToolApprovalWidgetData>) {
  const grantApproval = useGrantApproval()
  const denyApproval = useDenyApproval()
  const [done, setDone] = useState<'approved' | 'denied' | null>(null)
  const [outcome, setOutcome] = useState<string>('')

  const busy = grantApproval.isLoading || denyApproval.isLoading
  const paramEntries = Object.entries(data.params || {})

  const handleApprove = async () => {
    try {
      const res = await grantApproval.mutateAsync({ id: data.grant_id })
      const executed = (res?.grant?.details as { executed_result?: ExecutedResult } | undefined)
        ?.executed_result
      const line = executedOutcomeLine(executed)
      setOutcome(line)
      setDone('approved')
      if (executed && executed.success === false) {
        toast.error(line)
      } else {
        toast.success(line)
      }
    } catch {
      toast.error('Failed to approve the action')
    }
  }

  const handleDeny = async () => {
    try {
      await denyApproval.mutateAsync({ id: data.grant_id })
      setOutcome('Denied — the action will not run.')
      setDone('denied')
      toast.info('Action denied')
    } catch {
      toast.error('Failed to deny the action')
    }
  }

  return (
    <WidgetBase
      title={title || 'Action approval needed'}
      icon={<ShieldCheck className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={isLoading}
      error={error}
      onClose={onClose}
      onMaximize={onMaximize}
      onRefresh={onRefresh}
    >
      <div className="flex flex-col gap-3 p-3">
        <div>
          <p className="text-sm font-medium break-all">{data.action}</p>
          {data.message && (
            <p className="text-xs text-muted-foreground mt-0.5">{data.message}</p>
          )}
          {data.permission_level && (
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground mt-1">
              {data.permission_level}
            </p>
          )}
        </div>

        {/* AI-Act oversight banner — clone of the mission card's presentation. */}
        {(data.risk_tier || data.oversight_rationale) && (
          <div
            className="flex items-start gap-2 rounded border border-amber-300 bg-amber-50 px-2 py-1.5 dark:border-amber-800 dark:bg-amber-950/40"
            role="note"
            aria-label="Human oversight"
          >
            <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
            <div className="min-w-0">
              <p className="text-[11px] font-medium text-amber-800 dark:text-amber-300">
                {oversightTierLabel(data.risk_tier)}
                {data.risk_class ? ` · ${data.risk_class.replace(/_/g, ' ')}` : ''}
              </p>
              {data.oversight_rationale && (
                <p className="mt-0.5 text-[10px] text-amber-700 dark:text-amber-400">
                  {data.oversight_rationale}
                </p>
              )}
            </div>
          </div>
        )}

        {/* The exact call being approved — key/value digest, never raw JSON. */}
        {paramEntries.length > 0 && (
          <dl className="space-y-0.5 max-h-40 overflow-y-auto rounded border border-border bg-muted/40 px-2 py-1.5">
            {paramEntries.map(([key, value]) => (
              <div key={key} className="flex items-baseline gap-2 text-xs">
                <dt className="shrink-0 font-mono text-muted-foreground">{key}</dt>
                <dd className="min-w-0 truncate font-mono" title={String(value)}>
                  {String(value)}
                </dd>
              </div>
            ))}
          </dl>
        )}

        {done ? (
          <p className="text-sm text-muted-foreground" role="status">
            {outcome}
          </p>
        ) : (
          <div className="flex gap-2">
            <Button size="sm" disabled={busy} onClick={handleApprove} className="flex-1">
              <Check className="h-4 w-4 mr-1" />{' '}
              {grantApproval.isLoading ? 'Approving…' : 'Approve & run'}
            </Button>
            <Button size="sm" variant="outline" disabled={busy} onClick={handleDeny}>
              <X className="h-4 w-4 mr-1" /> Deny
            </Button>
          </div>
        )}
      </div>
    </WidgetBase>
  )
}

export const ToolApprovalWidgetDef: WidgetDefinition<ToolApprovalWidgetData> = {
  type: 'tool_approval',
  displayName: 'Action Approval',
  description: 'Approve or deny a confirmation-gated tool call',
  icon: ShieldCheck,
  component: ToolApprovalWidget,
  defaultSize: { width: 5, height: 4 },
  minSize: { width: 4, height: 3 },
  capabilities: ['refreshable'],
}

// Register the widget (importing this module auto-registers it)
registerWidget(ToolApprovalWidgetDef)
