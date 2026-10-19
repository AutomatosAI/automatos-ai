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
 *
 * PRD-260 T-005c: Play's chat renders this same card inline in Auto's reply and
 * passes onDecided, told once an approve or a deny went through, so Auto stops
 * waiting on the owner. Studio's canvas passes none.
 *
 * PRD-261 A-12c: the card is a protected surface. The exported widget wraps it
 * in ProtectedSurface (components/theme-host/protected-surface.tsx). What is
 * guaranteed: the card's own two boxes (the host and its inner wrapper) keep
 * every guarded property (display, visibility, opacity, clicks, position, the
 * transforms, masks, zoom, size) against every stylesheet rule, and the
 * server's scope check refuses any theme selector that names either wrapper,
 * so no theme rule reaches the card, or anything inside it, by name. What is a
 * review rule (docs/theme-host/css-layer-model.md): a generic selector such as
 * `button` that matches the card's elements without naming the card, a rule on
 * an ancestor, an element drawn over the card, and a theme's code in the page.
 */

import { useState } from 'react'
import { Check, ShieldCheck, X } from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import { Button } from '@/components/ui/button'
import { OversightNote } from '../oversight-note'
import { ProtectedSurface } from '@/components/theme-host/protected-surface'
import { useDenyApproval, useGrantApproval } from '@/hooks/use-approval-grants'
import type { ToolApprovalWidgetData, WidgetBaseProps, WidgetDefinition } from '../types'
import { toast } from 'sonner'


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

/** What the owner decided on the card. */
export type ApprovalDecision = 'approved' | 'denied'

export interface ToolApprovalWidgetProps extends WidgetBaseProps<ToolApprovalWidgetData> {
  /** Told once an approve or a deny went through (never after a failed call). */
  onDecided?: (decision: ApprovalDecision) => void
}

/** Every place the card renders gets it guarded (the canvas, a chat reply, a theme's page). */
export function ToolApprovalWidget(props: ToolApprovalWidgetProps) {
  return (
    <ProtectedSurface surface="tool-approval">
      <ToolApprovalCard {...props} />
    </ProtectedSurface>
  )
}

function ToolApprovalCard({
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
  onRefresh,
  onDecided,
}: ToolApprovalWidgetProps) {
  const grantApproval = useGrantApproval()
  const denyApproval = useDenyApproval()
  const [done, setDone] = useState<'approved' | 'denied' | null>(null)
  const [outcome, setOutcome] = useState<string>('')

  const busy = grantApproval.isLoading || denyApproval.isLoading
  const paramEntries = Object.entries(data.params || {})

  const handleApprove = async () => {
    try {
      const res = await grantApproval.mutateAsync(data.grant_id)
      const executed = (res?.grant?.details as { executed_result?: ExecutedResult } | undefined)
        ?.executed_result
      const line = executedOutcomeLine(executed)
      setOutcome(line)
      setDone('approved')
      onDecided?.('approved')
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
      await denyApproval.mutateAsync(data.grant_id)
      setOutcome('Denied — the action will not run.')
      setDone('denied')
      onDecided?.('denied')
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
          {data.question_md ? (
            <p className="text-xs mt-0.5 whitespace-pre-line break-words">{data.question_md}</p>
          ) : data.message && (
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
          <OversightNote tier={data.risk_tier} riskClass={data.risk_class} rationale={data.oversight_rationale} />
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
