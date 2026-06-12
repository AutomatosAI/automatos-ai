'use client'

/**
 * MissionApprovalWidget (PRD-163 S4)
 *
 * In-chat card for an auto-created mission that is awaiting approval: plan
 * summary, cost vs ceiling, an optional auto-proceed countdown, and
 * Approve / Reject. Approving starts execution; rejecting returns structured
 * feedback to the mission. Rendered via the widget router.
 */

import { useState, useEffect } from 'react'
import { ClipboardCheck, Check, X, Clock } from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import { Button } from '@/components/ui/button'
import { useApproveMission, useRejectMission } from '@/hooks/use-missions-api'
import type { WidgetBaseProps, MissionApprovalWidgetData, WidgetDefinition } from '../types'
import { toast } from 'sonner'

function useCountdown(deadlineIso?: string): number | null {
  const [secsLeft, setSecsLeft] = useState<number | null>(null)
  useEffect(() => {
    if (!deadlineIso) {
      setSecsLeft(null)
      return
    }
    const tick = () => {
      const ms = new Date(deadlineIso).getTime() - Date.now()
      setSecsLeft(Math.max(0, Math.round(ms / 1000)))
    }
    tick()
    const t = setInterval(tick, 1000)
    return () => clearInterval(t)
  }, [deadlineIso])
  return secsLeft
}

export function MissionApprovalWidget({
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
  onRefresh,
}: WidgetBaseProps<MissionApprovalWidgetData>) {
  const approve = useApproveMission()
  const reject = useRejectMission()
  const countdown = useCountdown(data.approval_deadline_at)
  const [done, setDone] = useState<'approved' | 'rejected' | null>(null)

  const busy = approve.isLoading || reject.isLoading

  const handleApprove = async () => {
    try {
      await approve.mutateAsync({ id: data.mission_id, body: {} })
      setDone('approved')
      toast.success('Mission approved — execution started')
      onClose?.()
    } catch {
      toast.error('Failed to approve mission')
    }
  }

  const handleReject = async () => {
    try {
      await reject.mutateAsync({ id: data.mission_id, body: { reason: 'Rejected from chat' } })
      setDone('rejected')
      toast.info('Mission plan rejected')
      onClose?.()
    } catch {
      toast.error('Failed to reject mission')
    }
  }

  return (
    <WidgetBase
      title={title || 'Mission plan — approval needed'}
      icon={<ClipboardCheck className="h-4 w-4" />}
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
          <p className="text-sm font-medium">{data.goal}</p>
          <p className="text-xs text-muted-foreground mt-0.5">
            {data.task_count} task{data.task_count === 1 ? '' : 's'}
            {typeof data.cost_estimate_usd === 'number' && (
              <>
                {' · est. $'}
                {data.cost_estimate_usd.toFixed(2)}
                {typeof data.cost_ceiling_usd === 'number' && (
                  <>
                    {' / $'}
                    {data.cost_ceiling_usd.toFixed(2)} ceiling
                  </>
                )}
              </>
            )}
          </p>
        </div>

        {data.tasks?.length > 0 && (
          <ol className="space-y-1 max-h-48 overflow-y-auto">
            {data.tasks.map((t, i) => (
              <li key={i} className="flex items-start gap-2 text-sm">
                <span className="text-xs text-muted-foreground mt-0.5 w-5 shrink-0">
                  {t.sequence ?? i + 1}.
                </span>
                <span className="flex-1">{t.title}</span>
                {t.agent_role && (
                  <span className="text-[10px] text-muted-foreground shrink-0">{t.agent_role}</span>
                )}
              </li>
            ))}
          </ol>
        )}

        {countdown !== null && !done && (
          <div className="flex items-center gap-1.5 text-xs text-amber-600">
            <Clock className="h-3 w-3" />
            Auto-proceeds in {countdown}s unless you act
          </div>
        )}

        {done ? (
          <p className="text-sm text-muted-foreground">Plan {done}.</p>
        ) : (
          <div className="flex gap-2">
            <Button size="sm" disabled={busy} onClick={handleApprove} className="flex-1">
              <Check className="h-4 w-4 mr-1" /> {approve.isLoading ? 'Starting…' : 'Approve & run'}
            </Button>
            <Button size="sm" variant="outline" disabled={busy} onClick={handleReject}>
              <X className="h-4 w-4 mr-1" /> Reject
            </Button>
          </div>
        )}
      </div>
    </WidgetBase>
  )
}

export const MissionApprovalWidgetDef: WidgetDefinition<MissionApprovalWidgetData> = {
  type: 'mission_approval',
  displayName: 'Mission Approval',
  description: 'Approve or reject an auto-created mission plan',
  icon: ClipboardCheck,
  component: MissionApprovalWidget,
  defaultSize: { width: 5, height: 5 },
  minSize: { width: 4, height: 3 },
  capabilities: ['refreshable'],
}

// Register the widget (importing this module auto-registers it)
registerWidget(MissionApprovalWidgetDef)
