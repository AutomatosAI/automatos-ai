'use client'

/**
 * MissionApprovalWidget (PRD-163 S4)
 *
 * In-chat card for an auto-created mission that is awaiting approval: plan
 * summary, cost vs ceiling, an optional auto-proceed countdown, and
 * Approve / Reject. Approving starts execution; rejecting returns structured
 * feedback to the mission. Rendered via the widget router.
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
 *
 * PRD-268 US-C20c: a plan that carries `approval_view` also shows who asked, the goal, its number,
 * the steps and the approval policy that held it, before the buttons, and answers keys 1 (Reject)
 * and 3 (Approve & run) while the card has focus (approval-view-parts.tsx). Without one it is today's.
 */

import { useState } from 'react'
import { ClipboardCheck, Check, X, Clock } from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import { Button } from '@/components/ui/button'
import { OversightNote } from '../oversight-note'
import { ApprovalViewParts, useApprovalCardBody } from '../approval-view-parts'
import { MISSION_APPROVAL_TITLE } from '@/lib/chat/approvals'
import { ProtectedSurface } from '@/components/theme-host/protected-surface'
import { useCountdown } from '@/hooks/use-countdown'
import { useApproveMission, useRejectMission, useUpdateMissionPlan } from '@/hooks/use-missions-api'
import type { WidgetBaseProps, MissionApprovalTask, MissionApprovalWidgetData, WidgetDefinition } from '../types'
import { roleEdit, withPlanPicks } from './role-edit'
import { MissionTaskList } from './task-list'
import { toast } from 'sonner'

/** Every place the card renders gets it guarded (the canvas, a chat reply, a theme's page). */
export function MissionApprovalWidget(props: WidgetBaseProps<MissionApprovalWidgetData>) {
  return (
    <ProtectedSurface surface="mission-approval">
      <MissionApprovalCard {...props} />
    </ProtectedSurface>
  )
}

function MissionApprovalCard({
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
  const updatePlan = useUpdateMissionPlan()
  const countdown = useCountdown(data.approval_deadline_at)
  const [done, setDone] = useState<'approved' | 'rejected' | null>(null)
  // F162 (b): the tasks as the server last answered, so an edit shows who now runs it.
  const [tasks, setTasks] = useState<MissionApprovalTask[]>(data.tasks ?? [])

  const busy = approve.isLoading || reject.isLoading || updatePlan.isLoading

  // PRD-163 S4/Q57: reassign a task's agent before approval. The edit PATCHes the
  // plan so the change persists onto the task row the dispatcher will execute.
  const commitRole = async (task: MissionApprovalTask, seq: number, value: string, original: string) => {
    const next = value.trim()
    if (!next || next === original.trim()) return
    try {
      const updated = await updatePlan.mutateAsync({
        id: data.mission_id,
        body: { task_edits: [roleEdit(task, seq, next)] },
      })
      setTasks((current) => withPlanPicks(current, updated?.plan))
      toast.success(`"${task.title}" reassigned to ${next}`)
    } catch {
      toast.error('Failed to update the plan')
    }
  }

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

  const heading = title || MISSION_APPROVAL_TITLE
  const card = useApprovalCardBody(data.approval_view, data.mission_id, heading, {
    refuse: handleReject,
    allowOnce: handleApprove,
    disabled: busy || done !== null,
  })

  return (
    <WidgetBase
      title={heading}
      icon={<ClipboardCheck className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={isLoading}
      error={error}
      onClose={onClose}
      onMaximize={onMaximize}
      onRefresh={onRefresh}
    >
      <div {...card.body}>
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

        {/* PRD-181 S5 (EU-AI-Act Art.14): risk tier + oversight rationale so the
            approver sees the risk classification and why a human is in the loop. */}
        {(data.risk_tier || data.oversight_rationale) && (
          <OversightNote tier={data.risk_tier} riskClass={data.risk_class} rationale={data.oversight_rationale} />
        )}

        <MissionTaskList tasks={tasks} disabled={busy || done !== null} onCommitRole={commitRole} />

        {countdown !== null && !done && (
          <div className="flex items-center gap-1.5 text-xs text-amber-600">
            <Clock className="h-3 w-3" />
            Auto-proceeds in {countdown}s unless you act
          </div>
        )}

        {card.view && <ApprovalViewParts view={card.view} ruleId={card.ruleId} />}

        {done ? (
          <p className="text-sm text-muted-foreground">Plan {done}.</p>
        ) : (
          <div className="flex gap-2">
            <Button size="sm" disabled={busy} onClick={handleApprove} className="flex-1" {...card.allowOnceKey}>
              <Check className="h-4 w-4 mr-1" /> {approve.isLoading ? 'Starting…' : 'Approve & run'}
            </Button>
            <Button size="sm" variant="outline" disabled={busy} onClick={handleReject} {...card.refuseKey}>
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
