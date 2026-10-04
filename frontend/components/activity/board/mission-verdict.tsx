'use client'

/**
 * PRD-252 D6 — a mission's card in Review is the mission's plan waiting for the
 * owner. Its buttons are the mission's own: Approve starts the mission through
 * the mission's approve endpoint (a ticket approval stranded it), and Reject
 * cancels it through the mission's reject endpoint. Changing the plan instead
 * happens on the mission's page. A step in Review is its mission's to check.
 * F291: Approve takes a note, which every step of the mission is given.
 */

import { useState } from 'react'
import Link from 'next/link'
import { useQueryClient } from '@tanstack/react-query'
import { CheckCircle2, XCircle } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { boardQueryKeys } from '@/hooks/use-board-tasks'
import { useApproveMission, useRejectMission } from '@/hooks/use-missions-api'
import { missionHref } from '@/lib/ticket-links'
import type { BoardTask } from '@/types/board'

const MISSION_PLAN = 'mission_plan'

function failure(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback
}

export function MissionVerdict({ task, onDecided }: { task: BoardTask; onDecided: () => void }) {
  const missionId = task.mission_id
  if (!missionId) return null
  if (task.review_reason !== MISSION_PLAN) {
    return (
      <Link href={missionHref(missionId) as any} className="text-sm text-primary underline-offset-2 hover:underline" data-testid="review-on-mission">
        Open the mission →
      </Link>
    )
  }
  return <MissionPlanVerdict missionId={missionId} onDecided={onDecided} />
}

function MissionPlanVerdict({ missionId, onDecided }: { missionId: string; onDecided: () => void }) {
  const [rejecting, setRejecting] = useState(false)
  const [reason, setReason] = useState('')
  const [note, setNote] = useState('')
  const approve = useApproveMission()
  const reject = useRejectMission()
  const queryClient = useQueryClient()
  const busy = approve.isLoading || reject.isLoading
  const why = reason.trim()

  const decided = (message: string) => {
    toast.success(message)
    void queryClient.invalidateQueries({ queryKey: boardQueryKeys.all })
    onDecided()
  }
  const approveNow = () =>
    approve.mutate({ id: missionId, body: note.trim() ? { note: note.trim() } : {} }, {
      onSuccess: () => decided('Plan approved. The mission is starting.'),
      onError: (err) => toast.error(failure(err, 'Could not approve the plan')),
    })
  const rejectNow = () =>
    reject.mutate({ id: missionId, body: { reason: why } }, {
      onSuccess: () => decided('Plan rejected. The mission is cancelled.'),
      onError: (err) => toast.error(failure(err, 'Could not reject the plan')),
    })

  return (
    <div className="space-y-3 pt-2 border-t border-border/30" data-testid="mission-verdict">
      <p className="text-xs text-muted-foreground">
        This is the mission&apos;s plan. Approve starts the mission; Reject cancels it. To change the plan instead,{' '}
        <Link href={missionHref(missionId) as any} className="text-primary underline-offset-2 hover:underline" data-testid="review-on-mission">
          open the mission
        </Link>
        .
      </p>
      {rejecting && (
        <label className="block space-y-1.5">
          <span className="text-xs font-medium">Why reject the plan?</span>
          <textarea
            autoFocus
            value={reason}
            disabled={busy}
            onChange={(e) => setReason(e.target.value)}
            rows={2}
            className="w-full resize-y rounded border border-border bg-background px-2 py-1.5 text-sm outline-none focus:border-primary"
          />
        </label>
      )}
      {!rejecting && (
        <label className="block space-y-1.5">
          <span className="text-xs font-medium">A note for every step (optional)</span>
          <textarea
            value={note}
            disabled={busy}
            onChange={(e) => setNote(e.target.value)}
            rows={2}
            data-testid="plan-note"
            className="w-full resize-y rounded border border-border bg-background px-2 py-1.5 text-sm outline-none focus:border-primary"
          />
        </label>
      )}
      <div className="flex gap-3">
        <Button variant="outline" className="flex-1" disabled={busy} onClick={() => setRejecting((open) => !open)}>
          {rejecting ? 'Keep the plan' : <><XCircle className="w-4 h-4 mr-2" /> Reject the plan</>}
        </Button>
        {rejecting ? (
          <Button variant="destructive" className="flex-1" disabled={busy || !why} onClick={rejectNow}>
            <XCircle className="w-4 h-4 mr-2" /> Reject and cancel the mission
          </Button>
        ) : (
          <Button className="flex-1" disabled={busy} onClick={approveNow}>
            <CheckCircle2 className="w-4 h-4 mr-2" /> Approve the plan
          </Button>
        )}
      </div>
    </div>
  )
}
