'use client'

/**
 * PRD-252 R2 — a review is Reject or Approve, and each says what it does.
 *
 * Reject needs the owner's words ("What's wrong?"). They lead the redo's brief
 * (services/ticket_redo.py); the viewer used to send no note, so the agent was
 * told "The owner sent it back without a note." Approve stays one click, with an
 * optional note kept on the ticket (F038: the API threw it away), and its button
 * names its effect from `planning_data.approval_action`. Before, "Approve &
 * Publish" was the only hint that an approval runs anything. Both confirm what
 * happened in a toast, and say why when they fail (they said nothing).
 */

import { useState } from 'react'
import { AlertCircle, CheckCircle2 } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { useApproveTask, useRejectTask } from '@/hooks/use-board-tasks'
import type { BoardTask } from '@/types/board'
import { approveEffect, approveLabel, approvedMessage } from './approval-effect'

// The API's own bounds (api/board_tasks.py): a reject note is read by a model,
// an approve note on a card.
const MAX_REJECT_NOTE_CHARS = 4000
const MAX_APPROVE_NOTE_CHARS = 1000

type Mode = 'idle' | 'reject' | 'approve-note'

function failure(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback
}

export function ReviewVerdict({ task, onDecided }: { task: BoardTask; onDecided: () => void }) {
  const [mode, setMode] = useState<Mode>('idle')
  const [text, setText] = useState('')
  const approve = useApproveTask()
  const reject = useRejectTask()
  const action = task.planning_data?.approval_action
  const busy = approve.isLoading || reject.isLoading
  const note = text.trim()
  const agent = task.assignee?.agent_name ?? 'the agent'

  const sendBack = () => {
    if (!note) return
    reject.mutate({ taskId: task.id, feedback: note }, {
      onSuccess: () => {
        toast.success(`Sent back to ${agent}. The redo starts from your note.`)
        onDecided()
      },
      onError: (err) => toast.error(failure(err, 'Could not send the ticket back')),
    })
  }

  const approveNow = () => {
    approve.mutate({ taskId: task.id, note: mode === 'approve-note' ? note || undefined : undefined }, {
      onSuccess: (result) => {
        toast.success(approvedMessage(result, task.id))
        onDecided()
      },
      onError: (err) => toast.error(failure(err, 'Could not approve the ticket')),
    })
  }

  const open = (next: Mode) => {
    setText('')
    setMode(next)
  }

  return (
    <div className="space-y-3 pt-2 border-t border-border/30" data-testid="review-verdict">
      <p className="text-xs text-muted-foreground" data-testid="approve-effect">
        <CheckCircle2 className="inline w-3.5 h-3.5 mr-1 text-primary align-[-2px]" />
        {approveEffect(action)}
      </p>

      {mode !== 'idle' && (
        <VerdictNote mode={mode} value={text} onChange={setText} disabled={busy} />
      )}

      <div className="flex gap-3">
        {mode === 'idle' ? (
          <Button variant="outline" className="flex-1" disabled={busy} onClick={() => open('reject')}>
            <AlertCircle className="w-4 h-4 mr-2" /> Reject
          </Button>
        ) : (
          <Button variant="outline" className="flex-1" disabled={busy} onClick={() => open('idle')}>
            Cancel
          </Button>
        )}
        {mode === 'reject' ? (
          <Button className="flex-1" disabled={busy || !note} onClick={sendBack}>
            <AlertCircle className="w-4 h-4 mr-2" /> Send back
          </Button>
        ) : (
          <Button className="flex-1" disabled={busy} onClick={approveNow}>
            <CheckCircle2 className="w-4 h-4 mr-2" /> {approveLabel(action)}
          </Button>
        )}
      </div>

      {mode === 'idle' && (
        <button
          type="button"
          className="text-[11px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
          onClick={() => open('approve-note')}
        >
          Add a note to the approval
        </button>
      )}
    </div>
  )
}

function VerdictNote({ mode, value, onChange, disabled }: {
  mode: Exclude<Mode, 'idle'>
  value: string
  onChange: (text: string) => void
  disabled: boolean
}) {
  const rejecting = mode === 'reject'
  return (
    <label className="block space-y-1.5">
      <span className="text-xs font-medium">
        {rejecting ? "What's wrong?" : 'Note on the approval (optional)'}
      </span>
      <textarea
        autoFocus
        value={value}
        disabled={disabled}
        required={rejecting}
        maxLength={rejecting ? MAX_REJECT_NOTE_CHARS : MAX_APPROVE_NOTE_CHARS}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        placeholder={rejecting ? 'Say what to fix. The agent reads this first.' : 'Kept on the ticket. The agent is not told.'}
        className="w-full resize-y rounded border border-border bg-background px-2 py-1.5 text-sm outline-none focus:border-primary"
      />
      <span className="block text-[11px] text-muted-foreground">
        {rejecting
          ? 'The ticket goes back to its agent, and the redo starts from your words.'
          : 'Your note stays on the ticket, beside its other notes.'}
      </span>
    </label>
  )
}
