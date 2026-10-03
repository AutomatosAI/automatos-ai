'use client'

/**
 * PRD-252 R2 — "Update ticket and re-queue": the brief agreed in the chat goes
 * onto the ticket as its brief, and the ticket goes back to its agent. The
 * dialog opens on the brief Auto proposed in its last reply, for the owner to
 * check and edit; the old brief and the last draft stay on the ticket.
 */

import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { useRebriefTask } from '@/hooks/use-board-tasks'
import { getChatMessages } from '@/lib/chat/api'
import { MAX_BRIEF_CHARS, discussionLabel, proposedBrief, type Discussion } from '@/lib/discussion'
import { useChatSessionStore } from '@/stores/chat-session-store'

interface RebriefDialogProps {
  discussion: Discussion
  open: boolean
  onOpenChange: (open: boolean) => void
  /** The ticket was updated and re-queued. */
  onDone: () => void
}

export function RebriefDialog({ discussion, open, onOpenChange, onDone }: RebriefDialogProps) {
  const { brief, setBrief, unread } = useProposedBrief(open)
  const rebrief = useRebriefTask()
  const label = discussionLabel(discussion)
  const agent = discussion.agentName ?? 'its agent'
  const text = brief.trim()

  const save = () => {
    rebrief.mutate({ taskId: discussion.ticketId, brief: text }, {
      onSuccess: () => {
        toast.success(`${label[0].toUpperCase()}${label.slice(1)} is back with ${agent}, working from the brief you agreed.`)
        onDone()
      },
      onError: (err) => toast.error(err instanceof Error && err.message ? err.message : 'Could not update the ticket'),
    })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg" data-testid="rebrief-dialog">
        <DialogHeader>
          <DialogTitle>Update {label} and re-queue</DialogTitle>
          <DialogDescription>
            This brief replaces the ticket&apos;s, and {agent} redoes the ticket from it. The old brief and the last
            draft stay on the ticket.
          </DialogDescription>
        </DialogHeader>
        <label className="block space-y-1.5">
          <span className="text-xs font-medium">The agreed brief</span>
          <textarea
            value={brief}
            onChange={(e) => setBrief(e.target.value)}
            rows={8}
            maxLength={MAX_BRIEF_CHARS}
            disabled={rebrief.isLoading}
            placeholder={unread ? 'Could not read the conversation. Paste or write the brief you agreed.' : 'The brief you agreed with Auto.'}
            className="w-full resize-y rounded border border-border bg-background px-2 py-1.5 text-sm outline-none focus:border-primary"
          />
        </label>
        <DialogFooter>
          <Button variant="outline" disabled={rebrief.isLoading} onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button disabled={!text || rebrief.isLoading} onClick={save}>
            Update and re-queue
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** The brief Auto proposed in the conversation on screen, read when the dialog opens.
 * `unread` says the conversation could not be read, so the owner writes the brief. */
function useProposedBrief(open: boolean) {
  const chatId = useChatSessionStore((s) => s.session.activeChatId)
  const [brief, setBrief] = useState('')
  const [unread, setUnread] = useState(false)
  useEffect(() => {
    if (!open || !chatId) return
    let cancelled = false
    getChatMessages(chatId)
      .then((messages) => {
        if (!cancelled) setBrief((typed) => typed || proposedBrief(messages))
      })
      .catch(() => {
        if (!cancelled) setUnread(true)
      })
    return () => {
      cancelled = true
    }
  }, [open, chatId])
  return { brief, setBrief, unread }
}
