'use client'

/**
 * PRD-252 R2 — the bar over the chat while it discusses a ticket.
 *
 * Discuss on a ticket opens `/chat?ticket=<id>`: a new conversation with the
 * ticket in its page context, so Auto reads the ticket before it answers. The
 * bar says which ticket is being discussed and carries "Update ticket and
 * re-queue". On a mission's ticket (D4) it points at the mission instead,
 * where the plan is decided.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { MessagesSquare, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useBoardTask } from '@/hooks/use-board-tasks'
import { CHAT_ROUTE, discussionLabel, discussionOf, type Discussion } from '@/lib/discussion'
import { missionHref, ticketHref } from '@/lib/ticket-links'
import { useChatSessionStore } from '@/stores/chat-session-store'
import { useDiscussionStore } from '@/stores/discussion-store'
import { RebriefDialog } from './rebrief-dialog'

const BAR_CLASS =
  'fixed inset-x-0 top-16 z-40 mx-auto flex w-fit max-w-[calc(100%-2rem)] flex-wrap items-center gap-x-3 gap-y-1 ' +
  'rounded-lg border border-border bg-background/95 px-3 py-2 text-xs shadow-lg backdrop-blur'

export function DiscussionBar() {
  const { discussion, missing } = useDiscussionFromLink()
  const end = useEndDiscussion()
  useEndOnAnotherConversation(end)
  const [rebriefing, setRebriefing] = useState(false)
  if (missing) {
    return (
      <div role="region" aria-label="Discussion" className={BAR_CLASS} data-testid="discussion-bar">
        <p>Ticket {missing} could not be opened here. It may be in another workspace.</p>
        <EndButton onEnd={end} />
      </div>
    )
  }
  if (!discussion) return null
  const label = discussionLabel(discussion)
  return (
    <div role="region" aria-label="Discussion" className={BAR_CLASS} data-testid="discussion-bar">
      <MessagesSquare className="h-4 w-4 shrink-0 text-primary" />
      <p className="min-w-0 max-w-[28rem] truncate">
        Discussing{' '}
        <Link href={ticketHref(discussion.ticketId) as any} className="font-medium underline-offset-2 hover:underline">
          {label}
        </Link>{' '}
        · {discussion.title}
      </p>
      {discussion.missionId ? (
        <Link href={missionHref(discussion.missionId) as any} className="text-primary underline-offset-2 hover:underline">
          Decide on the mission →
        </Link>
      ) : (
        <Button size="sm" className="h-7 text-xs" onClick={() => setRebriefing(true)}>
          Update ticket and re-queue
        </Button>
      )}
      <EndButton onEnd={end} />
      {discussion.agentName && !discussion.missionId && (
        <span className="basis-full text-muted-foreground">
          You are talking to Auto. To ask {discussion.agentName} instead, pick them in the message box.
        </span>
      )}
      {rebriefing && <RebriefDialog discussion={discussion} open onOpenChange={setRebriefing} onDone={end} />}
    </div>
  )
}

function EndButton({ onEnd }: { onEnd: () => void }) {
  return (
    <button
      type="button"
      aria-label="End the discussion"
      onClick={onEnd}
      className="rounded p-0.5 text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
    >
      <X className="h-3.5 w-3.5" />
    </button>
  )
}

/** `/chat?ticket=<id>` without `repo` (with it, the link opens a session's canvas)
 * discusses that ticket: once it has loaded, the discussion starts in a new
 * conversation. `missing` names a ticket that could not be loaded. */
function useDiscussionFromLink(): { discussion: Discussion | null; missing: string | null } {
  const params = useSearchParams()
  const ticketId = params?.get('repo') ? null : (params?.get('ticket') || '').replace(/[^0-9]/g, '') || null
  const { data: task, isError } = useBoardTask(ticketId)
  const hydrated = useChatSessionStore((s) => s.hydrated)
  const newDraft = useChatSessionStore((s) => s.newDraft)
  const { discussion, start, end } = useDiscussionStore()
  const opened = useRef<string | null>(null)
  useEffect(() => {
    if (!ticketId) {
      opened.current = null
      if (discussion) end()
      return
    }
    if (!task || !hydrated || opened.current === ticketId) return
    opened.current = ticketId
    start(discussionOf(task))
    newDraft()
  }, [ticketId, task, hydrated, discussion, start, end, newDraft])
  return { discussion: ticketId ? discussion : null, missing: ticketId && isError ? ticketId : null }
}

/** The discussion is its own conversation: the draft it opened, then the
 * conversation that draft becomes. That one is new: not already a tab, and not
 * yet named (the history panel names the chat it opens). Opening any other
 * conversation ends the discussion, so the ticket never reaches an unrelated chat. */
function useEndOnAnotherConversation(onLeave: () => void): void {
  const { discussion, chatId, adopt } = useDiscussionStore()
  const session = useChatSessionStore((s) => s.session)
  const before = useRef(session)
  useEffect(() => {
    const previous = before.current
    before.current = session
    const active = session.activeChatId
    if (!discussion || previous.activeChatId === active || active === chatId) return
    const draftNamed = chatId === null && previous.activeChatId === null && active !== null
      && !previous.openChatIds.includes(active) && !session.titles[active]
    if (draftNamed) adopt(active)
    else onLeave()
  }, [session, discussion, chatId, adopt, onLeave])
}

/** Ends the discussion: the ticket leaves the chat's context and its link. */
function useEndDiscussion(): () => void {
  const router = useRouter()
  const params = useSearchParams()
  const end = useDiscussionStore((s) => s.end)
  return useCallback(() => {
    end()
    const next = new URLSearchParams(params?.toString() ?? '')
    next.delete('ticket')
    const query = next.toString()
    router.replace(query ? `${CHAT_ROUTE}?${query}` : CHAT_ROUTE)
  }, [end, params, router])
}
