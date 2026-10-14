'use client'

import { type ReactNode, type RefObject, useEffect, useRef } from 'react'

import type { ChatMessage } from '@/types'

import { AutoReply } from './auto-reply'
import type { DeliverableRef } from './deliverables'
import { messageText } from './message-text'
import styles from './chat.module.css'

/**
 * PRD-260 §4 — the conversation (mock 01), ported from Automatos Play's
 * components/chat/MessageList.tsx: your messages outlined on the right, Auto's
 * replies on the left. It scrolls inside Play's frame (the frame is
 * viewport-bound), keeping the newest line in view while Auto answers unless
 * you scrolled up to read; a new message follows again. What follows the
 * conversation (a turn that failed before a reply began, a Mission Auto
 * suggests) sits under the list, inside the scroll.
 */

/** How close to the bottom counts as "reading the newest line". */
export const FOLLOW_SLACK_PX = 120

function nearBottom(scroller: HTMLElement): boolean {
  return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= FOLLOW_SLACK_PX
}

/** jsdom has no scrollIntoView, hence the check. */
function useFollow(messages: readonly ChatMessage[]): {
  scrollerRef: RefObject<HTMLDivElement>
  endRef: RefObject<HTMLDivElement>
} {
  const scrollerRef = useRef<HTMLDivElement>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const following = useRef(true)
  const count = messages.length

  useEffect(() => {
    const scroller = scrollerRef.current
    if (!scroller) return
    const onScroll = () => {
      following.current = nearBottom(scroller)
    }
    scroller.addEventListener('scroll', onScroll, { passive: true })
    return () => scroller.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    following.current = true
  }, [count])

  useEffect(() => {
    const end = endRef.current
    if (following.current && end && typeof end.scrollIntoView === 'function') end.scrollIntoView({ block: 'end' })
  }, [messages])

  return { scrollerRef, endRef }
}

export interface MessageListProps {
  readonly messages: readonly ChatMessage[]
  /** Auto is answering: the newest reply is the one being written. */
  readonly streaming: boolean
  /** The Deliverables a reply made live, by its id. */
  readonly deliverablesOf?: (replyId: string) => readonly DeliverableRef[]
  /** Try again, for the newest reply's snag box. */
  readonly onRetry?: () => void
  /** A spent trial blocked the newest turn: its reply shows no snag box. */
  readonly trialBlocked?: boolean
  /** Shown under the conversation, inside the scroll. */
  readonly after?: ReactNode
}

export function MessageList({ messages, streaming, deliverablesOf, onRetry, trialBlocked = false, after }: MessageListProps) {
  const { scrollerRef, endRef } = useFollow(messages)
  const newest = messages[messages.length - 1]?.id
  return (
    <div ref={scrollerRef} className={styles.messages}>
      <ol className={styles.messageList} aria-label="Conversation">
        {messages.map((message) => (
          <li key={message.id} className={message.role === 'user' ? styles.userItem : styles.replyItem}>
            {message.role === 'user' ? (
              <p className={styles.bubble}>{messageText(message)}</p>
            ) : (
              <AutoReply
                message={message}
                streaming={streaming && message.id === newest}
                deliverables={deliverablesOf?.(message.id)}
                onRetry={message.id === newest ? onRetry : undefined}
                snagHidden={trialBlocked && message.id === newest}
              />
            )}
          </li>
        ))}
      </ol>
      {after}
      <div ref={endRef} />
    </div>
  )
}
