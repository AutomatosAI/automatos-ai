'use client'

import type { ReactNode } from 'react'

import { type DeliverableRef, deliverablesFromReceipts, mergeDeliverables } from '@/lib/chat/deliverables'
import { activityLines, messageText, provenanceOf, savedTurnError } from '@/lib/chat/message-text'
import { receiptsOf } from '@/lib/chat/receipts'
import type { ChatMessage } from '@/types'

import { DeliverableCard } from './deliverable-card'
import { PlayMarkdown } from './play-markdown'
import { Receipts } from './receipts'
import { ReplyCards } from './reply-cards'
import { SnagBox } from './snag-box'
import { ToolLines } from './tool-lines'
import styles from './chat.module.css'

/**
 * PRD-260 §4 — one reply from Auto (mocks 01, 02 and 08), ported from
 * Automatos Play's components/chat/AutoReply.tsx onto the core's ChatMessage,
 * top to bottom: the "Auto" label, who wrote it when the turn didn't ("Auto ·
 * background", Studio's badge), the lines the platform puts above the answer
 * (a refused write, work not done), what he said before his tools and their
 * progress, his tool lines, the answer as markdown with a cursor while it
 * streams, the Deliverables the turn made, the approvals it raised (the core's
 * cards, T-005c), the core's withdraw offer and ticket cards it carries
 * (reply-cards.tsx, P260-FX-29), what he did (the receipts), a limit reached,
 * then a failure: in the snag box once the turn is over, as a plain line while
 * he answers. A failed turn the server saved is that failure, never an answer.
 * A reply you stopped keeps what it had and says so last (P260-FX-30, as the
 * approved Play app does; hooks/use-stopped-reply.ts keeps it on screen).
 * PRD-264 EUAI-09 (Art. 50(2)): the reply's root carries `data-ai-generated="true"`,
 * the same machine-readable mark the core chat puts on each assistant message.
 */

export const SPEAKER = 'Auto'
/** Under a reply you stopped (the approved Play app's words). */
export const STOPPED_NOTICE = 'You stopped this answer.'

const NO_DELIVERABLES: readonly DeliverableRef[] = []

function Lines({ lines, className }: { readonly lines: readonly string[]; readonly className: string }) {
  return (
    <>
      {lines.map((line, index) => (
        // Plain lines with no id; a reply only ever appends them.
        <p key={`${index}-${line}`} className={className}>
          {line}
        </p>
      ))}
    </>
  )
}

function Answer({ text, streaming }: { readonly text: string; readonly streaming: boolean }) {
  if (!text && !streaming) return null
  return (
    <div className={styles.answer}>
      {text && <PlayMarkdown text={text} />}
      {streaming && <span className={styles.cursor} aria-hidden="true" />}
    </div>
  )
}

function ErrorLine({ message }: { readonly message: string }) {
  return (
    <p className={styles.error}>
      <span className={styles.mark} aria-hidden="true">
        ✕
      </span>
      {message}
    </p>
  )
}

interface FailureProps {
  readonly message: string | null
  readonly streaming: boolean
  readonly onRetry?: () => void
  readonly hidden: boolean
}

/** The turn's failure in words: the snag box once it is over, a line while Auto still answers. */
function Failure({ message, streaming, onRetry, hidden }: FailureProps) {
  if (!message || hidden) return null
  return streaming ? <ErrorLine message={message} /> : <SnagBox lines={[message]} onRetry={onRetry} />
}

export interface AutoReplyProps {
  readonly message: ChatMessage
  /** This reply is the one being written now. */
  readonly streaming: boolean
  /** The Deliverables this reply made live (from the chat's data parts); its receipts add theirs. */
  readonly deliverables?: readonly DeliverableRef[]
  /** Try again on this reply's snag box: only the newest reply gets it. */
  readonly onRetry?: () => void
  /** A spent trial blocked this turn: the core's banner says so instead of the snag box. */
  readonly snagHidden?: boolean
  /** The approval cards this reply raised (approval-cards.tsx). */
  readonly approvals?: ReactNode
  /** You pressed Stop while Auto wrote this reply. */
  readonly stopped?: boolean
}

export function AutoReply({
  message,
  streaming,
  deliverables = NO_DELIVERABLES,
  onRetry,
  snagHidden = false,
  approvals,
  stopped = false,
}: AutoReplyProps) {
  const receipts = receiptsOf(message)
  const made = mergeDeliverables(deliverables, deliverablesFromReceipts(receipts))
  const notices = message.limitReached?.message ? [message.limitReached.message] : []
  const provenance = provenanceOf(message)
  const savedError = savedTurnError(message)
  return (
    <article
      className={styles.reply}
      aria-label={`${SPEAKER} said`}
      aria-busy={streaming}
      data-ai-generated="true"
    >
      <h2 className={styles.speaker}>{SPEAKER}</h2>
      <Lines lines={provenance ? [provenance] : []} className={styles.notice} />
      <Lines lines={message.receiptsAbove ?? []} className={styles.above} />
      <Lines lines={activityLines(message)} className={styles.activity} />
      <ToolLines calls={message.toolCalls} streaming={streaming} />
      {savedError === null && <Answer text={messageText(message)} streaming={streaming} />}
      {made.map((deliverable) => (
        <DeliverableCard key={deliverable.id} deliverable={deliverable} />
      ))}
      {approvals}
      <ReplyCards message={message} />
      <Receipts receipts={receipts} />
      <Lines lines={notices} className={styles.notice} />
      <Failure
        message={message.error?.message ?? savedError}
        streaming={streaming}
        onRetry={onRetry}
        hidden={snagHidden}
      />
      {stopped && <p className={styles.notice}>{STOPPED_NOTICE}</p>}
    </article>
  )
}
