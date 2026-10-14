'use client'

import type { ReactNode } from 'react'

import { receiptsOf } from '@/lib/chat/receipts'
import type { ChatMessage } from '@/types'

import { DeliverableCard } from './deliverable-card'
import { type DeliverableRef, deliverablesFromReceipts, mergeDeliverables } from './deliverables'
import { activityLines, messageText } from './message-text'
import { PlayMarkdown } from './play-markdown'
import { Receipts } from './receipts'
import { SnagBox } from './snag-box'
import { ToolLines } from './tool-lines'
import styles from './chat.module.css'

/**
 * PRD-260 §4 — one reply from Auto (mocks 01, 02 and 08), ported from
 * Automatos Play's components/chat/AutoReply.tsx onto the core's ChatMessage,
 * top to bottom: the "Auto" label, the lines the platform puts above the answer
 * (a refused write, work not done), what he said before his tools and their
 * progress, his tool lines, the answer as markdown with a cursor while it
 * streams, the Deliverables the turn made, the approvals it raised (the core's
 * cards, T-005c), what he did (the receipts), a limit reached, then a failure:
 * in the snag box once the turn is over, as a plain line while he answers.
 */

export const SPEAKER = 'Auto'

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
  readonly message: string | undefined
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
}

export function AutoReply({
  message,
  streaming,
  deliverables = NO_DELIVERABLES,
  onRetry,
  snagHidden = false,
  approvals,
}: AutoReplyProps) {
  const receipts = receiptsOf(message)
  const made = mergeDeliverables(deliverables, deliverablesFromReceipts(receipts))
  const notices = message.limitReached?.message ? [message.limitReached.message] : []
  return (
    <article className={styles.reply} aria-label={`${SPEAKER} said`} aria-busy={streaming}>
      <h2 className={styles.speaker}>{SPEAKER}</h2>
      <Lines lines={message.receiptsAbove ?? []} className={styles.above} />
      <Lines lines={activityLines(message)} className={styles.activity} />
      <ToolLines calls={message.toolCalls} streaming={streaming} />
      <Answer text={messageText(message)} streaming={streaming} />
      {made.map((deliverable) => (
        <DeliverableCard key={deliverable.id} deliverable={deliverable} />
      ))}
      {approvals}
      <Receipts receipts={receipts} />
      <Lines lines={notices} className={styles.notice} />
      <Failure message={message.error?.message} streaming={streaming} onRetry={onRetry} hidden={snagHidden} />
    </article>
  )
}
