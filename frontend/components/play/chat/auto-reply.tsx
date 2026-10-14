'use client'

import type { ChatMessage } from '@/types'

import { activityLines, messageText } from './message-text'
import { PlayMarkdown } from './play-markdown'
import styles from './chat.module.css'

/**
 * PRD-260 §4 — one reply from Auto (mocks 01 and 02), ported from Automatos
 * Play's components/chat/AutoReply.tsx onto the core's ChatMessage, top to
 * bottom: the "Auto" label, the lines the platform puts above the answer (a
 * refused write, work not done), what he said before his tools and their
 * progress, the answer as markdown with a cursor while it streams, then a limit
 * reached and an error in words.
 */

export const SPEAKER = 'Auto'

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

export interface AutoReplyProps {
  readonly message: ChatMessage
  /** This reply is the one being written now. */
  readonly streaming: boolean
}

export function AutoReply({ message, streaming }: AutoReplyProps) {
  const notices = message.limitReached?.message ? [message.limitReached.message] : []
  return (
    <article className={styles.reply} aria-label={`${SPEAKER} said`} aria-busy={streaming}>
      <h2 className={styles.speaker}>{SPEAKER}</h2>
      <Lines lines={message.receiptsAbove ?? []} className={styles.above} />
      <Lines lines={activityLines(message)} className={styles.activity} />
      <Answer text={messageText(message)} streaming={streaming} />
      <Lines lines={notices} className={styles.notice} />
      {message.error?.message && <ErrorLine message={message.error.message} />}
    </article>
  )
}
