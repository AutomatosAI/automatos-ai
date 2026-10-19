import { useId } from 'react'
import Link from 'next/link'

import {
  type ConversationGroup,
  type ConversationSummary,
  conversationTitle,
  groupConversations,
  lastActive,
  previewLine,
  relativeTime,
} from '@/lib/play/conversations'
import { chatHref } from '@/lib/ticket-links'

import styles from './conversations.module.css'

/**
 * PRD-260 §4 — the grouped list (Ember mock 03), ported from Automatos Play's
 * components/conversations/ConversationList.tsx: Today, This week, Earlier,
 * each a region named by its heading; a row is the title, the time and the
 * newest line, and opens the conversation in the chat (chatHref: the deep link
 * the core chat page and Play's chat both honour).
 */

export interface ConversationListProps {
  readonly chats: readonly ConversationSummary[]
  readonly now: Date
}

export function ConversationList({ chats, now }: ConversationListProps) {
  return (
    <>
      {groupConversations(chats, now).map((group) => (
        <Group key={group.label} group={group} now={now} />
      ))}
    </>
  )
}

function Group({ group, now }: { readonly group: ConversationGroup; readonly now: Date }) {
  const headingId = useId()
  return (
    <section className={styles.group} aria-labelledby={headingId}>
      <h2 id={headingId} className={styles.groupLabel}>
        {group.label}
      </h2>
      <ul className={styles.list}>
        {group.chats.map((chat) => (
          <li key={chat.id}>
            <Row chat={chat} now={now} />
          </li>
        ))}
      </ul>
    </section>
  )
}

function Row({ chat, now }: { readonly chat: ConversationSummary; readonly now: Date }) {
  const active = lastActive(chat)
  const preview = previewLine(chat)
  return (
    <Link href={chatHref(chat.id)} className={styles.row}>
      <span className={styles.rowTop}>
        <span className={styles.rowTitle}>{conversationTitle(chat)}</span>
        {active && (
          <time className={styles.time} dateTime={active.toISOString()}>
            {relativeTime(active, now)}
          </time>
        )}
      </span>
      {preview && <span className={styles.preview}>{preview}</span>}
    </Link>
  )
}
