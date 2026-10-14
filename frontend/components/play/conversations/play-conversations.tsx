'use client'

import { type ChangeEvent, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'

import { AutoMascot } from '@/components/mascot/auto-mascot'
import { MascotColumn, PageColumns, PageHeader } from '@/components/play/shell/page-columns'
import { type ConversationListState, useConversationList } from '@/hooks/use-conversation-list'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'
import { useStartChat } from '@/hooks/use-start-chat'
import { useIsPlay } from '@/hooks/use-studio-theme'
import { filterByTitle } from '@/lib/play/conversations'
import { CHAT_HREF } from '@/lib/ticket-links'

import { ConversationList } from './conversation-list'
import styles from './conversations.module.css'

/**
 * PRD-260 §4 — Play's Conversations page (Ember mock 03), ported from Automatos
 * Play's components/conversations/ConversationsScreen.tsx onto the core's
 * history (useConversationList: getChatHistory): every conversation grouped
 * into Today, This week and Earlier, a search on the title, and New chat. A row
 * opens the conversation in the chat (/chat?chatId=<id>). Auto idles in the
 * mascot column, in the same spot as in the chat so nothing jumps.
 *
 * app/conversations/page.tsx sends Classic and Studio to /chat on the server
 * (K10); PlayConversations does the same in the browser when the style leaves
 * Play on this page, and below 1024 px, where Play's pages don't render (D7)
 * and the core chat's own history is the list.
 */

export const CONVERSATIONS_TITLE = 'Conversations'
export const NEW_CHAT = 'New chat'
export const SEARCH_LABEL = 'Search conversations'
export const LOADING_LIST = 'Loading your conversations…'
export const NO_CONVERSATIONS = 'No conversations yet. Say hello to Auto and your first one starts.'
export const LIST_FAILED = "Your conversations didn't load. Try again in a moment."

export interface PlayConversationsProps {
  /** What "today" is; tests pin it. */
  readonly now?: Date
}

/** The page under Play at >= 1024 px; anywhere else, the chat. */
export function PlayConversations({ now }: PlayConversationsProps) {
  const isPlay = useIsPlay()
  const narrow = useIsTabletOrBelow()
  const router = useRouter()
  const elsewhere = !isPlay || narrow
  useEffect(() => {
    if (elsewhere) router.replace(CHAT_HREF)
  }, [elsewhere, router])
  return elsewhere ? null : <ConversationsScreen now={now} />
}

export function ConversationsScreen({ now }: PlayConversationsProps) {
  const list = useConversationList()
  const startChat = useStartChat()
  const [query, setQuery] = useState('')
  const onSearch = (event: ChangeEvent<HTMLInputElement>) => setQuery(event.target.value)
  const newChat = (
    <button type="button" className={styles.newChat} onClick={startChat}>
      {NEW_CHAT}
    </button>
  )
  return (
    <PageColumns
      mascot={
        <MascotColumn>
          <AutoMascot state="idle" />
        </MascotColumn>
      }
    >
      <PageHeader title={CONVERSATIONS_TITLE} action={newChat} />
      <div className={styles.body}>
        <div className={styles.page}>
          <input
            type="search"
            className={styles.search}
            aria-label={SEARCH_LABEL}
            placeholder={SEARCH_LABEL}
            value={query}
            onChange={onSearch}
          />
          <ListBody list={list} query={query} now={now ?? new Date()} />
        </div>
      </div>
    </PageColumns>
  )
}

interface ListBodyProps {
  readonly list: ConversationListState
  readonly query: string
  readonly now: Date
}

function ListBody({ list, query, now }: ListBodyProps) {
  if (list.status === 'loading') {
    return (
      <p className={styles.status} role="status">
        {LOADING_LIST}
      </p>
    )
  }
  if (list.status === 'failed') {
    return (
      <div className={styles.failed} role="alert">
        <p className={styles.error}>
          <span aria-hidden="true">✕</span> {LIST_FAILED}
        </p>
        <button type="button" className={styles.retry} onClick={list.retry}>
          Try again
        </button>
      </div>
    )
  }
  if (list.chats.length === 0) return <p className={styles.status}>{NO_CONVERSATIONS}</p>
  const shown = filterByTitle(list.chats, query)
  if (shown.length === 0) {
    return <p className={styles.status}>No conversation titles match “{query.trim()}”.</p>
  }
  return <ConversationList chats={shown} now={now} />
}
