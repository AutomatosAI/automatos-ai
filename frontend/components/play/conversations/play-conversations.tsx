'use client'

import { type ChangeEvent, useEffect, useMemo, useState } from 'react'
import {
  CHAT_HREF,
  type HostConversationListState,
  useConversationList,
  useHostRouter,
  useStartChat,
} from '@automatos/theme-sdk'
// PRD-261 A-14c seam: the SDK has no name for the theme layout yet (sdk-surface-map.md calls it
// host-only, but ThemeSlotRoute renders a route at any width and leaves the phone to the route).
import { useThemeLayout } from '@/hooks/use-studio-theme'

import { type ConversationSummary, filterByTitle, summaryOfHostChat } from '../../../lib/play/conversations'
import { AutoMascot } from '../../mascot/auto-mascot'
import { MascotColumn, PageColumns } from '../shell/page-columns'
import { ConversationList } from './conversation-list'
import styles from './conversations.module.css'

/**
 * PRD-260 §4 — Play's Conversations page (Ember mock 03), ported from Automatos
 * Play's components/conversations/ConversationsScreen.tsx onto the core's
 * history (the SDK's useConversationList: getChatHistory): every conversation grouped
 * into Today, This week and Earlier, a search on the title, and New chat. A row
 * opens the conversation in the chat (/chat?chatId=<id>). The title and New
 * chat head the list's own column, over the search box, as in the Play app
 * (P260-FX-14). Auto idles in the mascot column, in the same spot as in the
 * chat so nothing jumps.
 *
 * One status line sits under the search box for the page's life, so a screen
 * reader hears every change to it (P260-FX-24, M22): loading, no conversations
 * yet and no title matching are shown; while titles match, the count is said
 * to screen readers only, since the list itself shows it.
 *
 * app/conversations/page.tsx sends Classic and Studio to /chat on the server
 * (K10); PlayConversations does the same in the browser when the style leaves
 * Play on this page, and below 1024 px, where Play's pages don't render (D7)
 * and the core chat's own history is the list. Until the width is known (the
 * server, hydration) it shows nothing and sends nowhere, so a phone never asks
 * for the list (P260-FX-25, M69).
 */

export const CONVERSATIONS_TITLE = 'Conversations'
export const NEW_CHAT = 'New chat'
export const SEARCH_LABEL = 'Search conversations'
export const LOADING_LIST = 'Loading your conversations…'
export const NO_CONVERSATIONS = 'No conversations yet. Say hello to Auto and your first one starts.'
export const LIST_FAILED = "Your conversations didn't load. Try again in a moment."
/** sr-only: the Tailwind utility the core uses for words that are only for screen readers. */
const SCREEN_READER_ONLY = 'sr-only'

/** The status line while titles match: how many, for screen readers only. */
export function matchLine(count: number, query: string): string {
  const needle = query.trim()
  const one = count === 1
  if (!needle) return one ? '1 conversation.' : `${count} conversations.`
  return one ? `1 conversation matches “${needle}”.` : `${count} conversations match “${needle}”.`
}

/** The status line when no title matches the search: shown on the page. */
export function noMatchLine(query: string): string {
  return `No conversation titles match “${query.trim()}”.`
}

export interface PlayConversationsProps {
  /** What "today" is; tests pin it. */
  readonly now?: Date
}

/** The page under Play at >= 1024 px; anywhere else, the chat. */
export function PlayConversations({ now }: PlayConversationsProps) {
  const layout = useThemeLayout()
  const router = useHostRouter()
  const elsewhere = layout === 'fallback'
  useEffect(() => {
    if (elsewhere) router.replace(CHAT_HREF)
  }, [elsewhere, router])
  return layout === 'theme' ? <ConversationsScreen now={now} /> : null
}

export function ConversationsScreen({ now }: PlayConversationsProps) {
  const list = useConversationList()
  const { startChat, ready } = useStartChat()
  const [query, setQuery] = useState('')
  const onSearch = (event: ChangeEvent<HTMLInputElement>) => setQuery(event.target.value)
  const chats = useMemo(() => (list.status === 'loaded' ? list.chats.map(summaryOfHostChat) : []), [list])
  const shown = filterByTitle(chats, query)
  const line = statusLine(list, shown.length, query)
  return (
    <PageColumns
      mascot={
        <MascotColumn>
          <AutoMascot state="idle" />
        </MascotColumn>
      }
    >
      <div className={styles.body}>
        <div className={styles.page}>
          <div className={styles.header}>
            <h1 className={styles.title}>{CONVERSATIONS_TITLE}</h1>
            <button type="button" className={styles.newChat} onClick={startChat} disabled={!ready}>
              {NEW_CHAT}
            </button>
          </div>
          <input
            type="search"
            className={styles.search}
            aria-label={SEARCH_LABEL}
            placeholder={SEARCH_LABEL}
            value={query}
            onChange={onSearch}
          />
          <p className={line.hidden ? SCREEN_READER_ONLY : styles.status} role="status">
            {line.text}
          </p>
          <ListBody list={list} shown={shown} now={now ?? new Date()} />
        </div>
      </div>
    </PageColumns>
  )
}

interface StatusLine {
  readonly text: string
  /** Said to screen readers only: the list on the page already shows it. */
  readonly hidden: boolean
}

/** What the one status line under the search box says for the list and the search. */
function statusLine(list: HostConversationListState, shown: number, query: string): StatusLine {
  if (list.status === 'loading') return { text: LOADING_LIST, hidden: false }
  // The failure is the alert below; the line stays, empty, for the next change.
  if (list.status === 'failed') return { text: '', hidden: true }
  if (list.chats.length === 0) return { text: NO_CONVERSATIONS, hidden: false }
  if (shown === 0) return { text: noMatchLine(query), hidden: false }
  return { text: matchLine(shown, query), hidden: true }
}

interface ListBodyProps {
  readonly list: HostConversationListState
  /** The conversations whose titles match the search. */
  readonly shown: readonly ConversationSummary[]
  readonly now: Date
}

/** The failed state's alert, or the matching conversations; the status line says the rest. */
function ListBody({ list, shown, now }: ListBodyProps) {
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
  return shown.length > 0 ? <ConversationList chats={shown} now={now} /> : null
}
